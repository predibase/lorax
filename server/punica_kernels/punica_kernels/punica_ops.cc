#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>

#include "bgmv/bgmv_config.h"
#include "flashinfer_adapter/flashinfer_config.h"
#include "rms_norm/rms_norm.h"
#include "sgmv/sgmv.h"
#include "sgmv_flashinfer/sgmv_config.h"

namespace {

//====== utils ======

inline void check_shape(const torch::Tensor& a, const torch::Tensor& b,
                        const char* a_name, const char* b_name) {
  TORCH_CHECK(a.dim() == b.dim(), a_name, ".dim() != ", b_name, ".dim(). ",
              a.dim(), " vs ", b.dim());
  for (int i = 0; i < a.dim(); ++i) {
    TORCH_CHECK(a.size(i) == b.size(i), a_name, ".size(", i, ") != ", b_name,
                ".size(", i, ")");
  }
}

inline constexpr uint32_t pack_u16(uint16_t a, uint16_t b) {
  return (uint32_t(a) << 16) | uint32_t(b);
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

#define CHECK_DIM(d, x) \
  TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")

#define CHECK_SHAPE(a, b) check_shape(a, b, #a, #b)

#define CHECK_EQ(a, b) \
  TORCH_CHECK(a == b, "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

//====== dispatch pytorch dtype ======

#define _DISPATCH_SWITCH(scalar_type, ...) \
  [&]() -> bool {                          \
    switch (scalar_type) {                 \
      __VA_ARGS__                          \
      default:                             \
        return false;                      \
    }                                      \
  }()

#define _DISPATCH_CASE(enum_type, c_type_, ...) \
  case enum_type: {                             \
    using c_type = c_type_;                     \
    return __VA_ARGS__();                       \
  }

#define _DISPATCH_CASES(...)                                 \
  _DISPATCH_CASE(at::ScalarType::Half, nv_half, __VA_ARGS__) \
  _DISPATCH_CASE(at::ScalarType::BFloat16, nv_bfloat16, __VA_ARGS__)

#define DISPATCH_TORCH_DTYPE(scalar_type, ...) \
  _DISPATCH_SWITCH(scalar_type, _DISPATCH_CASES(__VA_ARGS__))

//====== flashinfer ======

void batch_decode(torch::Tensor o, torch::Tensor q, torch::Tensor kv_data,
                  torch::Tensor kv_indptr, torch::Tensor kv_indicies,
                  torch::Tensor last_page_offset, int layer_idx) {
  CHECK_INPUT(o);
  CHECK_INPUT(q);
  CHECK_INPUT(kv_data);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(kv_indicies);
  CHECK_INPUT(last_page_offset);

  CHECK_DIM(3, o);                 // [B, N, D]
  CHECK_DIM(3, q);                 // [B, N, D]
  CHECK_DIM(6, kv_data);           // [None, L, 2, N, P, D]
  CHECK_DIM(1, kv_indptr);         // [B+1]
  CHECK_DIM(1, kv_indicies);       // [None]
  CHECK_DIM(1, last_page_offset);  // [B]

  int num_layers = static_cast<int>(kv_data.size(1));
  int num_kv_heads = static_cast<int>(kv_data.size(3));
  int page_size = static_cast<int>(kv_data.size(4));
  int head_dim = static_cast<int>(kv_data.size(5));
  int batch_size = static_cast<int>(o.size(0));
  int num_qo_heads = static_cast<int>(o.size(1));
  CHECK_SHAPE(o, q);
  CHECK_EQ(kv_indptr.size(0), batch_size + 1);
  CHECK_EQ(last_page_offset.size(0), batch_size);

  bool ok = DISPATCH_TORCH_DTYPE(q.scalar_type(), [&] {
    FlashInferBatchDecodeKernel<c_type>(
        static_cast<c_type*>(o.data_ptr()), static_cast<c_type*>(q.data_ptr()),
        static_cast<c_type*>(kv_data.data_ptr()), kv_indptr.data_ptr<int32_t>(),
        kv_indicies.data_ptr<int32_t>(), last_page_offset.data_ptr<int32_t>(),
        head_dim, num_layers, layer_idx, num_qo_heads, num_kv_heads, page_size,
        batch_size);
    return true;
  });
  TORCH_CHECK(ok, "No suitable kernel.", " dtype=", q.scalar_type(),
              " head_dim=", head_dim);

#undef CASE
}

void init_kv(torch::Tensor kv_data, torch::Tensor kv_indptr,
             torch::Tensor kv_indicies, torch::Tensor last_page_offset,
             torch::Tensor k, torch::Tensor v, torch::Tensor seqlen_indptr,
             int layer_idx) {
  CHECK_INPUT(kv_data);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(kv_indicies);
  CHECK_INPUT(last_page_offset);
  CHECK_INPUT(k);
  CHECK_INPUT(v);
  CHECK_INPUT(seqlen_indptr);

  CHECK_DIM(6, kv_data);           // [None, L, 2, N, P, D]
  CHECK_DIM(1, kv_indptr);         // [B+1]
  CHECK_DIM(1, kv_indicies);       // [None]
  CHECK_DIM(1, last_page_offset);  // [B]
  CHECK_DIM(3, k);                 // [sum(seqlen_i), N, D]
  CHECK_DIM(3, v);                 // [sum(seqlen_i), N, D]
  CHECK_DIM(1, seqlen_indptr);     // [B+1]

  int num_layers = static_cast<int>(kv_data.size(1));
  int num_kv_heads = static_cast<int>(kv_data.size(3));
  int page_size = static_cast<int>(kv_data.size(4));
  int head_dim = static_cast<int>(kv_data.size(5));
  int batch_size = static_cast<int>(last_page_offset.size(0));
  CHECK_EQ(kv_indptr.size(0), batch_size + 1);
  CHECK_EQ(seqlen_indptr.size(0), batch_size + 1);

#define CASE(dim, _)                                                           \
  case dim:                                                                    \
    FlashInferInitKvKernel<dim, c_type>(                                       \
        static_cast<c_type*>(kv_data.data_ptr()),                              \
        kv_indptr.data_ptr<int32_t>(), kv_indicies.data_ptr<int32_t>(),        \
        last_page_offset.data_ptr<int32_t>(),                                  \
        static_cast<c_type*>(k.data_ptr()),                                    \
        static_cast<c_type*>(v.data_ptr()), seqlen_indptr.data_ptr<int32_t>(), \
        num_layers, layer_idx, num_kv_heads, page_size, batch_size);           \
    return true;

  bool ok = DISPATCH_TORCH_DTYPE(k.scalar_type(), [&] {
    switch (head_dim) {
      FOR_FlashInferBatchDecode_D(CASE);
      default:
        return false;
    }
  });
  TORCH_CHECK(ok, "No suitable kernel.", " dtype=", k.scalar_type(),
              " head_dim=", head_dim);
#undef CASE
}

void append_kv(torch::Tensor kv_data, torch::Tensor kv_indptr,
               torch::Tensor kv_indicies, torch::Tensor last_page_offset,
               torch::Tensor k, torch::Tensor v, int layer_idx) {
  CHECK_INPUT(kv_data);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(kv_indicies);
  CHECK_INPUT(last_page_offset);
  CHECK_INPUT(k);
  CHECK_INPUT(v);

  CHECK_DIM(6, kv_data);           // [None, L, 2, N, P, D]
  CHECK_DIM(1, kv_indptr);         // [B+1]
  CHECK_DIM(1, kv_indicies);       // [None]
  CHECK_DIM(1, last_page_offset);  // [B]
  CHECK_DIM(3, k);                 // [B, N, D]
  CHECK_DIM(3, v);                 // [B, N, D]

  int num_layers = static_cast<int>(kv_data.size(1));
  int num_kv_heads = static_cast<int>(kv_data.size(3));
  int page_size = static_cast<int>(kv_data.size(4));
  int head_dim = static_cast<int>(kv_data.size(5));
  int batch_size = static_cast<int>(k.size(0));
  CHECK_EQ(kv_indptr.size(0), batch_size + 1);
  CHECK_EQ(last_page_offset.size(0), batch_size);
  CHECK_SHAPE(k, v);

#define CASE(dim, _)                                                    \
  case dim:                                                             \
    FlashInferAppendKvKernel<dim, c_type>(                              \
        static_cast<c_type*>(kv_data.data_ptr()),                       \
        kv_indptr.data_ptr<int32_t>(), kv_indicies.data_ptr<int32_t>(), \
        last_page_offset.data_ptr<int32_t>(),                           \
        static_cast<c_type*>(k.data_ptr()),                             \
        static_cast<c_type*>(v.data_ptr()), num_layers, layer_idx,      \
        num_kv_heads, page_size, batch_size);                           \
    return true;

  bool ok = DISPATCH_TORCH_DTYPE(k.scalar_type(), [&] {
    switch (head_dim) {
      FOR_FlashInferBatchDecode_D(CASE);
      default:
        return false;
    }
  });
  TORCH_CHECK(ok, "No suitable kernel.", " dtype=", k.scalar_type(),
              " head_dim=", head_dim);
#undef CASE
}

//====== bgmv ======

template <typename T>
inline bool launch_bgmv_kernel(T* Y, const T* X, const T* W,
                               const int64_t* lora_indices,
                               uint16_t in_features, uint16_t out_features,
                               int64_t batch_size, int64_t num_layers,
                               int64_t layer_idx, float scale) {
  switch (pack_u16(in_features, out_features)) {
#define CASE_ONESIDE(_T, feat_in, feat_out)                           \
  case pack_u16(feat_in, feat_out):                                   \
    bgmv_kernel<feat_in, feat_out>(Y, X, W, lora_indices, batch_size, \
                                   num_layers, layer_idx, scale);     \
    break;
#define CASE(_T, narrow, wide)  \
  CASE_ONESIDE(T, narrow, wide) \
  CASE_ONESIDE(T, wide, narrow)

    FOR_BGMV_WIDE_NARROW(CASE, _)
#undef CASE
#undef CASE_ONESIDE
    default:
      return false;
  }

  return true;
}

void dispatch_bgmv(torch::Tensor y, torch::Tensor x, torch::Tensor w,
                   torch::Tensor indicies, int64_t layer_idx, float scale) {
  CHECK_INPUT(y);
  CHECK_INPUT(x);
  CHECK_INPUT(w);
  CHECK_INPUT(indicies);

  CHECK_DIM(2, y);
  CHECK_DIM(2, x);
  CHECK_DIM(4, w);
  CHECK_DIM(1, indicies);

  int64_t B = x.size(0);
  int64_t h_in = x.size(1);
  int64_t h_out = y.size(1);
  int64_t num_layers = w.size(1);
  CHECK_EQ(w.size(3), h_in);
  CHECK_EQ(w.size(2), h_out);
  CHECK_EQ(indicies.size(0), x.size(0));
  CHECK_EQ(y.size(0), x.size(0));
  bool ok = false;
  if (h_in < 65536 && h_out < 65536) {
    switch (x.scalar_type()) {
      case at::ScalarType::Half:
        ok = launch_bgmv_kernel(static_cast<nv_half*>(y.data_ptr()),
                                static_cast<nv_half*>(x.data_ptr()),
                                static_cast<nv_half*>(w.data_ptr()),
                                indicies.data_ptr<int64_t>(), h_in, h_out, B,
                                num_layers, layer_idx, scale);
        break;
      case at::ScalarType::BFloat16:
        ok = launch_bgmv_kernel(static_cast<nv_bfloat16*>(y.data_ptr()),
                                static_cast<nv_bfloat16*>(x.data_ptr()),
                                static_cast<nv_bfloat16*>(w.data_ptr()),
                                indicies.data_ptr<int64_t>(), h_in, h_out, B,
                                num_layers, layer_idx, scale);
        break;
      default:
        break;
    }
  }
  TORCH_CHECK(ok, "No suitable kernel.", " h_in=", h_in, " h_out=", h_out,
              " dtype=", x.scalar_type());
}

//====== sgmv ======

void dispatch_sgmv_cutlass(torch::Tensor y, torch::Tensor x,
                           torch::Tensor w_ptr, torch::Tensor s,
                           torch::Tensor tmp, int layer_idx) {
  CHECK_INPUT(y);
  CHECK_INPUT(x);
  CHECK_INPUT(w_ptr);
  CHECK_INPUT(s);
  CHECK_INPUT(tmp);

  CHECK_DIM(2, y);
  CHECK_DIM(2, x);
  CHECK_DIM(1, w_ptr);
  CHECK_DIM(1, s);
  CHECK_DIM(1, tmp);

  int num_problems = s.size(0) - 1;
  int d_in = x.size(1);
  int d_out = y.size(1);
  CHECK_EQ(tmp.size(0), static_cast<int64_t>(sgmv_tmp_size(num_problems)));
  bool ok = DISPATCH_TORCH_DTYPE(x.scalar_type(), [&] {
    return sgmv<c_type>((c_type*)y.data_ptr(), (c_type*)x.data_ptr(),
                        (c_type**)w_ptr.data_ptr(), s.data_ptr<int32_t>(),
                        tmp.data_ptr<uint8_t>(), num_problems, d_in, d_out,
                        layer_idx);
  });
  TORCH_CHECK(ok, "No suitable kernel.", " dtype=", x.scalar_type());
}

void dispatch_sgmv_shrink(torch::Tensor y, torch::Tensor x, torch::Tensor w_ptr,
                          torch::Tensor s, torch::Tensor tmp, int layer_idx) {
  CHECK_INPUT(y);
  CHECK_INPUT(x);
  CHECK_INPUT(w_ptr);
  CHECK_INPUT(s);
  CHECK_INPUT(tmp);

  CHECK_DIM(2, y);
  CHECK_DIM(2, x);
  CHECK_DIM(1, w_ptr);
  CHECK_DIM(1, s);
  CHECK_DIM(1, tmp);

  uint32_t num_problems = s.size(0) - 1;
  uint32_t d_in = x.size(1);
  uint32_t d_out = y.size(1);
  CHECK_EQ(tmp.scalar_type(), at::ScalarType::Byte);
  CHECK_EQ(tmp.size(0), 8 * 1024 * 1024);

#define CASE(_T, D_OUT)                                    \
  case D_OUT:                                              \
    return sgmv_shrink<c_type, D_OUT>(                     \
        (c_type*)y.data_ptr(), (c_type*)x.data_ptr(),      \
        (c_type**)w_ptr.data_ptr(), s.data_ptr<int32_t>(), \
        tmp.data_ptr<uint8_t>(), num_problems, d_in, layer_idx);

  bool ok = DISPATCH_TORCH_DTYPE(x.scalar_type(), [&] {
    switch (d_out) {
      FOR_SGMV_NARROW(CASE, c_type);
      default:
        return false;
    }
  });

#undef CASE
  TORCH_CHECK(ok, "No suitable kernel.", " dtype=", x.scalar_type(),
              " d_out=", d_out);
}

//====== rms_norm ======

void dispatch_rms_norm(torch::Tensor output, torch::Tensor input,
                       torch::Tensor weight, float epsilon) {
  CHECK_INPUT(output);
  CHECK_INPUT(input);
  CHECK_INPUT(weight);

  CHECK_DIM(2, input);
  CHECK_DIM(1, weight);
  CHECK_SHAPE(output, input);
  CHECK_EQ(input.size(input.dim() - 1), weight.size(0));
  CHECK_EQ(input.scalar_type(), weight.scalar_type());
  CHECK_EQ(input.scalar_type(), output.scalar_type());

  int rows = input.size(0);
  int columns = input.size(1);

  bool ok = DISPATCH_TORCH_DTYPE(input.scalar_type(), [&] {
    return rms_norm<c_type>(static_cast<c_type*>(output.data_ptr()),
                            static_cast<c_type*>(input.data_ptr()),
                            static_cast<c_type*>(weight.data_ptr()), rows,
                            columns, epsilon);
  });

  TORCH_CHECK(ok, "No suitable kernel.", " dtype=", input.scalar_type(),
              " columns=", columns);
}

}  // namespace

//====== pybind ======

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("batch_decode", &batch_decode, "");
  m.def("init_kv", &init_kv, "");
  m.def("append_kv", &append_kv, "");

  m.def("dispatch_bgmv", &dispatch_bgmv, "dispatch_bgmv");

  m.def("sgmv_cutlass", &dispatch_sgmv_cutlass, "");
  m.def("sgmv_cutlass_tmp_size", &sgmv_tmp_size, "");
  m.def("sgmv_shrink", &dispatch_sgmv_shrink, "");
  m.def("rms_norm", &dispatch_rms_norm, "");
}
