#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cutlass/util/host_tensor.h>
#include <torch/library.h>
#include <torch/version.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"

namespace
{
    int num_threadblocks = -1;
    template <typename GemmKernel>
    void run_grouped_gemm(const at::TensorList input,
                          const at::TensorList other,
                          const at::TensorList out,
                          bool segment)
    {
        using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;
        if (num_threadblocks == -1)
        {
            num_threadblocks = GemmGrouped::sufficient();
        }
        const int64_t num_matrices = input.size();
        const int64_t gemm_coord_size =
            num_matrices * ((int64_t)sizeof(cutlass::gemm::GemmCoord));
        // Number of gemm args not including *problem_sizes
        at::Tensor gemm_args =
            at::empty({num_matrices * 6 + gemm_coord_size},
                      at::TensorOptions().dtype(at::kLong).pinned_memory(true));

        // Obtain pointers for each argument (on host)
        int64_t *ld_A_data = gemm_args.data_ptr<int64_t>(); // Base pointer
        int64_t *ld_B_data = ld_A_data + num_matrices;
        int64_t *ld_C_data = ld_A_data + 2 * num_matrices;
        int64_t *ptr_A_data = ld_A_data + 3 * num_matrices;
        int64_t *ptr_B_data = ld_A_data + 4 * num_matrices;
        int64_t *ptr_C_data = ld_A_data + 5 * num_matrices;
        cutlass::gemm::GemmCoord *problem_sizes_data =
            reinterpret_cast<cutlass::gemm::GemmCoord *>(ld_A_data + 6 * num_matrices);

        // Set arguments into gemm_args from input args
        for (size_t i = 0; i < num_matrices; ++i)
        {
            auto new_in = input[i];
            auto new_other = other[i];
            auto new_out = out[i];
            auto m = new_in.size(0), k = new_other.size((int)(segment)),
                 n = new_out.size(1);

            problem_sizes_data[i] = cutlass::gemm::GemmCoord(m, n, k);

            ld_A_data[i] = GemmKernel::LayoutA::packed({m, k}).stride(0);
            ld_B_data[i] = GemmKernel::LayoutB::packed({k, n}).stride(0);
            ld_C_data[i] = GemmKernel::LayoutC::packed({m, n}).stride(0);

            ptr_A_data[i] = reinterpret_cast<int64_t>(new_in.data_ptr<float>());
            ptr_B_data[i] = reinterpret_cast<int64_t>(new_other.data_ptr<float>());
            ptr_C_data[i] = reinterpret_cast<int64_t>(new_out.data_ptr<float>());
        }

        // Transfer arguments to GPU
        gemm_args = gemm_args.to(out[0].device(), true);

        // Obtain pointers for each of arguments (on GPU)
        ld_A_data = gemm_args.data_ptr<int64_t>(); // Base pointer
        ld_B_data = ld_A_data + num_matrices;
        ld_C_data = ld_A_data + 2 * num_matrices;
        ptr_A_data = ld_A_data + 3 * num_matrices;
        ptr_B_data = ld_A_data + 4 * num_matrices;
        ptr_C_data = ld_A_data + 5 * num_matrices;
        problem_sizes_data =
            reinterpret_cast<cutlass::gemm::GemmCoord *>(ld_A_data + 6 * num_matrices);

        using EpilogueOutputOp = typename GemmKernel::Epilogue::OutputOp;
        typename EpilogueOutputOp::Params epilogue_op(1.0, 0.0);

        // Create GemmGrouped::Arguments using the arguments prepared above
        typename GemmGrouped::Arguments args(
            problem_sizes_data, num_matrices,
            /*threadblock_count=*/num_threadblocks, epilogue_op,
            reinterpret_cast<float **>(ptr_A_data),
            reinterpret_cast<float **>(ptr_B_data),
            reinterpret_cast<float **>(ptr_C_data),
            reinterpret_cast<float **>(ptr_C_data), ld_A_data, ld_B_data, ld_C_data,
            ld_C_data);

        GemmGrouped gemm;
        auto status =
            gemm.initialize(args, nullptr, at::cuda::getCurrentCUDAStream());
        TORCH_CHECK(status == cutlass::Status::kSuccess, "GroupedGEMM init failed");
        status = gemm.run(at::cuda::getCurrentCUDAStream());
        TORCH_CHECK(status == cutlass::Status::kSuccess, "GroupedGEMM run failed");

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Returns the amount of shared memory required per threadblock in
    // `GroupedGemmKernel`
    template <typename GroupedGemmKernel>
    int shared_memory_for_kernel()
    {
        return int(sizeof(typename GroupedGemmKernel::SharedStorage));
    }

    // Returns the bytes of shared memory available per SM on the GPU, or -1 on
    // error.
    cudaDeviceProp get_dev_prop()
    {
        cudaDeviceProp properties;
        int device_idx;
        C10_CUDA_CHECK(cudaGetDevice(&device_idx));
        C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device_idx));
        return properties;
    }
    cudaDeviceProp props;
    bool props_queried = false;

    void grouped_matmul_out_kernel(const at::TensorList input,
                                   const at::TensorList other,
                                   const at::TensorList out,
                                   bool segment)
    {
        if (!props_queried)
        {
            props = get_dev_prop();
            props_queried = true;
        }
        if (props.major < 8)
        {
            // Compute capability less than that of Ampere. No TF32 available.
            // note: we only support Volta and onwards
            using GemmKernel_Volta = typename cutlass::gemm::kernel::DefaultGemmGrouped<
                float,                                        // Element A
                cutlass::layout::RowMajor,                    // Layout A
                cutlass::ComplexTransform::kNone,             //
                1,                                            // Granularity A
                float,                                        // Element B
                cutlass::layout::RowMajor,                    // Layout B
                cutlass::ComplexTransform::kNone,             //
                1,                                            // Granularity B
                float,                                        // Element C&D
                cutlass::layout::RowMajor,                    // Layout C&D
                float,                                        // Element Accumulator
                cutlass::arch::OpClassSimt,                   // Operator Class Tag
                cutlass::arch::Sm70,                          // Architecture
                cutlass::gemm::GemmShape<128, 64, 8>,         // Threadblock-level Tile
                cutlass::gemm::GemmShape<64, 64, 8>,          // Warp-level Tile
                cutlass::gemm::GemmShape<1, 1, 1>,            // Warp-level Tile
                cutlass::epilogue::thread::LinearCombination< // Epilogue
                    float, 1, float, float>,                  //
                cutlass::gemm::threadblock::                  // Swizzling Operator
                GemmIdentityThreadblockSwizzle<8>,            //
                2                                             // Stages
                >::GemmKernel;
            run_grouped_gemm<GemmKernel_Volta>(input, other, out, segment);
        }
        else
        {
            // Compute capability at or beyond that of Ampere. TF32 is available.
            bool use_tf32;
#if TORCH_VERSION_MINOR >= 12 or TORCH_VERSION_MAJOR > 1
            use_tf32 = at::globalContext().float32MatmulPrecision() !=
                       at::Float32MatmulPrecision::HIGHEST;
#else
            use_tf32 = at::globalContext().allowTF32CuBLAS();
#endif
            if (use_tf32)
            {
                // TF32 is enabled
                using DefaultGemmKernel_TF32 =
                    typename cutlass::gemm::kernel::DefaultGemmGrouped<
                        float,                                        // Element A
                        cutlass::layout::RowMajor,                    // Layout A
                        cutlass::ComplexTransform::kNone,             //
                        1,                                            // Granularity A
                        float,                                        // Element B
                        cutlass::layout::RowMajor,                    // Layout B
                        cutlass::ComplexTransform::kNone,             //
                        1,                                            // Granularity B
                        float,                                        // Element C&D
                        cutlass::layout::RowMajor,                    // Layout C&D
                        float,                                        // Element Accumulator
                        cutlass::arch::OpClassTensorOp,               // Operator Class Tag
                        cutlass::arch::Sm80,                          // Architecture
                        cutlass::gemm::GemmShape<256, 128, 32>,       // Threadblock-level Tile
                        cutlass::gemm::GemmShape<64, 64, 32>,         // Warp-level Tile
                        cutlass::gemm::GemmShape<16, 8, 8>,           // Warp-level Tile
                        cutlass::epilogue::thread::LinearCombination< // Epilogue
                            float, 1, float, float>,                  //
                        cutlass::gemm::threadblock::                  // Swizzling Operator
                        GemmIdentityThreadblockSwizzle<8>,            //
                        3                                             // Stages
                        >::GemmKernel;
                int grouped_shared_mem =
                    shared_memory_for_kernel<DefaultGemmKernel_TF32>();
                if (grouped_shared_mem < props.sharedMemPerBlockOptin)
                {
                    // full size GPU
                    run_grouped_gemm<DefaultGemmKernel_TF32>(input, other, out, segment);
                }
                else
                {
                    // Smaller GPU
                    using SmallGemmKernel_TF32 =
                        typename cutlass::gemm::kernel::DefaultGemmGrouped<
                            float,                                        // Element A
                            cutlass::layout::RowMajor,                    // Layout A
                            cutlass::ComplexTransform::kNone,             //
                            1,                                            // Granularity A
                            float,                                        // Element B
                            cutlass::layout::RowMajor,                    // Layout B
                            cutlass::ComplexTransform::kNone,             //
                            1,                                            // Granularity B
                            float,                                        // Element C&D
                            cutlass::layout::RowMajor,                    // Layout C&D
                            float,                                        // Element Accumulator
                            cutlass::arch::OpClassTensorOp,               // Operator Class Tag
                            cutlass::arch::Sm80,                          // Architecture
                            cutlass::gemm::GemmShape<128, 64, 32>,        // Threadblock-level
                                                                          // Tile
                            cutlass::gemm::GemmShape<64, 64, 32>,         // Warp-level Tile
                            cutlass::gemm::GemmShape<16, 8, 8>,           // Warp-level Tile
                            cutlass::epilogue::thread::LinearCombination< // Epilogue
                                float, 1, float, float>,                  //
                            cutlass::gemm::threadblock::                  // Swizzling Operator
                            GemmIdentityThreadblockSwizzle<8>,            //
                            3                                             // Stages
                            >::GemmKernel;
                    run_grouped_gemm<SmallGemmKernel_TF32>(input, other, out, segment);
                }
            }
            else
            {
                // TF32 is manually disabled
                using DefaultGemmKernel_FP32 =
                    typename cutlass::gemm::kernel::DefaultGemmGrouped<
                        float,                                        // Element A
                        cutlass::layout::RowMajor,                    // Layout A
                        cutlass::ComplexTransform::kNone,             //
                        1,                                            // Granularity A
                        float,                                        // Element B
                        cutlass::layout::RowMajor,                    // Layout B
                        cutlass::ComplexTransform::kNone,             //
                        1,                                            // Granularity B
                        float,                                        // Element C&D
                        cutlass::layout::RowMajor,                    // Layout C&D
                        float,                                        // Element Accumulator
                        cutlass::arch::OpClassSimt,                   // Operator Class Tag
                        cutlass::arch::Sm80,                          // Architecture
                        cutlass::gemm::GemmShape<128, 64, 8>,         // Threadblock-level Tile
                        cutlass::gemm::GemmShape<64, 64, 8>,          // Warp-level Tile
                        cutlass::gemm::GemmShape<1, 1, 1>,            // Warp-level Tile
                        cutlass::epilogue::thread::LinearCombination< // Epilogue
                            float, 1, float, float>,                  //
                        cutlass::gemm::threadblock::                  // Swizzling Operator
                        GemmIdentityThreadblockSwizzle<8>,            //
                        3                                             // Stages
                        >::GemmKernel;
                int grouped_shared_mem =
                    shared_memory_for_kernel<DefaultGemmKernel_FP32>();
                if (grouped_shared_mem < props.sharedMemPerBlockOptin)
                {
                    // full size GPU
                    run_grouped_gemm<DefaultGemmKernel_FP32>(input, other, out, segment);
                }
                else
                {
                    // Smaller GPU
                    using SmallGemmKernel_FP32 =
                        typename cutlass::gemm::kernel::DefaultGemmGrouped<
                            float,                                        // Element A
                            cutlass::layout::RowMajor,                    // Layout A
                            cutlass::ComplexTransform::kNone,             //
                            1,                                            // Granularity A
                            float,                                        // Element B
                            cutlass::layout::RowMajor,                    // Layout B
                            cutlass::ComplexTransform::kNone,             //
                            1,                                            // Granularity B
                            float,                                        // Element C&D
                            cutlass::layout::RowMajor,                    // Layout C&D
                            float,                                        // Element Accumulator
                            cutlass::arch::OpClassSimt,                   // Operator Class Tag
                            cutlass::arch::Sm80,                          // Architecture
                            cutlass::gemm::GemmShape<64, 64, 8>,          // Threadblock-level
                                                                          // Tile
                            cutlass::gemm::GemmShape<64, 64, 8>,          // Warp-level Tile
                            cutlass::gemm::GemmShape<1, 1, 1>,            // Warp-level Tile
                            cutlass::epilogue::thread::LinearCombination< // Epilogue
                                float, 1, float, float>,                  //
                            cutlass::gemm::threadblock::                  // Swizzling Operator
                            GemmIdentityThreadblockSwizzle<8>,            //
                            3                                             // Stages
                            >::GemmKernel;
                    run_grouped_gemm<SmallGemmKernel_FP32>(input, other, out, segment);
                }
            }
        }
    }

    std::vector<at::Tensor> grouped_matmul_kernel(const at::TensorList input,
                                                  const at::TensorList other)
    {
        std::vector<at::Tensor> out(input.size());
        std::vector<at::Tensor> input_contiguous(input.size());
        std::vector<at::Tensor> other_contiguous(other.size());
        for (size_t i = 0; i < input.size(); ++i)
        {
            input_contiguous[i] = input[i].contiguous();
            other_contiguous[i] = other[i].contiguous();
            out[i] = input[i].new_empty({input[i].size(0), other[i].size(-1)});
        }
        grouped_matmul_out_kernel(input_contiguous, other_contiguous, out, false);

        return out;
    }

    at::Tensor segment_matmul_kernel(const at::Tensor &input,
                                     const at::Tensor &ptr,
                                     const at::Tensor &other)
    {
        const auto size = pyg::utils::size_from_ptr(ptr).cpu();
        // TODO (matthias) Allow for other types than `int64_t`.
        const auto sizes = at::IntArrayRef(size.data_ptr<int64_t>(), size.numel());
        const auto out = input.new_empty({input.size(0), other.size(-1)});

        // TODO (matthias) Better handle non-contiguous memory layouts.
        grouped_matmul_out_kernel(
            input.contiguous().split_with_sizes(/*split_size=*/sizes, /*dim=*/0),
            other.contiguous().split(/*split_size=*/1, /*dim=*/0),
            out.split_with_sizes(/*split_size=*/sizes, /*dim=*/0), true);

        return out;
    }

} // namespace

TORCH_LIBRARY_IMPL(pyg, CUDA, m)
{
    m.impl(TORCH_SELECTIVE_NAME("grouped_matmul"),
           TORCH_FN(grouped_matmul_kernel));
    m.impl(TORCH_SELECTIVE_NAME("segment_matmul"),
           TORCH_FN(segment_matmul_kernel));
}
