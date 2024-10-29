import os
from typing import Union

import torch
from loguru import logger

from lorax_server.utils.flash_attn_triton import triton_attention
from lorax_server.utils.import_utils import SYSTEM
from lorax_server.utils.state import FLASH_INFER

if os.getenv("USE_FLASH_ATTENTION", "").lower() == "false":
    raise ImportError("`USE_FLASH_ATTENTION` is false.")
HAS_FLASH_ATTN = False
HAS_FLASH_ATTN_V2_CUDA = False
HAS_FLASH_ATTN_V2_ROCM = False
ROCM_USE_FLASH_ATTN_V2_CK = False
ROCM_USE_FLASH_ATTN_V2_TRITON = False

if SYSTEM == "xpu":
    import intel_extension_for_pytorch as ipex

    def attention(
        q,
        k,
        v,
        out,
        cu_seqlens,
        max_s,
        softmax_scale,
        window_size_left=-1,
    ):
        if window_size_left <= 0 and window_size_left != -1:
            raise ValueError("`window_size_left` must be > 0 or -1")

        if window_size_left != -1:
            raise ValueError(
                f"XPU version of Flash Attention does not support window attention (window_size_left != -1, got window_size_left={window_size_left})."
            )
        return ipex.llm.functional.varlen_attention(
            q,
            k,
            v,
            out,
            cu_seqlens,
            cu_seqlens,
            max_s,
            max_s,
            0.0,
            softmax_scale,
            False,
            True,
            False,
            None,
        )


if SYSTEM in {"cuda", "rocm"}:
    if not torch.cuda.is_available():
        raise ImportError("CUDA is not available")

    major, minor = torch.cuda.get_device_capability()
    is_sm75 = major == 7 and minor == 5
    is_sm8x = major == 8 and minor >= 0
    is_sm90 = major == 9 and minor == 0
    is_sm94 = major == 9 and minor == 4

    if SYSTEM == "rocm":
        if (
            os.getenv("ROCM_USE_FLASH_ATTN_V2_TRITON", "").lower() == "true"
            or os.getenv("ROCM_USE_FLASH_ATTN_V2_TRITON", "0") == "1"
        ):
            ROCM_USE_FLASH_ATTN_V2_TRITON = True
            logger.info("ROCm: using Flash Attention 2 Triton implementation.")
        else:
            ROCM_USE_FLASH_ATTN_V2_CK = True
            logger.info("ROCm: using Flash Attention 2 Composable Kernel implementation.")

    try:
        try:
            import flash_attn_2_cuda
        except ImportError:
            architecture_suffix = f"-{SYSTEM}"
            raise ImportError(
                "Flash Attention V2 is not installed.\n"
                "Use the official Docker image (ghcr.io/predibase/lorax:latest) "
                f"or install flash attention v2 with `cd server && make install install-flash-attention-v2{architecture_suffix}`"
            )
        if SYSTEM == "cuda" and not (is_sm8x or is_sm90):
            raise ImportError(f"GPU with CUDA capability {major} {minor} is not supported for " "Flash Attention V2")
        elif SYSTEM == "rocm" and not (is_sm8x or is_sm90 or is_sm94):
            raise ImportError(
                f"AMD GPU with compute capability {major} {minor} is not supported for " "Flash Attention V2"
            )
        HAS_FLASH_ATTN_V2_CUDA = SYSTEM == "cuda"
        HAS_FLASH_ATTN_V2_ROCM = SYSTEM == "rocm"
    except ImportError as e:
        try:
            import flash_attn_cuda
        except ImportError:
            raise ImportError(
                "Flash Attention is not installed.\n"
                "Use the official Docker image (ghcr.io/predibase/lorax:latest) "
                "or install flash attention with `cd server && make install install-flash-attention`"
            ) from e

        if SYSTEM == "cuda" and not (is_sm75 or is_sm8x or is_sm90):
            raise ImportError(f"GPU with CUDA capability {major} {minor} is not supported") from e
        elif SYSTEM == "rocm":
            for idx in range(torch.cuda.device_count()):
                if "MI210" not in torch.cuda.get_device_name(idx) and "MI250" not in torch.cuda.get_device_name(idx):
                    raise ImportError(f"AMD GPU {torch.cuda.get_device_name(idx)} does not support flash-attention")

        logger.warning(f"Unable to use Flash Attention V2: {e}")
        HAS_FLASH_ATTN = True

if FLASH_INFER:

    def attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_cache: Union[torch.Tensor, None],
        value_cache: Union[torch.Tensor, None],
        cu_seqlens,
        max_s,
        softmax_scale,
        window_size_left=-1,
        causal=True,
        softcap=0.0,
        k_scale=1.0,
        v_scale=1.0,
        fp8_kv=False,
    ):
        from lorax_server.utils.flashinfer_attention import prefill_state, prefill_with_paged_kv_state

        if fp8_kv or (key_cache is None or value_cache is None):
            return prefill_state.get().forward(
                q,
                k,
                v,
                causal=causal,
                pos_encoding_mode="NONE",
                window_left=window_size_left,
                logits_soft_cap=softcap,
                sm_scale=softmax_scale,
            )

        return prefill_with_paged_kv_state.get().forward(
            q.contiguous(),
            causal=causal,
            paged_kv_cache=(key_cache, value_cache),
            logits_soft_cap=softcap,
            sm_scale=softmax_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            window_left=window_size_left,
        )

elif HAS_FLASH_ATTN_V2_CUDA:

    def attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cu_seqlens,
        max_s,
        softmax_scale,
        window_size_left=-1,
        causal=True,
        softcap=0.0,
        k_scale=1.0,
        v_scale=1.0,
        fp8_kv=False,
    ):
        if window_size_left <= 0 and window_size_left != -1:
            raise ValueError("`window_size_left` must be > 0 or -1")

        out = torch.empty_like(q)
        return flash_attn_2_cuda.varlen_fwd(
            q,
            k,
            v,
            out,
            cu_seqlens,
            cu_seqlens,
            None,
            None,
            None,
            max_s,
            max_s,
            0.0,
            softmax_scale,
            False,
            causal,
            window_size_left,
            0,
            False,
            None,
        )[0]

elif HAS_FLASH_ATTN_V2_ROCM and ROCM_USE_FLASH_ATTN_V2_CK:

    def attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cu_seqlens,
        max_s,
        softmax_scale,
        window_size_left=-1,
        causal=True,
        softcap=0.0,
        k_scale=1.0,
        v_scale=1.0,
        fp8_kv=False,
    ):
        if window_size_left <= 0 and window_size_left != -1:
            raise ValueError("`window_size_left` must be > 0 or -1")
        if window_size_left != -1:
            raise ValueError(
                f"RoCm version of Flash Attention v2 does not support window attention (window_size_left != -1, got window_size_left={window_size_left})."
            )

        # RoCm flash API does not take the window_size_left and window_size_right arguments.
        out = torch.empty_like(q)
        return flash_attn_2_cuda.varlen_fwd(
            q,
            k,
            v,
            out,
            cu_seqlens,
            cu_seqlens,
            max_s,
            max_s,
            0.0,
            softmax_scale,
            False,
            causal,
            False,
            None,
        )[0]

elif HAS_FLASH_ATTN_V2_ROCM and ROCM_USE_FLASH_ATTN_V2_TRITON:

    def attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cu_seqlens,
        max_s,
        softmax_scale,
        window_size_left=-1,
        causal=True,
        softcap=0.0,
        k_scale=1.0,
        v_scale=1.0,
        fp8_kv=False,
    ):
        out = torch.empty_like(q)
        output, _ = triton_attention(
            q,
            k,
            v,
            out,
            cu_seqlens,
            cu_seqlens,
            max_s,
            max_s,
            causal,
            softmax_scale,
        )
        return output

elif HAS_FLASH_ATTN:

    def attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cu_seqlens,
        max_s,
        softmax_scale,
        window_size_left=-1,
        causal=True,
        softcap=0.0,
        k_scale=1.0,
        v_scale=1.0,
        fp8_kv=False,
    ):
        if window_size_left != -1:
            raise NotImplementedError("window_size_left is only available with flash attn v2")

        # Flash attention v1 requires q, k and v to have the same number of heads
        if k.shape[1] != q.shape[1]:
            # MQA expand
            if k.shape[1] == 1:
                k = k.expand(-1, q.shape[1], -1)
            # Grouped attention reshape
            else:
                original_shape = k.shape
                k = (
                    k.unsqueeze(2)
                    .expand(-1, -1, q.shape[1] // k.shape[1], -1)
                    .reshape(original_shape[0], -1, original_shape[2])
                )
        if v.shape[1] != q.shape[1]:
            # MQA expand
            if v.shape[1] == 1:
                v = v.expand(-1, q.shape[1], -1)
            # Grouped attention reshape
            else:
                original_shape = v.shape
                v = (
                    v.unsqueeze(2)
                    .expand(-1, -1, q.shape[1] // v.shape[1], -1)
                    .reshape(original_shape[0], -1, original_shape[2])
                )

        out = torch.empty_like(q)
        flash_attn_cuda.fwd(
            q,
            k,
            v,
            out,
            cu_seqlens,
            cu_seqlens,
            max_s,
            max_s,
            0.0,
            softmax_scale,
            False,
            True,
            False,
            0,
            None,
        )
        return out

else:
    raise RuntimeError(
        f"Flash Attention is not available for this system: {SYSTEM}\n"
        f"If you are running on GPU, please check your environment to ensure CUDA is installed correctly.\n"
        f"You may also want to check envvars such as LD_PRELOAD for libraries that conflict with your device drivers.\n"
    )
