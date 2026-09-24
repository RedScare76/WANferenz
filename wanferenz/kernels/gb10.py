import sys

import torch


def pipeline_stages():
    return (
        2
        if torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 1)
        else 4
    )


def install():
    if pipeline_stages() != 2:
        return False
    import importlib

    kernel = importlib.import_module("wanferenz.kernels.reference")
    if getattr(kernel.fp8_gemm_kernel, "_v4_sm121", False):
        return True
    from wanferenz.kernels.fp8_vector import fp8_gemm_tiled_kernel

    original = kernel.fp8_gemm_kernel

    def safe_kernel(
        N, K, out_dtype="bfloat16", accum_dtype="float32", scale_dtype="float32"
    ):
        if out_dtype != "bfloat16" or accum_dtype != "float32":
            return original(N, K, out_dtype, accum_dtype, scale_dtype)
        return fp8_gemm_tiled_kernel(N, K, scale_dtype, (128, 2, 128))

    safe_kernel._v4_sm121 = True
    kernel.fp8_gemm_kernel = safe_kernel
    print(
        "[v4] SM 12.1 FP8 GEMM: two-stage pipeline (GB10 correctness fix)",
        file=sys.stderr,
        flush=True,
    )
    return True
