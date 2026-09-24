import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wanferenz.kernels.torch_ops as torch_ops


TILES = ((16, 64, 256), (16, 32, 128))

TILE_ENV = "V4_SM120_TILE"

VENDORED_BLOCK, VENDORED_THREADS, NUM_STAGES = 64, 256, 2

M_ATOM = 16


def kernel_smem(h_block, block, d, threads=VENDORED_THREADS):

    return 2 * (h_block * d + block * d + h_block * block) + 8 * threads


def smem_limit(device=None):

    p = torch.cuda.get_device_properties(0 if device is None else device)
    return (
        getattr(p, "shared_memory_per_block_optin", None) or p.shared_memory_per_block
    )


def padded_heads(h, h_block=M_ATOM):

    return -(-h // h_block) * h_block


def choose_tile(h, d, limit=None):

    if env := os.environ.get(TILE_ENV):
        return tuple(int(v) for v in env.split(","))
    limit = smem_limit() if limit is None else limit
    full = padded_heads(h)
    if kernel_smem(full, VENDORED_BLOCK, d, VENDORED_THREADS) <= limit:
        return full, VENDORED_BLOCK, VENDORED_THREADS
    for h_block, block, threads in TILES:
        if kernel_smem(h_block, block, d, threads) <= limit:
            return h_block, block, threads
    raise RuntimeError(
        f"v4 sm120: no tiling for h={h} d={d} fits {limit} B of shared memory (smallest candidate "
        f"{TILES[-1]} needs {kernel_smem(*TILES[-1][:2], d, TILES[-1][2])} B)"
    )


_KERNELS = {}


def sparse_attn_kernel(
    h: int,
    d: int,
    scale=None,
    h_block: int = M_ATOM,
    block: int = VENDORED_BLOCK,
    threads: int = VENDORED_THREADS,
    num_stages: int = NUM_STAGES,
):

    key = (h, d, scale, h_block, block, threads, num_stages)
    if key in _KERNELS:
        return _KERNELS[key]
    assert h % h_block == 0, (
        f"v4 sm120: h={h} not a multiple of h_block={h_block} (padded_heads)"
    )
    assert h_block % M_ATOM == 0, (
        f"v4 sm120: h_block={h_block} is not a multiple of {M_ATOM} (MMA)"
    )
    import tilelang
    import tilelang.language as T

    tilelang.set_log_level("WARNING")
    pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }
    BF16, FP32, INT32 = "bfloat16", "float32", "int32"

    b = T.symbolic("b")
    m = T.symbolic("m")
    n = T.symbolic("n")
    topk = T.symbolic("topk")
    if scale is None:
        scale = (1.0 / d) ** 0.5

    num_blocks = tilelang.cdiv(topk, block)
    n_h_blocks = h // h_block

    @tilelang.jit(pass_configs=pass_configs)
    def _build():

        @T.prim_func
        def sparse_attn_sm120_kernel_(
            q: T.Tensor[(b, m, h, d), BF16],
            kv: T.Tensor[(b, n, d), BF16],
            o: T.Tensor[(b, m, h, d), BF16],
            attn_sink: T.Tensor[(h,), FP32],
            topk_idxs: T.Tensor[(b, m, topk), INT32],
        ):
            with T.Kernel(m, b, n_h_blocks, threads=threads) as (bx, by, bz):
                q_shared = T.alloc_shared((h_block, d), BF16)
                kv_shared = T.alloc_shared((block, d), BF16)
                o_shared = T.alloc_shared((h_block, d), BF16)
                acc_s_cast = T.alloc_shared((h_block, block), BF16)

                idxs = T.alloc_fragment(block, INT32)
                acc_s = T.alloc_fragment((h_block, block), FP32)
                acc_o = T.alloc_fragment((h_block, d), FP32)
                scores_max = T.alloc_fragment(h_block, FP32)
                scores_max_prev = T.alloc_fragment(h_block, FP32)
                scores_scale = T.alloc_fragment(h_block, FP32)
                scores_sum = T.alloc_fragment(h_block, FP32)
                sum_exp = T.alloc_fragment(h_block, FP32)

                T.clear(acc_o)
                T.clear(sum_exp)
                T.fill(scores_max, -T.infinity(FP32))
                T.copy(q[by, bx, bz * h_block : (bz + 1) * h_block, :], q_shared)

                for t in T.Pipelined(num_blocks, num_stages=num_stages):
                    for i in T.Parallel(block):
                        idxs[i] = T.if_then_else(
                            t * block + i < topk, topk_idxs[by, bx, t * block + i], -1
                        )
                    for i, j in T.Parallel(block, d):
                        kv_shared[i, j] = T.if_then_else(
                            idxs[i] != -1, kv[by, idxs[i], j], 0
                        )
                    for i, j in T.Parallel(h_block, block):
                        acc_s[i, j] = T.if_then_else(
                            idxs[j] != -1, 0, -T.infinity(FP32)
                        )
                    T.gemm(
                        q_shared,
                        kv_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for i, j in T.Parallel(h_block, block):
                        acc_s[i, j] *= scale
                    T.copy(scores_max, scores_max_prev)
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(h_block):
                        scores_scale[i] = T.exp(scores_max_prev[i] - scores_max[i])
                    for i, j in T.Parallel(h_block, block):
                        acc_s[i, j] = T.exp(acc_s[i, j] - scores_max[i])
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(h_block):
                        sum_exp[i] = sum_exp[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)
                    for i, j in T.Parallel(h_block, d):
                        acc_o[i, j] *= scores_scale[i]
                    T.gemm(
                        acc_s_cast, kv_shared, acc_o, policy=T.GemmWarpPolicy.FullRow
                    )

                for i in T.Parallel(h_block):
                    sum_exp[i] += T.exp(attn_sink[bz * h_block + i] - scores_max[i])
                for i, j in T.Parallel(h_block, d):
                    acc_o[i, j] /= sum_exp[i]
                T.copy(acc_o, o_shared)
                T.copy(o_shared, o[by, bx, bz * h_block : (bz + 1) * h_block, :])

        return sparse_attn_sm120_kernel_

    _KERNELS[key] = _build()
    return _KERNELS[key]


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:

    b, s, h, d = q.size()
    h_block, block, threads = choose_tile(h, d)
    pad = padded_heads(h, h_block) - h
    if pad:
        q = torch.cat([q, q.new_zeros(b, s, pad, d)], dim=2)
        attn_sink = torch.cat([attn_sink, attn_sink.new_zeros(pad)])
    o = torch.empty_like(q)
    kernel = sparse_attn_kernel(q.size(2), d, softmax_scale, h_block, block, threads)
    kernel(q, kv, o, attn_sink, topk_idxs)
    if pad:
        o = o.narrow(2, 0, h).contiguous()
    return o


def sparse_attn_eager(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:

    return torch_ops.sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)


def install_sm120(h=64, d=512):

    if not torch.cuda.is_available():
        return None
    limit = smem_limit()
    if kernel_smem(padded_heads(h), VENDORED_BLOCK, d, VENDORED_THREADS) <= limit:
        return None
    import importlib

    kernel = importlib.import_module("wanferenz.kernels.reference")
    if getattr(kernel, "_v4_cpu_backend", False):
        return None
    kernel.sparse_attn = sparse_attn
    return choose_tile(h, d, limit)


def _smoke():

    assert torch.cuda.is_available(), "v4 sm120 smoke needs a CUDA device"
    torch.manual_seed(0)
    dev, h, d, n, topk = "cuda", 64, 512, 640, 640
    tile = choose_tile(h, d)
    print(
        f"smem limit {smem_limit()}  vendored {kernel_smem(h, VENDORED_BLOCK, d)}  "
        f"tile {tile} {kernel_smem(tile[0], tile[1], d, tile[2])}"
    )
    for s in (1, 33):
        q = torch.randn(1, s, h, d, dtype=torch.bfloat16, device=dev)
        kv = torch.randn(1, n, d, dtype=torch.bfloat16, device=dev)
        sink = torch.randn(h, dtype=torch.float32, device=dev)
        idx = torch.randint(0, n, (1, s, topk), dtype=torch.int32, device=dev)
        idx[..., ::7] = -1
        o = sparse_attn(q, kv, sink, idx, d**-0.5)
        ref = sparse_attn_eager(q, kv, sink, idx, d**-0.5)
        assert torch.isfinite(o.float()).all(), f"non-finite output at s={s}"
        print(
            f"s={s:<3d} out {tuple(o.shape)}  max|kernel-eager| "
            f"{(o.float() - ref.float()).abs().max().item():.3e}"
        )


if __name__ == "__main__":
    _smoke()
