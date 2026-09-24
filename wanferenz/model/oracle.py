import importlib.util
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wanferenz.kernels.torch_ops as torch_ops

from wanferenz.model.assets import reference_directory


REF_DIR = reference_directory()
INFERENCE_DIR = REF_DIR


ENCODING_DIR = REF_DIR

_REF = None


def reference_module():

    global _REF
    if _REF is None:
        torch_ops.install()
        if INFERENCE_DIR not in sys.path:
            sys.path.insert(0, INFERENCE_DIR)
        if torch_ops.backend() == "tilelang":
            import wanferenz.kernels.sparse_attention as sparse_attention

            sparse_attention.install_sm120()
            import wanferenz.kernels.gb10 as gb10

            gb10.install()
        spec = importlib.util.spec_from_file_location(
            "dsv4_model", os.path.join(INFERENCE_DIR, "architecture.py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["dsv4_model"] = mod
        spec.loader.exec_module(mod)

        import wanferenz.kernels.expert_dispatch as expert_dispatch

        expert_dispatch.install(mod)

        import wanferenz.kernels.fp4_grouped as fp4_grouped

        fp4_grouped.install(mod)

        import wanferenz.kernels.expert_batch as expert_batch

        expert_batch.install(mod)

        import wanferenz.kernels.short_context as short_context

        short_context.install(mod)

        import wanferenz.kernels.fp8_vector as fp8_vector

        fp8_vector.install(mod)
        _REF = mod
    return _REF


_CPU_ARGS = dict(
    dtype="bf16",
    scale_fmt=None,
    scale_dtype="fp32",
    expert_dtype=None,
    temperature=0.0,
    max_batch_size=2,
    max_seq_len=256,
    vocab_size=512,
    dim=256,
    n_layers=8,
    n_heads=4,
    o_groups=2,
    q_lora_rank=64,
    o_lora_rank=32,
    head_dim=128,
    rope_head_dim=64,
    window_size=16,
    index_head_dim=128,
    index_n_heads=4,
    index_topk=8,
    compress_ratios=(0, 0, 4, 8, 4, 8, 4, 0, 0, 0),
    original_seq_len=64,
    rope_factor=4,
    compress_rope_theta=160000,
    n_routed_experts=8,
    n_activated_experts=2,
    n_shared_experts=1,
    moe_inter_dim=64,
    n_hash_layers=2,
    score_func="sqrtsoftplus",
    route_scale=1.5,
    swiglu_limit=10.0,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    n_mtp_layers=2,
    dspark_block_size=3,
    dspark_noise_token_id=511,
    dspark_target_layer_ids=(5, 6, 7),
    dspark_markov_rank=16,
)


def miniature_parameters(**overrides):

    ref = reference_module()
    args = ref.ModelArgs(**{**_CPU_ARGS, **overrides})
    nope = args.head_dim - args.rope_head_dim
    assert nope > 0 and nope % 64 == 0, (
        f"head_dim-rope_head_dim={nope} must be a positive multiple of 64"
    )
    idx_d = args.index_head_dim
    assert idx_d & (idx_d - 1) == 0 and idx_d % 32 == 0, (
        f"index_head_dim={idx_d} must be a power of two and a multiple of 32"
    )
    assert idx_d > args.rope_head_dim, (
        f"index_head_dim={idx_d} must exceed rope_head_dim={args.rope_head_dim}"
    )
    assert args.n_heads % args.o_groups == 0, "n_heads must be divisible by o_groups"
    n_mtp = args.n_mtp_layers if args.dspark_block_size else 0
    assert len(args.compress_ratios) == args.n_layers + n_mtp, (
        f"compress_ratios must be {args.n_layers + n_mtp} long (n_layers + n_mtp_layers)"
    )
    assert all(r == 0 for r in args.compress_ratios[args.n_layers :]), (
        "DSpark layers must have compress_ratio 0"
    )
    assert all(r in (0, 4) or r > 4 for r in args.compress_ratios), (
        "compress ratios must be 0, 4, or > 4"
    )
    assert all(i < args.n_layers for i in args.dspark_target_layer_ids), (
        "dspark target layers must exist"
    )
    assert args.dspark_noise_token_id < args.vocab_size, "noise token must be in vocab"
    return args


def initialize_parameters(model, seed=0):

    ref = reference_module()
    norms = {
        f"{n}.weight" for n, m in model.named_modules() if isinstance(m, ref.RMSNorm)
    }
    ints = {
        f"{n}.tid2eid": m.weight.size(0)
        for n, m in model.named_modules()
        if isinstance(m, ref.Gate) and m.hash
    }
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in ints:
                p.random_(0, ints[name])
            elif name in norms:
                p.normal_(1.0, 0.02)
            else:
                p.normal_(0.0, 0.02)
    return model


def create_oracle(args=None, seed=0):

    ref = reference_module()
    args = args or miniature_parameters()
    with ref.set_dtype(torch.bfloat16):
        model = ref.Transformer(args)
    return initialize_parameters(model, seed).eval()


def _smoke():

    args = miniature_parameters()
    torch.manual_seed(0)
    prompt, steps = 33, 40
    x = torch.randint(0, args.vocab_size, (1, prompt + steps))
    model = create_oracle(args)

    output_ids, logits, main_hidden = model(x[:, :prompt])
    assert model.forward_spec(output_ids, main_hidden) is None, (
        "prefill forward_spec must return None"
    )
    print(
        f"prefill  logits {tuple(logits.shape)}  main_hidden {tuple(main_hidden.shape)}  "
        f"next {output_ids.tolist()}"
    )
    checksum = 0.0
    for i in range(prompt, prompt + steps):
        output_ids, logits, main_hidden = model(x[:, i : i + 1], i)
        output_ids, spec_logits, confidence = model.forward_spec(
            output_ids, main_hidden, i
        )
        for t in (logits, main_hidden, spec_logits, confidence):
            assert torch.isfinite(t).all(), f"non-finite tensor at step {i}"
        checksum += logits.float().sum().item() + spec_logits.float().sum().item()
    print(
        f"decode   x{steps}  spec_logits {tuple(spec_logits.shape)}  "
        f"output_ids {tuple(output_ids.shape)}  confidence {tuple(confidence.shape)}"
    )
    print(f"backend  {torch_ops.backend()}   checksum {checksum:.4f}")


if __name__ == "__main__":
    _smoke()
