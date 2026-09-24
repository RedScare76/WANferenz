import os

import torch

V4_MOE_DECODE = os.environ.get("V4_MOE_DECODE", "1") not in ("", "0")

_REF_FORWARD = None
_WORLD_SIZE = 1


def decode_forward(self, x, input_ids):

    shape = x.size()
    xv = x.view(-1, self.dim)
    if xv.size(0) != 1 or _WORLD_SIZE > 1:
        return _REF_FORWARD(self, x, input_ids)

    weights, indices = self.gate(xv, input_ids.flatten())
    sel = indices[0].tolist()
    if len(set(sel)) != len(sel):
        return _REF_FORWARD(self, x, input_ids)
    y = torch.zeros_like(xv, dtype=torch.float32)
    for k in sorted(range(len(sel)), key=lambda j: sel[j]):
        i = sel[k]
        if not (self.experts_start_idx <= i < self.experts_end_idx):
            continue
        y += self.experts[i](xv, weights[:, k, None])
    y += self.shared_experts(xv)
    return y.type_as(xv).view(shape)


def install(mod):

    global _REF_FORWARD, _WORLD_SIZE
    if not V4_MOE_DECODE or getattr(mod.MoE.forward, "_v4_decode_fast", False):
        return False
    if getattr(mod.MoE.forward, "_v4_grouped", False) or getattr(
        mod.MoE.forward, "_v4_multi", False
    ):
        return False
    _REF_FORWARD = mod.MoE.forward
    _WORLD_SIZE = int(getattr(mod, "world_size", 1) or 1)
    decode_forward._v4_decode_fast = True
    mod.MoE.forward = decode_forward
    return True
