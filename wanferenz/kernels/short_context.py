import os

import torch

V4_REF_SLIM = os.environ.get("V4_REF_SLIM", "0") not in ("", "0")
V4_REF_SLIM_NOQAT = os.environ.get("V4_REF_SLIM_NOQAT", "0") not in ("", "0")


_REF_INDEXER_FORWARD = None
_GET_COMPRESS_TOPK_IDXS = None
_REF_ACT_QUANT = None
_REF_FP4_ACT_QUANT = None


_ACTIVE = True
_ACTIVE_NOQAT = True


_JOB_MAX_POS = None


def _keep_compressor(indexer, ratio):

    if _JOB_MAX_POS is None:
        return True
    return (_JOB_MAX_POS // ratio) > indexer.index_topk


def slim_indexer_forward(self, x, qr, start_pos, offset):

    if not _ACTIVE:
        return _REF_INDEXER_FORWARD(self, x, qr, start_pos, offset)
    bsz, seqlen, _ = x.size()
    ratio = self.compress_ratio
    end_pos = start_pos + seqlen
    if end_pos // ratio > self.index_topk:
        return _REF_INDEXER_FORWARD(self, x, qr, start_pos, offset)
    if _keep_compressor(self, ratio):
        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.freqs_cis = self.freqs_cis
        self.compressor(x, start_pos)
    return _GET_COMPRESS_TOPK_IDXS(ratio, bsz, seqlen, start_pos, offset)


def slim_act_quant(
    x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False
):

    if _ACTIVE_NOQAT and inplace:
        return x
    return _REF_ACT_QUANT(x, block_size, scale_fmt, scale_dtype, inplace)


def slim_fp4_act_quant(x, block_size=32, inplace=False):

    if _ACTIVE_NOQAT and inplace:
        return x
    return _REF_FP4_ACT_QUANT(x, block_size, inplace)


def set_job_max_pos(n):

    global _JOB_MAX_POS
    _JOB_MAX_POS = None if n is None else int(n)


def set_active(on=True):

    global _ACTIVE
    _ACTIVE = bool(on)


def set_active_noqat(on=True):

    global _ACTIVE_NOQAT
    _ACTIVE_NOQAT = bool(on)


def install(mod, item1=None, item2=None):

    global \
        _REF_INDEXER_FORWARD, \
        _GET_COMPRESS_TOPK_IDXS, \
        _REF_ACT_QUANT, \
        _REF_FP4_ACT_QUANT
    i1 = V4_REF_SLIM if item1 is None else bool(item1)
    i2 = V4_REF_SLIM_NOQAT if item2 is None else bool(item2)
    took = {"indexer_skip": False, "noqat": False}
    if i1 and not getattr(mod.Indexer.forward, "_v4_ref_slim", False):
        _REF_INDEXER_FORWARD = mod.Indexer.forward
        _GET_COMPRESS_TOPK_IDXS = mod.get_compress_topk_idxs
        slim_indexer_forward._v4_ref_slim = True
        mod.Indexer.forward = slim_indexer_forward
        took["indexer_skip"] = True
    if i2 and not getattr(mod.act_quant, "_v4_ref_slim", False):
        _REF_ACT_QUANT = mod.act_quant
        _REF_FP4_ACT_QUANT = mod.fp4_act_quant
        slim_act_quant._v4_ref_slim = True
        slim_fp4_act_quant._v4_ref_slim = True
        mod.act_quant = slim_act_quant
        mod.fp4_act_quant = slim_fp4_act_quant
        took["noqat"] = True
    return took


def uninstall(mod):

    global \
        _REF_INDEXER_FORWARD, \
        _GET_COMPRESS_TOPK_IDXS, \
        _REF_ACT_QUANT, \
        _REF_FP4_ACT_QUANT
    gone = {"indexer_skip": False, "noqat": False}
    if (
        getattr(mod.Indexer.forward, "_v4_ref_slim", False)
        and _REF_INDEXER_FORWARD is not None
    ):
        mod.Indexer.forward = _REF_INDEXER_FORWARD
        _REF_INDEXER_FORWARD = _GET_COMPRESS_TOPK_IDXS = None
        gone["indexer_skip"] = True
    if getattr(mod.act_quant, "_v4_ref_slim", False) and _REF_ACT_QUANT is not None:
        mod.act_quant = _REF_ACT_QUANT
        mod.fp4_act_quant = _REF_FP4_ACT_QUANT
        _REF_ACT_QUANT = _REF_FP4_ACT_QUANT = None
        gone["noqat"] = True
    return gone
