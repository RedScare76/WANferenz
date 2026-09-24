import argparse
import glob
import json
import os
import shutil
import sys

import torch
from safetensors.torch import safe_open, save_file


from wanferenz.model.tensor_layout import cast_e2m1fn_to_e4m3fn, mapping


def _rename(name):
    if name.startswith("model."):
        name = name[len("model.") :]
    if name.startswith("mtp.") and ("emb" in name or name.endswith("head.weight")):
        return None
    name = name.replace("self_attn", "attn")
    name = name.replace("mlp", "ffn")
    name = name.replace("weight_scale_inv", "scale")
    name = name.replace("e_score_correction_bias", "bias")
    if any(x in name for x in ["hc", "attn_sink", "tie2eid", "ape"]):
        key = name.split(".")[-1]
    else:
        key = name.split(".")[-2]
    new_key, _ = mapping.get(key, (key, None))
    return name.replace(key, new_key)


def _check_pairs(files):
    owners = {}
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                renamed = _rename(name)
                if renamed is not None:
                    owners[renamed] = path
    bad = []
    for name, path in owners.items():
        if not name.endswith(".weight"):
            continue
        if "experts" not in name and not name.endswith("wo_a.weight"):
            continue
        scale = name.replace(".weight", ".scale")
        if scale in owners and owners[scale] != path:
            bad.append((name, os.path.basename(path), os.path.basename(owners[scale])))
    if bad:
        raise RuntimeError(
            f"{len(bad)} weight/scale pairs cross input partitions; first: {bad[0]}"
        )


def _convert_one(path, expert_dtype):
    state = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for old_name in handle.keys():
            name = _rename(old_name)
            if name is not None:
                state[name] = handle.get_tensor(old_name)

    for name in list(state):
        if name.endswith("wo_a.weight"):
            weight = state[name]
            scale_name = name.replace("weight", "scale")
            scale = state.pop(scale_name)
            weight = (
                weight.unflatten(0, (-1, 128)).unflatten(-1, (-1, 128)).float()
                * scale[:, None, :, None].float()
            )
            state[name] = weight.flatten(2, 3).flatten(0, 1).bfloat16()
        elif "experts" in name and state[name].dtype == torch.int8:
            if expert_dtype == "fp8":
                scale_name = name.replace("weight", "scale")
                weight = state.pop(name)
                scale = state.pop(scale_name)
                state[name], state[scale_name] = cast_e2m1fn_to_e4m3fn(weight, scale)
            else:
                state[name] = state[name].view(torch.float4_e2m1fn_x2)
    return state


def run_command(source, output, expert_dtype="fp4"):
    source = os.path.realpath(source)
    output = os.path.realpath(output)
    if source == output or output.startswith(source + os.sep):
        raise ValueError(
            "output must be separate from the read-only Hugging Face source"
        )
    files = sorted(glob.glob(os.path.join(source, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no input safetensors in {source}")
    os.makedirs(output, exist_ok=True)
    existing = glob.glob(os.path.join(output, "model*-mp*.safetensors"))
    if existing:
        raise FileExistsError(f"refusing existing converted output: {existing[0]}")

    _check_pairs(files)
    torch.set_num_threads(8)
    for index, path in enumerate(files):
        state = _convert_one(path, expert_dtype)
        target = os.path.join(output, f"model{index:05d}-mp1.safetensors")
        partial = target + ".partial"
        save_file(state, partial)
        os.replace(partial, target)
        print(
            f"[{index + 1}/{len(files)}] {os.path.basename(target)} {len(state)} tensors",
            flush=True,
        )

    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(os.path.join(source, name), os.path.join(output, name))
    config = os.path.join(source, "inference", "config.json")
    shutil.copyfile(config, os.path.join(output, "config.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-ckpt-path", required=True)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--expert-dtype", choices=["fp4", "fp8"], default="fp4")
    args = parser.parse_args()
    run_command(args.hf_ckpt_path, args.save_path, args.expert_dtype)
