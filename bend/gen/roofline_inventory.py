# Copyright (c) 2026 Gil Rodrigues
"""Read-only safetensors-header inventory for the roofline model.

Reads only the 8-byte length + JSON header of each .safetensors file (no tensor
data) under the model directories, checks shape x dtype == data_offsets span for
every tensor, and returns the per-tensor byte constants roofline.bend uses.

Inputs (required arguments):
- --target DIR: the target model directory (Qwen3.8-27B EXL3 4.00 bpw, the
  qwen38-27b-exl3 directory of the served models).
- --draft DIR: the draft model directory (Qwen3.8-27B DFlash2 EXL3 4.00 bpw, the
  dflash2-exl3 directory of the served models).
Output: the JSON constants, on stdout.

Regenerate and check the tracked input of bend/gen/roofline_impl.py:
    python3 -I -B bend/gen/roofline_inventory.py \
        --target TARGET --draft DRAFT > /tmp/x
    cmp /tmp/x bend/gen/roofline_inventory.json

Origin: the inventory script of the roofline work. Only the model directory
arguments, the duplicate tensor check and the explicit file encodings changed.
"""

import argparse
import json
import pathlib
import struct
import sys

DT = {
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "I16": 2,
    "I32": 4,
    "I8": 1,
    "U8": 1,
    "F64": 8,
    "I64": 8,
}


def load(model_dir: str) -> dict[str, int]:
    """Return the byte count of every tensor in a model directory.

    Args:
        model_dir: Directory holding the ``.safetensors`` files.

    Returns:
        Tensor byte counts by tensor name.

    Raises:
        SystemExit: If no file exists, a header is malformed, a size disagrees
            with its offsets or a tensor name repeats.

    """
    out: dict[str, int] = {}
    names = sorted(
        p.name
        for p in pathlib.Path(model_dir).glob("*.safetensors")
        if not p.name.startswith(".")
    )
    files = [f"{model_dir}/{name}" for name in names]
    if not files:
        msg = f"no safetensors in {model_dir}"
        raise SystemExit(msg)
    for f in files:
        add_header(f, out)
    return out


def add_header(f: str, out: dict[str, int]) -> None:
    """Add the checked byte count of every tensor in one file's header.

    Args:
        f: Path of the ``.safetensors`` file.
        out: Tensor byte counts by name, updated in place.

    Raises:
        SystemExit: If the header is malformed, a size disagrees with its
            offsets or a tensor name is already in ``out``.

    """
    with pathlib.Path(f).open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h: object = json.loads(fh.read(n))
    if not isinstance(h, dict):
        msg = f"header is not an object {f}"
        raise SystemExit(msg)
    for k, v in h.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        if not isinstance(k, str) or not isinstance(a, int) or not isinstance(b, int):
            msg = f"malformed header entry {f}:{k}"
            raise SystemExit(msg)
        nel = 1
        for s in v["shape"]:
            nel *= s
        if nel * DT[v["dtype"]] != b - a:
            msg = f"size mismatch {f}:{k}"
            raise SystemExit(msg)
        if k in out:
            msg = f"duplicate tensor {f}:{k}"
            raise SystemExit(msg)
        out[k] = b - a


def lin(tb: dict[str, int], p: str) -> int:
    """Return EXL3 linear streamed bytes: trellis + suh + svh.

    mul1 is a 4-byte scalar argument and is not counted.

    Args:
        tb: Tensor byte counts by name.
        p: Tensor name prefix of the linear layer.

    Returns:
        The streamed byte count.

    """
    return tb[p + ".trellis"] + tb[p + ".suh"] + tb[p + ".svh"]


def constants(target: str, draft: str) -> dict[str, int]:
    """Return the per-tensor-kind byte constants of the two models.

    Args:
        target: Target model directory.
        draft: Draft model directory.

    Returns:
        The byte constants keyed by roofline name.

    Raises:
        SystemExit: If a layer is of an unknown kind or not byte-identical to
            its kind's constants.

    """
    t, d = load(target), load(draft)
    with pathlib.Path(f"{target}/config.json").open(encoding="utf-8") as fh:
        cfg = json.load(fh)["text_config"]
    lp = "model.language_model.layers."
    raw_types: object = cfg["layer_types"]
    if not isinstance(raw_types, list):
        msg = "text_config.layer_types is not a list"
        raise SystemExit(msg)
    types = [x for x in raw_types if isinstance(x, str)]
    if len(types) != len(raw_types):
        msg = "text_config.layer_types holds a non-string entry"
        raise SystemExit(msg)
    gdn0 = types.index("linear_attention")
    att0 = types.index("full_attention")
    consts = {
        "mlp_gate": lin(t, f"{lp}0.mlp.gate_proj"),
        "mlp_up": lin(t, f"{lp}0.mlp.up_proj"),
        "mlp_down": lin(t, f"{lp}0.mlp.down_proj"),
        "ln_in": t[f"{lp}0.input_layernorm.weight"],
        "ln_post": t[f"{lp}0.post_attention_layernorm.weight"],
        "gdn_qkv": lin(t, f"{lp}{gdn0}.linear_attn.in_proj_qkv"),
        "gdn_z": lin(t, f"{lp}{gdn0}.linear_attn.in_proj_z"),
        "gdn_out": lin(t, f"{lp}{gdn0}.linear_attn.out_proj"),
        "gdn_a": t[f"{lp}{gdn0}.linear_attn.in_proj_a.weight"],
        "gdn_b": t[f"{lp}{gdn0}.linear_attn.in_proj_b.weight"],
        "gdn_conv1d": t[f"{lp}{gdn0}.linear_attn.conv1d.weight"],
        "gdn_alog": t[f"{lp}{gdn0}.linear_attn.A_log"],
        "gdn_dtb": t[f"{lp}{gdn0}.linear_attn.dt_bias"],
        "gdn_norm": t[f"{lp}{gdn0}.linear_attn.norm.weight"],
        "attn_q": lin(t, f"{lp}{att0}.self_attn.q_proj"),
        "attn_k": lin(t, f"{lp}{att0}.self_attn.k_proj"),
        "attn_v": lin(t, f"{lp}{att0}.self_attn.v_proj"),
        "attn_o": lin(t, f"{lp}{att0}.self_attn.o_proj"),
        "attn_qn": t[f"{lp}{att0}.self_attn.q_norm.weight"],
        "attn_kn": t[f"{lp}{att0}.self_attn.k_norm.weight"],
        "final_norm": t["model.language_model.norm.weight"],
        "head": lin(t, "lm_head"),
        "d_q": lin(d, "layers.0.self_attn.q_proj"),
        "d_k": lin(d, "layers.0.self_attn.k_proj"),
        "d_v": lin(d, "layers.0.self_attn.v_proj"),
        "d_o": lin(d, "layers.0.self_attn.o_proj"),
        "d_gate": lin(d, "layers.0.mlp.gate_proj"),
        "d_up": lin(d, "layers.0.mlp.up_proj"),
        "d_down": lin(d, "layers.0.mlp.down_proj"),
        "d_akp": d["layers.0.attention_conv.kernel_projection.weight"],
        "d_mkp": d["layers.0.mlp_conv.kernel_projection.weight"],
        "d_abk": d["layers.0.attention_conv.base_kernel"],
        "d_mbk": d["layers.0.mlp_conv.base_kernel"],
        "d_ln_in": d["layers.0.input_layernorm.weight"],
        "d_ln_post": d["layers.0.post_attention_layernorm.weight"],
        "d_qn": d["layers.0.self_attn.q_norm.weight"],
        "d_kn": d["layers.0.self_attn.k_norm.weight"],
        "d_norm": d["norm.weight"],
        "d_hnorm": d["hidden_norm.weight"],
        "d_fc": lin(d, "fc"),
        "d_hproj": d["candidate_selector.hidden_projection.weight"],
    }
    # Every layer of a kind carries byte-identical tensors (uniform 4.00 bpw):
    # the per-kind constants above stand for all 64 target / 5 draft layers.
    for i, kind in enumerate(types):
        p = f"{lp}{i}."
        same = [
            lin(t, p + "mlp.gate_proj") == consts["mlp_gate"],
            lin(t, p + "mlp.up_proj") == consts["mlp_up"],
            lin(t, p + "mlp.down_proj") == consts["mlp_down"],
        ]
        if kind == "linear_attention":
            same += [
                lin(t, p + "linear_attn.in_proj_qkv") == consts["gdn_qkv"],
                lin(t, p + "linear_attn.in_proj_z") == consts["gdn_z"],
                lin(t, p + "linear_attn.out_proj") == consts["gdn_out"],
                t[p + "linear_attn.in_proj_a.weight"] == consts["gdn_a"],
                t[p + "linear_attn.in_proj_b.weight"] == consts["gdn_b"],
                t[p + "linear_attn.conv1d.weight"] == consts["gdn_conv1d"],
            ]
        elif kind == "full_attention":
            same += [
                lin(t, p + f"self_attn.{x}_proj") == consts[f"attn_{x}"] for x in "qkvo"
            ]
        else:
            msg = f"layer {i}: unknown kind {kind}"
            raise SystemExit(msg)
        if not all(same):
            msg = f"layer {i} ({kind}) is not byte-identical to its kind's constants"
            raise SystemExit(msg)
    for i in range(cfg_draft_layers(draft)):
        p = f"layers.{i}."
        if [lin(d, p + f"self_attn.{x}_proj") for x in "qkvo"] != [
            consts["d_q"],
            consts["d_k"],
            consts["d_v"],
            consts["d_o"],
        ]:
            msg = f"draft layer {i} differs"
            raise SystemExit(msg)
    consts["n_layers"] = len(types)
    consts["n_gdn"] = types.count("linear_attention")
    consts["n_attn"] = types.count("full_attention")
    consts["target_total_file_bytes"] = sum(t.values())
    return consts


def cfg_draft_layers(draft: str) -> int:
    """Return the draft model's layer count from its config.

    Args:
        draft: Draft model directory.

    Returns:
        The ``num_hidden_layers`` value.

    Raises:
        SystemExit: If the value is not an integer.

    """
    with pathlib.Path(f"{draft}/config.json").open(encoding="utf-8") as fh:
        n = json.load(fh)["num_hidden_layers"]
    if not isinstance(n, int):
        msg = f"draft num_hidden_layers is not an integer: {n!r}"
        raise SystemExit(msg)
    return n


class Args(argparse.Namespace):
    """Parsed command line arguments."""

    target: str
    draft: str


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print the roofline byte constants.")
    parser.add_argument("--target", required=True, help="target model directory")
    parser.add_argument("--draft", required=True, help="draft model directory")
    args = parser.parse_args(namespace=Args())
    sys.stdout.write(json.dumps(constants(args.target, args.draft), indent=1) + "\n")
