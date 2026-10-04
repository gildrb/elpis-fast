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
    python3 -I -B bend/gen/roofline_inventory.py --target TARGET --draft DRAFT > /tmp/x
    cmp /tmp/x bend/gen/roofline_inventory.json

Origin: the inventory script of the roofline work. Only the model directory
arguments, the duplicate tensor check and the explicit file encodings changed.
"""

import argparse
import glob
import json
import struct

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
    out = {}
    files = sorted(glob.glob(f"{model_dir}/*.safetensors"))
    if not files:
        raise SystemExit(f"no safetensors in {model_dir}")
    for f in files:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        for k, v in h.items():
            if k == "__metadata__":
                continue
            a, b = v["data_offsets"]
            nel = 1
            for s in v["shape"]:
                nel *= s
            if nel * DT[v["dtype"]] != b - a:
                raise SystemExit(f"size mismatch {f}:{k}")
            if k in out:
                raise SystemExit(f"duplicate tensor {f}:{k}")
            out[k] = b - a
    return out


def lin(tb: dict[str, int], p: str) -> int:
    """EXL3 linear streamed bytes: trellis + suh + svh (mul1 is a 4-byte scalar argument)."""
    return tb[p + ".trellis"] + tb[p + ".suh"] + tb[p + ".svh"]


def constants(target: str, draft: str) -> dict[str, int]:
    t, d = load(target), load(draft)
    with open(f"{target}/config.json", encoding="utf-8") as fh:
        cfg = json.load(fh)["text_config"]
    L = "model.language_model.layers."
    types = cfg["layer_types"]
    gdn0 = types.index("linear_attention")
    att0 = types.index("full_attention")
    C = dict(
        mlp_gate=lin(t, f"{L}0.mlp.gate_proj"),
        mlp_up=lin(t, f"{L}0.mlp.up_proj"),
        mlp_down=lin(t, f"{L}0.mlp.down_proj"),
        ln_in=t[f"{L}0.input_layernorm.weight"],
        ln_post=t[f"{L}0.post_attention_layernorm.weight"],
        gdn_qkv=lin(t, f"{L}{gdn0}.linear_attn.in_proj_qkv"),
        gdn_z=lin(t, f"{L}{gdn0}.linear_attn.in_proj_z"),
        gdn_out=lin(t, f"{L}{gdn0}.linear_attn.out_proj"),
        gdn_a=t[f"{L}{gdn0}.linear_attn.in_proj_a.weight"],
        gdn_b=t[f"{L}{gdn0}.linear_attn.in_proj_b.weight"],
        gdn_conv1d=t[f"{L}{gdn0}.linear_attn.conv1d.weight"],
        gdn_alog=t[f"{L}{gdn0}.linear_attn.A_log"],
        gdn_dtb=t[f"{L}{gdn0}.linear_attn.dt_bias"],
        gdn_norm=t[f"{L}{gdn0}.linear_attn.norm.weight"],
        attn_q=lin(t, f"{L}{att0}.self_attn.q_proj"),
        attn_k=lin(t, f"{L}{att0}.self_attn.k_proj"),
        attn_v=lin(t, f"{L}{att0}.self_attn.v_proj"),
        attn_o=lin(t, f"{L}{att0}.self_attn.o_proj"),
        attn_qn=t[f"{L}{att0}.self_attn.q_norm.weight"],
        attn_kn=t[f"{L}{att0}.self_attn.k_norm.weight"],
        final_norm=t["model.language_model.norm.weight"],
        head=lin(t, "lm_head"),
        d_q=lin(d, "layers.0.self_attn.q_proj"),
        d_k=lin(d, "layers.0.self_attn.k_proj"),
        d_v=lin(d, "layers.0.self_attn.v_proj"),
        d_o=lin(d, "layers.0.self_attn.o_proj"),
        d_gate=lin(d, "layers.0.mlp.gate_proj"),
        d_up=lin(d, "layers.0.mlp.up_proj"),
        d_down=lin(d, "layers.0.mlp.down_proj"),
        d_akp=d["layers.0.attention_conv.kernel_projection.weight"],
        d_mkp=d["layers.0.mlp_conv.kernel_projection.weight"],
        d_abk=d["layers.0.attention_conv.base_kernel"],
        d_mbk=d["layers.0.mlp_conv.base_kernel"],
        d_ln_in=d["layers.0.input_layernorm.weight"],
        d_ln_post=d["layers.0.post_attention_layernorm.weight"],
        d_qn=d["layers.0.self_attn.q_norm.weight"],
        d_kn=d["layers.0.self_attn.k_norm.weight"],
        d_norm=d["norm.weight"],
        d_hnorm=d["hidden_norm.weight"],
        d_fc=lin(d, "fc"),
        d_hproj=d["candidate_selector.hidden_projection.weight"],
    )
    # Every layer of a kind carries byte-identical tensors (uniform 4.00 bpw):
    # the per-kind constants above stand for all 64 target / 5 draft layers.
    for i, kind in enumerate(types):
        p = f"{L}{i}."
        same = [
            lin(t, p + "mlp.gate_proj") == C["mlp_gate"],
            lin(t, p + "mlp.up_proj") == C["mlp_up"],
            lin(t, p + "mlp.down_proj") == C["mlp_down"],
        ]
        if kind == "linear_attention":
            same += [
                lin(t, p + "linear_attn.in_proj_qkv") == C["gdn_qkv"],
                lin(t, p + "linear_attn.in_proj_z") == C["gdn_z"],
                lin(t, p + "linear_attn.out_proj") == C["gdn_out"],
                t[p + "linear_attn.in_proj_a.weight"] == C["gdn_a"],
                t[p + "linear_attn.in_proj_b.weight"] == C["gdn_b"],
                t[p + "linear_attn.conv1d.weight"] == C["gdn_conv1d"],
            ]
        elif kind == "full_attention":
            same += [
                lin(t, p + f"self_attn.{x}_proj") == C[f"attn_{x}"] for x in "qkvo"
            ]
        else:
            raise SystemExit(f"layer {i}: unknown kind {kind}")
        if not all(same):
            raise SystemExit(
                f"layer {i} ({kind}) is not byte-identical to its kind's constants"
            )
    for i in range(cfg_draft_layers(draft)):
        p = f"layers.{i}."
        if [lin(d, p + f"self_attn.{x}_proj") for x in "qkvo"] != [
            C["d_q"],
            C["d_k"],
            C["d_v"],
            C["d_o"],
        ]:
            raise SystemExit(f"draft layer {i} differs")
    C["n_layers"] = len(types)
    C["n_gdn"] = types.count("linear_attention")
    C["n_attn"] = types.count("full_attention")
    C["target_total_file_bytes"] = sum(t.values())
    return C


def cfg_draft_layers(draft: str) -> int:
    with open(f"{draft}/config.json", encoding="utf-8") as fh:
        return json.load(fh)["num_hidden_layers"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print the roofline byte constants.")
    parser.add_argument("--target", required=True, help="target model directory")
    parser.add_argument("--draft", required=True, help="draft model directory")
    args = parser.parse_args()
    print(json.dumps(constants(args.target, args.draft), indent=1))
