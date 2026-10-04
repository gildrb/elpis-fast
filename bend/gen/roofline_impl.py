"""Emit bend/roofline.bend from the safetensors-header inventory.

Byte constants are written as decimal digit lists (Bin.dec) so the checker
evaluates them in binary.

Input: bend/gen/roofline_inventory.json (or --inventory PATH). It holds the byte
count of each tensor kind. bend/gen/roofline_inventory.py makes it from the
safetensors headers of the target and draft models.
Output: the Bend source, on stdout.

Regenerate and check the tracked file (from the repository root):
    python3 -I -B bend/gen/roofline_impl.py > /tmp/x && cmp /tmp/x bend/roofline.bend

Origin: the generator of the roofline work. Only the input path handling, the
generator lines of the header, the value check and the UTF-8 output changed.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description="Emit bend/roofline.bend on stdout.")
parser.add_argument(
    "--inventory",
    type=Path,
    default=ROOT / "bend/gen/roofline_inventory.json",
    help="inventory JSON (default: bend/gen/roofline_inventory.json)",
)
C = json.loads(parser.parse_args().inventory.read_text(encoding="utf-8"))


def dg(n: int) -> str:
    return "[" + ", ".join(f"{c}n" for c in str(n)) + "]"


def const(name: str, value: int, cite: str) -> str:
    if type(value) is not int or value < 0:
        raise SystemExit(f"roofline_impl: FAIL: {name}: {value!r} is not a byte count")
    return f"# {cite}\ndef {name}() -> N.Bin:\n  N.Bin.dec({dg(value)})\n"


TENSORS = [
    # (bend name, inventory key, citation)
    (
        "mlp_gate",
        "mlp_gate",
        "layers.*.mlp.gate_proj.{trellis [320,1088,64] I16, suh [5120], svh [17408] F16}",
    ),
    ("mlp_up", "mlp_up", "layers.*.mlp.up_proj.{trellis [320,1088,64] I16, suh, svh}"),
    (
        "mlp_down",
        "mlp_down",
        "layers.*.mlp.down_proj.{trellis [1088,320,64] I16, suh [17408], svh [5120]}",
    ),
    ("ln_in", "ln_in", "layers.*.input_layernorm.weight [5120] BF16"),
    ("ln_post", "ln_post", "layers.*.post_attention_layernorm.weight [5120] BF16"),
    (
        "gdn_qkv",
        "gdn_qkv",
        "linear_attn.in_proj_qkv.{trellis [320,640,64] I16, suh [5120], svh [10240]}",
    ),
    (
        "gdn_z",
        "gdn_z",
        "linear_attn.in_proj_z.{trellis [320,384,64] I16, suh [5120], svh [6144]}",
    ),
    (
        "gdn_out",
        "gdn_out",
        "linear_attn.out_proj.{trellis [384,320,64] I16, suh [6144], svh [5120]}",
    ),
    ("gdn_a", "gdn_a", "linear_attn.in_proj_a.weight [48,5120] F16"),
    ("gdn_b", "gdn_b", "linear_attn.in_proj_b.weight [48,5120] F16"),
    ("gdn_conv1d", "gdn_conv1d", "linear_attn.conv1d.weight [10240,1,4] BF16"),
    ("gdn_alog", "gdn_alog", "linear_attn.A_log [48] BF16"),
    ("gdn_dtb", "gdn_dtb", "linear_attn.dt_bias [48] BF16"),
    ("gdn_norm", "gdn_norm", "linear_attn.norm.weight [128] BF16"),
    (
        "attn_q",
        "attn_q",
        "self_attn.q_proj.{trellis [320,768,64] I16, suh [5120], svh [12288]}",
    ),
    (
        "attn_k",
        "attn_k",
        "self_attn.k_proj.{trellis [320,64,64] I16, suh [5120], svh [1024]}",
    ),
    (
        "attn_v",
        "attn_v",
        "self_attn.v_proj.{trellis [320,64,64] I16, suh [5120], svh [1024]}",
    ),
    (
        "attn_o",
        "attn_o",
        "self_attn.o_proj.{trellis [384,320,64] I16, suh [6144], svh [5120]}",
    ),
    ("attn_qn", "attn_qn", "self_attn.q_norm.weight [256] BF16"),
    ("attn_kn", "attn_kn", "self_attn.k_norm.weight [256] BF16"),
    ("final_norm", "final_norm", "model.language_model.norm.weight [5120] BF16"),
    (
        "head",
        "head",
        "lm_head.{trellis [320,15520,96] I16 (6 bpw), suh [5120], svh [248320] F16}",
    ),
    (
        "d_q",
        "d_q",
        "draft layers.*.self_attn.q_proj.{trellis [320,256,64] I16, suh [5120], svh [4096]}",
    ),
    (
        "d_k",
        "d_k",
        "draft layers.*.self_attn.k_proj.{trellis [320,64,64] I16, suh [5120], svh [1024]}",
    ),
    (
        "d_v",
        "d_v",
        "draft layers.*.self_attn.v_proj.{trellis [320,64,64] I16, suh [5120], svh [1024]}",
    ),
    (
        "d_o",
        "d_o",
        "draft layers.*.self_attn.o_proj.{trellis [256,320,64] I16, suh [4096], svh [5120]}",
    ),
    ("d_gate", "d_gate", "draft layers.*.mlp.gate_proj (as target)"),
    ("d_up", "d_up", "draft layers.*.mlp.up_proj (as target)"),
    ("d_down", "d_down", "draft layers.*.mlp.down_proj (as target)"),
    (
        "d_akp",
        "d_akp",
        "draft layers.*.attention_conv.kernel_projection.weight [1280,5120] F16",
    ),
    (
        "d_mkp",
        "d_mkp",
        "draft layers.*.mlp_conv.kernel_projection.weight [1280,5120] F16",
    ),
    ("d_abk", "d_abk", "draft layers.*.attention_conv.base_kernel [2,2,5120] F16"),
    ("d_mbk", "d_mbk", "draft layers.*.mlp_conv.base_kernel [2,2,5120] F16"),
    ("d_ln_in", "d_ln_in", "draft layers.*.input_layernorm.weight [5120] BF16"),
    (
        "d_ln_post",
        "d_ln_post",
        "draft layers.*.post_attention_layernorm.weight [5120] BF16",
    ),
    ("d_qn", "d_qn", "draft layers.*.self_attn.q_norm.weight [128] BF16"),
    ("d_kn", "d_kn", "draft layers.*.self_attn.k_norm.weight [128] BF16"),
    ("d_norm", "d_norm", "draft norm.weight [5120] BF16"),
    ("d_hnorm", "d_hnorm", "draft hidden_norm.weight [5120] BF16"),
    ("d_fc", "d_fc", "draft fc.{trellis [1600,320,64] I16, suh [25600], svh [5120]}"),
    (
        "d_hproj",
        "d_hproj",
        "draft candidate_selector.hidden_projection.weight [256,5120] F16",
    ),
]

HEAD = """# DRAM roofline of one greedy speculative verify round (M = 8 target rows)
# of the committed g7n engine: the bytes each round moves, as the linear form
# bytes(d, c, w) = c0 + kd*d + kc*c + kw*w of the committed depth d, the round's
# committed count c (1..8) and the draft window read w = min(d + 8, 2056)
# (roofline_bin.bend). GENERATED by bend/gen/roofline_impl.py from
# bend/gen/roofline_inventory.json: the safetensors headers (8-byte length +
# JSON header only, bend/gen/roofline_inventory.py) of
# /models/qwen38-27b-exl3 (target) and /models/dflash2-exl3 (draft); every
# layer of a kind is byte-identical there, so one constant per tensor kind.
# EXL3 linear = trellis + suh + svh (mul1 is a 4-byte kernel scalar).
# Traffic facts (file:line) are from the pristine engine (the stock
# ExLlamaV3 355c6ee exllamav3 package) plus patches/exl3{,-ext}:
#  - CQ3 KV (cache/quant.py:32-41, q_cache_kernels.cuh:9-23): per token per
#    layer K and V each 1024*3/8 words + 1024/32 fp16 scales = 448 B.
#  - Full attention reads all d + 8 tokens (3002 L181-227, lo = 0) after the
#    8 new rows are quantized in (attention.cpp:568).
#  - GDN state row 0: 48*128*128 fp32 (gated_delta_net.py:187-191); read once
#    by RULE_VERIFY (0003 L84), read + written once by batched_state_replay
#    (0003 L225-250) every round (0002 gdn_rewind.py L267-302).
#  - Conv state 10240 x (4 + 7) bf16: verify reads 4 entries and writes all 11
#    per channel (5001 L556, L598-602); rewind copies 4 (gdn.cu:1904-1921).
#  - Replay statics per verify row: conv_out 10240 bf16, g 48 fp32, beta 48
#    bf16 (0004); replay reads c rows of them.
#  - Draft: CQ3 cache, 5 layers x 8 kv heads x 128 = 896 B/token/layer,
#    sliding window 2048 + the 8-row block (dflash2.py:108-125).
#  - Draft sampling runs the full target lm_head on 8 rows (dflash2.py:260-292;
#    4001 not in the series), then hidden_projection (fp16.py:93-94), topk and a
#    gather of 1 + 16 codebook rows of 256 fp16 per draft row (dflash2.cu:166-236).
#  - update_kv_from_target: fc on c tap rows (5 x 5120 fp16), hidden_norm,
#    k/v proj + k_norm per draft layer, c tokens quantized in (dflash.py:82-139).
#  - Embeddings: CPU gathers, 8 x 5120 fp16 uploaded each for target and draft
#    (embedding.py:143-167).
# Counting rule (lower bound): every weight, state, cache and round-crossing
# buffer byte once per use; outputs once per write (eventual write-back);
# a read by the immediately following kernel of a buffer <= 4 MB and
# intra-phase activations are L2-resident and not counted.
import Base
import ./roofline_bin.bend as N

def k(ds: List<&2, Nat>) -> N.Bin:
  N.Bin.dec(ds)

def c(ds: List<&2, Nat>) -> N.Form:
  N.Form.const(N.Bin.dec(ds))

def times(n: Nat, +b: N.Bin) -> N.Bin:
  N.Bin.mul(N.Bin.small(n), b)

def plus(xs: List<&2, N.Bin>) -> N.Bin:
  match xs:
    case Nil{}:
      N.BE{}
    case Con{x, rest}:
      N.Bin.add(x, plus(rest))

# ---- tensor bytes (safetensors headers) ----
"""

BODY = """
# ---- traffic bytes (layouts above) ----
# CQ3 K + V bytes per token per target attention layer.
def kv_tok() -> N.Bin:
  k([8n, 9n, 6n])

# GDN recurrent row 0 (48 x 128 x 128 fp32).
def state_row() -> N.Bin:
  k([3n, 1n, 4n, 5n, 7n, 2n, 8n])

# conv verify: read 4 bf16 per channel; write 11 bf16 per channel.
def conv_read() -> N.Bin:
  k([8n, 1n, 9n, 2n, 0n])

def conv_write() -> N.Bin:
  k([2n, 2n, 5n, 2n, 8n, 0n])

# replay statics per verify row: conv_out 20480 + g 192 + beta 96.
def statics_row() -> N.Bin:
  k([2n, 0n, 7n, 6n, 8n])

# fp16 logits of 8 rows x 248320.
def logits() -> N.Bin:
  k([3n, 9n, 7n, 3n, 1n, 2n, 0n])

# fp16 tap export: 5 taps x 8 rows x 5120; refresh reads c rows x 25600 fp16.
def taps_write() -> N.Bin:
  k([4n, 0n, 9n, 6n, 0n, 0n])

def tap_row() -> N.Bin:
  k([5n, 1n, 2n, 0n, 0n])

# Draft CQ3 K + V bytes per token over its 5 layers.
def dkv_tok() -> N.Bin:
  k([4n, 4n, 8n, 0n])

# selector gathers: 7 draft rows x (1 + 16) rows x 256 fp16.
def gathers() -> N.Bin:
  k([6n, 0n, 9n, 2n, 8n])

# 8 x 5120 fp16 embedding upload.
def embed_up() -> N.Bin:
  k([8n, 1n, 9n, 2n, 0n])

# ---- per-layer groups ----
def mlp_layer() -> N.Bin:
  plus([mlp_gate(), mlp_up(), mlp_down()])

def gdn_proj() -> N.Bin:
  plus([gdn_qkv(), gdn_z(), gdn_out()])

def gdn_small_w() -> N.Bin:
  plus([gdn_a(), gdn_b(), gdn_conv1d(), gdn_alog(), gdn_dtb(), gdn_norm()])

def attn_proj() -> N.Bin:
  plus([attn_q(), attn_k(), attn_v(), attn_o()])

def draft_layer() -> N.Bin:
  plus([d_q(), d_k(), d_v(), d_o(), d_gate(), d_up(), d_down(), d_akp(), d_mkp(), d_abk(),
    d_mbk(), d_ln_in(), d_ln_post(), d_qn(), d_kn()])

# ---- phases: one form per kernel-trace-3 bucket (kernel_trace.py BUCKETS) ----
def ph_target_mlp_gemm() -> N.Form:
  N.Form.const(times(64n, mlp_layer()))

def ph_gdn_gemm() -> N.Form:
  N.Form.const(times(48n, gdn_proj()))

# ba + conv (gdn_conv_ba_kernel), recurrent rule (state row 0 read), statics write.
def ph_gdn_small() -> N.Form:
  N.Form.const(times(48n, plus([gdn_small_w(), state_row(), conv_read(), conv_write(),
    N.Bin.mul(N.Bin.small(8n), statics_row())])))

def ph_attn_proj_gemm() -> N.Form:
  N.Form.const(times(16n, attn_proj()))

# split + combine + rope/quant/norm kernels: reads d + 8 tokens, writes 8.
def ph_attention() -> N.Form:
  +kv = times(16n, kv_tok())
  N.Form{plus([times(16n, N.Bin.add(attn_qn(), attn_kn())), times(8n, kv), times(8n, kv)]),
    kv, N.BE{}, N.BE{}}

def ph_norms_residual() -> N.Form:
  N.Form.const(plus([times(64n, N.Bin.add(ln_in(), ln_post())), final_norm(), taps_write()]))

def ph_lm_head() -> N.Form:
  N.Form.const(N.Bin.add(head(), logits()))

# Draft forward: 5 layers + final norm; window read, 8 block tokens written.
def ph_draft_forward() -> N.Form:
  N.Form{N.Bin.add(N.Bin.add(times(5n, draft_layer()), d_norm()), times(8n, dkv_tok())),
    N.BE{}, N.BE{}, dkv_tok()}

def ph_draft_sample_head() -> N.Form:
  N.Form.const(plus([head(), logits(), d_hproj()]))

def ph_draft_sample_walk() -> N.Form:
  N.Form.const(gathers())

# state replay read + write, conv rewind read + write, c statics rows read.
def ph_rewind_replay() -> N.Form:
  N.Form{times(48n, N.Bin.add(times(2n, state_row()), times(2n, conv_read()))), N.BE{},
    times(48n, statics_row()), N.BE{}}

# fc + hidden_norm + 5 x (k, v, k_norm); c tap rows read, c draft tokens written.
def ph_draft_kv_refresh() -> N.Form:
  N.Form{plus([d_fc(), d_hnorm(), times(5n, plus([d_k(), d_v(), d_kn()]))]), N.BE{},
    N.Bin.add(tap_row(), dkv_tok()), N.BE{}}

def ph_memcpy() -> N.Form:
  N.Form.const(times(2n, embed_up()))

# Buckets whose floor is zero bytes under the counting rule (their inputs are
# L2-resident immediate reads): target_mlp_other, sampler, draft_sample_topk,
# draft_sample_other, and the idle time between kernels.
def phases() -> List<&2, N.Form>:
  [ph_target_mlp_gemm(), N.Form.zero(), ph_gdn_gemm(), ph_gdn_small(), ph_attn_proj_gemm(),
    ph_attention(), ph_norms_residual(), ph_lm_head(), N.Form.zero(), ph_draft_forward(),
    ph_draft_sample_head(), N.Form.zero(), ph_draft_sample_walk(), N.Form.zero(),
    ph_rewind_replay(), ph_draft_kv_refresh(), ph_memcpy(), N.Form.zero()]

# ---- round total: closed form by traffic class ----
def target_weights() -> N.Bin:
  plus([times(64n, plus([mlp_layer(), ln_in(), ln_post()])),
    times(48n, N.Bin.add(gdn_proj(), gdn_small_w())),
    times(16n, plus([attn_proj(), attn_qn(), attn_kn()])), final_norm(), head()])

def draft_weights() -> N.Bin:
  plus([times(5n, draft_layer()), d_norm(), head(), d_hproj(), d_fc(), d_hnorm(),
    times(5n, plus([d_k(), d_v(), d_kn()]))])

# 3 row-0 passes (verify read, replay read, replay write) per GDN layer.
def gdn_state() -> N.Bin:
  times(48n, times(3n, state_row()))

def conv_state() -> N.Bin:
  times(48n, plus([conv_read(), conv_write(), times(2n, conv_read())]))

def fixed_activations() -> N.Bin:
  plus([times(48n, times(8n, statics_row())), times(2n, logits()), taps_write(), gathers(),
    times(2n, embed_up())])

# c0: all fixed bytes, incl. the 8 new target KV tokens written and read back
# and the 8 draft block tokens written.
def round() -> N.Form:
  +kv = times(16n, kv_tok())
  N.Form{plus([target_weights(), draft_weights(), gdn_state(), conv_state(),
      fixed_activations(), times(16n, kv), times(8n, dkv_tok())]),
    kv,
    plus([times(48n, statics_row()), tap_row(), dkv_tok()]),
    dkv_tok()}

# ---- reference totals the trace prints (differential anchors) ----
# kernel_trace.py target_all_weight_gemms: mlp + gdn qkv/z/out + attn qkvo + lm_head.
def trace_target_gemm_bytes() -> N.Bin:
  plus([times(64n, mlp_layer()), times(48n, gdn_proj()), times(16n, attn_proj()), head()])

# kernel_trace.py draft_forward_linears: q, k, v, o, gate, up, down of 5 layers.
def trace_draft_linears() -> N.Bin:
  times(5n, plus([d_q(), d_k(), d_v(), d_o(), d_gate(), d_up(), d_down()]))
"""

out = [HEAD]
for name, key, cite in TENSORS:
    out.append(const(name, C[key], cite))
out.append(BODY)
sys.stdout.buffer.write("\n".join(out).encode("utf-8"))
