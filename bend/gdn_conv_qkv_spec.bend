# Independent reference for ext 5111 (bend/gdn_conv_qkv.bend): the stock long-chunk GDN prefill conv path it replaces
# (bsz 1, seq_len S > 256, so conv1d.py's Triton output / state kernels run: MAX_CUDA_SEQLEN = 32 < S). Sources, stock
# ExLlamaV3 355c6ee (conv1d.py unchanged by the series before 5111: exl3-ext.json "pre" hash a08e9909):
# G = modules/gated_delta_net.py (stock: mixed_qkv = qkv.transpose(1, 2).to(torch.bfloat16).contiguous()),
# C = modules/gated_delta_net_fn/conv1d.py (causal_conv1d_update -> causal_conv1d_update_slotted_triton, transpose_output
# = True), R = modules/gated_delta_net_fn/gated_delta_rule.py (q, k, v = torch.split(mixed_qkv, [k_dim, k_dim, v_dim], -1),
# made contiguous by fla's input_guard). Names: S = seq_len, K = k_dim, V = v_dim, dim = 2 K + V (qkv's last dim).
import Base

# G: qkv (1, S, dim) fp32 contiguous -> .transpose(1, 2).contiguous(): element j of the (dim, S) copy is qkv[(j % S), (j / S)],
# i.e. qkv's flat element (j % S) dim + j / S (the cast to bf16 is applied elementwise, round to nearest even)
def trans_src(+S: Nat, +dim: Nat, +j: Nat) -> Nat:
  Nat.add(Nat.mul(Nat.mod(j, S), dim), Nat.div(j, S))

# C:159 and C:224: x + (pid_b * dim + offs_d) * seq_len + x_t, pid_b = 0: the element the output / state kernel reads for
# channel d at input time t
def x_off(+S: Nat, +dim: Nat, +d: Nat, +t: Nat) -> Nat:
  Nat.add(Nat.mul(Nat.add(Nat.mul(0n, dim), d), S), t)

# C:173-177 (transpose_output): the output kernel stores the value of (position s, channel d) at out + (pid_b * seq_len
# + offs_s) * dim + offs_d of a fresh (1, S, dim) bf16 tensor; R's split views read column d of row s from there.

# R: torch.split(out, [K, K, V], -1): block b (0 = q, 1 = k, 2 = v) is the columns start(b) .. start(b) + width(b) - 1
def start(+K: Nat, +b: Nat) -> Nat:
  match b:
    case 0n:
      0n
    case 1n+c:
      match c:
        case 0n:
          K
        case 1n+e:
          Nat.add(K, K)

def width(+K: Nat, +V: Nat, +b: Nat) -> Nat:
  match b:
    case 0n:
      K
    case 1n+c:
      match c:
        case 0n:
          K
        case 1n+e:
          V

# .contiguous() of block b's (1, S, width) view: element (s, j) at offset s * width + j of its own tensor
def contig(+K: Nat, +V: Nat, +b: Nat, +s: Nat, +j: Nat) -> Nat:
  Nat.add(Nat.mul(s, width(K, V, b)), j)
