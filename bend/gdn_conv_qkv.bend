# Ext 5111: the GDN prefill conv reading the fp32 qkv projection directly and writing q / k / v as three contiguous
# bf16 blocks (modules/gated_delta_net_fn/conv1d.py _conv1d_prefill_qkv_output_kernel / _state_kernel and
# conv1d_prefill_qkv), transcribed over Nat. F = patches/exl3-ext/5111-gdn-prefill-fuse-nom4096.patch (conv1d.py line =
# F line + 250 for F:91-248). Only the x addressing, the in-kernel bf16 cast and the output addressing differ from the
# stock kernels (checked by gdn_conv_qkv_diff.py); this file models those index maps. Reference: gdn_conv_qkv_spec.bend;
# laws: gdn_conv_qkv_laws.bend. Names: S = seq_len, KD = k_dim, dim = qkv's last dim, s = position, d = channel, t = input
# time (x_t >= 0 lanes; the masked lanes read other = 0.0 in both kernels).
import Base

# F:139 and F:199: x + x_t * dim + offs_d (fp32 (seq_len, dim) qkv, then .to(tl.bfloat16), F:142, F:202)
def x_off(+dim: Nat, +d: Nat, +t: Nat) -> Nat:
  Nat.add(Nat.mul(t, dim), d)

# F:154 VD: tl.constexpr = dim - 2 * KD
def vd(+dim: Nat, +KD: Nat) -> Nat:
  Nat.sub(dim, Nat.mul(2n, KD))

# F:155-158 dst = where(d < KD, s * KD + d, where(d < 2 * KD, seq_len * KD + s * KD + (d - KD),
#                  2 * seq_len * KD + s * VD + (d - 2 * KD)))
def dst.pick(a: Bool, b: Bool, +S: Nat, +KD: Nat, +dim: Nat, +s: Nat, +d: Nat) -> Nat:
  match a:
    case True{}:
      Nat.add(Nat.mul(s, KD), d)
    case False{}:
      match b:
        case True{}:
          Nat.add(Nat.add(Nat.mul(S, KD), Nat.mul(s, KD)), Nat.sub(d, KD))
        case False{}:
          Nat.add(Nat.add(Nat.mul(Nat.mul(2n, S), KD), Nat.mul(s, vd(dim, KD))), Nat.sub(d, Nat.mul(2n, KD)))

def dst(+S: Nat, +KD: Nat, +dim: Nat, +s: Nat, +d: Nat) -> Nat:
  dst.pick(Nat.is_lt(d, KD), Nat.is_lt(d, Nat.mul(2n, KD)), S, KD, dim, s, d)

# F:245-247 q = buf[: seq_len * k_dim], k = buf[seq_len * k_dim: 2 * seq_len * k_dim], v = buf[2 * seq_len * k_dim:],
# each viewed (1, seq_len, width) contiguous
def view_base(+S: Nat, +KD: Nat, +b: Nat) -> Nat:
  match b:
    case 0n:
      0n
    case 1n+c:
      match c:
        case 0n:
          Nat.mul(S, KD)
        case 1n+e:
          Nat.mul(Nat.mul(2n, S), KD)
