# Ext 3032 (patches/exl3-ext/3032-attn-verify-int8qk-pv16.patch, EXL3_AV_FAST bit 1): the int8 QK^T operands of the
# verify split kernel, transcribed over Nat. K = exllamav3_ext/attn_verify.cu of the patched tree (OUT/patched). Covers:
# the Q k-slot order (qk8_slot), the bytes of the s8 K operand (k3_s8), the int8 q and fp32 q-scale addresses in shared
# memory, and the lanes the per-(row, group) q absmax reads. The group IMMA and i2f_exact are pattn8i_int.bend's
# (Impl.mma, Impl.i2f_exact: the same PTX string and the same function, checked by attn_fast_diff.py). Not modelled:
# the fp32 steps (q rotation, quantisation, the q scale, qsc8 * s, the group FMA, x scale_log2), the ldmatrix lane
# layout, the bit extraction of k3_s8 from the plane words (checked exhaustively against the exact kernel's plane
# decoding by attn_fast_diff.py), and PV16 (fp16 accumulation is floating point; its schedule is the exact kernel's).
# Reference: attn_fast_spec.bend; laws: attn_fast_laws.bend; source link: attn_fast_diff.py.
import Base
import ./attn_fast_spec.bend as Spec

def pickn(c: Bool, +x: Nat, +y: Nat) -> Nat:
  match c:
    case True{}:
      x
    case False{}:
      y

# K: return (d & 16) | ((d & 3) << 2) | ((d >> 2) & 3);  for d < 32: d & 16 = 16 ((d / 16) mod 2), (d & 3) << 2 =
# 4 (d mod 4), (d >> 2) & 3 = (d / 4) mod 4; the fields occupy disjoint bits (16, 12, 3), so | is +.
def qk8_slot(+d: Nat) -> Nat:
  Nat.add(Nat.mul(16n, Nat.mod(Nat.div(d, 16n), 2n)), Nat.add(Nat.mul(4n, Nat.mod(d, 4n)), Nat.mod(Nat.div(d, 4n), 4n)))

# K: k3_s8, byte i (code c = 2 x2 + x1 of the byte's dim): (x2s & 0x0C0C0C0Cu) holds 4 x2 and
# (((y << 1) | (y << 5)) & 0x02020202u) holds 2 x1 (disjoint bits), so u_i = 2 c; (u + 0x79797979u) adds 121 to every
# byte (no byte carries: law k_operand); ^ 0x80808080u flips bit 7 of each byte: v -> v + 128 below 128, v - 128 from
# 128 (v < 256).
def xor128(+v: Nat) -> Nat:
  pickn(Nat.is_lt(v, 128n), Nat.add(v, 128n), Nat.sub(v, 128n))

def kbyte(+c: Nat) -> Nat:
  xor128(Nat.add(Nat.mul(c, 2n), 121n))

# K: qrow = qs8 + r * QS8 + 32 * g; qrow[qk8_slot(8 * nt + 2 * t + e)] = q8 (QS8 = AV_HD + 16 = 272): the int8 q byte of
# (row r, group g, dim d) at byte r * 272 + 32 g + qk8_slot(d) (sum reordered), read back by the QK8 ldmatrix rows
# qs8 + (16 i + (lane & 15)) * QS8 + 32 * g + 16 * (lane >> 4).
def qaddr(+r: Nat, +g: Nat, +d: Nat) -> Nat:
  Nat.add(Nat.add(Nat.mul(g, 32n), qk8_slot(d)), Nat.mul(r, 272n))

# K: qsc8[g * AV_ROWS + r] (AV_ROWS = 48), written by the row's t == 0 lane, read as qsc8[g * AV_ROWS + 16 * i + gid]
# (+ 8 for the second m-tile half).
def qsidx(+g: Nat, +r: Nat) -> Nat:
  Nat.add(r, Nat.mul(g, 48n))

# XOR of the low bits of a and b (a, b < 2 in use)
def bxor(+a: Nat, +b: Nat) -> Nat:
  pickn(Nat.is_eq(Nat.mod(a, 2n), Nat.mod(b, 2n)), 0n, 1n)

# t ^ s for t, s < 4
def txor(+t: Nat, +s: Nat) -> Nat:
  Nat.add(bxor(t, s), Nat.mul(2n, bxor(Nat.div(t, 2n), Nat.div(s, 2n))))

# K: amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, s)) for s = 1, then 2: lane 4 gid + t reads lane
# (4 gid + t) ^ s = 4 gid + (t ^ s) (s < 4 touches the low two bits only).
def partner(+gid: Nat, +t: Nat, +s: Nat) -> Nat:
  Nat.add(txor(t, s), Nat.mul(gid, 4n))
