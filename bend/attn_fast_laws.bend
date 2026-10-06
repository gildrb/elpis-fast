# Obligations for the int8 QK^T operands of ext 3032 (attn_fast.bend, kernel exllamav3_ext/attn_verify.cu of the patched
# tree) against the reference attn_fast_spec.bend and the integer core of pattn8i_int.bend. Covered: the s8 K byte of
# every 3-bit code; the pairing of Q and K k-slots (one head dim per k-slot, both ways); one group IMMA is the exact
# integer group dot product, |J| <= 32 * 127 * 7 = 28448 < 2^22, and i2f_exact is exact on that range; the int8 q bytes
# and the q scales of a row are that row's own cells (addresses decode back to (row, group), inside the fp16 q region);
# the q absmax reads only the four lanes of its own fragment row, all four. Together: each score depends on its row's
# q and its token's K alone (row invariance of ext 3032 rests on this and on the unchanged schedule of
# attn_stride_laws.bend / attn_tree_laws.bend). NOT covered: the fp32 steps, the ldmatrix lane layout, the bit extraction
# of k3_s8 from the plane words (host C evidence in attn_fast_diff.py), PV16's fp16 accumulation.
import Base
import ./attn_fast.bend as Impl
import ./attn_fast_spec.bend as Spec
import ./pattn8i_int.bend as PImpl
import ./pattn8i_int_spec.bend as Z

# 1. K operand: for every code c < 8 the byte k3_s8 builds is the s8 value 2 c - 7 (canonical, |.| <= 7), and adding
# 0x79 to it carries out of no byte.
law k_operand:
  for +c: Nat
  for lt: {Nat.is_lt(c, 8n) == True{} : Bool}
  {Nat.is_lt(Nat.add(Nat.mul(c, 2n), 121n), 256n) == True{} : Bool} &
  {Spec.s8(Impl.kbyte(c)) == Spec.kval(c) : Z.Z} &
  {Nat.is_le(Z.mass(Spec.kval(c)), 7n) == True{} : Bool}

# 2. Slot pairing: Q stores head dim d of a group at k-slot qk8_slot(d), the K operand carries dim kdim(s) at k-slot s;
# the two maps are inverse bijections of 0..31, so at every k-slot the IMMA multiplies the q and K values of one dim and
# the group dot product is sum_d q8[d] (2 c_d - 7) in a permuted order (exact in integers).
law slot_pairing_q:
  for +d: Nat
  for lt: {Nat.is_lt(d, 32n) == True{} : Bool}
  {Nat.is_lt(Impl.qk8_slot(d), 32n) == True{} : Bool} &
  {Spec.kdim(Impl.qk8_slot(d)) == d : Nat}

law slot_pairing_k:
  for +s: Nat
  for lt: {Nat.is_lt(s, 32n) == True{} : Bool}
  {Nat.is_lt(Spec.kdim(s), 32n) == True{} : Bool} &
  {Impl.qk8_slot(Spec.kdim(s)) == s : Nat}

# 3. One group IMMA from a zero accumulator: 32 products of |q8| <= 127 (the clamp, pattn8i_int_laws.bend clamp_bound:
# the same kernel line) and |k| <= 7 (law 1) give the exact integer dot product J, |J| <= 28448 = 32 * (127 * 7) < 2^22.
law group_dot:
  for +q: List<&2, Z.Z>
  for +k: List<&2, Z.Z>
  for +ok: {Z.zip_ok(q, k, 127n, 7n) == True{} : Bool}
  for +n32: {Nat.is_eq(Z.len(k), 32n) == True{} : Bool}
  Z.zeq(PImpl.mma(Z.Z{0n, 0n}, q, k, Nat.sub(Nat.mul(32768n, 65536n), 1n)), Z.dot(q, k)) &
  {Nat.is_le(Z.abs(Z.dot(q, k)), Nat.mul(32n, Nat.mul(127n, 7n))) == True{} : Bool} &
  {Nat.is_lt(Nat.mul(32n, Nat.mul(127n, 7n)), Nat.mul(2048n, 2048n)) == True{} : Bool}

# 4. i2f_exact returns every group dot product exactly (h = 2^22 = Nat.mul(2048n, 2048n)).
law group_i2f:
  for +p: Nat
  for +n: Nat
  for +small: {Nat.is_le(Z.abs(Z.Z{p, n}), Nat.mul(32n, Nat.mul(127n, 7n))) == True{} : Bool}
  {PImpl.i2f_exact(Z.Z{p, n}, Nat.mul(2048n, 2048n)) == Some{Z.norm(Z.Z{p, n})} : Maybe<&2, Z.Z>}

# 5. Row-local q operands: the int8 q byte of (row r < 48, group g < 8, dim d < 32) decodes back to row r and to the
# in-row offset 32 g + qk8_slot(d) < 256 (so rows never share a byte), below 48 * 272; the q scale of (g, r) decodes to
# (g, r) below 8 * 48; the int8 rows and the scales fit the fp16 q region (48 * 264 halves) they replace.
law q_operand_rows:
  for +r: Nat
  for +g: Nat
  for +d: Nat
  for +rl: {Nat.is_lt(r, 48n) == True{} : Bool}
  for +gl: {Nat.is_lt(g, 8n) == True{} : Bool}
  for +dl: {Nat.is_lt(d, 32n) == True{} : Bool}
  {Nat.div(Impl.qaddr(r, g, d), 272n) == r : Nat} &
  {Nat.mod(Impl.qaddr(r, g, d), 272n) == Nat.add(Nat.mul(g, 32n), Impl.qk8_slot(d)) : Nat} &
  {Nat.is_lt(Nat.add(Nat.mul(g, 32n), Impl.qk8_slot(d)), 256n) == True{} : Bool} &
  {Nat.is_lt(Impl.qaddr(r, g, d), Nat.mul(48n, 272n)) == True{} : Bool}

law q_scale_rows:
  for +g: Nat
  for +r: Nat
  for +rl: {Nat.is_lt(r, 48n) == True{} : Bool}
  for +gl: {Nat.is_lt(g, 8n) == True{} : Bool}
  {Nat.mod(Impl.qsidx(g, r), 48n) == r : Nat} &
  {Nat.div(Impl.qsidx(g, r), 48n) == g : Nat} &
  {Nat.is_lt(Impl.qsidx(g, r), Nat.mul(8n, 48n)) == True{} : Bool}

law q_region:
  {Nat.is_le(Nat.add(Nat.mul(48n, 272n), Nat.mul(4n, Nat.mul(8n, 48n))), Nat.mul(Nat.mul(48n, 264n), 2n)) == True{} : Bool}

# 6. The per-(row, group) absmax: the shfl xor 1 and xor 2 partners of lane 4 gid + t are lanes of row gid, and the two
# steps cover the row's four lanes t, t ^ 1, t ^ 2, t ^ 3 (so the scale is the row's group max, from that row alone).
law amax_partners:
  for +gid: Nat
  for +t: Nat
  for tl: {Nat.is_lt(t, 4n) == True{} : Bool}
  {Nat.div(Impl.partner(gid, t, 1n), 4n) == gid : Nat} &
  {Nat.div(Impl.partner(gid, t, 2n), 4n) == gid : Nat} &
  {Spec.distinct4(t, Impl.txor(t, 1n), Impl.txor(t, 2n), Impl.txor(Impl.txor(t, 1n), 2n)) == True{} : Bool}
