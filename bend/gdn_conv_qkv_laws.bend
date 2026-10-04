# Obligations for ext 5111 (gdn_conv_qkv.bend) against the stock transpose / cast / split / contiguous path
# (gdn_conv_qkv_spec.bend). Together: every (channel d, time t) input the fused conv reads is the qkv element the
# stock transposed bf16 copy held at the address the stock kernel reads (same fp32 value, same round-to-nearest-even
# bf16 cast, law 1); and the fused output (s, d) lands at the offset of its q / k / v view at which the stock split +
# contiguous tensor holds (s, d) (laws 2-4), the three views tiling the buffer back to back (law 5), each view's
# (s, j) -> s width + j a bijection (law 6), so the store map is a bijection onto the buffer and every value the chunk
# rule reads is the stock one. The conv arithmetic between the load and the store is the stock kernels' text
# (gdn_conv_qkv_diff.py). Domain: dim = 2 KD + V (the model's split), t < S, s any, bsz 1. Floating point is opaque.
import Base
import ./gdn_conv_qkv.bend as Impl
import ./gdn_conv_qkv_spec.bend as Spec

# 1. Same input element: the stock kernel's address in the transposed copy holds qkv's element t dim + d, which is the
# fused kernel's fp32 address.
law x_src:
  for +S: Nat
  for +dim: Nat
  for +d: Nat
  for +t: Nat
  for tb: {Nat.is_lt(t, S) == True{} : Bool}
  {Spec.trans_src(S, dim, Spec.x_off(S, dim, d, t)) == Impl.x_off(dim, d, t) : Nat}

# 2-4. The fused store of (s, start(b) + j) is the q / k / v view's element (s, j): view base + s width + j.
law view_q:
  for +S: Nat
  for +K: Nat
  for +V: Nat
  for +dim: Nat
  for +s: Nat
  for +j: Nat
  for jb: {Nat.is_lt(j, K) == True{} : Bool}
  {Impl.dst(S, K, dim, s, Nat.add(Spec.start(K, 0n), j)) == Nat.add(Impl.view_base(S, K, 0n), Spec.contig(K, V, 0n, s, j)) : Nat}

law view_k:
  for +S: Nat
  for +K: Nat
  for +V: Nat
  for +dim: Nat
  for +s: Nat
  for +j: Nat
  for jb: {Nat.is_lt(j, K) == True{} : Bool}
  {Impl.dst(S, K, dim, s, Nat.add(Spec.start(K, 1n), j)) == Nat.add(Impl.view_base(S, K, 1n), Spec.contig(K, V, 1n, s, j)) : Nat}

law view_v:
  for +S: Nat
  for +K: Nat
  for +V: Nat
  for +dim: Nat
  for +s: Nat
  for +j: Nat
  for shape: {Nat.add(Nat.mul(2n, K), V) == dim : Nat}
  {Impl.dst(S, K, dim, s, Nat.add(Spec.start(K, 2n), j)) == Nat.add(Impl.view_base(S, K, 2n), Spec.contig(K, V, 2n, s, j)) : Nat}

# 5. The views tile the buffer: q at 0, k right after q's S K elements, v right after k's, v ending at S dim.
law views_tile:
  for +S: Nat
  for +K: Nat
  for +V: Nat
  for +dim: Nat
  for shape: {Nat.add(Nat.mul(2n, K), V) == dim : Nat}
  {Impl.view_base(S, K, 1n) == Nat.add(Impl.view_base(S, K, 0n), Nat.mul(S, Spec.width(K, V, 0n))) : Nat} &
  {Impl.view_base(S, K, 2n) == Nat.add(Impl.view_base(S, K, 1n), Nat.mul(S, Spec.width(K, V, 1n))) : Nat} &
  {Nat.add(Impl.view_base(S, K, 2n), Nat.mul(S, Spec.width(K, V, 2n))) == Nat.mul(S, dim) : Nat}

# 6. Within a view, (s, j) -> s width + j names (s, j) back (j < width), and every offset is some (o / width, o % width).
law contig_inj:
  for +K: Nat
  for +V: Nat
  for +b: Nat
  for +s: Nat
  for +j: Nat
  for jb: {Nat.is_lt(j, Spec.width(K, V, b)) == True{} : Bool}
  {Nat.div(Spec.contig(K, V, b, s, j), Spec.width(K, V, b)) == s : Nat} &
  {Nat.mod(Spec.contig(K, V, b, s, j), Spec.width(K, V, b)) == j : Nat}

law contig_onto:
  for +K: Nat
  for +V: Nat
  for +b: Nat
  for +o: Nat
  for wpos: {Nat.is_lt(0n, Spec.width(K, V, b)) == True{} : Bool}
  {Spec.contig(K, V, b, Nat.div(o, Spec.width(K, V, b)), Nat.mod(o, Spec.width(K, V, b))) == o : Nat} &
  {Nat.is_lt(Nat.mod(o, Spec.width(K, V, b)), Spec.width(K, V, b)) == True{} : Bool}
