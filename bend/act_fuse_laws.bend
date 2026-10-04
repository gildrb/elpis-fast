# Obligations for ext 5112 (gemm_wide_silu_kernel epilogue, act_fuse.bend) against the stock hgemm_recon + silu_mul
# path (act_fuse_spec.bend). Together: every output half2 the fused kernel stores sits at the same element pair
# (2 i, 2 i + 1) as act_mul_kernel_h's index i, and is the same term (the same fp16 operations on the same operands in
# the same dependency order, so the same rounding points), for either act_limit branch; the stock launch handles each
# i < numel / 2 once; and the fused route is taken only where stock hgemm_recon runs the 3011 wide kernel. The
# accumulator pair AccPair{row, col} is the same fp32 value in both kernels because their K loops are the same text
# (act_fuse_diff.py) with hgemm_wide's proven k_order. Domain: N = 2 n (N even; the route requires N = 17408), the
# output contiguous (ldc = N), col even (law 1). Floating point values are opaque terms.
import Base
import ./act_fuse.bend as Impl
import ./act_fuse_spec.bend as Spec
import ./hgemm_wide.bend as HW
import ./hgemm_wide_spec.bend as HWS

# 1. Store columns are pair-aligned: the fragment store (tile tn, thread tid, j) writes columns 2 c and 2 c + 1 of one row.
law epi_col_pair:
  for +tn: Nat
  for +tid: Nat
  for +j: Nat
  {HW.col_of(Nat.mul(tn, 128n), tid, j, 0n) == Nat.mul(Impl.pair_col(tn, tid, j), 2n) : Nat} &
  {HW.col_of(Nat.mul(tn, 128n), tid, j, 1n) == 1n+Nat.mul(Impl.pair_col(tn, tid, j), 2n) : Nat}

# 2. Pair index: with ldc = N = 2 n and column 2 c (c < n), the epilogue's element offset is 2 i with i = row n + c, and
# i names (row, c) back; every i is some (i / n, i % n) with i % n < n. So (row, c) <-> i is a bijection between the
# epilogue's pair stores of rows < M and act_mul_kernel_h's indices i < M n = numel / 2.
law pair_index:
  for +n: Nat
  for +row: Nat
  for +c: Nat
  for cn: {Nat.is_lt(c, n) == True{} : Bool}
  {Impl.epi_off(Nat.mul(n, 2n), row, Nat.mul(c, 2n)) == Nat.mul(Nat.add(Nat.mul(row, n), c), 2n) : Nat} &
  {Nat.div(Nat.add(Nat.mul(row, n), c), n) == row : Nat} &
  {Nat.mod(Nat.add(Nat.mul(row, n), c), n) == c : Nat}

law pair_onto:
  for +n: Nat
  for +i: Nat
  for npos: {Nat.is_lt(0n, n) == True{} : Bool}
  {Nat.add(Nat.mul(Nat.div(i, n), n), Nat.mod(i, n)) == i : Nat} &
  {Nat.is_lt(Nat.mod(i, n), n) == True{} : Bool}

# 3. The value: for both act_limit branches, the half2 the fused epilogue stores for (row, 2 c) is the term
# act_mul_kernel_h<ACT_SILU> stores at index row n + c after the wide kernel stored u.
law value_eq:
  for limit: Bool
  for +n: Nat
  for +row: Nat
  for +c: Nat
  for cn: {Nat.is_lt(c, n) == True{} : Bool}
  {Impl.epi(limit, Nat.mul(n, 2n), row, Nat.mul(c, 2n)) == Spec.act_mul(limit, Nat.mul(n, 2n), Nat.add(Nat.mul(row, n), c)) : Spec.H2}

# 4. The stock silu_mul launch handles every index once: thread (blk, tid < 256) handles blk 256 + tid, which names
# (blk, tid) back, and every i < numel / 2 is handled by block i / 256 < ceil(numel / 512), thread i % 256.
law stock_thread:
  for +blk: Nat
  for +tid: Nat
  for tb: {Nat.is_lt(tid, 256n) == True{} : Bool}
  {Nat.div(Spec.stock_idx(blk, tid), 256n) == blk : Nat} &
  {Nat.mod(Spec.stock_idx(blk, tid), 256n) == tid : Nat}

law stock_cover:
  for +i: Nat
  for +numel: Nat
  for live: {Nat.is_lt(i, Nat.div(numel, 2n)) == True{} : Bool}
  {Spec.stock_idx(Nat.div(i, 256n), Nat.mod(i, 256n)) == i : Nat} &
  {Nat.is_lt(Nat.div(i, 256n), Spec.stock_blocks(numel)) == True{} : Bool}

# 5. Route: the fused kernel runs only with both switches on, the device / layout / size / status checks passed, on the
# MLP gate/up shape K 5120 x N 17408 with M >= 1024 ...
law fused_scope:
  for mlp: Impl.Sw
  for wide: Impl.Sw
  for dev: Bool
  for +M: Nat
  for +K: Nat
  for +N: Nat
  for mfit: Bool
  for status: Bool
  for fused: {Impl.is_fused(Impl.route(mlp, wide, dev, M, K, N, mfit, status)) == True{} : Bool}
  {Impl.sw_on(mlp) == Impl.Fused{} : Impl.Route} &
  {Impl.sw_on(wide) == Impl.Fused{} : Impl.Route} &
  {Bool.and(dev, Bool.and(mfit, status)) == True{} : Bool} &
  {Nat.is_eq(K, 5120n) == True{} : Bool} &
  {Nat.is_eq(N, 17408n) == True{} : Bool} &
  {Nat.is_lt(M, 1024n) == False{} : Bool}

# ... where stock hgemm_recon -> hgemm_f16acc_try runs the 3011 wide kernel: the sm86 wide route of hgemm_wide.bend
# with the switch unset or "1" (sw_on(wide) == Fused) is the wide kernel for this shape at every M >= 1024 ...
law stock_wide_route:
  for +M: U32
  for long: {U32.is_ge(M, 1024) == True{} : Bool}
  {HW.route(HWS.HwUnset{}, 8, 6, 1, M, 5120, 17408, False{}) == HWS.HwWide{} : HWS.HwRoute} &
  {HW.route(HWS.HwOne{}, 8, 6, 1, M, 5120, 17408, False{}) == HWS.HwWide{} : HWS.HwRoute}

# ... and hgemm_f16acc_try's worthwhile() holds (the RTX 3090 has 82 <= 1088 SMs); covered()'s grid bound is the host's
# M <= 65535 * 128 check (mfit), its layout checks are implied by dev (not modelled); enabled() is status.
law stock_worthwhile:
  for +M: Nat
  for +nsm: Nat
  for long: {Nat.is_lt(M, 1024n) == False{} : Bool}
  for sms: {Nat.is_le(nsm, 1088n) == True{} : Bool}
  {Spec.worthwhile86(M, 17408n, nsm) == True{} : Bool}
