# Obligations for ext 5110g (prefill_merge.bend) against the 2048-row prefill schedule (prefill_merge_spec.bend), plus
# the chunk-size independence of the ext 3010 prefill attention rows (pattn_sched.bend) and of the 3011 GEMM route
# (hgemm_wide.bend). Domain of the schedule laws: the served text-only recurrent model, no prompt-cache page hit in the
# pieces involved, page-aligned kv position k = a P, max_chunk_size mc = c P with c >= 1, P >= 1. What they do NOT
# cover: see the per-law notes and the report (floating point of the forwards; kernels outside pattn / the wide GEMM).
import Base
import ./prefill_merge.bend as Impl
import ./prefill_merge_spec.bend as Spec
import ./pattn_sched.bend as P6
import ./hgemm_wide.bend as HW
import ./hgemm_wide_spec.bend as HWS

# 1. A merged step is exactly two 2048-row steps: the merged forward ends where the second of the two unmerged pieces
# ends (k + 2 mc), the first unmerged piece ends at k + mc, and neither is cut by the last-page split.
law merge_two_steps:
  for +a: Nat
  for +c: Nat
  for +T: Nat
  for +P: Nat
  for +G: Nat
  for pp: {Nat.is_lt(0n, P) == True{} : Bool}
  for cp: {Nat.is_lt(0n, c) == True{} : Bool}
  for merged: {Impl.merge_ok(True{}, Nat.mul(a, P), T, Nat.mul(c, P), P, G) == True{} : Bool}
  {Spec.step(Nat.mul(a, P), T, Nat.mul(c, P), P) == Nat.add(Nat.mul(a, P), Nat.mul(c, P)) : Nat} &
  {Spec.step(Spec.step(Nat.mul(a, P), T, Nat.mul(c, P), P), T, Nat.mul(c, P), P) == Nat.add(Nat.add(Nat.mul(a, P), Nat.mul(c, P)), Nat.mul(c, P)) : Nat} &
  {Impl.step(True{}, Nat.mul(a, P), T, Nat.mul(c, P), P, G) == Nat.add(Nat.add(Nat.mul(a, P), Nat.mul(c, P)), Nat.mul(c, P)) : Nat}

# 2. Otherwise the step is the 2048-row step.
law unmerged_same:
  for en: Bool
  for +k: Nat
  for +T: Nat
  for +mc: Nat
  for +P: Nat
  for +G: Nat
  for no: {Impl.merge_ok(en, k, T, mc, P, G) == False{} : Bool}
  {Impl.step(en, k, T, mc, P, G) == Spec.step(k, T, mc, P) : Nat}

# 3. The guard: a merge happens only for prompts with last_tok <= G (M4096_MAX_PROMPT = 131072, checked in the source by
# prefill_merge_diff.py); above it every step and the whole schedule are the 2048-row ones.
law guard:
  for en: Bool
  for +k: Nat
  for +T: Nat
  for +mc: Nat
  for +P: Nat
  for +G: Nat
  for merged: {Impl.merge_ok(en, k, T, mc, P, G) == True{} : Bool}
  {Nat.is_le(T, G) == True{} : Bool}

law above_guard_schedule:
  for fuel: Nat
  for +en: Bool
  for +k: Nat
  for +T: Nat
  for +mc: Nat
  for +P: Nat
  for +G: Nat
  for above: {Nat.is_le(T, G) == False{} : Bool}
  {Impl.visits(fuel, en, k, T, mc, P, G) == Spec.visits(fuel, k, T, mc, P) : List<&2, Nat>}

# 4. Same rows in the same order: a forward over m1 + m2 rows from a is the forward over m1 rows from a followed by the
# forward over m2 rows from a + m1 (so the merged forward's rows are the two pieces' rows, concatenated).
law rows_concat:
  for +m1: Nat
  for +m2: Nat
  for +a: Nat
  {Spec.rows(Nat.add(m1, m2), a) == List.append(&2, Nat, Spec.rows(m1, a), Spec.rows(m2, Nat.add(a, m1))) : List<&2, Nat>}

# 5. What the schedule law does not give: recurrent-checkpoint stash positions. On a small instance of the same code
# (mc = 4, P = 2, G = 64, last_tok 64, pp interval 16), a prefill resumed from a cached prefix ending at 4 (= mc mod 2 mc)
# stashes at 20, 36, 52 on the 2048-row schedule and at none of them merged (the merged steps land on multiples of 2 mc
# only); from 0 (cold) both stash at 16, 32, 48.
law checkpoint_positions:
  {Spec.any_boundary(Spec.visits(40n, 4n, 64n, 4n, 2n), 4n, 16n, 56n) == [20n, 36n, 52n] : List<&2, Nat>} &
  {Spec.any_boundary(Impl.visits(40n, True{}, 4n, 64n, 4n, 2n, 64n), 4n, 16n, 56n) == [] : List<&2, Nat>} &
  {Spec.any_boundary(Spec.visits(40n, 0n, 64n, 4n, 2n), 0n, 16n, 56n) == [16n, 32n, 48n] : List<&2, Nat>} &
  {Spec.any_boundary(Impl.visits(40n, True{}, 0n, 64n, 4n, 2n, 64n), 0n, 16n, 56n) == [16n, 32n, 48n] : List<&2, Nat>}

# 6. The 3010 prefill attention row does not depend on its chunk: two chunkings (L, q, qb) and (L', q', qb') that put
# the row's 16-block at the same absolute start, both full and inside their chunk, give the same tile loop, mask state
# and kept keys in every iteration (so 4096-row and 2048-row chunks attend identically; with the 3020 laws, so does the
# 8-warp kernel).
law pattn_chunk_invariant:
  for +L: Nat
  for +q: Nat
  for +qb: Nat
  for +L2: Nat
  for +q2: Nat
  for +qb2: Nat
  for +r: Nat
  for +it: Nat
  for +p: Nat
  for rb: {Nat.is_lt(r, 16n) == True{} : Bool}
  for full: {Nat.is_le(Nat.mul(1n+qb, 16n), q) == True{} : Bool}
  for full2: {Nat.is_le(Nat.mul(1n+qb2, 16n), q2) == True{} : Bool}
  for len: {Nat.is_le(q, L) == True{} : Bool}
  for len2: {Nat.is_le(q2, L2) == True{} : Bool}
  for same: {P6.q_abs0(L, q, qb) == P6.q_abs0(L2, q2, qb2) : Nat}
  {P6.visits(L, q, qb, r, it, p) == P6.visits(L2, q2, qb2, r, it, p) : Bool}

# 7. The 3011 wide-GEMM route is the same for M = 2048 and M = 4096 (every switch value, device, batch and shape); its
# per-element order (hgemm_wide k_order) does not involve M.
law gemm_route_m:
  for env: HWS.HwEnv
  for +major: U32
  for +minor: U32
  for +batch: U32
  for +K: U32
  for +N: U32
  for +f: Bool
  {HW.route(env, major, minor, batch, 2048, K, N, f) == HW.route(env, major, minor, batch, 4096, K, N, f) : HWS.HwRoute}
