# Obligations for the ext 3022 ping-pong K / V pipeline of pattn8i_kernel (pattn8i_pipe.bend) against the two-stage
# checker pattn8i_pipe_spec.bend and the ext 3020 / 3021c schedule (pattn8_sched.bend). Together they say: in every
# iteration the stage the warps read holds exactly that iteration's tile, issued in the previous epoch (or the
# prologue) and never overwritten in between; the next tile always goes to the other stage, so no load races a read;
# and each warp computes exactly the tiles of 3020's trace in loop order (so, with 3020 = 3010 per warp, the 3010
# order). With the per-thread code textually unchanged by 3022 (pattn8i_pipe_diff.py), every output element combines
# the same values in the same order with the same rounding points as 3021c: bit-identical outputs. Domain: any
# ntiles_cta; floating point is outside this model (values are opaque, only which tile / which stage / which order).
import Base
import ./pattn8i_pipe.bend as P
import ./pattn8i_pipe_spec.bend as S
import ./pattn8_sched.bend as P8
import ./pattn8_sched_spec.bend as Spec

# 1. Stages: `it & 1` is a stage, `st ^ 1` is the stage of tile it + 1 and never tile it's.
law stage_next:
  for +it: Nat
  {Nat.is_lt(P.stage(it), 2n) == True{} : Bool} &
  {P.other(P.stage(it)) == P.stage(1n+it) : Nat} &
  {Nat.is_ne(P.stage(1n+it), P.stage(it)) == True{} : Bool}

# 2. Race-free and fresh: the prologue plus every iteration of the CTA loop pass the checker, for every loop bound.
law pipe_safe:
  for +nt: Nat
  {S.run(P.loop(nt), P.pro_c0(nt), 0n) == True{} : Bool}

# 3. Same computations in the same order: the iterations in which a warp computes are 3020's trace.
law pipe_trace:
  for +L: Nat
  for +q: Nat
  for +qc: Nat
  for +b: Nat
  {P.trace(L, q, qc, b) == P8.trace(L, q, qc, b) : List<&2, Nat>}

# 4. Composed with pattn8_sched_laws.trace_eq: a live warp computes 3010's loop 0 .. ntiles - 1 of block 4 qc + b.
law pipe_trace_3010:
  for +L: Nat
  for +q: Nat
  for +qc: Nat
  for +b: Nat
  for blt: {Nat.is_lt(b, 4n) == True{} : Bool}
  for live: {Nat.is_lt(P8.p0(qc, b), q) == True{} : Bool}
  {P.trace(L, q, qc, b) == Spec.trace(L, q, Nat.add(Nat.mul(qc, 4n), b)) : List<&2, Nat>}

# 5. The checker is not vacuous: it rejects 3021c's single stage driven as a ping-pong (tile 1 issued into stage 0
# while tile 0 is read from it), a read of a tile never staged, and a stage index past the two stages.
law reject_same_stage:
  {S.run([P.Ep{0n, 0n, True{}, 1n, 0n}, P.Ep{1n, 0n, False{}, 2n, 0n}], 1n, 0n) == False{} : Bool}

law reject_stale:
  {S.run([P.Ep{0n, 0n, False{}, 1n, 1n}, P.Ep{1n, 1n, False{}, 2n, 0n}], 1n, 0n) == False{} : Bool}

law reject_range:
  {S.run([P.Ep{0n, 0n, True{}, 1n, 2n}, P.Ep{1n, 2n, False{}, 2n, 0n}], 1n, 0n) == False{} : Bool}
