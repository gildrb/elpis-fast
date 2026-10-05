# Proofs of pattn8i_pipe_laws.bend (ext 3022 ping-pong pipeline of pattn8i_kernel). Nat lemmas from
# gemm_m16_partition_proof.bend (Ar); 3020's trace law from pattn8_sched_laws.bend (proven in pattn8_sched_proof.bend,
# imported so the composed claim is discharged).
import Base
import ./pattn8i_pipe.bend as P
import ./pattn8i_pipe_spec.bend as S
import ./pattn8i_pipe_laws.bend as Laws
import ./pattn8_sched.bend as P8
import ./pattn8_sched_spec.bend as Spec
import ./pattn8_sched_laws.bend as L8
import ./pattn8_sched_proof.bend as P8P
import ./gemm_m16_partition_proof.bend as Ar

# ---------------------------------------------------------------------------------------------
# Helpers

# x && y holds when both do
def pp_and(-x: Bool, -y: Bool, hx: {x == True{} : Bool}, hy: {y == True{} : Bool}) -> {Bool.and(x, y) == True{} : Bool}:
  %Equal.sym(Bool, x, True{}, hx) : {Bool.and(_, y) == True{} : Bool}
  hy

# Parity of Nat.divmod.go with divisor 2 (m counts down 1, 0; r counts up 0, 1): the remainder after one more step is
# the other stage, from either phase and any quotient accumulators.
def pp_PA(+n: Nat, +d: Nat, +e: Nat) -> Type:
  {P.other(Pair.snd(Nat, Nat, Nat.divmod.go(n, 1n, d, 0n))) == Pair.snd(Nat, Nat, Nat.divmod.go(n, 0n, e, 1n)) : Nat}

def pp_PB(+n: Nat, +d: Nat, +e: Nat) -> Type:
  {P.other(Pair.snd(Nat, Nat, Nat.divmod.go(n, 0n, d, 1n))) == Pair.snd(Nat, Nat, Nat.divmod.go(n, 1n, e, 0n)) : Nat}

def pp_par(+n: Nat, +d: Nat, +e: Nat) -> pp_PA(n, d, e) & pp_PB(n, d, e):
  match n:
    case 0n:
      ({==}, {==})
    case 1n+k:
      (Ar.ard_snd(pp_PA(k, d, 1n+e), pp_PB(k, d, 1n+e), pp_par(k, d, 1n+e)),
       Ar.ard_fst(pp_PA(k, 1n+d, e), pp_PB(k, 1n+d, e), pp_par(k, 1n+d, e)))

# other(stage(it)) == stage(it + 1)
def pp_next(+it: Nat) -> {P.other(P.stage(it)) == P.stage(1n+it) : Nat}:
  Ar.ard_fst(pp_PA(it, 0n, 0n), pp_PB(it, 0n, 0n), pp_par(it, 0n, 0n))

# the other stage is a stage and differs from s
def pp_wr(+s: Nat) -> {S.wr_ok(s, P.other(s)) == True{} : Bool}:
  match s:
    case 0n:
      {==}
    case 1n+p:
      {==}

# writing t into stage s and reading s back gives t
def pp_get_put(+s: Nat, +t: Nat, +c0: Nat, +c1: Nat) -> {S.get(s, S.put0(s, t, c0), S.put1(s, t, c1)) == t : Nat}:
  match s:
    case 0n:
      {==}
    case 1n+p:
      {==}

# stage(it) holds tile it: its equality test passes
def pp_rd_eq(+it: Nat, +c0: Nat, +c1: Nat, inv: {S.get(P.stage(it), c0, c1) == 1n+it : Nat})
  -> {Nat.is_eq(S.get(P.stage(it), c0, c1), 1n+it) == True{} : Bool}:
  %Equal.sym(Nat, S.get(P.stage(it), c0, c1), 1n+it, inv) : {Nat.is_eq(_, 1n+it) == True{} : Bool}
  Ar.ar_eq_refl(1n+it)

# the read of iteration it passes when its stage holds tile it
def pp_rd(+it: Nat, +c0: Nat, +c1: Nat, inv: {S.get(P.stage(it), c0, c1) == 1n+it : Nat})
  -> {S.rd_ok(it, P.stage(it), c0, c1) == True{} : Bool}:
  pp_and(Nat.is_lt(P.stage(it), 2n), Nat.is_eq(S.get(P.stage(it), c0, c1), 1n+it),
    Ar.ard_dm_lt(it, 2n, {==}), pp_rd_eq(it, c0, c1, inv))

# the other stage differs from s
def pp_ne(+s: Nat) -> {Nat.is_ne(P.other(s), s) == True{} : Bool}:
  match s:
    case 0n:
      {==}
    case 1n+p:
      {==}

# after tile it + 1 is written into other(stage(it)), stage(it + 1) holds it
def pp_inv1(+it: Nat, +c0: Nat, +c1: Nat)
  -> {S.get(P.stage(1n+it), S.put0(P.other(P.stage(it)), 2n+it, c0), S.put1(P.other(P.stage(it)), 2n+it, c1)) == 2n+it : Nat}:
  %pp_next(it) : {S.get(_, S.put0(P.other(P.stage(it)), 2n+it, c0), S.put1(P.other(P.stage(it)), 2n+it, c1)) == 2n+it : Nat}
  pp_get_put(P.other(P.stage(it)), 2n+it, c0, c1)

# stage(it + 1) differs from stage(it)
def pp_next_ne(+it: Nat) -> {Nat.is_ne(P.stage(1n+it), P.stage(it)) == True{} : Bool}:
  %pp_next(it) : {Nat.is_ne(_, P.stage(it)) == True{} : Bool}
  pp_ne(P.stage(it))

# 1 + it < it + (2 + j)
def pp_lt_more(+it: Nat, +j: Nat) -> {Nat.is_lt(1n+it, Nat.add(it, 2n+j)) == True{} : Bool}:
  match it:
    case 0n:
      {==}
    case 1n+p:
      pp_lt_more(p, j)

# ---------------------------------------------------------------------------------------------
# 1. Stages

def Laws.stage_next(it):
  (Ar.ard_dm_lt(it, 2n, {==}), pp_next(it), pp_next_ne(it))

# ---------------------------------------------------------------------------------------------
# 2. Race-free and fresh

# One epoch of iteration it with its write flag c = (it + 1 < nt); ih: the rest of the loop passes from the state the
# write leaves.
def pp_ep(c: Bool, +j: Nat, +it: Nat, +nt: Nat, +c0: Nat, +c1: Nat,
    hc: {Nat.is_lt(1n+it, nt) == c : Bool},
    e: {Nat.add(it, 1n+j) == nt : Nat},
    inv: {S.get(P.stage(it), c0, c1) == 1n+it : Nat},
    ih: {S.run(P.epochs(j, 1n+it, nt), S.put0(P.other(P.stage(it)), 2n+it, c0), S.put1(P.other(P.stage(it)), 2n+it, c1)) == True{} : Bool})
  -> {S.run(Con{P.Ep{it, P.stage(it), c, 1n+it, P.other(P.stage(it))}, P.epochs(j, 1n+it, nt)}, c0, c1) == True{} : Bool}:
  match c j:
    case True{} _:
      pp_and(S.rd_ok(it, P.stage(it), c0, c1),
        Bool.and(S.wr_ok(P.stage(it), P.other(P.stage(it))),
          S.run(P.epochs(j, 1n+it, nt), S.put0(P.other(P.stage(it)), 2n+it, c0), S.put1(P.other(P.stage(it)), 2n+it, c1))),
        pp_rd(it, c0, c1, inv),
        pp_and(S.wr_ok(P.stage(it), P.other(P.stage(it))),
          S.run(P.epochs(j, 1n+it, nt), S.put0(P.other(P.stage(it)), 2n+it, c0), S.put1(P.other(P.stage(it)), 2n+it, c1)),
          pp_wr(P.stage(it)), ih))
    case False{} 0n:
      pp_and(S.rd_ok(it, P.stage(it), c0, c1), True{}, pp_rd(it, c0, c1, inv), {==})
    case False{} 1n+jp:
      +lt = Equal.trans(Bool, Nat.is_lt(1n+it, nt), Nat.is_lt(1n+it, Nat.add(it, 2n+jp)), True{},
        Equal.cong(Nat, Bool, z => Nat.is_lt(1n+it, z), nt, Nat.add(it, 2n+jp), Equal.sym(Nat, Nat.add(it, 2n+jp), nt, e)),
        pp_lt_more(it, jp))
      Empty.absurd({S.run(Con{P.Ep{it, P.stage(it), False{}, 1n+it, P.other(P.stage(it))}, P.epochs(1n+jp, 1n+it, nt)}, c0, c1) == True{} : Bool},
        Ar.ar_false_true(Equal.trans(Bool, False{}, Nat.is_lt(1n+it, nt), True{}, Equal.sym(Bool, Nat.is_lt(1n+it, nt), False{}, hc), lt)))

# Iterations it .. it + k - 1 of a loop with bound nt = it + k pass from any state whose stage(it) holds tile it.
def pp_go(+k: Nat, +it: Nat, +nt: Nat, +c0: Nat, +c1: Nat,
    +e: {Nat.add(it, k) == nt : Nat},
    inv: {S.get(P.stage(it), c0, c1) == 1n+it : Nat})
  -> {S.run(P.epochs(k, it, nt), c0, c1) == True{} : Bool}:
  match k:
    case 0n:
      {==}
    case 1n+j:
      +s1 = P.other(P.stage(it))
      +e1 = Equal.trans(Nat, Nat.add(1n+it, j), Nat.add(it, 1n+j), nt,
        Equal.sym(Nat, Nat.add(it, 1n+j), 1n+Nat.add(it, j), Ar.ar_add_succ_r(it, j)), e)
      pp_ep(Nat.is_lt(1n+it, nt), j, it, nt, c0, c1, {==}, e, inv,
        pp_go(j, 1n+it, nt, S.put0(s1, 2n+it, c0), S.put1(s1, 2n+it, c1), e1, pp_inv1(it, c0, c1)))

def Laws.pipe_safe(nt):
  match nt:
    case 0n:
      {==}
    case 1n+m:
      pp_go(1n+m, 0n, 1n+m, 1n, 0n, {==}, {==})

# ---------------------------------------------------------------------------------------------
# 3-4. Compute trace

# the warp's computing iterations of epochs(k, it, nt) are 3020's steps(k, it, nw)
def pp_acts(+k: Nat, +it: Nat, +nt: Nat, +nw: Nat) -> {P.acts(P.epochs(k, it, nt), nw) == P8.steps(k, it, nw) : List<&2, Nat>}:
  match k:
    case 0n:
      {==}
    case 1n+j:
      Equal.cong(List<&2, Nat>, List<&2, Nat>, z => P8.steps.keep(Nat.is_lt(it, nw), it, z),
        P.acts(P.epochs(j, 1n+it, nt), nw), P8.steps(j, 1n+it, nw), pp_acts(j, 1n+it, nt, nw))

def Laws.pipe_trace(L, q, qc, b):
  pp_acts(P8.ntiles_cta(L, q, qc), 0n, P8.ntiles_cta(L, q, qc), P8.ntiles(L, q, qc, b))

def Laws.pipe_trace_3010(L, q, qc, b, blt, live):
  Equal.trans(List<&2, Nat>, P.trace(L, q, qc, b), P8.trace(L, q, qc, b), Spec.trace(L, q, Nat.add(Nat.mul(qc, 4n), b)),
    Laws.pipe_trace(L, q, qc, b), L8.trace_eq(L, q, qc, b, blt, live))

# ---------------------------------------------------------------------------------------------
# 5. Negative cases

def Laws.reject_same_stage():
  {==}

def Laws.reject_stale():
  {==}

def Laws.reject_range():
  {==}
