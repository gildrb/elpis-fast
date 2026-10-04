# Proofs of prefill_merge_laws.bend (ext 5110g merged prefill pieces; chunk independence of pattn rows and the wide-GEMM
# route). Nat lemmas from gemm_m16_partition_proof.bend (Ar) and attn_stride_proof.bend (SP).
import Base
import ./prefill_merge.bend as Impl
import ./prefill_merge_spec.bend as Spec
import ./prefill_merge_laws.bend as Laws
import ./pattn_sched.bend as P6
import ./attn_stride.bend as Stride
import ./hgemm_wide.bend as HW
import ./hgemm_wide_spec.bend as HWS
import ./gemm_m16_partition_proof.bend as Ar
import ./attn_stride_proof.bend as SP

# ---------------------------------------------------------------------------------------------
# Helpers

# no cut when the second test fails
def pm_nocut_s(+x: Bool, +y: Bool, +l: Nat, +e: Nat, hy: {y == False{} : Bool})
  -> {Spec.stop_at.pick(Bool.and(x, y), l, e) == e : Nat}:
  match x:
    case False{}:
      {==}
    case True{}:
      %Equal.sym(Bool, y, False{}, hy) : {Spec.stop_at.pick(_, l, e) == e : Nat}
      {==}

def pm_nocut_i(+x: Bool, +y: Bool, +l: Nat, +e: Nat, hy: {y == False{} : Bool})
  -> {Impl.cut.pick(Bool.and(x, y), l, e) == e : Nat}:
  match x:
    case False{}:
      {==}
    case True{}:
      %Equal.sym(Bool, y, False{}, hy) : {Impl.cut.pick(_, l, e) == e : Nat}
      {==}

def pm_cut_eq(c: Bool, +l: Nat, +e: Nat) -> {Impl.cut.pick(c, l, e) == Spec.stop_at.pick(c, l, e) : Nat}:
  match c:
    case True{}:
      {==}
    case False{}:
      {==}

# (x + m <= y + m) == (m + x <= m + y)
def pm_comm_le(+x: Nat, +m: Nat, +y: Nat) -> {Nat.is_le(Nat.add(x, m), Nat.add(y, m)) == Nat.is_le(Nat.add(m, x), Nat.add(m, y)) : Bool}:
  %Ar.ar_add_comm(m, x) : {Nat.is_le(_, Nat.add(y, m)) == Nat.is_le(Nat.add(m, x), Nat.add(m, y)) : Bool}
  %Ar.ar_add_comm(m, y) : {Nat.is_le(Nat.add(m, x), _) == Nat.is_le(Nat.add(m, x), Nat.add(m, y)) : Bool}
  {==}

# x < l gives (l <= x) == False
def pm_lt_nle(+x: Nat, +l: Nat, h: {Nat.is_lt(x, l) == True{} : Bool}) -> {Nat.is_le(l, x) == False{} : Bool}:
  Ar.p1l_not_true(Nat.is_le(l, x), Ar.ard_rt(Nat.is_lt(x, l), Bool.not(Nat.is_le(l, x)), Ar.ar_lt_not_le(x, l), h))

# x < x + m for m > 0
def pm_lt_add(+x: Nat, +m: Nat, pos: {Nat.is_lt(0n, m) == True{} : Bool}) -> {Nat.is_lt(x, Nat.add(x, m)) == True{} : Bool}:
  %Ar.ar_add_zero_r(x) : {Nat.is_lt(_, Nat.add(x, m)) == True{} : Bool}
  Ar.ard_tr(Nat.is_lt(Nat.add(x, 0n), Nat.add(x, m)), Nat.is_lt(0n, m), Ar.ar_lt_add_cancel(x, 0n, m), pos)

# e - k == m with m > 0 gives e == k + m
def pm_sub_eq.pick(b: Bool, +e: Nat, +k: Nat, +m: Nat, hb: {Nat.is_le(k, e) == b : Bool}, h: {Nat.sub(e, k) == m : Nat},
    pos: {Nat.is_lt(0n, m) == True{} : Bool}) -> {e == Nat.add(k, m) : Nat}:
  match b:
    case True{}:
      %h : {e == Nat.add(k, _) : Nat}
      Equal.sym(Nat, Nat.add(k, Nat.sub(e, k)), e, Ar.ar_sub_add(e, k, hb))
    case False{}:
      +lt = Equal.trans(Bool, Nat.is_lt(e, k), Bool.not(Nat.is_le(k, e)), True{}, Ar.ar_lt_not_le(e, k),
        Equal.cong(Bool, Bool, z => Bool.not(z), Nat.is_le(k, e), False{}, hb))
      +z0 = Equal.trans(Nat, m, Nat.sub(e, k), 0n, Equal.sym(Nat, Nat.sub(e, k), m, h), SP.sp_sub_le0(e, k, Ar.ar_lt_le(e, k, lt)))
      Empty.absurd({e == Nat.add(k, m) : Nat}, Ar.ar_false_true(Equal.trans(Bool, False{}, Nat.is_lt(0n, m), True{},
        Equal.cong(Nat, Bool, z => Nat.is_lt(0n, z), 0n, m, Equal.sym(Nat, m, 0n, z0)), pos)))

def pm_sub_eq(+e: Nat, +k: Nat, +m: Nat, h: {Nat.sub(e, k) == m : Nat}, pos: {Nat.is_lt(0n, m) == True{} : Bool})
  -> {e == Nat.add(k, m) : Nat}:
  pm_sub_eq.pick(Nat.is_le(k, e), e, k, m, {==}, h, pos)

# x <= T - m with x > 0 gives x + m <= T
def pm_fit(+x: Nat, +m: Nat, +T: Nat, hx: {Nat.is_le(x, Nat.sub(T, m)) == True{} : Bool}, xpos: {Nat.is_lt(0n, x) == True{} : Bool})
  -> {Nat.is_le(Nat.add(x, m), T) == True{} : Bool}:
  +hl = hx
  +sp = Ar.ar_lt_le_trans(0n, x, Nat.sub(T, m), xpos, hl)
  +mlt = Ar.ard_rt(Nat.is_lt(0n, Nat.sub(T, m)), Nat.is_lt(Nat.add(m, 0n), T), Ar.ar_lt_sub_iff(T, m, 0n), sp)
  +mle = Ar.ar_lt_le(m, T, Equal.trans(Bool, Nat.is_lt(m, T), Nat.is_lt(Nat.add(m, 0n), T), True{},
    Equal.cong(Nat, Bool, z => Nat.is_lt(z, T), m, Nat.add(m, 0n), Equal.sym(Nat, Nat.add(m, 0n), m, Ar.ar_add_zero_r(m))), mlt))
  +eT = Equal.trans(Nat, Nat.add(Nat.sub(T, m), m), Nat.add(m, Nat.sub(T, m)), T, Ar.ar_add_comm(Nat.sub(T, m), m), Ar.ar_sub_add(T, m, mle))
  +le1 = Ar.ard_tr(Nat.is_le(Nat.add(x, m), Nat.add(Nat.sub(T, m), m)), Nat.is_le(x, Nat.sub(T, m)),
    Equal.trans(Bool, Nat.is_le(Nat.add(x, m), Nat.add(Nat.sub(T, m), m)), Nat.is_le(Nat.add(m, x), Nat.add(m, Nat.sub(T, m))), Nat.is_le(x, Nat.sub(T, m)),
      pm_comm_le(x, m, Nat.sub(T, m)),
      Ar.ar_le_add_cancel(m, x, Nat.sub(T, m))), hl)
  Equal.trans(Bool, Nat.is_le(Nat.add(x, m), T), Nat.is_le(Nat.add(x, m), Nat.add(Nat.sub(T, m), m)), True{},
    Equal.cong(Nat, Bool, z => Nat.is_le(Nat.add(x, m), z), T, Nat.add(Nat.sub(T, m), m), Equal.sym(Nat, Nat.add(Nat.sub(T, m), m), T, eT)), le1)

# (x + m < y + m) == (m + x < m + y)
def pm_comm_lt(+x: Nat, +m: Nat, +y: Nat) -> {Nat.is_lt(Nat.add(x, m), Nat.add(y, m)) == Nat.is_lt(Nat.add(m, x), Nat.add(m, y)) : Bool}:
  %Ar.ar_add_comm(m, x) : {Nat.is_lt(_, Nat.add(y, m)) == Nat.is_lt(Nat.add(m, x), Nat.add(m, y)) : Bool}
  %Ar.ar_add_comm(m, y) : {Nat.is_lt(Nat.add(m, x), _) == Nat.is_lt(Nat.add(m, x), Nat.add(m, y)) : Bool}
  {==}

def pm_le_cong(-x: Nat, -x2: Nat, -y: Nat, -y2: Nat, ex: {x == x2 : Nat}, ey: {y == y2 : Nat})
  -> {Nat.is_le(x, y) == Nat.is_le(x2, y2) : Bool}:
  Equal.trans(Bool, Nat.is_le(x, y), Nat.is_le(x2, y), Nat.is_le(x2, y2),
    Equal.cong(Nat, Bool, z => Nat.is_le(z, y), x, x2, ex), Equal.cong(Nat, Bool, z => Nat.is_le(x2, z), y, y2, ey))

# ---------------------------------------------------------------------------------------------
# 1. A merged step is two unmerged steps

def Laws.merge_two_steps(a, c, T, P, G, pp, cp, merged):
  +pl = pp
  +cq = cp
  +m = merged
  +k = Nat.mul(a, P)
  +mc = Nat.mul(c, P)
  +M2 = Nat.mul(2n, mc)
  +E0 = Impl.base_end(k, T, mc, P)
  +lpb = Nat.mul(Nat.div(T, P), P)
  +R3 = Bool.and(Nat.is_le(Nat.add(E0, mc), Nat.sub(T, M2)), Nat.is_eq(Nat.mod(mc, P), 0n))
  +R2 = Bool.and(Nat.is_eq(Nat.sub(E0, k), mc), R3)
  +R1 = Bool.and(Nat.is_eq(Nat.mod(k, M2), 0n), R2)
  +r1 = Ar.ar_and_r(Nat.is_le(T, G), R1, m)
  +r2 = Ar.ar_and_r(Nat.is_eq(Nat.mod(k, M2), 0n), R2, r1)
  +he = Ar.ar_and_l(Nat.is_eq(Nat.sub(E0, k), mc), R3, r2)
  +r3 = Ar.ar_and_r(Nat.is_eq(Nat.sub(E0, k), mc), R3, r2)
  +hr = Ar.ar_and_l(Nat.is_le(Nat.add(E0, mc), Nat.sub(T, M2)), Nat.is_eq(Nat.mod(mc, P), 0n), r3)
  +mcpos = Ar.ard_tr(Nat.is_lt(0n, mc), Nat.is_lt(0n, c), Ar.ar_mul_lt_cancel(0n, c, P, pl), cq)
  +e0 = pm_sub_eq(E0, k, mc, Ar.ar_eq_true(Nat.sub(E0, k), mc, he), mcpos)
  +e1 = Nat.add(k, mc)
  +e2 = Nat.add(e1, mc)
  +hr2 = Equal.trans(Bool, Nat.is_le(e2, Nat.sub(T, M2)), Nat.is_le(Nat.add(E0, mc), Nat.sub(T, M2)), True{},
    Equal.cong(Nat, Bool, z => Nat.is_le(Nat.add(z, mc), Nat.sub(T, M2)), e1, E0, Equal.sym(Nat, E0, e1, e0)), hr)
  +e2pos = Ar.ar_lt_le_trans(0n, mc, e2, mcpos, Ar.ar_le_add_l(e1, mc))
  +fit = pm_fit(e2, M2, T, hr2, e2pos)
  +h1c = Ar.ard_rt(Nat.is_lt(0n, c), Nat.is_le(1n, c), Ar.ar_lt_succ_le(0n, c), cq)
  +pmc = Equal.trans(Bool, Nat.is_le(P, mc), Nat.is_le(Nat.mul(1n, P), mc), True{},
    Equal.cong(Nat, Bool, z => Nat.is_le(z, mc), P, Nat.mul(1n, P), Equal.sym(Nat, Nat.mul(1n, P), P, Ar.ar_mul_one_l(P))),
    Ar.ar_le_mul_mono(1n, c, P, h1c))
  +mcM2 = Ar.ar_le_add_r(mc, Nat.add(mc, 0n))
  +rr = Nat.mod(T, P)
  +rrM2 = Ar.ar_lt_le_trans(rr, P, M2, Ar.ard_dm_lt(T, P, pl), Ar.ar_le_trans(P, mc, M2, pmc, mcM2))
  +Tlt = Equal.trans(Bool, Nat.is_lt(T, Nat.add(lpb, M2)), Nat.is_lt(Nat.add(lpb, rr), Nat.add(lpb, M2)), True{},
    Equal.cong(Nat, Bool, z => Nat.is_lt(z, Nat.add(lpb, M2)), T, Nat.add(lpb, rr), Equal.sym(Nat, Nat.add(lpb, rr), T, Ar.ard_dm_eq(T, P, pl))),
    Ar.ard_tr(Nat.is_lt(Nat.add(lpb, rr), Nat.add(lpb, M2)), Nat.is_lt(rr, M2), Ar.ar_lt_add_cancel(lpb, rr, M2), rrM2))
  +e2M = Ar.ar_le_lt_trans(Nat.add(e2, M2), T, Nat.add(lpb, M2), fit, Tlt)
  +e2l = Ar.ard_tr(Nat.is_lt(e2, lpb), Nat.is_lt(Nat.add(e2, M2), Nat.add(lpb, M2)),
    Equal.sym(Bool, Nat.is_lt(Nat.add(e2, M2), Nat.add(lpb, M2)), Nat.is_lt(e2, lpb),
      Equal.trans(Bool, Nat.is_lt(Nat.add(e2, M2), Nat.add(lpb, M2)), Nat.is_lt(Nat.add(M2, e2), Nat.add(M2, lpb)), Nat.is_lt(e2, lpb),
        pm_comm_lt(e2, M2, lpb), Ar.ar_lt_add_cancel(M2, e2, lpb))), e2M)
  +e1l = Ar.ar_lt_trans(e1, e2, lpb, pm_lt_add(e1, mc, mcpos), e2l)
  +n1 = pm_lt_nle(e1, lpb, e1l)
  +n2 = pm_lt_nle(e2, lpb, e2l)
  +pa = Equal.trans(Nat, Spec.step(k, T, mc, P), Spec.stop_at.pick(Bool.and(Nat.is_lt(k, lpb), Nat.is_le(lpb, e1)), lpb, e1), e1,
    Equal.cong(Nat, Nat, z => Spec.stop_at.pick(Bool.and(Nat.is_lt(k, lpb), Nat.is_le(lpb, z)), lpb, z), E0, e1, e0),
    pm_nocut_s(Nat.is_lt(k, lpb), Nat.is_le(lpb, e1), lpb, e1, n1))
  +q = Nat.add(Nat.add(a, c), c)
  +eqm = Equal.trans(Nat, Nat.mul(q, P), Nat.add(Nat.mul(Nat.add(a, c), P), mc), e2, Ar.ar_mul_add_l(Nat.add(a, c), c, P),
    Equal.cong(Nat, Nat, z => Nat.add(z, mc), Nat.mul(Nat.add(a, c), P), e1, Ar.ar_mul_add_l(a, c, P)))
  +dq = Equal.trans(Nat, Nat.div(e2, P), Nat.div(Nat.mul(q, P), P), q,
    Equal.cong(Nat, Nat, z => Nat.div(z, P), e2, Nat.mul(q, P), Equal.sym(Nat, Nat.mul(q, P), e2, eqm)),
    Equal.trans(Nat, Nat.div(Nat.mul(q, P), P), Nat.div(Nat.add(Nat.mul(q, P), 0n), P), q,
      Equal.cong(Nat, Nat, z => Nat.div(z, P), Nat.mul(q, P), Nat.add(Nat.mul(q, P), 0n), Equal.sym(Nat, Nat.add(Nat.mul(q, P), 0n), Nat.mul(q, P), Ar.ar_add_zero_r(Nat.mul(q, P)))),
      Ar.p7_div_q(q, 0n, P, pl)))
  +al2 = Equal.trans(Nat, Nat.mul(Nat.div(e2, P), P), Nat.mul(q, P), e2, Equal.cong(Nat, Nat, z => Nat.mul(z, P), Nat.div(e2, P), q, dq), eqm)
  +e2T = Ar.ar_le_trans(e2, Nat.add(e2, M2), T, Ar.ar_le_add_r(e2, M2), fit)
  +pe1 = Equal.trans(Nat, Spec.piece_end(e1, T, mc, P), Nat.min(e2, T), e2,
    Equal.cong(Nat, Nat, z => Nat.min(z, T), Nat.mul(Nat.div(e2, P), P), e2, al2), Ar.ar_min_left(e2, T, e2T))
  +pb1 = Equal.trans(Nat, Spec.step(e1, T, mc, P), Spec.stop_at.pick(Bool.and(Nat.is_lt(e1, lpb), Nat.is_le(lpb, e2)), lpb, e2), e2,
    Equal.cong(Nat, Nat, z => Spec.stop_at.pick(Bool.and(Nat.is_lt(e1, lpb), Nat.is_le(lpb, z)), lpb, z), Spec.piece_end(e1, T, mc, P), e2, pe1),
    pm_nocut_s(Nat.is_lt(e1, lpb), Nat.is_le(lpb, e2), lpb, e2, n2))
  +pb = Equal.trans(Nat, Spec.step(Spec.step(k, T, mc, P), T, mc, P), Spec.step(e1, T, mc, P), e2,
    Equal.cong(Nat, Nat, z => Spec.step(z, T, mc, P), Spec.step(k, T, mc, P), e1, pa), pb1)
  +pc1 = Equal.cong(Bool, Nat, z => Impl.cut(k, T, P, Impl.widen(z, E0, mc)), Impl.merge_ok(True{}, k, T, mc, P, G), True{}, m)
  +pc2 = Equal.cong(Nat, Nat, z => Impl.cut(k, T, P, Nat.add(z, mc)), E0, e1, e0)
  +pc3 = pm_nocut_i(Nat.is_lt(k, lpb), Nat.is_le(lpb, e2), lpb, e2, n2)
  (pa, pb,
   Equal.trans(Nat, Impl.step(True{}, k, T, mc, P, G), Impl.cut(k, T, P, Nat.add(E0, mc)), e2, pc1,
     Equal.trans(Nat, Impl.cut(k, T, P, Nat.add(E0, mc)), Impl.cut(k, T, P, e2), e2, pc2, pc3)))

# ---------------------------------------------------------------------------------------------
# 2-3. Unmerged steps, the guard, the schedule above the guard

def Laws.unmerged_same(en, k, T, mc, P, G, no):
  %Equal.sym(Bool, Impl.merge_ok(en, k, T, mc, P, G), False{}, no)
    : {Impl.cut(k, T, P, Impl.widen(_, Impl.base_end(k, T, mc, P), mc)) == Spec.step(k, T, mc, P) : Nat}
  pm_cut_eq(Bool.and(Nat.is_lt(k, Nat.mul(Nat.div(T, P), P)), Nat.is_le(Nat.mul(Nat.div(T, P), P), Impl.base_end(k, T, mc, P))),
    Nat.mul(Nat.div(T, P), P), Impl.base_end(k, T, mc, P))

def pm_rest(+k: Nat, +T: Nat, +mc: Nat, +P: Nat) -> Bool:
  Bool.and(Nat.is_eq(Nat.mod(k, Nat.mul(2n, mc)), 0n),
    Bool.and(Nat.is_eq(Nat.sub(Impl.base_end(k, T, mc, P), k), mc),
      Bool.and(Nat.is_le(Nat.add(Impl.base_end(k, T, mc, P), mc), Nat.sub(T, Nat.mul(2n, mc))), Nat.is_eq(Nat.mod(mc, P), 0n))))

def Laws.guard(en, k, T, mc, P, G, merged):
  Ar.ar_and_l(Nat.is_le(T, G), pm_rest(k, T, mc, P), Ar.ar_and_r(en, Bool.and(Nat.is_le(T, G), pm_rest(k, T, mc, P)), merged))

def pm_nm(+en: Bool, +k: Nat, +T: Nat, +mc: Nat, +P: Nat, +G: Nat, above: {Nat.is_le(T, G) == False{} : Bool})
  -> {Impl.merge_ok(en, k, T, mc, P, G) == False{} : Bool}:
  Equal.trans(Bool, Impl.merge_ok(en, k, T, mc, P, G), Bool.and(en, False{}), False{},
    Equal.cong(Bool, Bool, z => Bool.and(en, Bool.and(z, pm_rest(k, T, mc, P))), Nat.is_le(T, G), False{}, above),
    Ar.p6_and_false(en))

def pm_go(fuel: Nat, done: Bool, +en: Bool, +k: Nat, +T: Nat, +mc: Nat, +P: Nat, +G: Nat, +above: {Nat.is_le(T, G) == False{} : Bool})
  -> {Impl.visits.go(fuel, done, en, k, T, mc, P, G) == Spec.visits.go(fuel, done, k, T, mc, P) : List<&2, Nat>}:
  match fuel:
    case 0n:
      {==}
    case 1n+f:
      match done:
        case True{}:
          {==}
        case False{}:
          +s = Impl.step(en, k, T, mc, P, G)
          +t = Spec.step(k, T, mc, P)
          +se = Laws.unmerged_same(en, k, T, mc, P, G, pm_nm(en, k, T, mc, P, G, above))
          Equal.trans(List<&2, Nat>, Con{s, Impl.visits.go(f, Nat.is_le(T, s), en, s, T, mc, P, G)}, Con{s, Spec.visits.go(f, Nat.is_le(T, s), s, T, mc, P)},
            Con{t, Spec.visits.go(f, Nat.is_le(T, t), t, T, mc, P)},
            Equal.cong(List<&2, Nat>, List<&2, Nat>, z => Con{s, z}, Impl.visits.go(f, Nat.is_le(T, s), en, s, T, mc, P, G), Spec.visits.go(f, Nat.is_le(T, s), s, T, mc, P),
              pm_go(f, Nat.is_le(T, s), en, s, T, mc, P, G, above)),
            Equal.cong(Nat, List<&2, Nat>, z => Con{z, Spec.visits.go(f, Nat.is_le(T, z), z, T, mc, P)}, s, t, se))

def Laws.above_guard_schedule(fuel, en, k, T, mc, P, G, above):
  +ab = above
  pm_go(fuel, Nat.is_le(T, k), en, k, T, mc, P, G, ab)

# ---------------------------------------------------------------------------------------------
# 4-5. Rows, checkpoint positions

def Laws.rows_concat(m1, m2, a):
  match m1:
    case 0n:
      Equal.cong(Nat, List<&2, Nat>, z => Spec.rows(m2, z), a, Nat.add(a, 0n), Equal.sym(Nat, Nat.add(a, 0n), a, Ar.ar_add_zero_r(a)))
    case 1n+j:
      +ih = Laws.rows_concat(j, m2, 1n+a)
      +e = Equal.cong(Nat, List<&2, Nat>, z => Spec.rows(m2, z), Nat.add(1n+a, j), Nat.add(a, 1n+j),
        Equal.sym(Nat, Nat.add(a, 1n+j), 1n+Nat.add(a, j), Ar.ar_add_succ_r(a, j)))
      Equal.cong(List<&2, Nat>, List<&2, Nat>, z => Con{a, z}, Spec.rows(Nat.add(j, m2), 1n+a),
        List.append(&2, Nat, Spec.rows(j, 1n+a), Spec.rows(m2, Nat.add(a, 1n+j))),
        Equal.trans(List<&2, Nat>, Spec.rows(Nat.add(j, m2), 1n+a), List.append(&2, Nat, Spec.rows(j, 1n+a), Spec.rows(m2, Nat.add(1n+a, j))),
          List.append(&2, Nat, Spec.rows(j, 1n+a), Spec.rows(m2, Nat.add(a, 1n+j))), ih,
          Equal.cong(List<&2, Nat>, List<&2, Nat>, z => List.append(&2, Nat, Spec.rows(j, 1n+a), z), Spec.rows(m2, Nat.add(1n+a, j)),
            Spec.rows(m2, Nat.add(a, 1n+j)), e)))

def Laws.checkpoint_positions():
  ({==}, {==}, {==}, {==})

# ---------------------------------------------------------------------------------------------
# 6. Chunk independence of a pattn row

def pm_mk(m: Bool, +A: Nat, +r: Nat, +p: Nat) -> Bool:
  match m:
    case False{}:
      True{}
    case True{}:
      Nat.is_le(p, Nat.add(A, r))

def pm_V(+A: Nat, +r: Nat, +it: Nat, +p: Nat) -> Bool:
  Bool.and(Nat.is_le(P6.n0(it), p), Bool.and(Nat.is_lt(p, Nat.add(P6.n0(it), 32n)),
    Bool.and(Nat.is_lt(it, Stride.cdiv(Nat.add(A, 16n), 32n)), pm_mk(Nat.is_lt(A, Nat.add(P6.n0(it), 31n)), A, r, p))))

def pm_hi(+L: Nat, +q: Nat, +qb: Nat, full: {Nat.is_le(Nat.mul(1n+qb, 16n), q) == True{} : Bool}, len: {Nat.is_le(q, L) == True{} : Bool})
  -> {Nat.is_le(Nat.add(P6.q_abs0(L, q, qb), 16n), L) == True{} : Bool}:
  +D = Nat.sub(L, q)
  +B = Nat.mul(qb, 16n)
  +f2 = Equal.trans(Bool, Nat.is_le(Nat.add(B, 16n), q), Nat.is_le(Nat.add(16n, B), q), True{},
    Equal.cong(Nat, Bool, z => Nat.is_le(z, q), Nat.add(B, 16n), Nat.add(16n, B), Ar.ar_add_comm(B, 16n)), full)
  +le1 = Ar.ard_tr(Nat.is_le(Nat.add(D, Nat.add(B, 16n)), Nat.add(D, q)), Nat.is_le(Nat.add(B, 16n), q), Ar.ar_le_add_cancel(D, Nat.add(B, 16n), q), f2)
  +eL = Equal.trans(Nat, Nat.add(D, q), Nat.add(q, D), L, Ar.ar_add_comm(D, q), Ar.ar_sub_add(L, q, len))
  Equal.trans(Bool, Nat.is_le(Nat.add(Nat.add(D, B), 16n), L), Nat.is_le(Nat.add(D, Nat.add(B, 16n)), Nat.add(D, q)), True{},
    pm_le_cong(Nat.add(Nat.add(D, B), 16n), Nat.add(D, Nat.add(B, 16n)), L, Nat.add(D, q), Ar.ar_add_assoc(D, B, 16n), Equal.sym(Nat, Nat.add(D, q), L, eL)),
    le1)

def pm_nhi(+L: Nat, +q: Nat, +qb: Nat, full: {Nat.is_le(Nat.mul(1n+qb, 16n), q) == True{} : Bool}, len: {Nat.is_le(q, L) == True{} : Bool})
  -> {P6.n_hi(L, q, qb) == Nat.add(P6.q_abs0(L, q, qb), 16n) : Nat}:
  Ar.ar_min_left(Nat.add(P6.q_abs0(L, q, qb), 16n), L, pm_hi(L, q, qb, full, len))

def pm_mask(m: Bool, +L: Nat, +q: Nat, +qb: Nat, +r: Nat, +p: Nat, rb: {Nat.is_lt(r, 16n) == True{} : Bool},
    full: {Nat.is_le(Nat.mul(1n+qb, 16n), q) == True{} : Bool}, len: {Nat.is_le(q, L) == True{} : Bool})
  -> {P6.mask_ok(m, L, q, qb, r, p) == pm_mk(m, P6.q_abs0(L, q, qb), r, p) : Bool}:
  match m:
    case False{}:
      {==}
    case True{}:
      +A = P6.q_abs0(L, q, qb)
      +hi = pm_hi(L, q, qb, full, len)
      +ltr = Ar.ard_tr(Nat.is_lt(Nat.add(A, r), Nat.add(A, 16n)), Nat.is_lt(r, 16n), Ar.ar_lt_add_cancel(A, r, 16n), rb)
      +aL = Ar.ar_lt_le_trans(Nat.add(A, r), Nat.add(A, 16n), L, ltr, hi)
      SP.sp_keep(Nat.is_le(p, Nat.add(A, r)), Nat.is_lt(p, L), hs => Ar.ar_le_lt_trans(p, Nat.add(A, r), L, hs, aL))

def pm_vis(+L: Nat, +q: Nat, +qb: Nat, +r: Nat, +it: Nat, +p: Nat, +rb: {Nat.is_lt(r, 16n) == True{} : Bool},
    +full: {Nat.is_le(Nat.mul(1n+qb, 16n), q) == True{} : Bool}, +len: {Nat.is_le(q, L) == True{} : Bool})
  -> {P6.visits(L, q, qb, r, it, p) == pm_V(P6.q_abs0(L, q, qb), r, it, p) : Bool}:
  +A = P6.q_abs0(L, q, qb)
  %Equal.sym(Nat, P6.n_hi(L, q, qb), Nat.add(A, 16n), pm_nhi(L, q, qb, full, len))
    : {Bool.and(Nat.is_le(P6.n0(it), p), Bool.and(Nat.is_lt(p, Nat.add(P6.n0(it), 32n)), Bool.and(Nat.is_lt(it, Stride.cdiv(_, 32n)),
        P6.mask_ok(P6.masked(L, q, qb, it), L, q, qb, r, p)))) == pm_V(A, r, it, p) : Bool}
  %Equal.sym(Bool, P6.mask_ok(P6.masked(L, q, qb, it), L, q, qb, r, p), pm_mk(P6.masked(L, q, qb, it), A, r, p),
      pm_mask(P6.masked(L, q, qb, it), L, q, qb, r, p, rb, full, len))
    : {Bool.and(Nat.is_le(P6.n0(it), p), Bool.and(Nat.is_lt(p, Nat.add(P6.n0(it), 32n)), Bool.and(Nat.is_lt(it, Stride.cdiv(Nat.add(A, 16n), 32n)),
        _))) == pm_V(A, r, it, p) : Bool}
  {==}

def Laws.pattn_chunk_invariant(L, q, qb, L2, q2, qb2, r, it, p, rb, full, full2, len, len2, same):
  +rbb = rb
  +f1 = full
  +f2 = full2
  +l1 = len
  +l2 = len2
  +A1 = P6.q_abs0(L, q, qb)
  +A2 = P6.q_abs0(L2, q2, qb2)
  Equal.trans(Bool, P6.visits(L, q, qb, r, it, p), pm_V(A1, r, it, p), P6.visits(L2, q2, qb2, r, it, p),
    pm_vis(L, q, qb, r, it, p, rbb, f1, l1),
    Equal.trans(Bool, pm_V(A1, r, it, p), pm_V(A2, r, it, p), P6.visits(L2, q2, qb2, r, it, p),
      Equal.cong(Nat, Bool, z => pm_V(z, r, it, p), A1, A2, same),
      Equal.sym(Bool, P6.visits(L2, q2, qb2, r, it, p), pm_V(A2, r, it, p), pm_vis(L2, q2, qb2, r, it, p, rbb, f2, l2))))

# ---------------------------------------------------------------------------------------------
# 7. Wide-GEMM route at M = 2048 and 4096

def Laws.gemm_route_m(env, major, minor, batch, K, N, f):
  match env:
    case HWS.HwUnset{}:
      {==}
    case HWS.HwOne{}:
      {==}
    case HWS.HwZero{}:
      {==}
    case HWS.HwOther{}:
      {==}
