# Proofs of pattn8_sched_laws.bend (ext 3020 8-warp prefill attention against ext 3010). Nat lemmas from
# gemm_m16_partition_proof.bend (Ar) and attn_stride_proof.bend (SP); 3010's own coverage laws from pattn_sched_laws.bend
# (proven in pattn_sched_proof.bend, imported here so the composed claims are discharged).
import Base
import ./pattn8_sched.bend as P8
import ./pattn8_sched_spec.bend as Spec
import ./pattn8_sched_laws.bend as Laws
import ./pattn_sched.bend as P6
import ./pattn_sched_spec.bend as S6
import ./pattn_sched_laws.bend as L6
import ./pattn_sched_proof.bend as PSP
import ./gemm_m16_partition_proof.bend as Ar
import ./attn_stride_proof.bend as SP
import ./attn_stride.bend as Stride

# ---------------------------------------------------------------------------------------------
# Arithmetic helpers

# a <= c from a == a2 and a2 <= c
def q8_le_rw_l(-a: Nat, -a2: Nat, -c: Nat, e: {a == a2 : Nat}, h: {Nat.is_le(a2, c) == True{} : Bool})
  -> {Nat.is_le(a, c) == True{} : Bool}:
  %Equal.sym(Nat, a, a2, e) : {Nat.is_le(_, c) == True{} : Bool}
  h

# x <= y gives x + k <= y + k
def q8_le_add_mono(+x: Nat, +y: Nat, +k: Nat, h: {Nat.is_le(x, y) == True{} : Bool})
  -> {Nat.is_le(Nat.add(x, k), Nat.add(y, k)) == True{} : Bool}:
  %Ar.ar_add_comm(k, x) : {Nat.is_le(_, Nat.add(y, k)) == True{} : Bool}
  %Ar.ar_add_comm(k, y) : {Nat.is_le(Nat.add(k, x), _) == True{} : Bool}
  Ar.ard_tr(Nat.is_le(Nat.add(k, x), Nat.add(k, y)), Nat.is_le(x, y), Ar.ar_le_add_cancel(k, x, y), h)

# x <= y gives min(x, L) <= min(y, L)
def q8_min_mono(+x: Nat, +y: Nat, +L: Nat, h: {Nat.is_le(x, y) == True{} : Bool})
  -> {Nat.is_le(Nat.min(x, L), Nat.min(y, L)) == True{} : Bool}:
  match x y L:
    case 0n _ _:
      Ar.ar_h_le_zero(Nat.min(y, L))
    case 1n+a 0n _:
      Empty.absurd({Nat.is_le(Nat.min(1n+a, L), 0n) == True{} : Bool}, Ar.ar_false_true(h))
    case 1n+a 1n+c 0n:
      {==}
    case 1n+a 1n+c 1n+l:
      q8_min_mono(a, c, l, h)

# x <= y gives ceil(x / 32) <= ceil(y / 32)
def q8_cdiv_mono(+x: Nat, +y: Nat, h: {Nat.is_le(x, y) == True{} : Bool})
  -> {Nat.is_le(Stride.cdiv(x, 32n), Stride.cdiv(y, 32n)) == True{} : Bool}:
  %Equal.sym(Nat, Nat.sub(Nat.add(x, 32n), 1n), Nat.add(x, 31n), SP.sp_pred(x, 31n))
    : {Nat.is_le(Nat.div(_, 32n), Nat.div(Nat.sub(Nat.add(y, 32n), 1n), 32n)) == True{} : Bool}
  %Equal.sym(Nat, Nat.sub(Nat.add(y, 32n), 1n), Nat.add(y, 31n), SP.sp_pred(y, 31n))
    : {Nat.is_le(Nat.div(Nat.add(x, 31n), 32n), Nat.div(_, 32n)) == True{} : Bool}
  Ar.ar_div_mono(Nat.add(x, 31n), Nat.add(y, 31n), 32n, {==}, q8_le_add_mono(x, y, 31n, h))

# (x && s) == s when s implies x
def q8_keep_r(+x: Bool, +s: Bool, h: {s == True{} : Bool} -> {x == True{} : Bool})
  -> {Bool.and(x, s) == s : Bool}:
  match x s:
    case False{} False{}:
      {==}
    case False{} True{}:
      Empty.absurd({Bool.and(False{}, True{}) == True{} : Bool}, Ar.ar_false_true(h({==})))
    case True{} False{}:
      {==}
    case True{} True{}:
      {==}

# b < 4 gives w % 4 style bounds: the warp's block index is below 4 for every w
def q8_wb_lt(+w: Nat) -> {Nat.is_lt(P8.wb(w), 4n) == True{} : Bool}:
  Ar.ard_dm_lt(w, 4n, {==})

# w < 8 gives w / 4 < 2
def q8_wh_lt(+w: Nat, wlt: {Nat.is_lt(w, 8n) == True{} : Bool}) -> {Nat.is_lt(P8.wh(w), 2n) == True{} : Bool}:
  Ar.ard_tr(Nat.is_lt(Nat.div(w, 4n), 2n), Nat.is_lt(w, 8n), Ar.ar_div_lt(2n, w, 4n, {==}), wlt)

# p0(qc, b) == (4 qc + b) * 16
def q8_p0(+qc: Nat, +b: Nat) -> {P8.p0(qc, b) == Nat.mul(Nat.add(Nat.mul(qc, 4n), b), 16n) : Nat}:
  Equal.sym(Nat, Nat.mul(Nat.add(Nat.mul(qc, 4n), b), 16n), Nat.add(Nat.mul(Nat.mul(qc, 4n), 16n), Nat.mul(b, 16n)),
    Ar.ar_mul_add_l(Nat.mul(qc, 4n), b, 16n))

# p0(qc, b) + r == 3010's position of row r of block 4 qc + b
def q8_pos(+qc: Nat, +b: Nat, +r: Nat)
  -> {Nat.add(P8.p0(qc, b), r) == P6.q_pos(Nat.add(Nat.mul(qc, 4n), b), r) : Nat}:
  Equal.cong(Nat, Nat, z => Nat.add(z, r), P8.p0(qc, b), Nat.mul(Nat.add(Nat.mul(qc, 4n), b), 16n), q8_p0(qc, b))

# ---------------------------------------------------------------------------------------------
# 1-3. Warp map

def Laws.warp_map(kvh, pair, qc, w, pb, wlt):
  +h = P8.wh(w)
  +hlt = q8_wh_lt(w, wlt)
  +P2 = Nat.mul(pair, 2n)
  +a1 = Ar.ard_tr(Nat.is_lt(Nat.add(P2, h), Nat.add(P2, 2n)), Nat.is_lt(h, 2n), Ar.ar_lt_add_cancel(P2, h, 2n), hlt)
  +le6 = Ar.ar_le_mul_mono(1n+pair, 3n, 2n, Ar.ard_rt(Nat.is_lt(pair, 3n), Nat.is_le(1n+pair, 3n), Ar.ar_lt_succ_le(pair, 3n), pb))
  +le6b = q8_le_rw_l(Nat.add(P2, 2n), Nat.add(2n, P2), 6n, Ar.ar_add_comm(P2, 2n), le6)
  (Ar.ar_lt_le_trans(Nat.add(P2, h), Nat.add(P2, 2n), 6n, a1, le6b),
   Ar.ar_add_assoc(Nat.mul(kvh, 6n), P2, h),
   Equal.trans(Nat, P8.p0(qc, P8.wb(w)), Nat.mul(Nat.add(Nat.mul(qc, 4n), P8.wb(w)), 16n),
     Nat.add(Nat.mul(Nat.add(Nat.mul(qc, 4n), P8.wb(w)), 16n), 0n),
     q8_p0(qc, P8.wb(w)),
     Equal.sym(Nat, Nat.add(Nat.mul(Nat.add(Nat.mul(qc, 4n), P8.wb(w)), 16n), 0n), Nat.mul(Nat.add(Nat.mul(qc, 4n), P8.wb(w)), 16n),
       Ar.ar_add_zero_r(Nat.mul(Nat.add(Nat.mul(qc, 4n), P8.wb(w)), 16n)))))

def Laws.warp_map_inj(pair, qc, w, wlt):
  +h = P8.wh(w)
  +v = P8.wb(w)
  +hl = q8_wh_lt(w, wlt)
  +vl = q8_wb_lt(w)
  +mh = Ar.p7_mod_r(pair, h, 2n, hl)
  +mv = Ar.p7_mod_r(qc, v, 4n, vl)
  (Ar.p7_div_q(pair, h, 2n, hl),
   (%Equal.sym(Nat, Nat.mod(Nat.add(Nat.mul(pair, 2n), h), 2n), h, mh)
      : {Nat.add(Nat.mul(_, 4n), Nat.mod(Nat.add(Nat.mul(qc, 4n), v), 4n)) == w : Nat}
    %Equal.sym(Nat, Nat.mod(Nat.add(Nat.mul(qc, 4n), v), 4n), v, mv)
      : {Nat.add(Nat.mul(h, 4n), _) == w : Nat}
    Ar.ard_dm_eq(w, 4n, {==})),
   Ar.p7_div_q(qc, v, 4n, vl))

def Laws.warp_map_onto(w6, qb, w6b):
  +u = Nat.mod(w6, 2n)
  +v = Nat.mod(qb, 4n)
  +ult = Ar.ard_dm_lt(w6, 2n, {==})
  +vlt = Ar.ard_dm_lt(qb, 4n, {==})
  +du = Ar.p7_div_q(u, v, 4n, vlt)
  +dv = Ar.p7_mod_r(u, v, 4n, vlt)
  (Ar.ard_tr(Nat.is_lt(Nat.div(w6, 2n), 3n), Nat.is_lt(w6, 6n), Ar.ar_div_lt(3n, w6, 2n, {==}), w6b),
   Ar.p7_mr_lt(u, v, 2n, 4n, ult, vlt),
   (%Equal.sym(Nat, Nat.div(Nat.add(Nat.mul(u, 4n), v), 4n), u, du) : {Nat.add(Nat.mul(Nat.div(w6, 2n), 2n), _) == w6 : Nat}
    Ar.ard_dm_eq(w6, 2n, {==})),
   (%Equal.sym(Nat, Nat.mod(Nat.add(Nat.mul(u, 4n), v), 4n), v, dv) : {Nat.add(Nat.mul(Nat.div(qb, 4n), 4n), _) == qb : Nat}
    Ar.ard_dm_eq(qb, 4n, {==})))

# ---------------------------------------------------------------------------------------------
# 4-5. Grid and idle warps

def Laws.grid_cover(q, qb, live):
  +d = Nat.div(qb, 4n)
  +e64 = Ar.ar_mul_assoc(d, 4n, 16n)
  +le = Ar.ar_le_mul_mono(Nat.mul(d, 4n), qb, 16n, Ar.ard_le_mul_div(qb, 4n, {==}))
  +le2 = q8_le_rw_l(Nat.mul(d, 64n), Nat.mul(Nat.mul(d, 4n), 16n), Nat.mul(qb, 16n), Equal.sym(Nat, Nat.mul(Nat.mul(d, 4n), 16n), Nat.mul(d, 64n), e64), le)
  SP.sp_lt_cdiv(d, q, 64n, {==}, Ar.ar_le_lt_trans(Nat.mul(d, 64n), Nat.mul(qb, 16n), q, le2, live))

def Laws.idle_warp(L, q, qc, b, dead):
  %Equal.sym(Bool, Nat.is_lt(P8.p0(qc, b), q), False{}, dead) : {P8.ntiles.pick(_, L, q, qc, b) == 0n : Nat}
  {==}

# ---------------------------------------------------------------------------------------------
# 6-8. Bounds, mask state, tile counts

def Laws.q_abs_eq(L, q, qc, b):
  Equal.cong(Nat, Nat, z => Nat.add(Nat.sub(L, q), z), P8.p0(qc, b), Nat.mul(Nat.add(Nat.mul(qc, 4n), b), 16n), q8_p0(qc, b))

def Laws.masked_eq(L, q, qc, b, it):
  Equal.cong(Nat, Bool, z => Nat.is_lt(z, Nat.add(P8.n0(it), 31n)), P8.q_abs0(L, q, qc, b),
    P6.q_abs0(L, q, Nat.add(Nat.mul(qc, 4n), b)), Laws.q_abs_eq(L, q, qc, b))

def Laws.ntiles_eq(L, q, qc, b, live):
  %Equal.sym(Bool, Nat.is_lt(P8.p0(qc, b), q), True{}, live) : {P8.ntiles.pick(_, L, q, qc, b) == P6.ntiles(L, q, Nat.add(Nat.mul(qc, 4n), b)) : Nat}
  Equal.cong(Nat, Nat, z => Stride.cdiv(Nat.min(Nat.add(z, 16n), L), 32n), P8.q_abs0(L, q, qc, b),
    P6.q_abs0(L, q, Nat.add(Nat.mul(qc, 4n), b)), Laws.q_abs_eq(L, q, qc, b))

# the warp's key bound q_abs0 + 16 is at most the CTA's (L - q) + c0 + 64
def q8_hi_le(+L: Nat, +q: Nat, +qc: Nat, +b: Nat, blt: {Nat.is_lt(b, 4n) == True{} : Bool})
  -> {Nat.is_le(Nat.add(P8.q_abs0(L, q, qc, b), 16n), Nat.add(Nat.add(Nat.sub(L, q), P8.c0(qc)), 64n)) == True{} : Bool}:
  +D = Nat.sub(L, q)
  +C = P8.c0(qc)
  +B = Nat.mul(b, 16n)
  +e1 = Ar.ar_add_assoc(D, Nat.add(C, B), 16n)
  +e2 = Equal.cong(Nat, Nat, z => Nat.add(D, z), Nat.add(Nat.add(C, B), 16n), Nat.add(C, Nat.add(B, 16n)), Ar.ar_add_assoc(C, B, 16n))
  +e3 = Equal.sym(Nat, Nat.add(Nat.add(D, C), Nat.add(B, 16n)), Nat.add(D, Nat.add(C, Nat.add(B, 16n))), Ar.ar_add_assoc(D, C, Nat.add(B, 16n)))
  +ex = Equal.trans(Nat, Nat.add(Nat.add(D, Nat.add(C, B)), 16n), Nat.add(D, Nat.add(C, Nat.add(B, 16n))), Nat.add(Nat.add(D, C), Nat.add(B, 16n)),
    Equal.trans(Nat, Nat.add(Nat.add(D, Nat.add(C, B)), 16n), Nat.add(D, Nat.add(Nat.add(C, B), 16n)), Nat.add(D, Nat.add(C, Nat.add(B, 16n))), e1, e2), e3)
  +b16 = Ar.ar_le_mul_mono(1n+b, 4n, 16n, Ar.ard_rt(Nat.is_lt(b, 4n), Nat.is_le(1n+b, 4n), Ar.ar_lt_succ_le(b, 4n), blt))
  +b16b = q8_le_rw_l(Nat.add(B, 16n), Nat.add(16n, B), 64n, Ar.ar_add_comm(B, 16n), b16)
  +inner = Ar.ard_tr(Nat.is_le(Nat.add(Nat.add(D, C), Nat.add(B, 16n)), Nat.add(Nat.add(D, C), 64n)), Nat.is_le(Nat.add(B, 16n), 64n),
    Ar.ar_le_add_cancel(Nat.add(D, C), Nat.add(B, 16n), 64n), b16b)
  q8_le_rw_l(Nat.add(Nat.add(D, Nat.add(C, B)), 16n), Nat.add(Nat.add(D, C), Nat.add(B, 16n)), Nat.add(Nat.add(D, C), 64n), ex, inner)

def q8_fit_pick(live: Bool, +L: Nat, +q: Nat, +qc: Nat, +b: Nat, blt: {Nat.is_lt(b, 4n) == True{} : Bool})
  -> {Nat.is_le(P8.ntiles.pick(live, L, q, qc, b), P8.ntiles_cta(L, q, qc)) == True{} : Bool}:
  match live:
    case False{}:
      Ar.ar_h_le_zero(P8.ntiles_cta(L, q, qc))
    case True{}:
      q8_cdiv_mono(P8.n_hi(L, q, qc, b), P8.n_hi_cta(L, q, qc),
        q8_min_mono(Nat.add(P8.q_abs0(L, q, qc, b), 16n), Nat.add(Nat.add(Nat.sub(L, q), P8.c0(qc)), 64n), L, q8_hi_le(L, q, qc, b, blt)))

def Laws.ntiles_fit(L, q, qc, b, blt):
  q8_fit_pick(Nat.is_lt(P8.p0(qc, b), q), L, q, qc, b, blt)

# ---------------------------------------------------------------------------------------------
# 9. The warp's iteration trace

# s < n gives n - s == 1 + (n - (s + 1))
def q8_sub_step(+n: Nat, +s: Nat, h: {Nat.is_lt(s, n) == True{} : Bool}) -> {Nat.sub(n, s) == 1n+Nat.sub(n, 1n+s) : Nat}:
  match n s:
    case 0n _:
      Empty.absurd({Nat.sub(0n, s) == 1n+Nat.sub(0n, 1n+s) : Nat}, Ar.ar_false_true(Equal.trans(Bool, False{}, Nat.is_lt(s, 0n), True{}, Equal.sym(Bool, Nat.is_lt(s, 0n), False{}, Ar.ar_h_lt_zero(s)), h)))
    case 1n+m 0n:
      %Equal.sym(Nat, Nat.sub(m, 0n), m, SP.sp_sub_zero(m)) : {1n+m == 1n+_ : Nat}
      {==}
    case 1n+m 1n+t:
      q8_sub_step(m, t, h)

# n <= s: the loop from s keeps nothing
def q8_steps_nil(k: Nat, +s: Nat, +n: Nat, +h: {Nat.is_le(n, s) == True{} : Bool}) -> {P8.steps(k, s, n) == Nil{} : List<&2, Nat>}:
  match k:
    case 0n:
      {==}
    case 1n+j:
      %Equal.sym(Bool, Nat.is_lt(s, n), False{}, Ar.p6_le_nlt(s, n, h)) : {P8.steps.keep(_, s, P8.steps(j, 1n+s, n)) == Nil{} : List<&2, Nat>}
      q8_steps_nil(j, 1n+s, n, Ar.ar_le_trans(n, s, 1n+s, h, Ar.ar_h_le_succ(s)))

def q8_steps_pick(c: Bool, j: Nat, +s: Nat, +n: Nat, hc: {Nat.is_lt(s, n) == c : Bool},
    ih: {P8.steps(j, 1n+s, n) == Spec.upto(Nat.sub(n, 1n+s), 1n+s) : List<&2, Nat>})
  -> {P8.steps.keep(c, s, P8.steps(j, 1n+s, n)) == Spec.upto(Nat.sub(n, s), s) : List<&2, Nat>}:
  match c:
    case True{}:
      %Equal.sym(Nat, Nat.sub(n, s), 1n+Nat.sub(n, 1n+s), q8_sub_step(n, s, hc)) : {Con{s, P8.steps(j, 1n+s, n)} == Spec.upto(_, s) : List<&2, Nat>}
      Equal.cong(List<&2, Nat>, List<&2, Nat>, z => Con{s, z}, P8.steps(j, 1n+s, n), Spec.upto(Nat.sub(n, 1n+s), 1n+s), ih)
    case False{}:
      +nle = Ar.p6_nlt_le(s, n, hc)
      +e0a = SP.sp_sub_le0(n, 1n+s, Ar.ar_le_trans(n, s, 1n+s, nle, Ar.ar_h_le_succ(s)))
      +e0b = SP.sp_sub_le0(n, s, nle)
      Equal.trans(List<&2, Nat>, P8.steps(j, 1n+s, n), Spec.upto(Nat.sub(n, 1n+s), 1n+s), Spec.upto(Nat.sub(n, s), s), ih,
        Equal.trans(List<&2, Nat>, Spec.upto(Nat.sub(n, 1n+s), 1n+s), Nil{}, Spec.upto(Nat.sub(n, s), s),
          Equal.cong(Nat, List<&2, Nat>, z => Spec.upto(z, 1n+s), Nat.sub(n, 1n+s), 0n, e0a),
          Equal.sym(List<&2, Nat>, Spec.upto(Nat.sub(n, s), s), Nil{},
            Equal.cong(Nat, List<&2, Nat>, z => Spec.upto(z, s), Nat.sub(n, s), 0n, e0b))))

# n <= s + k: the loop s .. s + k - 1 keeping it < n is [s .. n - 1] (empty when n <= s)
def q8_steps_upto(+k: Nat, +s: Nat, +n: Nat, h2: {Nat.is_le(n, Nat.add(s, k)) == True{} : Bool})
  -> {P8.steps(k, s, n) == Spec.upto(Nat.sub(n, s), s) : List<&2, Nat>}:
  match k:
    case 0n:
      +h2b = Equal.trans(Bool, Nat.is_le(n, s), Nat.is_le(n, Nat.add(s, 0n)), True{},
        Equal.cong(Nat, Bool, z => Nat.is_le(n, z), s, Nat.add(s, 0n), Equal.sym(Nat, Nat.add(s, 0n), s, Ar.ar_add_zero_r(s))), h2)
      %Equal.sym(Nat, Nat.sub(n, s), 0n, SP.sp_sub_le0(n, s, h2b)) : {Nil{} == Spec.upto(_, s) : List<&2, Nat>}
      {==}
    case 1n+j:
      +h2j = Equal.trans(Bool, Nat.is_le(n, Nat.add(1n+s, j)), Nat.is_le(n, Nat.add(s, 1n+j)), True{},
        Equal.cong(Nat, Bool, z => Nat.is_le(n, z), Nat.add(1n+s, j), Nat.add(s, 1n+j), Equal.sym(Nat, Nat.add(s, 1n+j), 1n+Nat.add(s, j), Ar.ar_add_succ_r(s, j))),
        h2)
      q8_steps_pick(Nat.is_lt(s, n), j, s, n, {==}, q8_steps_upto(j, 1n+s, n, h2j))

def Laws.trace_eq(L, q, qc, b, blt, live):
  +n = P8.ntiles(L, q, qc, b)
  +k = P8.ntiles_cta(L, q, qc)
  +st = q8_steps_upto(k, 0n, n, Laws.ntiles_fit(L, q, qc, b, blt))
  +en = Laws.ntiles_eq(L, q, qc, b, live)
  %en : {P8.trace(L, q, qc, b) == Spec.upto(_, 0n) : List<&2, Nat>}
  %SP.sp_sub_zero(n) : {P8.trace(L, q, qc, b) == Spec.upto(_, 0n) : List<&2, Nat>}
  st

# ---------------------------------------------------------------------------------------------
# 10-11. Kept keys and causal coverage

def q8_mask_eq(m: Bool, +L: Nat, +q: Nat, +qc: Nat, +b: Nat, +r: Nat, +p: Nat)
  -> {P8.mask_ok(m, L, q, qc, b, r, p) == P6.mask_ok(m, L, q, Nat.add(Nat.mul(qc, 4n), b), r, p) : Bool}:
  match m:
    case False{}:
      {==}
    case True{}:
      Equal.cong(Nat, Bool, z => Bool.and(Nat.is_le(p, Nat.add(z, r)), Nat.is_lt(p, L)), P8.q_abs0(L, q, qc, b),
        P6.q_abs0(L, q, Nat.add(Nat.mul(qc, 4n), b)), Laws.q_abs_eq(L, q, qc, b))

def Laws.visits_eq(L, q, qc, b, r, it, p, blt, live):
  +qb = Nat.add(Nat.mul(qc, 4n), b)
  +n8 = P8.ntiles(L, q, qc, b)
  +k = P8.ntiles_cta(L, q, qc)
  +fit = Laws.ntiles_fit(L, q, qc, b, blt)
  +keep = q8_keep_r(Nat.is_lt(it, k), Nat.is_lt(it, n8), h => Ar.ar_lt_le_trans(it, n8, k, h, fit))
  +M8 = P8.mask_ok(P8.masked(L, q, qc, b, it), L, q, qc, b, r, p)
  %Equal.sym(Bool, Bool.and(Nat.is_lt(it, k), Nat.is_lt(it, n8)), Nat.is_lt(it, n8), keep)
    : {Bool.and(Nat.is_le(P8.n0(it), p), Bool.and(Nat.is_lt(p, Nat.add(P8.n0(it), 32n)), Bool.and(_, M8))) == P6.visits(L, q, qb, r, it, p) : Bool}
  %Equal.sym(Nat, n8, P6.ntiles(L, q, qb), Laws.ntiles_eq(L, q, qc, b, live))
    : {Bool.and(Nat.is_le(P8.n0(it), p), Bool.and(Nat.is_lt(p, Nat.add(P8.n0(it), 32n)), Bool.and(Nat.is_lt(it, _), M8))) == P6.visits(L, q, qb, r, it, p) : Bool}
  %Equal.sym(Bool, M8, P6.mask_ok(P8.masked(L, q, qc, b, it), L, q, qb, r, p), q8_mask_eq(P8.masked(L, q, qc, b, it), L, q, qc, b, r, p))
    : {Bool.and(Nat.is_le(P8.n0(it), p), Bool.and(Nat.is_lt(p, Nat.add(P8.n0(it), 32n)), Bool.and(Nat.is_lt(it, P6.ntiles(L, q, qb)), _))) == P6.visits(L, q, qb, r, it, p) : Bool}
  %Equal.sym(Bool, P8.masked(L, q, qc, b, it), P6.masked(L, q, qb, it), Laws.masked_eq(L, q, qc, b, it))
    : {Bool.and(Nat.is_le(P8.n0(it), p), Bool.and(Nat.is_lt(p, Nat.add(P8.n0(it), 32n)), Bool.and(Nat.is_lt(it, P6.ntiles(L, q, qb)), P6.mask_ok(_, L, q, qb, r, p)))) == P6.visits(L, q, qb, r, it, p) : Bool}
  {==}

def Laws.attend_once8(L, q, qc, b, r, p, blt, rb, row, len):
  +qb = Nat.add(Nat.mul(qc, 4n), b)
  +rw = row
  +live = Ar.ar_le_lt_trans(P8.p0(qc, b), Nat.add(P8.p0(qc, b), r), q, Ar.ar_le_add_r(P8.p0(qc, b), r), rw)
  +epos = q8_pos(qc, b, r)
  +row6 = Equal.trans(Bool, Nat.is_lt(P6.q_pos(qb, r), q), Nat.is_lt(Nat.add(P8.p0(qc, b), r), q), True{},
    Equal.cong(Nat, Bool, z => Nat.is_lt(z, q), P6.q_pos(qb, r), Nat.add(P8.p0(qc, b), r), Equal.sym(Nat, Nat.add(P8.p0(qc, b), r), P6.q_pos(qb, r), epos)), rw)
  %Equal.sym(Bool, P8.visits(L, q, qc, b, r, Nat.div(p, 32n), p), P6.visits(L, q, qb, r, Nat.div(p, 32n), p), Laws.visits_eq(L, q, qc, b, r, Nat.div(p, 32n), p, blt, live))
    : {S6.once(_) == S6.once(S6.sees(L, q, Nat.add(P8.p0(qc, b), r), p)) : Nat}
  %Equal.sym(Nat, Nat.add(P8.p0(qc, b), r), P6.q_pos(qb, r), epos) : {S6.once(P6.visits(L, q, qb, r, Nat.div(p, 32n), p)) == S6.once(S6.sees(L, q, _, p)) : Nat}
  L6.attend_once(L, q, qb, r, p, rb, row6, len)

def Laws.tile_unique8(L, q, qc, b, r, it, p, blt, live):
  %Equal.sym(Bool, P8.visits(L, q, qc, b, r, it, p), P6.visits(L, q, Nat.add(Nat.mul(qc, 4n), b), r, it, p), Laws.visits_eq(L, q, qc, b, r, it, p, blt, live))
    : {Bool.and(_, Nat.is_ne(it, Nat.div(p, 32n))) == False{} : Bool}
  L6.tile_unique(L, q, Nat.add(Nat.mul(qc, 4n), b), r, it, p)

# ---------------------------------------------------------------------------------------------
# 12-13. q prologue rows and tile issue

def Laws.q_rows(kvh, pair, qc, w, pos, posb):
  +row = P8.qrow(w, pos)
  +dw = Ar.p7_div_q(w, pos, 16n, posb)
  +ep = Equal.trans(Nat, P8.prow_pos(row), Nat.sub(row, Nat.mul(w, 16n)), pos,
    Equal.cong(Nat, Nat, z => Nat.sub(row, Nat.mul(z, 16n)), P8.prow_w(row), w, dw),
    Ar.ar_add_sub(Nat.mul(w, 16n), pos))
  (Equal.cong(Nat, Nat, z => Nat.add(Nat.add(Nat.mul(kvh, 6n), Nat.mul(pair, 2n)), Nat.div(z, 4n)), P8.prow_w(row), w, dw),
   (%Equal.sym(Nat, P8.prow_pos(row), pos, ep)
      : {Nat.add(Nat.add(P8.c0(qc), Nat.mul(Nat.mod(P8.prow_w(row), 4n), 16n)), _) == Nat.add(P8.p0(qc, P8.wb(w)), pos) : Nat}
    %Equal.sym(Nat, P8.prow_w(row), w, dw)
      : {Nat.add(Nat.add(P8.c0(qc), Nat.mul(Nat.mod(_, 4n), 16n)), pos) == Nat.add(P8.p0(qc, P8.wb(w)), pos) : Nat}
    {==}))

def Laws.issue_once(T, tid, i, c, tb, cb):
  +tl = tb
  +pos = Ar.ard_pos(tid, T, tl)
  +ec = Ar.ar_add_comm(Nat.mul(i, T), tid)
  +lec = Ar.ard_le_mul_div(c, T, pos)
  (%ec : {Nat.div(_, T) == i : Nat}
   Ar.p7_div_q(i, tid, T, tl),
   %ec : {Nat.mod(_, T) == tid : Nat}
   Ar.p7_mod_r(i, tid, T, tl),
   %Ar.ar_add_comm(Nat.mul(Nat.div(c, T), T), Nat.mod(c, T)) : {_ == c : Nat}
   Ar.ard_dm_eq(c, T, pos),
   SP.sp_lt_cdiv(Nat.div(c, T), 1024n, T, pos, Ar.ar_le_lt_trans(Nat.mul(Nat.div(c, T), T), c, 1024n, lec, cb)))
