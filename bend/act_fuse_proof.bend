# Proofs of act_fuse_laws.bend (ext 5112 fused SiLU-gate epilogue against stock hgemm_recon + silu_mul). Nat lemmas
# from gemm_m16_partition_proof.bend (Ar) and attn_stride_proof.bend (SP).
import Base
import ./act_fuse.bend as Impl
import ./act_fuse_spec.bend as Spec
import ./act_fuse_laws.bend as Laws
import ./hgemm_wide.bend as HW
import ./hgemm_wide_spec.bend as HWS
import ./gemm_m16_partition_proof.bend as Ar
import ./attn_stride_proof.bend as SP

# ---------------------------------------------------------------------------------------------
# 1. Pair-aligned store columns

# P * 2 == the store column without its element offset e
def af_col2(+tn: Nat, +tid: Nat, +j: Nat)
  -> {Nat.mul(Impl.pair_col(tn, tid, j), 2n) ==
      Nat.add(Nat.add(Nat.add(Nat.mul(tn, 128n), Nat.mul(HW.wn(tid), 64n)), Nat.mul(j, 8n)), Nat.mul(HW.tq(tid), 2n)) : Nat}:
  +a = Nat.mul(tn, 64n)
  +w = Nat.mul(HW.wn(tid), 32n)
  +b = Nat.mul(j, 4n)
  +t = HW.tq(tid)
  +d1 = Equal.trans(Nat, Nat.mul(Nat.add(a, w), 2n), Nat.add(Nat.mul(a, 2n), Nat.mul(w, 2n)), Nat.add(Nat.mul(tn, 128n), Nat.mul(HW.wn(tid), 64n)),
    Ar.ar_mul_add_l(a, w, 2n),
    Ar.p8_add_cong(Nat.mul(a, 2n), Nat.mul(tn, 128n), Nat.mul(w, 2n), Nat.mul(HW.wn(tid), 64n), Ar.ar_mul_assoc(tn, 64n, 2n), Ar.ar_mul_assoc(HW.wn(tid), 32n, 2n)))
  +d2 = Equal.trans(Nat, Nat.mul(Nat.add(Nat.add(a, w), b), 2n), Nat.add(Nat.mul(Nat.add(a, w), 2n), Nat.mul(b, 2n)),
    Nat.add(Nat.add(Nat.mul(tn, 128n), Nat.mul(HW.wn(tid), 64n)), Nat.mul(j, 8n)),
    Ar.ar_mul_add_l(Nat.add(a, w), b, 2n),
    Ar.p8_add_cong(Nat.mul(Nat.add(a, w), 2n), Nat.add(Nat.mul(tn, 128n), Nat.mul(HW.wn(tid), 64n)), Nat.mul(b, 2n), Nat.mul(j, 8n), d1, Ar.ar_mul_assoc(j, 4n, 2n)))
  Equal.trans(Nat, Nat.mul(Nat.add(Nat.add(Nat.add(a, w), b), t), 2n), Nat.add(Nat.mul(Nat.add(Nat.add(a, w), b), 2n), Nat.mul(t, 2n)),
    Nat.add(Nat.add(Nat.add(Nat.mul(tn, 128n), Nat.mul(HW.wn(tid), 64n)), Nat.mul(j, 8n)), Nat.mul(t, 2n)),
    Ar.ar_mul_add_l(Nat.add(Nat.add(a, w), b), t, 2n),
    Ar.p8_add_cong(Nat.mul(Nat.add(Nat.add(a, w), b), 2n), Nat.add(Nat.add(Nat.mul(tn, 128n), Nat.mul(HW.wn(tid), 64n)), Nat.mul(j, 8n)),
      Nat.mul(t, 2n), Nat.mul(t, 2n), d2, {==}))

def Laws.epi_col_pair(tn, tid, j):
  +X = Nat.add(Nat.add(Nat.add(Nat.mul(tn, 128n), Nat.mul(HW.wn(tid), 64n)), Nat.mul(j, 8n)), Nat.mul(HW.tq(tid), 2n))
  +P2 = Nat.mul(Impl.pair_col(tn, tid, j), 2n)
  +e = af_col2(tn, tid, j)
  (Equal.trans(Nat, Nat.add(X, 0n), X, P2, Ar.ar_add_zero_r(X), Equal.sym(Nat, P2, X, e)),
   Equal.trans(Nat, Nat.add(X, 1n), 1n+Nat.add(X, 0n), 1n+P2, Ar.ar_add_succ_r(X, 0n),
     Equal.cong(Nat, Nat, z => 1n+z, Nat.add(X, 0n), P2, Equal.trans(Nat, Nat.add(X, 0n), X, P2, Ar.ar_add_zero_r(X), Equal.sym(Nat, P2, X, e)))))

# ---------------------------------------------------------------------------------------------
# 2. Pair index

# row (2 n) + 2 c == (row n + c) 2
def af_off(+n: Nat, +row: Nat, +c: Nat)
  -> {Impl.epi_off(Nat.mul(n, 2n), row, Nat.mul(c, 2n)) == Nat.mul(Nat.add(Nat.mul(row, n), c), 2n) : Nat}:
  Equal.trans(Nat, Nat.add(Nat.mul(row, Nat.mul(n, 2n)), Nat.mul(c, 2n)), Nat.add(Nat.mul(Nat.mul(row, n), 2n), Nat.mul(c, 2n)),
    Nat.mul(Nat.add(Nat.mul(row, n), c), 2n),
    Equal.cong(Nat, Nat, z => Nat.add(z, Nat.mul(c, 2n)), Nat.mul(row, Nat.mul(n, 2n)), Nat.mul(Nat.mul(row, n), 2n),
      Equal.sym(Nat, Nat.mul(Nat.mul(row, n), 2n), Nat.mul(row, Nat.mul(n, 2n)), Ar.ar_mul_assoc(row, n, 2n))),
    Equal.sym(Nat, Nat.mul(Nat.add(Nat.mul(row, n), c), 2n), Nat.add(Nat.mul(Nat.mul(row, n), 2n), Nat.mul(c, 2n)), Ar.ar_mul_add_l(Nat.mul(row, n), c, 2n)))

def Laws.pair_index(n, row, c, cn):
  +cl = cn
  (af_off(n, row, c), Ar.p7_div_q(row, c, n, cl), Ar.p7_mod_r(row, c, n, cl))

def Laws.pair_onto(n, i, npos):
  +np = npos
  (Ar.ard_dm_eq(i, n, np), Ar.ard_dm_lt(i, n, np))

# ---------------------------------------------------------------------------------------------
# 3. Value terms

def af_pick(limit: Bool, +x: Spec.H2, +y: Spec.H2) -> {Impl.epi.pick(limit, x, y) == Spec.act_mul.pick(limit, x, y) : Spec.H2}:
  match limit:
    case True{}:
      {==}
    case False{}:
      {==}

def Laws.value_eq(limit, n, row, c, cn):
  +lm = limit
  +N = Nat.mul(n, 2n)
  +C = Nat.mul(c, 2n)
  +i = Nat.add(Nat.mul(row, n), c)
  +O = Nat.add(Nat.mul(row, N), C)
  +G = {Spec.GateAt{O} : Spec.H2}
  +small = Ar.ard_tr(Nat.is_lt(C, N), Nat.is_lt(c, n), Ar.ar_mul_lt_cancel(c, n, 2n, {==}), cn)
  +dq = Ar.p7_div_q(row, C, N, small)
  +mr = Ar.p7_mod_r(row, C, N, small)
  +s1 = af_pick(lm, Impl.silu_h2(G), Spec.Rn2{Spec.AccPair{row, C}})
  +s2a = Equal.cong(Nat, Spec.H2, z => Spec.act_mul.pick(lm, Spec.silu(G), Spec.Rn2{Spec.AccPair{z, C}}), row, Nat.div(O, N),
    Equal.sym(Nat, Nat.div(O, N), row, dq))
  +s2b = Equal.cong(Nat, Spec.H2, z => Spec.act_mul.pick(lm, Spec.silu(G), Spec.Rn2{Spec.AccPair{Nat.div(O, N), z}}), C, Nat.mod(O, N),
    Equal.sym(Nat, Nat.mod(O, N), C, mr))
  +s3 = Equal.cong(Nat, Spec.H2, z => Spec.act_mul.pick(lm, Spec.silu(Spec.GateAt{z}), Spec.Rn2{Spec.AccPair{Nat.div(z, N), Nat.mod(z, N)}}),
    O, Nat.mul(i, 2n), af_off(n, row, c))
  Equal.trans(Spec.H2, Impl.epi(lm, N, row, C), Spec.act_mul.pick(lm, Spec.silu(G), Spec.Rn2{Spec.AccPair{row, C}}), Spec.act_mul(lm, N, i), s1,
    Equal.trans(Spec.H2, Spec.act_mul.pick(lm, Spec.silu(G), Spec.Rn2{Spec.AccPair{row, C}}),
      Spec.act_mul.pick(lm, Spec.silu(G), Spec.Rn2{Spec.AccPair{Nat.div(O, N), Nat.mod(O, N)}}), Spec.act_mul(lm, N, i),
      Equal.trans(Spec.H2, Spec.act_mul.pick(lm, Spec.silu(G), Spec.Rn2{Spec.AccPair{row, C}}),
        Spec.act_mul.pick(lm, Spec.silu(G), Spec.Rn2{Spec.AccPair{Nat.div(O, N), C}}),
        Spec.act_mul.pick(lm, Spec.silu(G), Spec.Rn2{Spec.AccPair{Nat.div(O, N), Nat.mod(O, N)}}), s2a, s2b),
      s3))

# ---------------------------------------------------------------------------------------------
# 4. Stock launch

def Laws.stock_thread(blk, tid, tb):
  +tl = tb
  (Ar.p7_div_q(blk, tid, 256n, tl), Ar.p7_mod_r(blk, tid, 256n, tl))

def Laws.stock_cover(i, numel, live):
  +d = Nat.div(i, 256n)
  +x = Nat.mul(i, 2n)
  +h1 = Ar.ard_rt(Nat.is_lt(i, Nat.div(numel, 2n)), Nat.is_le(1n+i, Nat.div(numel, 2n)), Ar.ar_lt_succ_le(i, Nat.div(numel, 2n)), live)
  +h2 = Ar.ar_le_trans(Nat.mul(1n+i, 2n), Nat.mul(Nat.div(numel, 2n), 2n), numel, Ar.ar_le_mul_mono(1n+i, Nat.div(numel, 2n), 2n, h1), Ar.ard_le_mul_div(numel, 2n, {==}))
  +xlt = Ar.ard_tr(Nat.is_lt(x, numel), Nat.is_le(1n+x, numel), Ar.ar_lt_succ_le(x, numel),
    Ar.ar_le_trans(1n+x, 1n+1n+x, numel, Ar.ar_h_le_succ(1n+x), h2))
  +le1 = Ar.ar_le_mul_mono(Nat.mul(d, 256n), i, 2n, Ar.ard_le_mul_div(i, 256n, {==}))
  +le2 = Equal.trans(Bool, Nat.is_le(Nat.mul(d, 512n), x), Nat.is_le(Nat.mul(Nat.mul(d, 256n), 2n), x), True{},
    Equal.cong(Nat, Bool, z => Nat.is_le(z, x), Nat.mul(d, 512n), Nat.mul(Nat.mul(d, 256n), 2n),
      Equal.sym(Nat, Nat.mul(Nat.mul(d, 256n), 2n), Nat.mul(d, 512n), Ar.ar_mul_assoc(d, 256n, 2n))), le1)
  (Ar.ard_dm_eq(i, 256n, {==}),
   SP.sp_lt_cdiv(d, numel, 512n, {==}, Ar.ar_le_lt_trans(Nat.mul(d, 512n), x, numel, le2, xlt)))

# ---------------------------------------------------------------------------------------------
# 5. Route

def af_bool(b: Bool, h: {Impl.is_fused(Impl.route.bool(b)) == True{} : Bool}) -> {b == True{} : Bool}:
  match b:
    case True{}:
      {==}
    case False{}:
      Empty.absurd({False{} == True{} : Bool}, Ar.ar_false_true(h))

def AfShape(dev: Bool, mfit: Bool, status: Bool, +M: Nat, +K: Nat, +N: Nat) -> Type:
  {Bool.and(dev, Bool.and(mfit, status)) == True{} : Bool} &
  {Nat.is_eq(K, 5120n) == True{} : Bool} &
  {Nat.is_eq(N, 17408n) == True{} : Bool} &
  {Nat.is_lt(M, 1024n) == False{} : Bool}

def af_shape(dev: Bool, mfit: Bool, status: Bool, +M: Nat, +K: Nat, +N: Nat,
    h: {Impl.is_fused(Impl.route.shape(dev, M, K, N, mfit, status)) == True{} : Bool})
  -> AfShape(dev, mfit, status, M, K, N):
  match dev mfit status:
    case False{} _ _:
      Empty.absurd(AfShape(False{}, mfit, status, M, K, N), Ar.ar_false_true(h))
    case True{} False{} _:
      Empty.absurd(AfShape(True{}, False{}, status, M, K, N), Ar.ar_false_true(h))
    case True{} True{} False{}:
      Empty.absurd(AfShape(True{}, True{}, False{}, M, K, N), Ar.ar_false_true(h))
    case True{} True{} True{}:
      +r1 = Bool.and(Nat.is_eq(N, 17408n), Bool.not(Nat.is_lt(M, 1024n)))
      +x = af_bool(Bool.and(Nat.is_eq(K, 5120n), r1), h)
      +y = Ar.ar_and_r(Nat.is_eq(K, 5120n), r1, x)
      ({==},
       Ar.ar_and_l(Nat.is_eq(K, 5120n), r1, x),
       Ar.ar_and_l(Nat.is_eq(N, 17408n), Bool.not(Nat.is_lt(M, 1024n)), y),
       Ar.p1l_not_true(Nat.is_lt(M, 1024n), Ar.ar_and_r(Nat.is_eq(N, 17408n), Bool.not(Nat.is_lt(M, 1024n)), y)))

def AfEnv(mlp: Impl.Sw, wide: Impl.Sw, dev: Bool, mfit: Bool, status: Bool, +M: Nat, +K: Nat, +N: Nat) -> Type:
  {Impl.sw_on(mlp) == Impl.Fused{} : Impl.Route} & {Impl.sw_on(wide) == Impl.Fused{} : Impl.Route} & AfShape(dev, mfit, status, M, K, N)

def af_env(mlp: Impl.Sw, wide: Impl.Sw, dev: Bool, mfit: Bool, status: Bool, +M: Nat, +K: Nat, +N: Nat,
    h: {Impl.is_fused(Impl.route(mlp, wide, dev, M, K, N, mfit, status)) == True{} : Bool})
  -> AfEnv(mlp, wide, dev, mfit, status, M, K, N):
  match mlp wide:
    case Impl.SwUnset{} Impl.SwUnset{}:
      ({==}, {==}, af_shape(dev, mfit, status, M, K, N, h))
    case Impl.SwUnset{} Impl.SwOne{}:
      ({==}, {==}, af_shape(dev, mfit, status, M, K, N, h))
    case Impl.SwOne{} Impl.SwUnset{}:
      ({==}, {==}, af_shape(dev, mfit, status, M, K, N, h))
    case Impl.SwOne{} Impl.SwOne{}:
      ({==}, {==}, af_shape(dev, mfit, status, M, K, N, h))
    case Impl.SwUnset{} Impl.SwZero{}:
      Empty.absurd(AfEnv(Impl.SwUnset{}, Impl.SwZero{}, dev, mfit, status, M, K, N), Ar.ar_false_true(h))
    case Impl.SwUnset{} Impl.SwOther{}:
      Empty.absurd(AfEnv(Impl.SwUnset{}, Impl.SwOther{}, dev, mfit, status, M, K, N), Ar.ar_false_true(h))
    case Impl.SwOne{} Impl.SwZero{}:
      Empty.absurd(AfEnv(Impl.SwOne{}, Impl.SwZero{}, dev, mfit, status, M, K, N), Ar.ar_false_true(h))
    case Impl.SwOne{} Impl.SwOther{}:
      Empty.absurd(AfEnv(Impl.SwOne{}, Impl.SwOther{}, dev, mfit, status, M, K, N), Ar.ar_false_true(h))
    case Impl.SwZero{} _:
      Empty.absurd(AfEnv(Impl.SwZero{}, wide, dev, mfit, status, M, K, N), Ar.ar_false_true(h))
    case Impl.SwOther{} _:
      Empty.absurd(AfEnv(Impl.SwOther{}, wide, dev, mfit, status, M, K, N), Ar.ar_false_true(h))

def Laws.fused_scope(mlp, wide, dev, M, K, N, mfit, status, fused):
  af_env(mlp, wide, dev, mfit, status, M, K, N, fused)

def Laws.stock_wide_route(M, long):
  +lg = long
  (%Equal.sym(Bool, U32.is_ge(M, 1024), True{}, lg) : {HW.route_on(True{}, True{}, _, True{}) == HWS.HwWide{} : HWS.HwRoute}
   {==},
   %Equal.sym(Bool, U32.is_ge(M, 1024), True{}, lg) : {HW.route_on(True{}, True{}, _, True{}) == HWS.HwWide{} : HWS.HwRoute}
   {==})

def Laws.stock_worthwhile(M, nsm, long, sms):
  +ge = Ar.p6_nlt_le(M, 1024n, long)
  +l384 = Ar.p6_le_nlt(M, 384n, Ar.ar_le_trans(384n, 1024n, M, {==}, ge))
  +a = Nat.sub(Nat.add(M, 128n), 1n)
  +cd = Nat.div(a, 128n)
  +ha = Equal.trans(Bool, Nat.is_le(1024n, a), Nat.is_le(1024n, Nat.add(M, 127n)), True{},
    Equal.cong(Nat, Bool, z => Nat.is_le(1024n, z), a, Nat.add(M, 127n), SP.sp_pred(M, 127n)),
    Ar.ar_le_trans(1024n, M, Nat.add(M, 127n), ge, Ar.ar_le_add_r(M, 127n)))
  +cd8 = Ar.ard_rt(Nat.is_le(Nat.mul(8n, 128n), a), Nat.is_le(8n, cd), Ar.ar_div_le(8n, a, 128n, {==}), ha)
  +le8 = Ar.ar_le_mul_mono(8n, cd, 136n, cd8)
  %Equal.sym(Bool, Nat.is_lt(M, 384n), False{}, l384)
    : {Bool.and(Bool.not(_), Nat.is_le(nsm, Nat.mul(cd, Nat.div(17408n, 128n)))) == True{} : Bool}
  Ar.ar_le_trans(nsm, 1088n, Nat.mul(cd, 136n), sms, le8)
