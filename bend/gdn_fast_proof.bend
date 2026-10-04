# Proofs of the 5101 fused GDN verify mapping laws (gdn_fast_laws.bend) about the Bend model
# gdn_fast.bend. Structural evidence only: no unsafe definitions, axioms, holes or foreign proof
# code. The fixed-size thread maps (512 = 4 * 128 threads, 4 replay blocks of 128) are decided by
# the checker evaluating a Bool check over the whole finite domain (gf_all), then instantiated per
# element (gf_all_get); every other law is proven for all parameters by induction, reusing the
# m16 finisher law finish_jobs_once for the strided loops and the Nat library of
# gemm_m16_partition_proof.bend. Every def precedes its uses.
import Base
import ./gdn_fast.bend as Impl
import ./gdn_fast_spec.bend as Spec
import ./gdn_fast_laws.bend as Laws
import ./gdn_replay.bend as GR
import ./gemm_m16_partition.bend as M16
import ./gemm_m16_partition_spec.bend as MS
import ./gemm_m16_partition_laws.bend as ML
import ./gemm_m16_partition_proof.bend as Ar

# ==== finite-domain reflection ====

# f holds on every i < n.
def gf_all(~f: Nat -> Bool, +n: Nat) -> Bool:
  match n:
    case 0n:
      True{}
    case 1n+p:
      Bool.and(f(p), gf_all(~f, p))

# t < 1 + p and t != p give t < p.
def gf_lt_ne(+t: Nat, +p: Nat, tr: {Nat.is_lt(t, 1n+p) == True{} : Bool},
    ne: {Nat.is_eq(t, p) == False{} : Bool}) -> {Nat.is_lt(t, p) == True{} : Bool}:
  match t p:
    case 0n 0n:
      Empty.absurd({Nat.is_lt(0n, 0n) == True{} : Bool}, Ar.ar_true_false(ne))
    case 0n 1n+q:
      {==}
    case 1n+u 0n:
      Empty.absurd({Nat.is_lt(1n+u, 0n) == True{} : Bool}, Ar.ar_false_true(Equal.trans(Bool, False{}, Nat.is_lt(u, 0n), True{},
        Equal.sym(Bool, Nat.is_lt(u, 0n), False{}, Ar.ar_h_lt_zero(u)), tr)))
    case 1n+u 1n+q:
      gf_lt_ne(u, q, tr, ne)

def gf_pick_eq(~f: Nat -> Bool, +t: Nat, +p: Nat, e: Bool, ee: {Nat.is_eq(t, p) == e : Bool},
    tr: {Nat.is_lt(t, 1n+p) == True{} : Bool},
    h: {Bool.and(f(p), gf_all(~f, p)) == True{} : Bool},
    rec: {Nat.is_lt(t, p) == True{} : Bool} -> {f(t) == True{} : Bool})
  -> {f(t) == True{} : Bool}:
  match e:
    case True{}:
      +tp = Ar.ar_eq_true(t, p, ee)
      %Equal.sym(Nat, t, p, tp) : {f(_) == True{} : Bool}
      Ar.ar_and_l(f(p), gf_all(~f, p), h)
    case False{}:
      rec(gf_lt_ne(t, p, tr, ee))

def gf_all_get(~f: Nat -> Bool, +n: Nat, +t: Nat, h: {gf_all(~f, n) == True{} : Bool},
    tr: {Nat.is_lt(t, n) == True{} : Bool}) -> {f(t) == True{} : Bool}:
  match n:
    case 0n:
      Empty.absurd({f(t) == True{} : Bool}, Ar.ar_false_true(Equal.trans(Bool, False{}, Nat.is_lt(t, 0n), True{},
        Equal.sym(Bool, Nat.is_lt(t, 0n), False{}, Ar.ar_h_lt_zero(t)), tr)))
    case 1n+p:
      +hh = h
      gf_pick_eq(~f, t, p, Nat.is_eq(t, p), {==}, tr, hh,
        r => gf_all_get(~f, p, t, Ar.ar_and_r(f(p), gf_all(~f, p), hh), r))

# f holds on every a + i, i < n (the window [a, a + n)).
def gf_rng(~f: Nat -> Bool, +a: Nat, +n: Nat) -> Bool:
  match n:
    case 0n:
      True{}
    case 1n+p:
      Bool.and(f(Nat.add(p, a)), gf_rng(~f, a, p))

# f on every i < a and on the window [a, a + b) gives f on every i < b + a. The checker's
# evaluation cost grows steeply with the size of the domain and its elements, so a large
# finite check is decided window by window (each window in its own def) and glued here.
def gf_glue(~f: Nat -> Bool, +b: Nat, +a: Nat, r: {gf_rng(~f, a, b) == True{} : Bool},
    h: {gf_all(~f, a) == True{} : Bool}) -> {gf_all(~f, Nat.add(b, a)) == True{} : Bool}:
  match b:
    case 0n:
      h
    case 1n+p:
      +rr = r
      +x = Ar.ar_and_l(f(Nat.add(p, a)), gf_rng(~f, a, p), rr)
      +y = gf_glue(~f, p, a, Ar.ar_and_r(f(Nat.add(p, a)), gf_rng(~f, a, p), rr), h)
      %Equal.sym(Bool, f(Nat.add(p, a)), True{}, x) :
        {Bool.and(_, gf_all(~f, Nat.add(p, a))) == True{} : Bool}
      %Equal.sym(Bool, gf_all(~f, Nat.add(p, a)), True{}, y) : {Bool.and(True{}, _) == True{} : Bool}
      {==}

# gf_glue at a literal size n == b + a: callers state the window sizes as literals, so the checker
# compares the glued check with the next window's syntactically instead of re-evaluating it.
def gf_glue_at(~f: Nat -> Bool, +b: Nat, +a: Nat, +n: Nat, eq: {Nat.add(b, a) == n : Nat},
    r: {gf_rng(~f, a, b) == True{} : Bool}, h: {gf_all(~f, a) == True{} : Bool})
  -> {gf_all(~f, n) == True{} : Bool}:
  %eq : {gf_all(~f, _) == True{} : Bool}
  gf_glue(~f, b, a, r, h)

# ==== (1) thread maps ====

def gf_ov(+t: Nat) -> Bool:
  Bool.and(Nat.is_lt(Impl.kslice(t), 4n), Bool.and(Nat.is_lt(Impl.vcol(0n, t), 128n),
    Nat.is_eq(Spec.verify_thread(Impl.kslice(t), Impl.vcol(0n, t)), t)))

def gf_ov_w(x: Nat) -> Bool:
  gf_ov(x)

# The 512-thread check, decided in 16 windows of 32 threads (one def each).
def gf_ov_w1() -> {gf_rng(~gf_ov_w, 32n, 32n) == True{} : Bool}:
  {==}

def gf_ov_w2() -> {gf_rng(~gf_ov_w, 64n, 32n) == True{} : Bool}:
  {==}

def gf_ov_w3() -> {gf_rng(~gf_ov_w, 96n, 32n) == True{} : Bool}:
  {==}

def gf_ov_w4() -> {gf_rng(~gf_ov_w, 128n, 32n) == True{} : Bool}:
  {==}

def gf_ov_w5() -> {gf_rng(~gf_ov_w, 160n, 32n) == True{} : Bool}:
  {==}

def gf_ov_w6() -> {gf_rng(~gf_ov_w, 192n, 32n) == True{} : Bool}:
  {==}

def gf_ov_w7() -> {gf_rng(~gf_ov_w, 224n, 32n) == True{} : Bool}:
  {==}

def gf_ov_w8() -> {gf_rng(~gf_ov_w, 256n, 32n) == True{} : Bool}:
  {==}

def gf_ov_w9() -> {gf_rng(~gf_ov_w, 288n, 32n) == True{} : Bool}:
  {==}

def gf_ov_w10() -> {gf_rng(~gf_ov_w, 320n, 32n) == True{} : Bool}:
  {==}

def gf_ov_w11() -> {gf_rng(~gf_ov_w, 352n, 32n) == True{} : Bool}:
  {==}

def gf_ov_w12() -> {gf_rng(~gf_ov_w, 384n, 32n) == True{} : Bool}:
  {==}

def gf_ov_w13() -> {gf_rng(~gf_ov_w, 416n, 32n) == True{} : Bool}:
  {==}

def gf_ov_w14() -> {gf_rng(~gf_ov_w, 448n, 32n) == True{} : Bool}:
  {==}

def gf_ov_w15() -> {gf_rng(~gf_ov_w, 480n, 32n) == True{} : Bool}:
  {==}

def gf_ov_c1() -> {gf_all(~gf_ov_w, 32n) == True{} : Bool}:
  {==}

def gf_ov_c2() -> {gf_all(~gf_ov_w, 64n) == True{} : Bool}:
  gf_glue_at(~gf_ov_w, 32n, 32n, 64n, {==}, gf_ov_w1(), gf_ov_c1())

def gf_ov_c3() -> {gf_all(~gf_ov_w, 96n) == True{} : Bool}:
  gf_glue_at(~gf_ov_w, 32n, 64n, 96n, {==}, gf_ov_w2(), gf_ov_c2())

def gf_ov_c4() -> {gf_all(~gf_ov_w, 128n) == True{} : Bool}:
  gf_glue_at(~gf_ov_w, 32n, 96n, 128n, {==}, gf_ov_w3(), gf_ov_c3())

def gf_ov_c5() -> {gf_all(~gf_ov_w, 160n) == True{} : Bool}:
  gf_glue_at(~gf_ov_w, 32n, 128n, 160n, {==}, gf_ov_w4(), gf_ov_c4())

def gf_ov_c6() -> {gf_all(~gf_ov_w, 192n) == True{} : Bool}:
  gf_glue_at(~gf_ov_w, 32n, 160n, 192n, {==}, gf_ov_w5(), gf_ov_c5())

def gf_ov_c7() -> {gf_all(~gf_ov_w, 224n) == True{} : Bool}:
  gf_glue_at(~gf_ov_w, 32n, 192n, 224n, {==}, gf_ov_w6(), gf_ov_c6())

def gf_ov_c8() -> {gf_all(~gf_ov_w, 256n) == True{} : Bool}:
  gf_glue_at(~gf_ov_w, 32n, 224n, 256n, {==}, gf_ov_w7(), gf_ov_c7())

def gf_ov_c9() -> {gf_all(~gf_ov_w, 288n) == True{} : Bool}:
  gf_glue_at(~gf_ov_w, 32n, 256n, 288n, {==}, gf_ov_w8(), gf_ov_c8())

def gf_ov_c10() -> {gf_all(~gf_ov_w, 320n) == True{} : Bool}:
  gf_glue_at(~gf_ov_w, 32n, 288n, 320n, {==}, gf_ov_w9(), gf_ov_c9())

def gf_ov_c11() -> {gf_all(~gf_ov_w, 352n) == True{} : Bool}:
  gf_glue_at(~gf_ov_w, 32n, 320n, 352n, {==}, gf_ov_w10(), gf_ov_c10())

def gf_ov_c12() -> {gf_all(~gf_ov_w, 384n) == True{} : Bool}:
  gf_glue_at(~gf_ov_w, 32n, 352n, 384n, {==}, gf_ov_w11(), gf_ov_c11())

def gf_ov_c13() -> {gf_all(~gf_ov_w, 416n) == True{} : Bool}:
  gf_glue_at(~gf_ov_w, 32n, 384n, 416n, {==}, gf_ov_w12(), gf_ov_c12())

def gf_ov_c14() -> {gf_all(~gf_ov_w, 448n) == True{} : Bool}:
  gf_glue_at(~gf_ov_w, 32n, 416n, 448n, {==}, gf_ov_w13(), gf_ov_c13())

def gf_ov_c15() -> {gf_all(~gf_ov_w, 480n) == True{} : Bool}:
  gf_glue_at(~gf_ov_w, 32n, 448n, 480n, {==}, gf_ov_w14(), gf_ov_c14())

def gf_ov_c16() -> {gf_all(~gf_ov_w, 512n) == True{} : Bool}:
  gf_glue_at(~gf_ov_w, 32n, 480n, 512n, {==}, gf_ov_w15(), gf_ov_c15())

def gf_ov_all() -> {gf_all(~gf_ov_w, 512n) == True{} : Bool}:
  gf_ov_c16()

def Laws.own_verify_into(t, tr):
  gf_all_get(~gf_ov_w, 512n, t, gf_ov_all(), tr)

# ==== (2), (4) strided loops: a flat group of 8 G threads is the m16 finisher grid of G blocks ====

# Threads < k + 8 b of the stride-8G loop == finisher blocks < b plus warps < k of block b.
def gf_warps(+k: Nat, +b: Nat, +x: Nat, +G: Nat, +J: Nat,
    base: {Impl.stride_hits(Nat.mul(8n, b), x, Nat.mul(8n, G), J) == M16.job_hits(b, x, G, J) : Nat})
  -> {Impl.stride_hits(Nat.add(k, Nat.mul(8n, b)), x, Nat.mul(8n, G), J) ==
    Nat.add(M16.job_hits(b, x, G, J), M16.warp_hits(k, b, x, G, J)) : Nat}:
  match k:
    case 0n:
      +jb = M16.job_hits(b, x, G, J)
      Equal.trans(Nat, Impl.stride_hits(Nat.mul(8n, b), x, Nat.mul(8n, G), J), jb, Nat.add(jb, 0n),
        base, Equal.sym(Nat, Nat.add(jb, 0n), jb, Ar.ar_add_zero_r(jb)))
    case 1n+j:
      +j1 = j
      +a = M16.job_hits(b, x, G, J)
      +w = M16.warp_hits(j1, b, x, G, J)
      +h = MS.bit(MS.member(M16.jobloop(b, j1, G, J), x))
      Equal.trans(Nat, Nat.add(Impl.stride_hits(Nat.add(j1, Nat.mul(8n, b)), x, Nat.mul(8n, G), J), h),
        Nat.add(Nat.add(a, w), h), Nat.add(a, Nat.add(w, h)),
        Ar.p8_add_cong(Impl.stride_hits(Nat.add(j1, Nat.mul(8n, b)), x, Nat.mul(8n, G), J), Nat.add(a, w), h, h,
          gf_warps(j1, b, x, G, J, base), {==}),
        Ar.ar_add_assoc(a, w, h))

def gf_blocks(+b: Nat, +x: Nat, +G: Nat, +J: Nat)
  -> {Impl.stride_hits(Nat.mul(8n, b), x, Nat.mul(8n, G), J) == M16.job_hits(b, x, G, J) : Nat}:
  match b:
    case 0n:
      {==}
    case 1n+p:
      +p1 = p
      %Equal.sym(Nat, Nat.mul(8n, 1n+p1), Nat.add(8n, Nat.mul(8n, p1)), Ar.ar_mul_succ_r(8n, p1)) :
        {Impl.stride_hits(_, x, Nat.mul(8n, G), J) == M16.job_hits(1n+p1, x, G, J) : Nat}
      gf_warps(8n, p1, x, G, J, gf_blocks(p1, x, G, J))

# 8 G threads, stride 8 G, bound J: every x < J visited exactly once, none past the end.
def gf_stride(+G: Nat, +x: Nat, +J: Nat, grid: {Nat.is_lt(0n, G) == True{} : Bool})
  -> {Impl.stride_hits(Nat.mul(8n, G), x, Nat.mul(8n, G), J) == MS.bit(Nat.is_lt(x, J)) : Nat}:
  Equal.trans(Nat, Impl.stride_hits(Nat.mul(8n, G), x, Nat.mul(8n, G), J), M16.job_hits(G, x, G, J),
    MS.bit(Nat.is_lt(x, J)), gf_blocks(G, x, G, J), ML.finish_jobs_once(x, G, J, grid))

def Laws.red_exactly_once(x, S):
  gf_stride(64n, x, Nat.mul(S, 128n), {==})

def Laws.task_exactly_once(x, S):
  gf_stride(2n, x, Nat.mul(2n, S), {==})

# a - (a / b) * b == a % b
def gf_sub_mod(+a: Nat, +b: Nat, pos: {Nat.is_lt(0n, b) == True{} : Bool})
  -> {Nat.sub(a, Nat.mul(Nat.div(a, b), b)) == Nat.mod(a, b) : Nat}:
  %Ar.ard_dm_eq(a, b, pos) : {Nat.sub(_, Nat.mul(Nat.div(a, b), b)) == Nat.mod(a, b) : Nat}
  Ar.ar_add_sub(Nat.mul(Nat.div(a, b), b), Nat.mod(a, b))

def gf_fst(-A: Type, -B: Type, p: A & B) -> A:
  (x, y) = p
  x

def gf_snd(-A: Type, -B: Type, p: A & B) -> B:
  (x, y) = p
  y

def Laws.red_decode(s, c, cr):
  +c1 = cr
  +q = gf_fst({Nat.div(Nat.add(Nat.mul(s, 128n), c), 128n) == s : Nat},
    {Nat.mod(Nat.add(Nat.mul(s, 128n), c), 128n) == c : Nat}, Ar.ar_div_unique(s, c, 128n, c1))
  +e = Nat.add(Nat.mul(s, 128n), c)
  (q,
   Equal.trans(Nat, Nat.sub(e, Nat.mul(Nat.div(e, 128n), 128n)), Nat.sub(e, Nat.mul(s, 128n)), c,
     Equal.cong(Nat, Nat, z => Nat.sub(e, Nat.mul(z, 128n)), Nat.div(e, 128n), s, q),
     Ar.ar_add_sub(Nat.mul(s, 128n), c)))

def gf_mod_lt(+e: Nat, +b: Nat, +pos: {Nat.is_lt(0n, b) == True{} : Bool})
  -> {Nat.is_lt(Nat.sub(e, Nat.mul(Nat.div(e, b), b)), b) == True{} : Bool}:
  %Equal.sym(Nat, Nat.sub(e, Nat.mul(Nat.div(e, b), b)), Nat.mod(e, b), gf_sub_mod(e, b, pos)) : {Nat.is_lt(_, b) == True{} : Bool}
  Ar.ard_dm_lt(e, b, pos)

def gf_mod_eq(+e: Nat, +b: Nat, +pos: {Nat.is_lt(0n, b) == True{} : Bool})
  -> {Nat.add(Nat.mul(Nat.div(e, b), b), Nat.sub(e, Nat.mul(Nat.div(e, b), b))) == e : Nat}:
  %Equal.sym(Nat, Nat.sub(e, Nat.mul(Nat.div(e, b), b)), Nat.mod(e, b), gf_sub_mod(e, b, pos)) : {Nat.add(Nat.mul(Nat.div(e, b), b), _) == e : Nat}
  Ar.ard_dm_eq(e, b, pos)

def Laws.red_decode_range(e, S, er):
  (Ar.ard_tr(Nat.is_lt(Nat.div(e, 128n), S), Nat.is_lt(e, Nat.mul(S, 128n)), Ar.ar_div_lt(S, e, 128n, {==}), er),
   gf_mod_lt(e, 128n, {==}),
   gf_mod_eq(e, 128n, {==}))

def Laws.task_decode(s, ab, abr):
  Ar.ar_div_unique(s, ab, 2n, abr)

def Laws.task_decode_range(x, S, xr):
  +x2 = Ar.ard_tr(Nat.is_lt(x, Nat.mul(S, 2n)), Nat.is_lt(x, Nat.mul(2n, S)),
    Equal.cong(Nat, Bool, z => Nat.is_lt(x, z), Nat.mul(S, 2n), Nat.mul(2n, S), Ar.ar_mul_comm(S, 2n)), xr)
  (Ar.ard_tr(Nat.is_lt(Nat.div(x, 2n), S), Nat.is_lt(x, Nat.mul(S, 2n)), Ar.ar_div_lt(S, x, 2n, {==}), x2),
   Ar.ard_dm_lt(x, 2n, {==}),
   Ar.ard_dm_eq(x, 2n, {==}))

# ==== (3) norm rows ====

def gf_bit_step(+j: Nat, +k: Nat)
  -> {Nat.add(MS.bit(Nat.is_lt(k, j)), MS.bit(Nat.is_eq(j, k))) == MS.bit(Nat.is_lt(k, 1n+j)) : Nat}:
  match j k:
    case 0n 0n:
      {==}
    case 0n 1n+kp:
      match kp:
        case 0n:
          {==}
        case 1n+kq:
          {==}
    case 1n+jp 0n:
      {==}
    case 1n+jp 1n+kp:
      gf_bit_step(jp, kp)

def gf_bit_step_s(+s: Bool, +j: Nat, +k: Nat)
  -> {Nat.add(MS.bit(Bool.and(s, Nat.is_lt(k, j))), MS.bit(Bool.and(s, Nat.is_eq(j, k)))) ==
    MS.bit(Bool.and(s, Nat.is_lt(k, 1n+j))) : Nat}:
  match s:
    case False{}:
      {==}
    case True{}:
      gf_bit_step(j, k)

# w < S && w == s  ==  s < S && w == s
def gf_norm_term(+w: Nat, +s: Nat, +S: Nat, e: Bool, ee: {Nat.is_eq(w, s) == e : Bool})
  -> {MS.bit(Bool.and(Nat.is_lt(w, S), Nat.is_eq(w, s))) == MS.bit(Bool.and(Nat.is_lt(s, S), Nat.is_eq(w, s))) : Nat}:
  match e:
    case True{}:
      +ws = Ar.ar_eq_true(w, s, ee)
      %ws : {MS.bit(Bool.and(Nat.is_lt(w, S), Nat.is_eq(w, s))) == MS.bit(Bool.and(Nat.is_lt(_, S), Nat.is_eq(w, s))) : Nat}
      {==}
    case False{}:
      %Equal.sym(Bool, Nat.is_eq(w, s), False{}, ee) : {MS.bit(Bool.and(Nat.is_lt(w, S), _)) == MS.bit(Bool.and(Nat.is_lt(s, S), _)) : Nat}
      %Ar.ar_h_and_false_r(Nat.is_lt(w, S)) : {MS.bit(_) == MS.bit(Bool.and(Nat.is_lt(s, S), False{})) : Nat}
      %Ar.ar_h_and_false_r(Nat.is_lt(s, S)) : {MS.bit(False{}) == MS.bit(_) : Nat}
      {==}

def gf_norm_zero(+L: Bool, +s: Nat) -> {0n == MS.bit(Bool.and(L, Nat.is_lt(s, 0n))) : Nat}:
  %Equal.sym(Bool, Nat.is_lt(s, 0n), False{}, Ar.ar_h_lt_zero(s)) : {0n == MS.bit(Bool.and(L, _)) : Nat}
  %Ar.ar_h_and_false_r(L) : {0n == MS.bit(_) : Nat}
  {==}

def gf_norm(+n: Nat, +s: Nat, +S: Nat)
  -> {Impl.norm_hits(n, s, S) == MS.bit(Bool.and(Nat.is_lt(s, S), Nat.is_lt(s, n))) : Nat}:
  match n:
    case 0n:
      gf_norm_zero(Nat.is_lt(s, S), s)
    case 1n+w:
      +w1 = w
      +L = Nat.is_lt(s, S)
      Equal.trans(Nat, Nat.add(Impl.norm_hits(w1, s, S), MS.bit(Bool.and(Nat.is_lt(w1, S), Nat.is_eq(w1, s)))),
        Nat.add(MS.bit(Bool.and(L, Nat.is_lt(s, w1))), MS.bit(Bool.and(L, Nat.is_eq(w1, s)))),
        MS.bit(Bool.and(L, Nat.is_lt(s, 1n+w1))),
        Ar.p8_add_cong(Impl.norm_hits(w1, s, S), MS.bit(Bool.and(L, Nat.is_lt(s, w1))),
          MS.bit(Bool.and(Nat.is_lt(w1, S), Nat.is_eq(w1, s))), MS.bit(Bool.and(L, Nat.is_eq(w1, s))),
          gf_norm(w1, s, S), gf_norm_term(w1, s, S, Nat.is_eq(w1, s), {==})),
        gf_bit_step_s(L, w1, s))

def gf_keep(+a: Bool, +b: Bool, h: {a == True{} : Bool} -> {b == True{} : Bool})
  -> {Bool.and(a, b) == a : Bool}:
  match a:
    case False{}:
      {==}
    case True{}:
      h({==})

def Laws.norm_exactly_once(s, S, sr):
  +s1 = sr
  Equal.trans(Nat, Impl.norm_hits(16n, s, S), MS.bit(Bool.and(Nat.is_lt(s, S), Nat.is_lt(s, 16n))),
    Spec.covered(s, S), gf_norm(16n, s, S),
    Equal.cong(Bool, Nat, z => MS.bit(z), Bool.and(Nat.is_lt(s, S), Nat.is_lt(s, 16n)), Nat.is_lt(s, S),
      gf_keep(Nat.is_lt(s, S), Nat.is_lt(s, 16n), h => Ar.ar_lt_le_trans(s, S, 16n, h, s1))))

# f2 holds on every (i / 128, i % 128), i < n.
def gf_all2(~f2: Nat -> Nat -> Bool, +n: Nat) -> Bool:
  match n:
    case 0n:
      True{}
    case 1n+p:
      Bool.and(f2(Nat.div(p, 128n), Nat.mod(p, 128n)), gf_all2(~f2, p))

def gf_pick2(~f2: Nat -> Nat -> Bool, +t: Nat, +p: Nat, e: Bool, ee: {Nat.is_eq(t, p) == e : Bool},
    tr: {Nat.is_lt(t, 1n+p) == True{} : Bool},
    h: {Bool.and(f2(Nat.div(p, 128n), Nat.mod(p, 128n)), gf_all2(~f2, p)) == True{} : Bool},
    rec: {Nat.is_lt(t, p) == True{} : Bool} -> {f2(Nat.div(t, 128n), Nat.mod(t, 128n)) == True{} : Bool})
  -> {f2(Nat.div(t, 128n), Nat.mod(t, 128n)) == True{} : Bool}:
  match e:
    case True{}:
      +tp = Ar.ar_eq_true(t, p, ee)
      %Equal.sym(Nat, t, p, tp) : {f2(Nat.div(_, 128n), Nat.mod(_, 128n)) == True{} : Bool}
      Ar.ar_and_l(f2(Nat.div(p, 128n), Nat.mod(p, 128n)), gf_all2(~f2, p), h)
    case False{}:
      rec(gf_lt_ne(t, p, tr, ee))

def gf_all2_get(~f2: Nat -> Nat -> Bool, +n: Nat, +t: Nat, h: {gf_all2(~f2, n) == True{} : Bool},
    tr: {Nat.is_lt(t, n) == True{} : Bool}) -> {f2(Nat.div(t, 128n), Nat.mod(t, 128n)) == True{} : Bool}:
  match n:
    case 0n:
      Empty.absurd({f2(Nat.div(t, 128n), Nat.mod(t, 128n)) == True{} : Bool}, Ar.ar_false_true(Equal.trans(Bool, False{}, Nat.is_lt(t, 0n), True{},
        Equal.sym(Bool, Nat.is_lt(t, 0n), False{}, Ar.ar_h_lt_zero(t)), tr)))
    case 1n+p:
      +hh = h
      gf_pick2(~f2, t, p, Nat.is_eq(t, p), {==}, tr, hh,
        r => gf_all2_get(~f2, p, t, Ar.ar_and_r(f2(Nat.div(p, 128n), Nat.mod(p, 128n)), gf_all2(~f2, p), hh), r))

# The gf_all window machinery (gf_rng, gf_glue, gf_glue_at) for gf_all2.
def gf_rng2(~f2: Nat -> Nat -> Bool, +a: Nat, +n: Nat) -> Bool:
  match n:
    case 0n:
      True{}
    case 1n+p:
      Bool.and(f2(Nat.div(Nat.add(p, a), 128n), Nat.mod(Nat.add(p, a), 128n)), gf_rng2(~f2, a, p))

def gf_glue2(~f2: Nat -> Nat -> Bool, +b: Nat, +a: Nat, r: {gf_rng2(~f2, a, b) == True{} : Bool},
    h: {gf_all2(~f2, a) == True{} : Bool}) -> {gf_all2(~f2, Nat.add(b, a)) == True{} : Bool}:
  match b:
    case 0n:
      h
    case 1n+p:
      +rr = r
      +x = Ar.ar_and_l(f2(Nat.div(Nat.add(p, a), 128n), Nat.mod(Nat.add(p, a), 128n)), gf_rng2(~f2, a, p), rr)
      +y = gf_glue2(~f2, p, a, Ar.ar_and_r(f2(Nat.div(Nat.add(p, a), 128n), Nat.mod(Nat.add(p, a), 128n)),
        gf_rng2(~f2, a, p), rr), h)
      %Equal.sym(Bool, f2(Nat.div(Nat.add(p, a), 128n), Nat.mod(Nat.add(p, a), 128n)), True{}, x) :
        {Bool.and(_, gf_all2(~f2, Nat.add(p, a))) == True{} : Bool}
      %Equal.sym(Bool, gf_all2(~f2, Nat.add(p, a)), True{}, y) : {Bool.and(True{}, _) == True{} : Bool}
      {==}

def gf_glue2_at(~f2: Nat -> Nat -> Bool, +b: Nat, +a: Nat, +n: Nat, eq: {Nat.add(b, a) == n : Nat},
    r: {gf_rng2(~f2, a, b) == True{} : Bool}, h: {gf_all2(~f2, a) == True{} : Bool})
  -> {gf_all2(~f2, n) == True{} : Bool}:
  %eq : {gf_all2(~f2, _) == True{} : Bool}
  gf_glue2(~f2, b, a, r, h)

# The whole check at the size term the laws use (m, e.g. 4 * 128, is not normalized: -m).
def gf_all2_lit(~f2: Nat -> Nat -> Bool, +n: Nat, -m: Nat, eq: {n == m : Nat},
    h: {gf_all2(~f2, n) == True{} : Bool}) -> {gf_all2(~f2, m) == True{} : Bool}:
  %eq : {gf_all2(~f2, _) == True{} : Bool}
  h

# j * w + i < R * w for j < R, i < w.
def gf_row_lt(+j: Nat, +i: Nat, +R: Nat, +w: Nat, jr: {Nat.is_lt(j, R) == True{} : Bool},
    ir: {Nat.is_lt(i, w) == True{} : Bool})
  -> {Nat.is_lt(Nat.add(Nat.mul(j, w), i), Nat.mul(R, w)) == True{} : Bool}:
  +M = Nat.mul(j, w)
  +s1 = Ar.ard_tr(Nat.is_lt(Nat.add(M, i), Nat.add(M, w)), Nat.is_lt(i, w), Ar.ar_lt_add_cancel(M, i, w), ir)
  +s2 = Equal.trans(Bool, Nat.is_lt(Nat.add(M, i), Nat.add(w, M)), Nat.is_lt(Nat.add(M, i), Nat.add(M, w)), True{},
    Equal.cong(Nat, Bool, z => Nat.is_lt(Nat.add(M, i), z), Nat.add(w, M), Nat.add(M, w), Ar.ar_add_comm(w, M)), s1)
  +le = Ar.ar_le_mul_mono(1n+j, R, w, Ar.ard_rt(Nat.is_lt(j, R), Nat.is_le(1n+j, R), Ar.ar_lt_succ_le(j, R), jr))
  Ar.ar_lt_le_trans(Nat.add(M, i), Nat.add(w, M), Nat.mul(R, w), s2, le)

# From f2 on the whole [0, R * 128) to f2(j, i) for j < R, i < 128.
def gf_all2_at(~f2: Nat -> Nat -> Bool, +R: Nat, +j: Nat, +i: Nat, h: {gf_all2(~f2, Nat.mul(R, 128n)) == True{} : Bool},
    jr: {Nat.is_lt(j, R) == True{} : Bool}, +ir: {Nat.is_lt(i, 128n) == True{} : Bool})
  -> {f2(j, i) == True{} : Bool}:
  +x = Nat.add(Nat.mul(j, 128n), i)
  +q = gf_fst({Nat.div(x, 128n) == j : Nat}, {Nat.mod(x, 128n) == i : Nat}, Ar.ar_div_unique(j, i, 128n, ir))
  +r = gf_snd({Nat.div(x, 128n) == j : Nat}, {Nat.mod(x, 128n) == i : Nat}, Ar.ar_div_unique(j, i, 128n, ir))
  +got = gf_all2_get(~f2, Nat.mul(R, 128n), x, h, gf_row_lt(j, i, R, 128n, jr, ir))
  %q : {f2(_, i) == True{} : Bool}
  %r : {f2(Nat.div(x, 128n), _) == True{} : Bool}
  got

def gf_ovo(+k: Nat, +v: Nat) -> Bool:
  Bool.and(Nat.is_lt(Spec.verify_thread(k, v), 512n),
    Bool.and(Nat.is_eq(Impl.kslice(Spec.verify_thread(k, v)), k),
      Nat.is_eq(Impl.vcol(0n, Spec.verify_thread(k, v)), v)))

def gf_ovo_w(x: Nat, y: Nat) -> Bool:
  gf_ovo(x, y)

# The 512-entry check of gf_ovo, decided in 16 windows of 32 (one def each).
def gf_ovo_r1() -> {gf_rng2(~gf_ovo_w, 32n, 32n) == True{} : Bool}:
  {==}

def gf_ovo_r2() -> {gf_rng2(~gf_ovo_w, 64n, 32n) == True{} : Bool}:
  {==}

def gf_ovo_r3() -> {gf_rng2(~gf_ovo_w, 96n, 32n) == True{} : Bool}:
  {==}

def gf_ovo_r4() -> {gf_rng2(~gf_ovo_w, 128n, 32n) == True{} : Bool}:
  {==}

def gf_ovo_r5() -> {gf_rng2(~gf_ovo_w, 160n, 32n) == True{} : Bool}:
  {==}

def gf_ovo_r6() -> {gf_rng2(~gf_ovo_w, 192n, 32n) == True{} : Bool}:
  {==}

def gf_ovo_r7() -> {gf_rng2(~gf_ovo_w, 224n, 32n) == True{} : Bool}:
  {==}

def gf_ovo_r8() -> {gf_rng2(~gf_ovo_w, 256n, 32n) == True{} : Bool}:
  {==}

def gf_ovo_r9() -> {gf_rng2(~gf_ovo_w, 288n, 32n) == True{} : Bool}:
  {==}

def gf_ovo_r10() -> {gf_rng2(~gf_ovo_w, 320n, 32n) == True{} : Bool}:
  {==}

def gf_ovo_r11() -> {gf_rng2(~gf_ovo_w, 352n, 32n) == True{} : Bool}:
  {==}

def gf_ovo_r12() -> {gf_rng2(~gf_ovo_w, 384n, 32n) == True{} : Bool}:
  {==}

def gf_ovo_r13() -> {gf_rng2(~gf_ovo_w, 416n, 32n) == True{} : Bool}:
  {==}

def gf_ovo_r14() -> {gf_rng2(~gf_ovo_w, 448n, 32n) == True{} : Bool}:
  {==}

def gf_ovo_r15() -> {gf_rng2(~gf_ovo_w, 480n, 32n) == True{} : Bool}:
  {==}

def gf_ovo_c1() -> {gf_all2(~gf_ovo_w, 32n) == True{} : Bool}:
  {==}

def gf_ovo_c2() -> {gf_all2(~gf_ovo_w, 64n) == True{} : Bool}:
  gf_glue2_at(~gf_ovo_w, 32n, 32n, 64n, {==}, gf_ovo_r1(), gf_ovo_c1())

def gf_ovo_c3() -> {gf_all2(~gf_ovo_w, 96n) == True{} : Bool}:
  gf_glue2_at(~gf_ovo_w, 32n, 64n, 96n, {==}, gf_ovo_r2(), gf_ovo_c2())

def gf_ovo_c4() -> {gf_all2(~gf_ovo_w, 128n) == True{} : Bool}:
  gf_glue2_at(~gf_ovo_w, 32n, 96n, 128n, {==}, gf_ovo_r3(), gf_ovo_c3())

def gf_ovo_c5() -> {gf_all2(~gf_ovo_w, 160n) == True{} : Bool}:
  gf_glue2_at(~gf_ovo_w, 32n, 128n, 160n, {==}, gf_ovo_r4(), gf_ovo_c4())

def gf_ovo_c6() -> {gf_all2(~gf_ovo_w, 192n) == True{} : Bool}:
  gf_glue2_at(~gf_ovo_w, 32n, 160n, 192n, {==}, gf_ovo_r5(), gf_ovo_c5())

def gf_ovo_c7() -> {gf_all2(~gf_ovo_w, 224n) == True{} : Bool}:
  gf_glue2_at(~gf_ovo_w, 32n, 192n, 224n, {==}, gf_ovo_r6(), gf_ovo_c6())

def gf_ovo_c8() -> {gf_all2(~gf_ovo_w, 256n) == True{} : Bool}:
  gf_glue2_at(~gf_ovo_w, 32n, 224n, 256n, {==}, gf_ovo_r7(), gf_ovo_c7())

def gf_ovo_c9() -> {gf_all2(~gf_ovo_w, 288n) == True{} : Bool}:
  gf_glue2_at(~gf_ovo_w, 32n, 256n, 288n, {==}, gf_ovo_r8(), gf_ovo_c8())

def gf_ovo_c10() -> {gf_all2(~gf_ovo_w, 320n) == True{} : Bool}:
  gf_glue2_at(~gf_ovo_w, 32n, 288n, 320n, {==}, gf_ovo_r9(), gf_ovo_c9())

def gf_ovo_c11() -> {gf_all2(~gf_ovo_w, 352n) == True{} : Bool}:
  gf_glue2_at(~gf_ovo_w, 32n, 320n, 352n, {==}, gf_ovo_r10(), gf_ovo_c10())

def gf_ovo_c12() -> {gf_all2(~gf_ovo_w, 384n) == True{} : Bool}:
  gf_glue2_at(~gf_ovo_w, 32n, 352n, 384n, {==}, gf_ovo_r11(), gf_ovo_c11())

def gf_ovo_c13() -> {gf_all2(~gf_ovo_w, 416n) == True{} : Bool}:
  gf_glue2_at(~gf_ovo_w, 32n, 384n, 416n, {==}, gf_ovo_r12(), gf_ovo_c12())

def gf_ovo_c14() -> {gf_all2(~gf_ovo_w, 448n) == True{} : Bool}:
  gf_glue2_at(~gf_ovo_w, 32n, 416n, 448n, {==}, gf_ovo_r13(), gf_ovo_c13())

def gf_ovo_c15() -> {gf_all2(~gf_ovo_w, 480n) == True{} : Bool}:
  gf_glue2_at(~gf_ovo_w, 32n, 448n, 480n, {==}, gf_ovo_r14(), gf_ovo_c14())

def gf_ovo_c16() -> {gf_all2(~gf_ovo_w, 512n) == True{} : Bool}:
  gf_glue2_at(~gf_ovo_w, 32n, 480n, 512n, {==}, gf_ovo_r15(), gf_ovo_c15())

def gf_ovo_all() -> {gf_all2(~gf_ovo_w, Nat.mul(4n, 128n)) == True{} : Bool}:
  gf_all2_lit(~gf_ovo_w, 512n, Nat.mul(4n, 128n), {==}, gf_ovo_c16())

def Laws.own_verify_onto(k, v, kr, vr):
  gf_all2_at(~gf_ovo_w, 4n, k, v, gf_ovo_all(), kr, vr)

def gf_ori(+b: Nat, +t: Nat) -> Bool:
  Bool.and(Nat.is_lt(Impl.kslice(t), 4n), Bool.and(Nat.is_lt(Impl.vcol(Nat.mul(b, 32n), t), 128n),
    Bool.and(Nat.is_eq(Spec.replay_block(Impl.vcol(Nat.mul(b, 32n), t)), b),
      Nat.is_eq(Spec.replay_thread(Impl.kslice(t), Impl.vcol(Nat.mul(b, 32n), t)), t))))

def gf_ori_w(x: Nat, y: Nat) -> Bool:
  gf_ori(x, y)

# The 512-entry check of gf_ori, decided in 16 windows of 32 (one def each).
def gf_ori_r1() -> {gf_rng2(~gf_ori_w, 32n, 32n) == True{} : Bool}:
  {==}

def gf_ori_r2() -> {gf_rng2(~gf_ori_w, 64n, 32n) == True{} : Bool}:
  {==}

def gf_ori_r3() -> {gf_rng2(~gf_ori_w, 96n, 32n) == True{} : Bool}:
  {==}

def gf_ori_r4() -> {gf_rng2(~gf_ori_w, 128n, 32n) == True{} : Bool}:
  {==}

def gf_ori_r5() -> {gf_rng2(~gf_ori_w, 160n, 32n) == True{} : Bool}:
  {==}

def gf_ori_r6() -> {gf_rng2(~gf_ori_w, 192n, 32n) == True{} : Bool}:
  {==}

def gf_ori_r7() -> {gf_rng2(~gf_ori_w, 224n, 32n) == True{} : Bool}:
  {==}

def gf_ori_r8() -> {gf_rng2(~gf_ori_w, 256n, 32n) == True{} : Bool}:
  {==}

def gf_ori_r9() -> {gf_rng2(~gf_ori_w, 288n, 32n) == True{} : Bool}:
  {==}

def gf_ori_r10() -> {gf_rng2(~gf_ori_w, 320n, 32n) == True{} : Bool}:
  {==}

def gf_ori_r11() -> {gf_rng2(~gf_ori_w, 352n, 32n) == True{} : Bool}:
  {==}

def gf_ori_r12() -> {gf_rng2(~gf_ori_w, 384n, 32n) == True{} : Bool}:
  {==}

def gf_ori_r13() -> {gf_rng2(~gf_ori_w, 416n, 32n) == True{} : Bool}:
  {==}

def gf_ori_r14() -> {gf_rng2(~gf_ori_w, 448n, 32n) == True{} : Bool}:
  {==}

def gf_ori_r15() -> {gf_rng2(~gf_ori_w, 480n, 32n) == True{} : Bool}:
  {==}

def gf_ori_c1() -> {gf_all2(~gf_ori_w, 32n) == True{} : Bool}:
  {==}

def gf_ori_c2() -> {gf_all2(~gf_ori_w, 64n) == True{} : Bool}:
  gf_glue2_at(~gf_ori_w, 32n, 32n, 64n, {==}, gf_ori_r1(), gf_ori_c1())

def gf_ori_c3() -> {gf_all2(~gf_ori_w, 96n) == True{} : Bool}:
  gf_glue2_at(~gf_ori_w, 32n, 64n, 96n, {==}, gf_ori_r2(), gf_ori_c2())

def gf_ori_c4() -> {gf_all2(~gf_ori_w, 128n) == True{} : Bool}:
  gf_glue2_at(~gf_ori_w, 32n, 96n, 128n, {==}, gf_ori_r3(), gf_ori_c3())

def gf_ori_c5() -> {gf_all2(~gf_ori_w, 160n) == True{} : Bool}:
  gf_glue2_at(~gf_ori_w, 32n, 128n, 160n, {==}, gf_ori_r4(), gf_ori_c4())

def gf_ori_c6() -> {gf_all2(~gf_ori_w, 192n) == True{} : Bool}:
  gf_glue2_at(~gf_ori_w, 32n, 160n, 192n, {==}, gf_ori_r5(), gf_ori_c5())

def gf_ori_c7() -> {gf_all2(~gf_ori_w, 224n) == True{} : Bool}:
  gf_glue2_at(~gf_ori_w, 32n, 192n, 224n, {==}, gf_ori_r6(), gf_ori_c6())

def gf_ori_c8() -> {gf_all2(~gf_ori_w, 256n) == True{} : Bool}:
  gf_glue2_at(~gf_ori_w, 32n, 224n, 256n, {==}, gf_ori_r7(), gf_ori_c7())

def gf_ori_c9() -> {gf_all2(~gf_ori_w, 288n) == True{} : Bool}:
  gf_glue2_at(~gf_ori_w, 32n, 256n, 288n, {==}, gf_ori_r8(), gf_ori_c8())

def gf_ori_c10() -> {gf_all2(~gf_ori_w, 320n) == True{} : Bool}:
  gf_glue2_at(~gf_ori_w, 32n, 288n, 320n, {==}, gf_ori_r9(), gf_ori_c9())

def gf_ori_c11() -> {gf_all2(~gf_ori_w, 352n) == True{} : Bool}:
  gf_glue2_at(~gf_ori_w, 32n, 320n, 352n, {==}, gf_ori_r10(), gf_ori_c10())

def gf_ori_c12() -> {gf_all2(~gf_ori_w, 384n) == True{} : Bool}:
  gf_glue2_at(~gf_ori_w, 32n, 352n, 384n, {==}, gf_ori_r11(), gf_ori_c11())

def gf_ori_c13() -> {gf_all2(~gf_ori_w, 416n) == True{} : Bool}:
  gf_glue2_at(~gf_ori_w, 32n, 384n, 416n, {==}, gf_ori_r12(), gf_ori_c12())

def gf_ori_c14() -> {gf_all2(~gf_ori_w, 448n) == True{} : Bool}:
  gf_glue2_at(~gf_ori_w, 32n, 416n, 448n, {==}, gf_ori_r13(), gf_ori_c13())

def gf_ori_c15() -> {gf_all2(~gf_ori_w, 480n) == True{} : Bool}:
  gf_glue2_at(~gf_ori_w, 32n, 448n, 480n, {==}, gf_ori_r14(), gf_ori_c14())

def gf_ori_c16() -> {gf_all2(~gf_ori_w, 512n) == True{} : Bool}:
  gf_glue2_at(~gf_ori_w, 32n, 480n, 512n, {==}, gf_ori_r15(), gf_ori_c15())

def gf_ori_all() -> {gf_all2(~gf_ori_w, Nat.mul(4n, 128n)) == True{} : Bool}:
  gf_all2_lit(~gf_ori_w, 512n, Nat.mul(4n, 128n), {==}, gf_ori_c16())

def Laws.own_replay_into(b, t, br, tr):
  gf_all2_at(~gf_ori_w, 4n, b, t, gf_ori_all(), br, tr)

def gf_oro(+k: Nat, +v: Nat) -> Bool:
  Bool.and(Nat.is_lt(Spec.replay_block(v), 4n), Bool.and(Nat.is_lt(Spec.replay_thread(k, v), 128n),
    Bool.and(Nat.is_eq(Impl.kslice(Spec.replay_thread(k, v)), k),
      Nat.is_eq(Impl.vcol(Nat.mul(Spec.replay_block(v), 32n), Spec.replay_thread(k, v)), v))))

def gf_oro_w(x: Nat, y: Nat) -> Bool:
  gf_oro(x, y)

# The 512-entry check of gf_oro, decided in 16 windows of 32 (one def each).
def gf_oro_r1() -> {gf_rng2(~gf_oro_w, 32n, 32n) == True{} : Bool}:
  {==}

def gf_oro_r2() -> {gf_rng2(~gf_oro_w, 64n, 32n) == True{} : Bool}:
  {==}

def gf_oro_r3() -> {gf_rng2(~gf_oro_w, 96n, 32n) == True{} : Bool}:
  {==}

def gf_oro_r4() -> {gf_rng2(~gf_oro_w, 128n, 32n) == True{} : Bool}:
  {==}

def gf_oro_r5() -> {gf_rng2(~gf_oro_w, 160n, 32n) == True{} : Bool}:
  {==}

def gf_oro_r6() -> {gf_rng2(~gf_oro_w, 192n, 32n) == True{} : Bool}:
  {==}

def gf_oro_r7() -> {gf_rng2(~gf_oro_w, 224n, 32n) == True{} : Bool}:
  {==}

def gf_oro_r8() -> {gf_rng2(~gf_oro_w, 256n, 32n) == True{} : Bool}:
  {==}

def gf_oro_r9() -> {gf_rng2(~gf_oro_w, 288n, 32n) == True{} : Bool}:
  {==}

def gf_oro_r10() -> {gf_rng2(~gf_oro_w, 320n, 32n) == True{} : Bool}:
  {==}

def gf_oro_r11() -> {gf_rng2(~gf_oro_w, 352n, 32n) == True{} : Bool}:
  {==}

def gf_oro_r12() -> {gf_rng2(~gf_oro_w, 384n, 32n) == True{} : Bool}:
  {==}

def gf_oro_r13() -> {gf_rng2(~gf_oro_w, 416n, 32n) == True{} : Bool}:
  {==}

def gf_oro_r14() -> {gf_rng2(~gf_oro_w, 448n, 32n) == True{} : Bool}:
  {==}

def gf_oro_r15() -> {gf_rng2(~gf_oro_w, 480n, 32n) == True{} : Bool}:
  {==}

def gf_oro_c1() -> {gf_all2(~gf_oro_w, 32n) == True{} : Bool}:
  {==}

def gf_oro_c2() -> {gf_all2(~gf_oro_w, 64n) == True{} : Bool}:
  gf_glue2_at(~gf_oro_w, 32n, 32n, 64n, {==}, gf_oro_r1(), gf_oro_c1())

def gf_oro_c3() -> {gf_all2(~gf_oro_w, 96n) == True{} : Bool}:
  gf_glue2_at(~gf_oro_w, 32n, 64n, 96n, {==}, gf_oro_r2(), gf_oro_c2())

def gf_oro_c4() -> {gf_all2(~gf_oro_w, 128n) == True{} : Bool}:
  gf_glue2_at(~gf_oro_w, 32n, 96n, 128n, {==}, gf_oro_r3(), gf_oro_c3())

def gf_oro_c5() -> {gf_all2(~gf_oro_w, 160n) == True{} : Bool}:
  gf_glue2_at(~gf_oro_w, 32n, 128n, 160n, {==}, gf_oro_r4(), gf_oro_c4())

def gf_oro_c6() -> {gf_all2(~gf_oro_w, 192n) == True{} : Bool}:
  gf_glue2_at(~gf_oro_w, 32n, 160n, 192n, {==}, gf_oro_r5(), gf_oro_c5())

def gf_oro_c7() -> {gf_all2(~gf_oro_w, 224n) == True{} : Bool}:
  gf_glue2_at(~gf_oro_w, 32n, 192n, 224n, {==}, gf_oro_r6(), gf_oro_c6())

def gf_oro_c8() -> {gf_all2(~gf_oro_w, 256n) == True{} : Bool}:
  gf_glue2_at(~gf_oro_w, 32n, 224n, 256n, {==}, gf_oro_r7(), gf_oro_c7())

def gf_oro_c9() -> {gf_all2(~gf_oro_w, 288n) == True{} : Bool}:
  gf_glue2_at(~gf_oro_w, 32n, 256n, 288n, {==}, gf_oro_r8(), gf_oro_c8())

def gf_oro_c10() -> {gf_all2(~gf_oro_w, 320n) == True{} : Bool}:
  gf_glue2_at(~gf_oro_w, 32n, 288n, 320n, {==}, gf_oro_r9(), gf_oro_c9())

def gf_oro_c11() -> {gf_all2(~gf_oro_w, 352n) == True{} : Bool}:
  gf_glue2_at(~gf_oro_w, 32n, 320n, 352n, {==}, gf_oro_r10(), gf_oro_c10())

def gf_oro_c12() -> {gf_all2(~gf_oro_w, 384n) == True{} : Bool}:
  gf_glue2_at(~gf_oro_w, 32n, 352n, 384n, {==}, gf_oro_r11(), gf_oro_c11())

def gf_oro_c13() -> {gf_all2(~gf_oro_w, 416n) == True{} : Bool}:
  gf_glue2_at(~gf_oro_w, 32n, 384n, 416n, {==}, gf_oro_r12(), gf_oro_c12())

def gf_oro_c14() -> {gf_all2(~gf_oro_w, 448n) == True{} : Bool}:
  gf_glue2_at(~gf_oro_w, 32n, 416n, 448n, {==}, gf_oro_r13(), gf_oro_c13())

def gf_oro_c15() -> {gf_all2(~gf_oro_w, 480n) == True{} : Bool}:
  gf_glue2_at(~gf_oro_w, 32n, 448n, 480n, {==}, gf_oro_r14(), gf_oro_c14())

def gf_oro_c16() -> {gf_all2(~gf_oro_w, 512n) == True{} : Bool}:
  gf_glue2_at(~gf_oro_w, 32n, 480n, 512n, {==}, gf_oro_r15(), gf_oro_c15())

def gf_oro_all() -> {gf_all2(~gf_oro_w, Nat.mul(4n, 128n)) == True{} : Bool}:
  gf_all2_lit(~gf_oro_w, 512n, Nat.mul(4n, 128n), {==}, gf_oro_c16())

def Laws.own_replay_onto(k, v, kr, vr):
  gf_all2_at(~gf_oro_w, 4n, k, v, gf_oro_all(), kr, vr)

# ==== (R) the shared token routine ====

def gf_outs(~S: Data, ~X: Data, ~O: Data, ~step: S -> X -> S, ~out: S -> X -> O,
    xs: List<&2, X>, +s: S)
  -> {Impl.outs(~S, ~O, Impl.run(~S, ~X, ~O, ~step, ~out, List.length(&2, X, xs), xs, s)) ==
    Spec.outputs(~S, ~X, ~O, ~step, ~out, xs, s) : List<&2, O>}:
  match xs:
    case Nil{}:
      {==}
    case +x <> rest:
      %gf_outs(~S, ~X, ~O, ~step, ~out, rest, step(s, x)) :
        {out(step(s, x), x) <> Impl.outs(~S, ~O, Impl.run(~S, ~X, ~O, ~step, ~out, List.length(&2, X, rest), rest, step(s, x))) ==
          out(step(s, x), x) <> _ : List<&2, O>}
      {==}

def Laws.routine_verify_outputs(S, X, O, step, out, xs, s0):
  gf_outs(~S, ~X, ~O, ~step, ~out, xs, s0)

def gf_cf(~S: Data, ~X: Data, ~O: Data, ~step: S -> X -> S, ~out: S -> X -> O, c: Nat,
    xs: List<&2, X>, +s: S)
  -> {Impl.final(~S, ~O, Impl.run(~S, ~X, ~O, ~step, ~out, c, xs, s), s) ==
    Spec.fold(~S, ~X, ~step, List.take(&2, X, xs, c), s) : S}:
  match c:
    case 0n:
      match xs:
        case Nil{}:
          {==}
        case x <> rest:
          {==}
    case 1n+m:
      match xs:
        case Nil{}:
          {==}
        case +x <> rest:
          gf_cf(~S, ~X, ~O, ~step, ~out, m, rest, step(s, x))

def Laws.routine_commit_fold(S, X, O, step, out, c, xs, s0):
  gf_cf(~S, ~X, ~O, ~step, ~out, c, xs, s0)

# run(c) is the first c records of the full run.
def gf_take(~S: Data, ~X: Data, ~O: Data, ~step: S -> X -> S, ~out: S -> X -> O, c: Nat,
    xs: List<&2, X>, +s: S)
  -> {Impl.run(~S, ~X, ~O, ~step, ~out, c, xs, s) ==
    List.take(&2, Impl.Tok<S, O>, Impl.run(~S, ~X, ~O, ~step, ~out, List.length(&2, X, xs), xs, s), c) : List<&2, Impl.Tok<S, O>>}:
  match c:
    case 0n:
      match xs:
        case Nil{}:
          {==}
        case x <> rest:
          {==}
    case 1n+m:
      match xs:
        case Nil{}:
          {==}
        case +x <> rest:
          %gf_take(~S, ~X, ~O, ~step, ~out, m, rest, step(s, x)) :
            {Impl.Tok{step(s, x), out(step(s, x), x)} <> Impl.run(~S, ~X, ~O, ~step, ~out, m, rest, step(s, x)) ==
              Impl.Tok{step(s, x), out(step(s, x), x)} <> _
              : List<&2, Impl.Tok<S, O>>}
          {==}

def Laws.commit_is_verify_prefix(S, X, O, step, out, c, xs, s0):
  Equal.cong(List<&2, Impl.Tok<S, O>>, S, ts => Impl.final(~S, ~O, ts, s0),
    Impl.run(~S, ~X, ~O, ~step, ~out, c, xs, s0),
    List.take(&2, Impl.Tok<S, O>, Impl.run(~S, ~X, ~O, ~step, ~out, List.length(&2, X, xs), xs, s0), c),
    gf_take(~S, ~X, ~O, ~step, ~out, c, xs, s0))

# Reference: outputs of the suffix from the prefix fold are the suffix of the outputs.
def gf_resume(~S: Data, ~X: Data, ~O: Data, ~step: S -> X -> S, ~out: S -> X -> O, c: Nat,
    xs: List<&2, X>, +s: S)
  -> {Spec.outputs(~S, ~X, ~O, ~step, ~out, List.drop(&2, X, xs, c), Spec.fold(~S, ~X, ~step, List.take(&2, X, xs, c), s)) ==
    List.drop(&2, O, Spec.outputs(~S, ~X, ~O, ~step, ~out, xs, s), c) : List<&2, O>}:
  match c:
    case 0n:
      match xs:
        case Nil{}:
          {==}
        case +x <> rest:
          {==}
    case 1n+m:
      match xs:
        case Nil{}:
          {==}
        case +x <> rest:
          gf_resume(~S, ~X, ~O, ~step, ~out, m, rest, step(s, x))

def Laws.resume_outputs(S, X, O, step, out, c, xs, s0):
  +cm = Impl.commit(~S, ~X, ~O, ~step, ~out, c, xs, s0)
  +fd = Spec.fold(~S, ~X, ~step, List.take(&2, X, xs, c), s0)
  +dx = List.drop(&2, X, xs, c)
  Equal.trans(List<&2, O>, Impl.verify_outputs(~S, ~X, ~O, ~step, ~out, dx, cm),
    Spec.outputs(~S, ~X, ~O, ~step, ~out, dx, cm),
    List.drop(&2, O, Impl.verify_outputs(~S, ~X, ~O, ~step, ~out, xs, s0), c),
    gf_outs(~S, ~X, ~O, ~step, ~out, dx, cm),
    Equal.trans(List<&2, O>, Spec.outputs(~S, ~X, ~O, ~step, ~out, dx, cm),
      Spec.outputs(~S, ~X, ~O, ~step, ~out, dx, fd),
      List.drop(&2, O, Impl.verify_outputs(~S, ~X, ~O, ~step, ~out, xs, s0), c),
      Equal.cong(S, List<&2, O>, z => Spec.outputs(~S, ~X, ~O, ~step, ~out, dx, z), cm, fd,
        gf_cf(~S, ~X, ~O, ~step, ~out, c, xs, s0)),
      Equal.trans(List<&2, O>, Spec.outputs(~S, ~X, ~O, ~step, ~out, dx, fd),
        List.drop(&2, O, Spec.outputs(~S, ~X, ~O, ~step, ~out, xs, s0), c),
        List.drop(&2, O, Impl.verify_outputs(~S, ~X, ~O, ~step, ~out, xs, s0), c),
        gf_resume(~S, ~X, ~O, ~step, ~out, c, xs, s0),
        Equal.cong(List<&2, O>, List<&2, O>, l => List.drop(&2, O, l, c),
          Spec.outputs(~S, ~X, ~O, ~step, ~out, xs, s0),
          Impl.verify_outputs(~S, ~X, ~O, ~step, ~out, xs, s0),
          Equal.sym(List<&2, O>, Impl.verify_outputs(~S, ~X, ~O, ~step, ~out, xs, s0),
            Spec.outputs(~S, ~X, ~O, ~step, ~out, xs, s0), gf_outs(~S, ~X, ~O, ~step, ~out, xs, s0))))))

def gf_gr_replay(~S: Data, ~X: Data, ~O: Data, ~step: S -> X -> S, ~out: S -> X -> O, c: Nat,
    xs: List<&2, X>, +s: S)
  -> {GR.replay(~S, ~X, ~step, c, xs, s) == Impl.final(~S, ~O, Impl.run(~S, ~X, ~O, ~step, ~out, c, xs, s), s) : S}:
  match c:
    case 0n:
      {==}
    case 1n+m:
      match xs:
        case Nil{}:
          {==}
        case +x <> rest:
          gf_gr_replay(~S, ~X, ~O, ~step, ~out, m, rest, step(s, x))

def gf_gr_verify(~S: Data, ~X: Data, ~O: Data, ~step: S -> X -> S, ~out: S -> X -> O,
    xs: List<&2, X>, +s: S)
  -> {GR.verify(~S, ~X, ~O, ~step, ~out, xs, s) == Spec.outputs(~S, ~X, ~O, ~step, ~out, xs, s) : List<&2, O>}:
  match xs:
    case Nil{}:
      {==}
    case +x <> rest:
      %gf_gr_verify(~S, ~X, ~O, ~step, ~out, rest, step(s, x)) :
        {out(step(s, x), x) <> GR.verify(~S, ~X, ~O, ~step, ~out, rest, step(s, x)) == out(step(s, x), x) <> _ : List<&2, O>}
      {==}

def Laws.routine_is_gdn_replay(S, X, O, step, out, c, xs, s0):
  (gf_gr_replay(~S, ~X, ~O, ~step, ~out, c, xs, s0),
   Equal.trans(List<&2, O>, GR.verify(~S, ~X, ~O, ~step, ~out, xs, s0), Spec.outputs(~S, ~X, ~O, ~step, ~out, xs, s0),
     Impl.verify_outputs(~S, ~X, ~O, ~step, ~out, xs, s0),
     gf_gr_verify(~S, ~X, ~O, ~step, ~out, xs, s0),
     Equal.sym(List<&2, O>, Impl.verify_outputs(~S, ~X, ~O, ~step, ~out, xs, s0),
       Spec.outputs(~S, ~X, ~O, ~step, ~out, xs, s0), gf_outs(~S, ~X, ~O, ~step, ~out, xs, s0))))

# ==== (5) conv channels ====

def gf_not_true(b: Bool, h: {Bool.not(b) == True{} : Bool}) -> {b == False{} : Bool}:
  match b:
    case False{}:
      {==}
    case True{}:
      Empty.absurd({True{} == False{} : Bool}, Ar.ar_false_true(h))

# t < n gives (n <= t) == False
def gf_le_false(+t: Nat, +n: Nat, h: {Nat.is_lt(t, n) == True{} : Bool}) -> {Nat.is_le(n, t) == False{} : Bool}:
  gf_not_true(Nat.is_le(n, t), Equal.trans(Bool, Bool.not(Nat.is_le(n, t)), Nat.is_lt(t, n), True{},
    Equal.sym(Bool, Nat.is_lt(t, n), Bool.not(Nat.is_le(n, t)), Ar.ar_lt_not_le(t, n)), h))

# n <= t gives (t < n) == False
def gf_lt_false(+t: Nat, +n: Nat, h: {Nat.is_le(n, t) == True{} : Bool}) -> {Nat.is_lt(t, n) == False{} : Bool}:
  %Equal.sym(Bool, Nat.is_lt(t, n), Bool.not(Nat.is_le(n, t)), Ar.ar_lt_not_le(t, n)) : {_ == False{} : Bool}
  %Equal.sym(Bool, Nat.is_le(n, t), True{}, h) : {Bool.not(_) == False{} : Bool}
  {==}

# (t < n) == False gives n <= t
def gf_le_of(+t: Nat, +n: Nat, h: {Nat.is_lt(t, n) == False{} : Bool}) -> {Nat.is_le(n, t) == True{} : Bool}:
  Ar.p8_not_false(Nat.is_le(n, t), Equal.trans(Bool, Bool.not(Nat.is_le(n, t)), Nat.is_lt(t, n), False{},
    Equal.sym(Bool, Nat.is_lt(t, n), Bool.not(Nat.is_le(n, t)), Ar.ar_lt_not_le(t, n)), h))

def gf_or_l(a: Bool, -b: Bool, bf: {b == False{} : Bool}, h: {Bool.or(a, b) == True{} : Bool}) -> {a == True{} : Bool}:
  match a:
    case True{}:
      {==}
    case False{}:
      Empty.absurd({False{} == True{} : Bool}, Ar.ar_false_ne(b, bf, h))

def gf_or_true_r(a: Bool) -> {Bool.or(a, True{}) == True{} : Bool}:
  match a:
    case True{}:
      {==}
    case False{}:
      {==}

# sub(M + t, k) == M + i when k + i == t
def gf_sub_k(+M: Nat, +t: Nat, +i: Nat, +k: Nat, et: {Nat.add(k, i) == t : Nat})
  -> {Nat.sub(Nat.add(M, t), k) == Nat.add(M, i) : Nat}:
  %et : {Nat.sub(Nat.add(M, _), k) == Nat.add(M, i) : Nat}
  %Equal.sym(Nat, Nat.add(M, Nat.add(k, i)), Nat.add(k, Nat.add(M, i)), Ar.ar_h_add_swap(M, k, i)) : {Nat.sub(_, k) == Nat.add(M, i) : Nat}
  Ar.ar_add_sub(k, Nat.add(M, i))

# chan in the three thread ranges
def gf_chanA(+Nk: Nat, +g: Nat, +h: Nat, +t: Nat, ta: {Nat.is_lt(t, 128n) == True{} : Bool})
  -> {Impl.chan(Nk, g, h, t) == Nat.add(Nat.mul(Impl.kh(g, h), 128n), t) : Nat}:
  %Equal.sym(Bool, Nat.is_lt(t, 128n), True{}, ta) : {Impl.pick(_, Nat.add(Nat.mul(Impl.kh(g, h), 128n), t),
    Impl.pick(Nat.is_lt(t, 256n), Nat.sub(Nat.add(Nat.mul(Nat.add(Nk, Impl.kh(g, h)), 128n), t), 128n),
      Nat.sub(Nat.add(Nat.add(Nat.mul(Nat.mul(2n, Nk), 128n), Nat.mul(h, 128n)), t), 256n))) ==
    Nat.add(Nat.mul(Impl.kh(g, h), 128n), t) : Nat}
  {==}

def gf_chanB(+Nk: Nat, +g: Nat, +h: Nat, +t: Nat, +i: Nat, ta: {Nat.is_lt(t, 128n) == False{} : Bool},
    tb: {Nat.is_lt(t, 256n) == True{} : Bool}, +et: {Nat.add(128n, i) == t : Nat})
  -> {Impl.chan(Nk, g, h, t) == Nat.add(Nat.mul(Nat.add(Nk, Impl.kh(g, h)), 128n), i) : Nat}:
  +M = Nat.mul(Nat.add(Nk, Impl.kh(g, h)), 128n)
  %Equal.sym(Bool, Nat.is_lt(t, 128n), False{}, ta) : {Impl.pick(_, Nat.add(Nat.mul(Impl.kh(g, h), 128n), t),
    Impl.pick(Nat.is_lt(t, 256n), Nat.sub(Nat.add(M, t), 128n),
      Nat.sub(Nat.add(Nat.add(Nat.mul(Nat.mul(2n, Nk), 128n), Nat.mul(h, 128n)), t), 256n))) ==
    Nat.add(M, i) : Nat}
  %Equal.sym(Bool, Nat.is_lt(t, 256n), True{}, tb) : {Impl.pick(_, Nat.sub(Nat.add(M, t), 128n),
      Nat.sub(Nat.add(Nat.add(Nat.mul(Nat.mul(2n, Nk), 128n), Nat.mul(h, 128n)), t), 256n)) ==
    Nat.add(M, i) : Nat}
  gf_sub_k(M, t, i, 128n, et)

# 2 * Nk == Nk + Nk
def gf_two(+Nk: Nat) -> {Nat.mul(2n, Nk) == Nat.add(Nk, Nk) : Nat}:
  Equal.cong(Nat, Nat, z => Nat.add(Nk, z), Nat.add(Nk, 0n), Nk, Ar.ar_add_zero_r(Nk))

def gf_chanC(+Nk: Nat, +g: Nat, +h: Nat, +t: Nat, +i: Nat, ta: {Nat.is_lt(t, 128n) == False{} : Bool},
    tb: {Nat.is_lt(t, 256n) == False{} : Bool}, +et: {Nat.add(256n, i) == t : Nat})
  -> {Impl.chan(Nk, g, h, t) == Nat.add(Nat.mul(Nat.add(Nat.add(Nk, Nk), h), 128n), i) : Nat}:
  +M = Nat.add(Nat.mul(Nat.mul(2n, Nk), 128n), Nat.mul(h, 128n))
  %Equal.sym(Bool, Nat.is_lt(t, 128n), False{}, ta) : {Impl.pick(_, Nat.add(Nat.mul(Impl.kh(g, h), 128n), t),
    Impl.pick(Nat.is_lt(t, 256n), Nat.sub(Nat.add(Nat.mul(Nat.add(Nk, Impl.kh(g, h)), 128n), t), 128n),
      Nat.sub(Nat.add(M, t), 256n))) ==
    Nat.add(Nat.mul(Nat.add(Nat.add(Nk, Nk), h), 128n), i) : Nat}
  %Equal.sym(Bool, Nat.is_lt(t, 256n), False{}, tb) : {Impl.pick(_, Nat.sub(Nat.add(Nat.mul(Nat.add(Nk, Impl.kh(g, h)), 128n), t), 128n),
      Nat.sub(Nat.add(M, t), 256n)) ==
    Nat.add(Nat.mul(Nat.add(Nat.add(Nk, Nk), h), 128n), i) : Nat}
  +N2 = Nat.add(Nk, Nk)
  +eqM = Equal.trans(Nat, M, Nat.add(Nat.mul(N2, 128n), Nat.mul(h, 128n)), Nat.mul(Nat.add(N2, h), 128n),
    Equal.cong(Nat, Nat, z => Nat.add(Nat.mul(z, 128n), Nat.mul(h, 128n)), Nat.mul(2n, Nk), N2, gf_two(Nk)),
    Equal.sym(Nat, Nat.mul(Nat.add(N2, h), 128n), Nat.add(Nat.mul(N2, 128n), Nat.mul(h, 128n)), Ar.ar_mul_add_l(N2, h, 128n)))
  Equal.trans(Nat, Nat.sub(Nat.add(M, t), 256n), Nat.add(M, i), Nat.add(Nat.mul(Nat.add(N2, h), 128n), i),
    gf_sub_k(M, t, i, 256n, et),
    Equal.cong(Nat, Nat, z => Nat.add(z, i), M, Nat.mul(Nat.add(N2, h), 128n), eqM))

# The reference storer of the channel j * 128 + i (i < 128), given the row kind.
def gf_sbq(+Nk: Nat, +g: Nat, +j: Nat, +i: Nat, -isq: Bool, -isk: Bool,
    eq: {Nat.is_lt(j, Nk) == isq : Bool}, ek: {Nat.is_lt(j, Nat.add(Nk, Nk)) == isk : Bool},
    +ir: {Nat.is_lt(i, 128n) == True{} : Bool})
  -> {Spec.store_block(Nk, g, Nat.add(Nat.mul(j, 128n), i)) == Spec.store_block.row(Nk, g, j, isq, isk) : Nat}:
  +x = Nat.add(Nat.mul(j, 128n), i)
  +q = gf_fst({Nat.div(x, 128n) == j : Nat}, {Nat.mod(x, 128n) == i : Nat}, Ar.ar_div_unique(j, i, 128n, ir))
  %eq : {Spec.store_block(Nk, g, x) == Spec.store_block.row(Nk, g, j, _, isk) : Nat}
  %ek : {Spec.store_block(Nk, g, x) == Spec.store_block.row(Nk, g, j, Nat.is_lt(j, Nk), _) : Nat}
  Equal.cong(Nat, Nat, z => Spec.store_block.row(Nk, g, z, Nat.is_lt(z, Nk), Nat.is_lt(z, Nat.add(Nk, Nk))),
    Nat.div(x, 128n), j, q)

def gf_stq(+Nk: Nat, +j: Nat, +i: Nat, -isq: Bool, -isk: Bool,
    eq: {Nat.is_lt(j, Nk) == isq : Bool}, ek: {Nat.is_lt(j, Nat.add(Nk, Nk)) == isk : Bool},
    +ir: {Nat.is_lt(i, 128n) == True{} : Bool})
  -> {Spec.store_thread(Nk, Nat.add(Nat.mul(j, 128n), i)) == Nat.add(Spec.store_thread.base(isq, isk), i) : Nat}:
  +x = Nat.add(Nat.mul(j, 128n), i)
  +q = gf_fst({Nat.div(x, 128n) == j : Nat}, {Nat.mod(x, 128n) == i : Nat}, Ar.ar_div_unique(j, i, 128n, ir))
  +r = gf_snd({Nat.div(x, 128n) == j : Nat}, {Nat.mod(x, 128n) == i : Nat}, Ar.ar_div_unique(j, i, 128n, ir))
  %eq : {Spec.store_thread(Nk, x) == Nat.add(Spec.store_thread.base(_, isk), i) : Nat}
  %ek : {Spec.store_thread(Nk, x) == Nat.add(Spec.store_thread.base(Nat.is_lt(j, Nk), _), i) : Nat}
  %r : {Spec.store_thread(Nk, x) == Nat.add(Spec.store_thread.base(Nat.is_lt(j, Nk), Nat.is_lt(j, Nat.add(Nk, Nk))), _) : Nat}
  Equal.cong(Nat, Nat, z => Nat.add(Spec.store_thread.base(Nat.is_lt(z, Nk), Nat.is_lt(z, Nat.add(Nk, Nk))), Nat.mod(x, 128n)),
    Nat.div(x, 128n), j, q)

# kh < Nk for h < Nk * g
def gf_kh_lt(+Nk: Nat, +g: Nat, +h: Nat, +gpos: {Nat.is_lt(0n, g) == True{} : Bool},
    hr: {Nat.is_lt(h, Nat.mul(Nk, g)) == True{} : Bool}) -> {Nat.is_lt(Impl.kh(g, h), Nk) == True{} : Bool}:
  Ar.ard_tr(Nat.is_lt(Nat.div(h, g), Nk), Nat.is_lt(h, Nat.mul(Nk, g)), Ar.ar_div_lt(Nk, h, g, gpos), hr)

# An owner is the last v head of its k head.
def gf_last_h(+g: Nat, +h: Nat, +gpos: {Nat.is_lt(0n, g) == True{} : Bool},
    own: {Impl.owner(g, h) == True{} : Bool}) -> {Spec.last(g, Impl.kh(g, h)) == h : Nat}:
  +e = Ar.ar_eq_true(Nat.mod(h, g), Nat.sub(g, 1n), own)
  %e : {Nat.add(Nat.mul(Nat.div(h, g), g), _) == h : Nat}
  Ar.ard_dm_eq(h, g, gpos)

def gf_lt_succ(+n: Nat) -> {Nat.is_lt(n, 1n+n) == True{} : Bool}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      gf_lt_succ(p)

# The last v head of k head k0 is an owner of k0.
def gf_gm1(+g: Nat, gpos: {Nat.is_lt(0n, g) == True{} : Bool}) -> {Nat.is_lt(Nat.sub(g, 1n), g) == True{} : Bool}:
  match g:
    case 0n:
      Empty.absurd({Nat.is_lt(Nat.sub(0n, 1n), 0n) == True{} : Bool}, Ar.ar_false_true(gpos))
    case 1n+gp:
      %Equal.sym(Nat, Nat.sub(gp, 0n), gp, Ar.ar_sub_zero_r(gp)) : {Nat.is_lt(_, 1n+gp) == True{} : Bool}
      gf_lt_succ(gp)

def gf_last_kh(+g: Nat, +k0: Nat, +gpos: {Nat.is_lt(0n, g) == True{} : Bool})
  -> {Impl.kh(g, Spec.last(g, k0)) == k0 : Nat} & {Impl.owner(g, Spec.last(g, k0)) == True{} : Bool}:
  +x = Nat.add(Nat.mul(k0, g), Nat.sub(g, 1n))
  (gf_fst({Nat.div(x, g) == k0 : Nat}, {Nat.mod(x, g) == Nat.sub(g, 1n) : Nat}, Ar.ar_div_unique(k0, Nat.sub(g, 1n), g, gf_gm1(g, gpos))),
   Ar.p8_eq_of(Nat.mod(x, g), Nat.sub(g, 1n), gf_snd({Nat.div(x, g) == k0 : Nat}, {Nat.mod(x, g) == Nat.sub(g, 1n) : Nat}, Ar.ar_div_unique(k0, Nat.sub(g, 1n), g, gf_gm1(g, gpos)))))

# last(g, k0) < K * g for k0 < K
def gf_last_lt(+g: Nat, +k0: Nat, +K: Nat, +gpos: {Nat.is_lt(0n, g) == True{} : Bool},
    kr: {Nat.is_lt(k0, K) == True{} : Bool}) -> {Nat.is_lt(Spec.last(g, k0), Nat.mul(K, g)) == True{} : Bool}:
  gf_row_lt(k0, Nat.sub(g, 1n), K, g, kr, gf_gm1(g, gpos))

def gf_store_owner(+g: Nat, +h: Nat, +t: Nat, tb: {Nat.is_lt(t, 256n) == True{} : Bool},
    st: {Impl.store(g, h, t) == True{} : Bool}) -> {Impl.owner(g, h) == True{} : Bool}:
  gf_or_l(Impl.owner(g, h), Nat.is_le(256n, t), gf_le_false(t, 256n, tb), st)

def gf_2nk_le(+Nk: Nat, +H: Nat) -> {Nat.is_le(Nat.add(Nk, Nk), Nat.add(Nat.add(Nk, Nk), H)) == True{} : Bool}:
  Ar.ar_le_add_r(Nat.add(Nk, Nk), H)

def gf_nk_le(+Nk: Nat, +H: Nat) -> {Nat.is_le(Nk, Nat.add(Nat.add(Nk, Nk), H)) == True{} : Bool}:
  Ar.ar_le_trans(Nk, Nat.add(Nk, Nk), Nat.add(Nat.add(Nk, Nk), H), Ar.ar_le_add_r(Nk, Nk), Ar.ar_le_add_r(Nat.add(Nk, Nk), H))

# ---- (5a) into, per thread range ----
def gf_into_A(+Nk: Nat, +g: Nat, +h: Nat, +t: Nat, +gpos: {Nat.is_lt(0n, g) == True{} : Bool},
    +hr: {Nat.is_lt(h, Nat.mul(Nk, g)) == True{} : Bool}, +ta: {Nat.is_lt(t, 128n) == True{} : Bool},
    st: {Impl.store(g, h, t) == True{} : Bool})
  -> {Spec.store_block(Nk, g, Impl.chan(Nk, g, h, t)) == h : Nat} &
     {Spec.store_thread(Nk, Impl.chan(Nk, g, h, t)) == t : Nat} &
     {Nat.is_lt(Impl.chan(Nk, g, h, t), Spec.F(Nk, Nat.mul(Nk, g))) == True{} : Bool}:
  +k = Impl.kh(g, h)
  +R = Nat.add(Nat.add(Nk, Nk), Nat.mul(Nk, g))
  +x = Nat.add(Nat.mul(k, 128n), t)
  +ch = gf_chanA(Nk, g, h, t, ta)
  +kl = gf_kh_lt(Nk, g, h, gpos, hr)
  +own = gf_store_owner(g, h, t, Ar.ar_lt_trans(t, 128n, 256n, ta, {==}), st)
  (Equal.trans(Nat, Spec.store_block(Nk, g, Impl.chan(Nk, g, h, t)), Spec.store_block(Nk, g, x), h,
     Equal.cong(Nat, Nat, z => Spec.store_block(Nk, g, z), Impl.chan(Nk, g, h, t), x, ch),
     Equal.trans(Nat, Spec.store_block(Nk, g, x), Spec.last(g, k), h,
       gf_sbq(Nk, g, k, t, True{}, Nat.is_lt(k, Nat.add(Nk, Nk)), kl, {==}, ta),
       gf_last_h(g, h, gpos, own))),
   Equal.trans(Nat, Spec.store_thread(Nk, Impl.chan(Nk, g, h, t)), Spec.store_thread(Nk, x), t,
     Equal.cong(Nat, Nat, z => Spec.store_thread(Nk, z), Impl.chan(Nk, g, h, t), x, ch),
     gf_stq(Nk, k, t, True{}, Nat.is_lt(k, Nat.add(Nk, Nk)), kl, {==}, ta)),
   Equal.trans(Bool, Nat.is_lt(Impl.chan(Nk, g, h, t), Nat.mul(R, 128n)), Nat.is_lt(x, Nat.mul(R, 128n)), True{},
     Equal.cong(Nat, Bool, z => Nat.is_lt(z, Nat.mul(R, 128n)), Impl.chan(Nk, g, h, t), x, ch),
     gf_row_lt(k, t, R, 128n, Ar.ar_lt_le_trans(k, Nk, R, kl, gf_nk_le(Nk, Nat.mul(Nk, g))), ta)))

def gf_into_B(+Nk: Nat, +g: Nat, +h: Nat, +t: Nat, +gpos: {Nat.is_lt(0n, g) == True{} : Bool},
    +hr: {Nat.is_lt(h, Nat.mul(Nk, g)) == True{} : Bool}, +ta: {Nat.is_lt(t, 128n) == False{} : Bool},
    +tb: {Nat.is_lt(t, 256n) == True{} : Bool}, st: {Impl.store(g, h, t) == True{} : Bool})
  -> {Spec.store_block(Nk, g, Impl.chan(Nk, g, h, t)) == h : Nat} &
     {Spec.store_thread(Nk, Impl.chan(Nk, g, h, t)) == t : Nat} &
     {Nat.is_lt(Impl.chan(Nk, g, h, t), Spec.F(Nk, Nat.mul(Nk, g))) == True{} : Bool}:
  +k = Impl.kh(g, h)
  +j = Nat.add(Nk, k)
  +i = Nat.sub(t, 128n)
  +R = Nat.add(Nat.add(Nk, Nk), Nat.mul(Nk, g))
  +x = Nat.add(Nat.mul(j, 128n), i)
  +et = Ar.ar_sub_add(t, 128n, gf_le_of(t, 128n, ta))
  +ir = Ar.ard_rt(Nat.is_lt(Nat.add(128n, i), Nat.add(128n, 128n)), Nat.is_lt(i, 128n), Ar.ar_lt_add_cancel(128n, i, 128n),
    Equal.trans(Bool, Nat.is_lt(Nat.add(128n, i), 256n), Nat.is_lt(t, 256n), True{},
      Equal.cong(Nat, Bool, z => Nat.is_lt(z, 256n), Nat.add(128n, i), t, et), tb))
  +ch = gf_chanB(Nk, g, h, t, i, ta, tb, et)
  +kl = gf_kh_lt(Nk, g, h, gpos, hr)
  +jq = gf_lt_false(j, Nk, Ar.ar_le_add_r(Nk, k))
  +jk = Ar.ard_tr(Nat.is_lt(j, Nat.add(Nk, Nk)), Nat.is_lt(k, Nk), Ar.ar_lt_add_cancel(Nk, k, Nk), kl)
  +own = gf_store_owner(g, h, t, tb, st)
  (Equal.trans(Nat, Spec.store_block(Nk, g, Impl.chan(Nk, g, h, t)), Spec.store_block(Nk, g, x), h,
     Equal.cong(Nat, Nat, z => Spec.store_block(Nk, g, z), Impl.chan(Nk, g, h, t), x, ch),
     Equal.trans(Nat, Spec.store_block(Nk, g, x), Spec.last(g, Nat.sub(j, Nk)), h,
       gf_sbq(Nk, g, j, i, False{}, True{}, jq, jk, ir),
       Equal.trans(Nat, Spec.last(g, Nat.sub(j, Nk)), Spec.last(g, k), h,
         Equal.cong(Nat, Nat, z => Spec.last(g, z), Nat.sub(j, Nk), k, Ar.ar_add_sub(Nk, k)),
         gf_last_h(g, h, gpos, own)))),
   Equal.trans(Nat, Spec.store_thread(Nk, Impl.chan(Nk, g, h, t)), Spec.store_thread(Nk, x), t,
     Equal.cong(Nat, Nat, z => Spec.store_thread(Nk, z), Impl.chan(Nk, g, h, t), x, ch),
     Equal.trans(Nat, Spec.store_thread(Nk, x), Nat.add(128n, i), t,
       gf_stq(Nk, j, i, False{}, True{}, jq, jk, ir), et)),
   Equal.trans(Bool, Nat.is_lt(Impl.chan(Nk, g, h, t), Nat.mul(R, 128n)), Nat.is_lt(x, Nat.mul(R, 128n)), True{},
     Equal.cong(Nat, Bool, z => Nat.is_lt(z, Nat.mul(R, 128n)), Impl.chan(Nk, g, h, t), x, ch),
     gf_row_lt(j, i, R, 128n, Ar.ar_lt_le_trans(j, Nat.add(Nk, Nk), R, jk, gf_2nk_le(Nk, Nat.mul(Nk, g))), ir)))

def gf_into_C(+Nk: Nat, +g: Nat, +h: Nat, +t: Nat,
    +hr: {Nat.is_lt(h, Nat.mul(Nk, g)) == True{} : Bool}, +tr: {Nat.is_lt(t, 384n) == True{} : Bool},
    +ta: {Nat.is_lt(t, 128n) == False{} : Bool}, +tb: {Nat.is_lt(t, 256n) == False{} : Bool})
  -> {Spec.store_block(Nk, g, Impl.chan(Nk, g, h, t)) == h : Nat} &
     {Spec.store_thread(Nk, Impl.chan(Nk, g, h, t)) == t : Nat} &
     {Nat.is_lt(Impl.chan(Nk, g, h, t), Spec.F(Nk, Nat.mul(Nk, g))) == True{} : Bool}:
  +N2 = Nat.add(Nk, Nk)
  +j = Nat.add(N2, h)
  +i = Nat.sub(t, 256n)
  +R = Nat.add(N2, Nat.mul(Nk, g))
  +x = Nat.add(Nat.mul(j, 128n), i)
  +et = Ar.ar_sub_add(t, 256n, gf_le_of(t, 256n, tb))
  +ir = Ar.ard_rt(Nat.is_lt(Nat.add(256n, i), Nat.add(256n, 128n)), Nat.is_lt(i, 128n), Ar.ar_lt_add_cancel(256n, i, 128n),
    Equal.trans(Bool, Nat.is_lt(Nat.add(256n, i), 384n), Nat.is_lt(t, 384n), True{},
      Equal.cong(Nat, Bool, z => Nat.is_lt(z, 384n), Nat.add(256n, i), t, et), tr))
  +ch = gf_chanC(Nk, g, h, t, i, ta, tb, et)
  +jq = gf_lt_false(j, Nk, Ar.ar_le_trans(Nk, N2, j, Ar.ar_le_add_r(Nk, Nk), Ar.ar_le_add_r(N2, h)))
  +jk = gf_lt_false(j, N2, Ar.ar_le_add_r(N2, h))
  (Equal.trans(Nat, Spec.store_block(Nk, g, Impl.chan(Nk, g, h, t)), Spec.store_block(Nk, g, x), h,
     Equal.cong(Nat, Nat, z => Spec.store_block(Nk, g, z), Impl.chan(Nk, g, h, t), x, ch),
     Equal.trans(Nat, Spec.store_block(Nk, g, x), Nat.sub(j, N2), h,
       gf_sbq(Nk, g, j, i, False{}, False{}, jq, jk, ir), Ar.ar_add_sub(N2, h))),
   Equal.trans(Nat, Spec.store_thread(Nk, Impl.chan(Nk, g, h, t)), Spec.store_thread(Nk, x), t,
     Equal.cong(Nat, Nat, z => Spec.store_thread(Nk, z), Impl.chan(Nk, g, h, t), x, ch),
     Equal.trans(Nat, Spec.store_thread(Nk, x), Nat.add(256n, i), t,
       gf_stq(Nk, j, i, False{}, False{}, jq, jk, ir), et)),
   Equal.trans(Bool, Nat.is_lt(Impl.chan(Nk, g, h, t), Nat.mul(R, 128n)), Nat.is_lt(x, Nat.mul(R, 128n)), True{},
     Equal.cong(Nat, Bool, z => Nat.is_lt(z, Nat.mul(R, 128n)), Impl.chan(Nk, g, h, t), x, ch),
     gf_row_lt(j, i, R, 128n, Ar.ard_tr(Nat.is_lt(j, R), Nat.is_lt(h, Nat.mul(Nk, g)), Ar.ar_lt_add_cancel(N2, h, Nat.mul(Nk, g)), hr), ir)))

def gf_into(+Nk: Nat, +g: Nat, +h: Nat, +t: Nat, a: Bool, +ea: {Nat.is_lt(t, 128n) == a : Bool},
    b: Bool, +eb: {Nat.is_lt(t, 256n) == b : Bool}, +gpos: {Nat.is_lt(0n, g) == True{} : Bool},
    +hr: {Nat.is_lt(h, Nat.mul(Nk, g)) == True{} : Bool}, +tr: {Nat.is_lt(t, 384n) == True{} : Bool},
    st: {Impl.store(g, h, t) == True{} : Bool})
  -> {Spec.store_block(Nk, g, Impl.chan(Nk, g, h, t)) == h : Nat} &
     {Spec.store_thread(Nk, Impl.chan(Nk, g, h, t)) == t : Nat} &
     {Nat.is_lt(Impl.chan(Nk, g, h, t), Spec.F(Nk, Nat.mul(Nk, g))) == True{} : Bool}:
  match a:
    case True{}:
      gf_into_A(Nk, g, h, t, gpos, hr, ea, st)
    case False{}:
      match b:
        case True{}:
          gf_into_B(Nk, g, h, t, gpos, hr, ea, eb, st)
        case False{}:
          gf_into_C(Nk, g, h, t, hr, tr, ea, eb)

def Laws.conv_store_into(Nk, g, h, t, gpos, hr, tr, st):
  gf_into(Nk, g, h, t, Nat.is_lt(t, 128n), {==}, Nat.is_lt(t, 256n), {==}, gpos, hr, tr, st)

# ---- (5a, 5b) onto ----
def gf_sbc(+Nk: Nat, +g: Nat, +c: Nat, -isq: Bool, -isk: Bool,
    eq: {Nat.is_lt(Nat.div(c, 128n), Nk) == isq : Bool}, ek: {Nat.is_lt(Nat.div(c, 128n), Nat.add(Nk, Nk)) == isk : Bool})
  -> {Spec.store_block(Nk, g, c) == Spec.store_block.row(Nk, g, Nat.div(c, 128n), isq, isk) : Nat}:
  %eq : {Spec.store_block(Nk, g, c) == Spec.store_block.row(Nk, g, Nat.div(c, 128n), _, isk) : Nat}
  %ek : {Spec.store_block(Nk, g, c) == Spec.store_block.row(Nk, g, Nat.div(c, 128n), Nat.is_lt(Nat.div(c, 128n), Nk), _) : Nat}
  {==}

def gf_stc(+Nk: Nat, +c: Nat, -isq: Bool, -isk: Bool,
    eq: {Nat.is_lt(Nat.div(c, 128n), Nk) == isq : Bool}, ek: {Nat.is_lt(Nat.div(c, 128n), Nat.add(Nk, Nk)) == isk : Bool})
  -> {Spec.store_thread(Nk, c) == Nat.add(Spec.store_thread.base(isq, isk), Nat.mod(c, 128n)) : Nat}:
  %eq : {Spec.store_thread(Nk, c) == Nat.add(Spec.store_thread.base(_, isk), Nat.mod(c, 128n)) : Nat}
  %ek : {Spec.store_thread(Nk, c) == Nat.add(Spec.store_thread.base(Nat.is_lt(Nat.div(c, 128n), Nk), _), Nat.mod(c, 128n)) : Nat}
  {==}

def gf_onto_gen(+Nk: Nat, +g: Nat, +c: Nat, +hB: Nat, +tT: Nat,
    +hb: {Spec.store_block(Nk, g, c) == hB : Nat}, +ht: {Spec.store_thread(Nk, c) == tT : Nat},
    p1: {Nat.is_lt(hB, Nat.mul(Nk, g)) == True{} : Bool}, p2: {Nat.is_lt(tT, 384n) == True{} : Bool},
    p3: {Impl.store(g, hB, tT) == True{} : Bool}, p4: {Impl.chan(Nk, g, hB, tT) == c : Nat})
  -> {Nat.is_lt(Spec.store_block(Nk, g, c), Nat.mul(Nk, g)) == True{} : Bool} &
     {Nat.is_lt(Spec.store_thread(Nk, c), 384n) == True{} : Bool} &
     {Impl.store(g, Spec.store_block(Nk, g, c), Spec.store_thread(Nk, c)) == True{} : Bool} &
     {Impl.chan(Nk, g, Spec.store_block(Nk, g, c), Spec.store_thread(Nk, c)) == c : Nat}:
  +sb = Spec.store_block(Nk, g, c)
  +sc = Spec.store_thread(Nk, c)
  (Equal.trans(Bool, Nat.is_lt(sb, Nat.mul(Nk, g)), Nat.is_lt(hB, Nat.mul(Nk, g)), True{},
     Equal.cong(Nat, Bool, z => Nat.is_lt(z, Nat.mul(Nk, g)), sb, hB, hb), p1),
   Equal.trans(Bool, Nat.is_lt(sc, 384n), Nat.is_lt(tT, 384n), True{},
     Equal.cong(Nat, Bool, z => Nat.is_lt(z, 384n), sc, tT, ht), p2),
   Equal.trans(Bool, Impl.store(g, sb, sc), Impl.store(g, hB, sc), True{},
     Equal.cong(Nat, Bool, z => Impl.store(g, z, sc), sb, hB, hb),
     Equal.trans(Bool, Impl.store(g, hB, sc), Impl.store(g, hB, tT), True{},
       Equal.cong(Nat, Bool, z => Impl.store(g, hB, z), sc, tT, ht), p3)),
   Equal.trans(Nat, Impl.chan(Nk, g, sb, sc), Impl.chan(Nk, g, hB, sc), c,
     Equal.cong(Nat, Nat, z => Impl.chan(Nk, g, z, sc), sb, hB, hb),
     Equal.trans(Nat, Impl.chan(Nk, g, hB, sc), Impl.chan(Nk, g, hB, tT), c,
       Equal.cong(Nat, Nat, z => Impl.chan(Nk, g, hB, z), sc, tT, ht), p4)))

def gf_owner_store(+g: Nat, +h: Nat, +t: Nat, own: {Impl.owner(g, h) == True{} : Bool})
  -> {Impl.store(g, h, t) == True{} : Bool}:
  Equal.cong(Bool, Bool, b => Bool.or(b, Nat.is_le(256n, t)), Impl.owner(g, h), True{}, own)

def gf_onto(+Nk: Nat, +g: Nat, +c: Nat, a: Bool, +ea: {Nat.is_lt(Nat.div(c, 128n), Nk) == a : Bool},
    b: Bool, +eb: {Nat.is_lt(Nat.div(c, 128n), Nat.add(Nk, Nk)) == b : Bool},
    +gpos: {Nat.is_lt(0n, g) == True{} : Bool},
    +cr: {Nat.is_lt(c, Spec.F(Nk, Nat.mul(Nk, g))) == True{} : Bool})
  -> {Nat.is_lt(Spec.store_block(Nk, g, c), Nat.mul(Nk, g)) == True{} : Bool} &
     {Nat.is_lt(Spec.store_thread(Nk, c), 384n) == True{} : Bool} &
     {Impl.store(g, Spec.store_block(Nk, g, c), Spec.store_thread(Nk, c)) == True{} : Bool} &
     {Impl.chan(Nk, g, Spec.store_block(Nk, g, c), Spec.store_thread(Nk, c)) == c : Nat}:
  match a:
    case True{}:
      +j = Nat.div(c, 128n)
      +i = Nat.mod(c, 128n)
      +N2 = Nat.add(Nk, Nk)
      +R = Nat.add(N2, Nat.mul(Nk, g))
      +ir = Ar.ard_dm_lt(c, 128n, {==})
      +ce = Ar.ard_dm_eq(c, 128n, {==})
      +jR = Ar.ard_tr(Nat.is_lt(j, R), Nat.is_lt(c, Nat.mul(R, 128n)), Ar.ar_div_lt(R, c, 128n, {==}), cr)
      +hB = Spec.last(g, j)
      +khq = gf_fst({Impl.kh(g, hB) == j : Nat}, {Impl.owner(g, hB) == True{} : Bool}, gf_last_kh(g, j, gpos))
      +own = gf_snd({Impl.kh(g, hB) == j : Nat}, {Impl.owner(g, hB) == True{} : Bool}, gf_last_kh(g, j, gpos))
      gf_onto_gen(Nk, g, c, hB, i, gf_sbc(Nk, g, c, True{}, b, ea, eb), gf_stc(Nk, c, True{}, b, ea, eb),
        gf_last_lt(g, j, Nk, gpos, ea), Ar.ar_lt_trans(i, 128n, 384n, ir, {==}), gf_owner_store(g, hB, i, own),
        Equal.trans(Nat, Impl.chan(Nk, g, hB, i), Nat.add(Nat.mul(Impl.kh(g, hB), 128n), i), c,
          gf_chanA(Nk, g, hB, i, ir),
          Equal.trans(Nat, Nat.add(Nat.mul(Impl.kh(g, hB), 128n), i), Nat.add(Nat.mul(j, 128n), i), c,
            Equal.cong(Nat, Nat, z => Nat.add(Nat.mul(z, 128n), i), Impl.kh(g, hB), j, khq), ce)))
    case False{}:
      match b:
        case True{}:
          +j = Nat.div(c, 128n)
          +i = Nat.mod(c, 128n)
          +N2 = Nat.add(Nk, Nk)
          +R = Nat.add(N2, Nat.mul(Nk, g))
          +ir = Ar.ard_dm_lt(c, 128n, {==})
          +ce = Ar.ard_dm_eq(c, 128n, {==})
          +jR = Ar.ard_tr(Nat.is_lt(j, R), Nat.is_lt(c, Nat.mul(R, 128n)), Ar.ar_div_lt(R, c, 128n, {==}), cr)
          +k0 = Nat.sub(j, Nk)
          +jeq = Ar.ar_sub_add(j, Nk, gf_le_of(j, Nk, ea))
          +k0l = Ar.ard_rt(Nat.is_lt(Nat.add(Nk, k0), N2), Nat.is_lt(k0, Nk), Ar.ar_lt_add_cancel(Nk, k0, Nk),
            Equal.trans(Bool, Nat.is_lt(Nat.add(Nk, k0), N2), Nat.is_lt(j, N2), True{},
              Equal.cong(Nat, Bool, z => Nat.is_lt(z, N2), Nat.add(Nk, k0), j, jeq), eb))
          +hB = Spec.last(g, k0)
          +tT = Nat.add(128n, i)
          +khq = gf_fst({Impl.kh(g, hB) == k0 : Nat}, {Impl.owner(g, hB) == True{} : Bool}, gf_last_kh(g, k0, gpos))
          +own = gf_snd({Impl.kh(g, hB) == k0 : Nat}, {Impl.owner(g, hB) == True{} : Bool}, gf_last_kh(g, k0, gpos))
          +ta = gf_lt_false(tT, 128n, Ar.ar_le_add_r(128n, i))
          +tb = Ar.ard_tr(Nat.is_lt(tT, Nat.add(128n, 128n)), Nat.is_lt(i, 128n), Ar.ar_lt_add_cancel(128n, i, 128n), ir)
          gf_onto_gen(Nk, g, c, hB, tT, gf_sbc(Nk, g, c, False{}, True{}, ea, eb), gf_stc(Nk, c, False{}, True{}, ea, eb),
            gf_last_lt(g, k0, Nk, gpos, k0l),
            Ar.ard_tr(Nat.is_lt(tT, Nat.add(128n, 256n)), Nat.is_lt(i, 256n), Ar.ar_lt_add_cancel(128n, i, 256n),
              Ar.ar_lt_trans(i, 128n, 256n, ir, {==})),
            gf_owner_store(g, hB, tT, own),
            Equal.trans(Nat, Impl.chan(Nk, g, hB, tT), Nat.add(Nat.mul(Nat.add(Nk, Impl.kh(g, hB)), 128n), i), c,
              gf_chanB(Nk, g, hB, tT, i, ta, tb, {==}),
              Equal.trans(Nat, Nat.add(Nat.mul(Nat.add(Nk, Impl.kh(g, hB)), 128n), i), Nat.add(Nat.mul(Nat.add(Nk, k0), 128n), i), c,
                Equal.cong(Nat, Nat, z => Nat.add(Nat.mul(Nat.add(Nk, z), 128n), i), Impl.kh(g, hB), k0, khq),
                Equal.trans(Nat, Nat.add(Nat.mul(Nat.add(Nk, k0), 128n), i), Nat.add(Nat.mul(j, 128n), i), c,
                  Equal.cong(Nat, Nat, z => Nat.add(Nat.mul(z, 128n), i), Nat.add(Nk, k0), j, jeq), ce))))
        case False{}:
          +j = Nat.div(c, 128n)
          +i = Nat.mod(c, 128n)
          +N2 = Nat.add(Nk, Nk)
          +R = Nat.add(N2, Nat.mul(Nk, g))
          +ir = Ar.ard_dm_lt(c, 128n, {==})
          +ce = Ar.ard_dm_eq(c, 128n, {==})
          +jR = Ar.ard_tr(Nat.is_lt(j, R), Nat.is_lt(c, Nat.mul(R, 128n)), Ar.ar_div_lt(R, c, 128n, {==}), cr)
          +hB = Nat.sub(j, N2)
          +tT = Nat.add(256n, i)
          +jeq = Ar.ar_sub_add(j, N2, gf_le_of(j, N2, eb))
          +hl = Ar.ard_rt(Nat.is_lt(Nat.add(N2, hB), R), Nat.is_lt(hB, Nat.mul(Nk, g)), Ar.ar_lt_add_cancel(N2, hB, Nat.mul(Nk, g)),
            Equal.trans(Bool, Nat.is_lt(Nat.add(N2, hB), R), Nat.is_lt(j, R), True{},
              Equal.cong(Nat, Bool, z => Nat.is_lt(z, R), Nat.add(N2, hB), j, jeq), jR))
          +l256 = Ar.ar_le_add_r(256n, i)
          +ta = gf_lt_false(tT, 128n, Ar.ar_le_trans(128n, 256n, tT, {==}, l256))
          +tb = gf_lt_false(tT, 256n, l256)
          gf_onto_gen(Nk, g, c, hB, tT, gf_sbc(Nk, g, c, False{}, False{}, ea, eb), gf_stc(Nk, c, False{}, False{}, ea, eb),
            hl,
            Ar.ard_tr(Nat.is_lt(tT, Nat.add(256n, 128n)), Nat.is_lt(i, 128n), Ar.ar_lt_add_cancel(256n, i, 128n), ir),
            Equal.trans(Bool, Impl.store(g, hB, tT), Bool.or(Impl.owner(g, hB), True{}), True{},
              Equal.cong(Bool, Bool, z => Bool.or(Impl.owner(g, hB), z), Nat.is_le(256n, tT), True{}, l256),
              gf_or_true_r(Impl.owner(g, hB))),
            Equal.trans(Nat, Impl.chan(Nk, g, hB, tT), Nat.add(Nat.mul(Nat.add(N2, hB), 128n), i), c,
              gf_chanC(Nk, g, hB, tT, i, ta, tb, {==}),
              Equal.trans(Nat, Nat.add(Nat.mul(Nat.add(N2, hB), 128n), i), Nat.add(Nat.mul(j, 128n), i), c,
                Equal.cong(Nat, Nat, z => Nat.add(Nat.mul(z, 128n), i), Nat.add(N2, hB), j, jeq), ce)))

def Laws.conv_store_onto(Nk, g, c, gpos, cr):
  gf_onto(Nk, g, c, Nat.is_lt(Nat.div(c, 128n), Nk), {==}, Nat.is_lt(Nat.div(c, 128n), Nat.add(Nk, Nk)), {==}, gpos, cr)

# ---- (5d) v privacy ----
def gf_qk_small(+Nk: Nat, +g: Nat, +h: Nat, +t: Nat, a: Bool, +ea: {Nat.is_lt(t, 128n) == a : Bool},
    +gpos: {Nat.is_lt(0n, g) == True{} : Bool}, +hr: {Nat.is_lt(h, Nat.mul(Nk, g)) == True{} : Bool},
    +tb: {Nat.is_lt(t, 256n) == True{} : Bool})
  -> {Nat.is_lt(Impl.chan(Nk, g, h, t), Nat.mul(Nat.add(Nk, Nk), 128n)) == True{} : Bool}:
  match a:
    case True{}:
      +k = Impl.kh(g, h)
      +N2 = Nat.add(Nk, Nk)
      +kl = gf_kh_lt(Nk, g, h, gpos, hr)
      +x = Nat.add(Nat.mul(k, 128n), t)
      Equal.trans(Bool, Nat.is_lt(Impl.chan(Nk, g, h, t), Nat.mul(N2, 128n)), Nat.is_lt(x, Nat.mul(N2, 128n)), True{},
        Equal.cong(Nat, Bool, z => Nat.is_lt(z, Nat.mul(N2, 128n)), Impl.chan(Nk, g, h, t), x, gf_chanA(Nk, g, h, t, ea)),
        gf_row_lt(k, t, N2, 128n, Ar.ar_lt_le_trans(k, Nk, N2, kl, Ar.ar_le_add_r(Nk, Nk)), ea))
    case False{}:
      +k = Impl.kh(g, h)
      +N2 = Nat.add(Nk, Nk)
      +kl = gf_kh_lt(Nk, g, h, gpos, hr)
      +i = Nat.sub(t, 128n)
      +et = Ar.ar_sub_add(t, 128n, gf_le_of(t, 128n, ea))
      +ir = Ar.ard_rt(Nat.is_lt(Nat.add(128n, i), Nat.add(128n, 128n)), Nat.is_lt(i, 128n), Ar.ar_lt_add_cancel(128n, i, 128n),
        Equal.trans(Bool, Nat.is_lt(Nat.add(128n, i), 256n), Nat.is_lt(t, 256n), True{},
          Equal.cong(Nat, Bool, z => Nat.is_lt(z, 256n), Nat.add(128n, i), t, et), tb))
      +j = Nat.add(Nk, k)
      +x = Nat.add(Nat.mul(j, 128n), i)
      Equal.trans(Bool, Nat.is_lt(Impl.chan(Nk, g, h, t), Nat.mul(N2, 128n)), Nat.is_lt(x, Nat.mul(N2, 128n)), True{},
        Equal.cong(Nat, Bool, z => Nat.is_lt(z, Nat.mul(N2, 128n)), Impl.chan(Nk, g, h, t), x, gf_chanB(Nk, g, h, t, i, ea, tb, et)),
        gf_row_lt(j, i, N2, 128n, Ar.ard_tr(Nat.is_lt(j, N2), Nat.is_lt(k, Nk), Ar.ar_lt_add_cancel(Nk, k, Nk), kl), ir))

def gf_vp(+Nk: Nat, +g: Nat, +h: Nat, +t: Nat, b: Bool, +eb: {Nat.is_lt(t, 256n) == b : Bool},
    +gpos: {Nat.is_lt(0n, g) == True{} : Bool}, +hr: {Nat.is_lt(h, Nat.mul(Nk, g)) == True{} : Bool},
    +tr: {Nat.is_lt(t, 384n) == True{} : Bool},
    isv: {Nat.is_le(Nat.mul(Nat.add(Nk, Nk), 128n), Impl.chan(Nk, g, h, t)) == True{} : Bool})
  -> {Spec.store_block(Nk, g, Impl.chan(Nk, g, h, t)) == h : Nat}:
  match b:
    case True{}:
      +lt = gf_qk_small(Nk, g, h, t, Nat.is_lt(t, 128n), {==}, gpos, hr, eb)
      Empty.absurd({Spec.store_block(Nk, g, Impl.chan(Nk, g, h, t)) == h : Nat},
        Ar.ar_false_ne(Nat.is_le(Nat.mul(Nat.add(Nk, Nk), 128n), Impl.chan(Nk, g, h, t)),
          gf_le_false(Impl.chan(Nk, g, h, t), Nat.mul(Nat.add(Nk, Nk), 128n), lt), isv))
    case False{}:
      +st = Equal.trans(Bool, Impl.store(g, h, t), Bool.or(Impl.owner(g, h), True{}), True{},
        Equal.cong(Bool, Bool, z => Bool.or(Impl.owner(g, h), z), Nat.is_le(256n, t), True{}, gf_le_of(t, 256n, eb)),
        gf_or_true_r(Impl.owner(g, h)))
      gf_fst({Spec.store_block(Nk, g, Impl.chan(Nk, g, h, t)) == h : Nat},
        {Spec.store_thread(Nk, Impl.chan(Nk, g, h, t)) == t : Nat} &
        {Nat.is_lt(Impl.chan(Nk, g, h, t), Spec.F(Nk, Nat.mul(Nk, g))) == True{} : Bool},
        Laws.conv_store_into(Nk, g, h, t, gpos, hr, tr, st))

def Laws.conv_v_private(Nk, g, h, t, gpos, hr, tr, isv):
  gf_vp(Nk, g, h, t, Nat.is_lt(t, 256n), {==}, gpos, hr, tr, isv)

# ---- (5c) consumers, owner, counter ----
def gf_eq_sym(+a: Nat, +b: Nat) -> {Nat.is_eq(a, b) == Nat.is_eq(b, a) : Bool}:
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      {==}
    case 1n+p 0n:
      {==}
    case 1n+p 1n+q:
      gf_eq_sym(p, q)

def Laws.conv_qk_consumers(g, h, k0, gpos):
  +gp = gpos
  +q = Nat.div(h, g)
  +M = Nat.mul(k0, g)
  %Equal.sym(Nat, Nat.add(M, g), Nat.add(g, M), Ar.ar_add_comm(M, g)) : {Nat.is_eq(q, k0) == Bool.and(Nat.is_le(M, h), Nat.is_lt(h, _)) : Bool}
  %Ar.ar_div_lt(1n+k0, h, g, gp) : {Nat.is_eq(q, k0) == Bool.and(Nat.is_le(M, h), _) : Bool}
  %Equal.sym(Bool, Nat.is_le(M, h), Nat.is_le(k0, q), Ar.ar_div_le(k0, h, g, gp)) : {Nat.is_eq(q, k0) == Bool.and(_, Nat.is_lt(q, 1n+k0)) : Bool}
  %Ar.p8_eq_iv(k0, q) : {Nat.is_eq(q, k0) == _ : Bool}
  gf_eq_sym(q, k0)

def gf_ol_fwd(+g: Nat, +h: Nat, +k0: Nat, +gpos: {Nat.is_lt(0n, g) == True{} : Bool},
    +ha: {Bool.and(Impl.owner(g, h), Nat.is_eq(Impl.kh(g, h), k0)) == True{} : Bool})
  -> {Nat.is_eq(h, Spec.last(g, k0)) == True{} : Bool}:
  +own = Ar.ar_and_l(Impl.owner(g, h), Nat.is_eq(Impl.kh(g, h), k0), ha)
  +keq = Ar.ar_eq_true(Impl.kh(g, h), k0, Ar.ar_and_r(Impl.owner(g, h), Nat.is_eq(Impl.kh(g, h), k0), ha))
  Ar.p8_eq_of(h, Spec.last(g, k0),
    Equal.trans(Nat, h, Spec.last(g, Impl.kh(g, h)), Spec.last(g, k0),
      Equal.sym(Nat, Spec.last(g, Impl.kh(g, h)), h, gf_last_h(g, h, gpos, own)),
      Equal.cong(Nat, Nat, z => Spec.last(g, z), Impl.kh(g, h), k0, keq)))

def gf_ol_bwd(+g: Nat, +h: Nat, +k0: Nat, +gpos: {Nat.is_lt(0n, g) == True{} : Bool},
    hb: {Nat.is_eq(h, Spec.last(g, k0)) == True{} : Bool})
  -> {Bool.and(Impl.owner(g, h), Nat.is_eq(Impl.kh(g, h), k0)) == True{} : Bool}:
  +L = Spec.last(g, k0)
  +e = Ar.ar_eq_true(h, L, hb)
  +khq = gf_fst({Impl.kh(g, L) == k0 : Nat}, {Impl.owner(g, L) == True{} : Bool}, gf_last_kh(g, k0, gpos))
  +own = gf_snd({Impl.kh(g, L) == k0 : Nat}, {Impl.owner(g, L) == True{} : Bool}, gf_last_kh(g, k0, gpos))
  %Equal.sym(Nat, h, L, e) : {Bool.and(Impl.owner(g, _), Nat.is_eq(Impl.kh(g, _), k0)) == True{} : Bool}
  %Equal.sym(Bool, Impl.owner(g, L), True{}, own) : {Bool.and(_, Nat.is_eq(Impl.kh(g, L), k0)) == True{} : Bool}
  %Equal.sym(Nat, Impl.kh(g, L), k0, khq) : {Bool.and(True{}, Nat.is_eq(_, k0)) == True{} : Bool}
  Ar.ar_eq_refl(k0)

def Laws.conv_owner_last(g, h, k0, gpos):
  +gp = gpos
  Ar.ar_bool_ext(Bool.and(Impl.owner(g, h), Nat.is_eq(Impl.kh(g, h), k0)), Nat.is_eq(h, Spec.last(g, k0)),
    x => gf_ol_fwd(g, h, k0, gp, x), y => gf_ol_bwd(g, h, k0, gp, y))

def gf_lm(+g: Nat, +h: Nat, +k0: Nat, gpos: {Nat.is_lt(0n, g) == True{} : Bool},
    hc: {Spec.consumes(g, k0, h) == True{} : Bool})
  -> {Nat.is_le(h, Spec.last(g, k0)) == True{} : Bool} & {Spec.consumes(g, k0, Spec.last(g, k0)) == True{} : Bool}:
  match g:
    case 0n:
      Empty.absurd({Nat.is_le(h, Spec.last(0n, k0)) == True{} : Bool} & {Spec.consumes(0n, k0, Spec.last(0n, k0)) == True{} : Bool},
        Ar.ar_false_true(gpos))
    case 1n+gp:
      +g1 = {1n+gp : Nat}
      +M = Nat.mul(k0, g1)
      +hl = Ar.ar_and_r(Nat.is_le(M, h), Nat.is_lt(h, Nat.add(M, g1)), hc)
      +hl2 = Equal.trans(Bool, Nat.is_lt(h, 1n+Nat.add(M, gp)), Nat.is_lt(h, Nat.add(M, g1)), True{},
        Equal.cong(Nat, Bool, z => Nat.is_lt(h, z), 1n+Nat.add(M, gp), Nat.add(M, g1),
          Equal.sym(Nat, Nat.add(M, g1), 1n+Nat.add(M, gp), Ar.ar_add_succ_r(M, gp))), hl)
      +le = Ar.ard_rt(Nat.is_lt(h, 1n+Nat.add(M, gp)), Nat.is_le(1n+h, 1n+Nat.add(M, gp)), Ar.ar_lt_succ_le(h, 1n+Nat.add(M, gp)), hl2)
      +sz = Ar.ar_sub_zero_r(gp)
      (Equal.trans(Bool, Nat.is_le(h, Nat.add(M, Nat.sub(gp, 0n))), Nat.is_le(h, Nat.add(M, gp)), True{},
         Equal.cong(Nat, Bool, z => Nat.is_le(h, Nat.add(M, z)), Nat.sub(gp, 0n), gp, sz), le),
       Equal.trans(Bool, Spec.consumes(g1, k0, Spec.last(g1, k0)), Spec.consumes(g1, k0, Nat.add(M, gp)), True{},
         Equal.cong(Nat, Bool, z => Spec.consumes(g1, k0, Nat.add(M, z)), Nat.sub(gp, 0n), gp, sz),
         Ar.ar_and_intro(Nat.is_le(M, Nat.add(M, gp)), Nat.is_lt(Nat.add(M, gp), Nat.add(M, g1)),
           Ar.ar_le_add_r(M, gp),
           Ar.ard_rt(Nat.is_lt(gp, g1), Nat.is_lt(Nat.add(M, gp), Nat.add(M, g1)),
             Equal.sym(Bool, Nat.is_lt(Nat.add(M, gp), Nat.add(M, g1)), Nat.is_lt(gp, g1), Ar.ar_lt_add_cancel(M, gp, g1)),
             gf_lt_succ(gp)))))

def Laws.conv_last_max(g, h, k0, gpos, hc):
  gf_lm(g, h, k0, gpos, hc)

def gf_ctr_ok(+k0: Nat) -> Bool:
  Bool.and(Nat.is_eq(Impl.ctr_adds(Nat.mul(16n, 3n), 3n, k0), 2n), Nat.is_eq(Impl.owners(Nat.mul(16n, 3n), 3n, k0), 1n))

def gf_ctr_ok_w(x: Nat) -> Bool:
  gf_ctr_ok(x)

def gf_ctr_all() -> {gf_all(~gf_ctr_ok_w, 16n) == True{} : Bool}:
  {==}

def gf_ctr(+k0: Nat, +kr: {Nat.is_lt(k0, 16n) == True{} : Bool})
  -> {Impl.ctr_adds(Nat.mul(16n, 3n), 3n, k0) == 2n : Nat} & {Impl.owners(Nat.mul(16n, 3n), 3n, k0) == 1n : Nat}:
  +ok = gf_all_get(~gf_ctr_ok_w, 16n, k0, gf_ctr_all(), kr)
  +a = Impl.ctr_adds(Nat.mul(16n, 3n), 3n, k0)
  +o = Impl.owners(Nat.mul(16n, 3n), 3n, k0)
  (Ar.ar_eq_true(a, 2n, Ar.ar_and_l(Nat.is_eq(a, 2n), Nat.is_eq(o, 1n), ok)),
   Ar.ar_eq_true(o, 1n, Ar.ar_and_r(Nat.is_eq(a, 2n), Nat.is_eq(o, 1n), ok)))

def Laws.conv_ctr_adds(k0, kr):
  gf_ctr(k0, kr)

def gf_wait2(a: Nat, ar: {Nat.is_le(a, 2n) == True{} : Bool})
  -> {Impl.wait_exits(3n, a) == Nat.is_eq(a, 2n) : Bool}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      match p:
        case 0n:
          {==}
        case 1n+q:
          match q:
            case 0n:
              {==}
            case 1n+r:
              Empty.absurd({Impl.wait_exits(3n, 1n+1n+1n+r) == Nat.is_eq(1n+1n+1n+r, 2n) : Bool}, Ar.ar_false_true(ar))

def gf_cw1(+a: Nat, +n: Nat, e: {n == 2n : Nat}, a2: {Nat.is_le(a, 2n) == True{} : Bool})
  -> {Impl.wait_exits(3n, a) == Nat.is_eq(a, n) : Bool}:
  %Equal.sym(Nat, n, 2n, e) : {Impl.wait_exits(3n, a) == Nat.is_eq(a, _) : Bool}
  gf_wait2(a, a2)

def gf_cw2(+n: Nat, e: {n == 2n : Nat}) -> {Impl.ctr_end(3n, n) == 0n : Nat}:
  %Equal.sym(Nat, n, 2n, e) : {Impl.ctr_end(3n, _) == 0n : Nat}
  {==}

def Laws.conv_ctr_wait(k0, a, kr, ar):
  +e = gf_fst({Impl.ctr_adds(Nat.mul(16n, 3n), 3n, k0) == 2n : Nat}, {Impl.owners(Nat.mul(16n, 3n), 3n, k0) == 1n : Nat}, gf_ctr(k0, kr))
  +a2 = Equal.trans(Bool, Nat.is_le(a, 2n), Nat.is_le(a, Impl.ctr_adds(Nat.mul(16n, 3n), 3n, k0)), True{},
    Equal.cong(Nat, Bool, z => Nat.is_le(a, z), 2n, Impl.ctr_adds(Nat.mul(16n, 3n), 3n, k0),
      Equal.sym(Nat, Impl.ctr_adds(Nat.mul(16n, 3n), 3n, k0), 2n, e)), ar)
  (gf_cw1(a, Impl.ctr_adds(Nat.mul(16n, 3n), 3n, k0), e, a2),
   gf_cw2(Impl.ctr_adds(Nat.mul(16n, 3n), 3n, k0), e))
