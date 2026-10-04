# Proofs of the ext 5108 loader laws (gdn_replay_gather_laws.bend) about the Bend model gdn_replay_gather.bend.
# Structural evidence only: no unsafe definitions, axioms, holes or foreign proof code. The finite coverage and
# thread facts are decided by the checker's evaluation; the address and lane laws are proven for symbolic
# operands from the Nat library of gemm_m16_partition_proof.bend.
import Base
import ./gdn_replay_gather.bend as Impl
import ./gdn_replay_gather_spec.bend as Spec
import ./gdn_replay_gather_laws.bend as Laws
import ./gemm_m16_partition_proof.bend as Ar

# The finite checks are decided by evaluation under a let: a non-empty context makes the checker's evaluation
# lazy (no normal form of every Data argument at each binding), which costs about a tenth of the closed one.
def Laws.gather_cover():
  +lazy : Nat = 0n
  {==}

def Laws.gather_threads():
  +lazy : Nat = 0n
  {==}

# ==== gather_all_S: the checks at S' + 1 tokens imply those at S', so the largest S' decides them all ====

# a && b, from both.
def ga_and_i(a: Bool, -b: Bool, ha: {a == True{} : Bool}, +hb: {b == True{} : Bool})
  -> {Bool.and(a, b) == True{} : Bool}:
  match a:
    case False{}:
      Empty.absurd({Bool.and(False{}, b) == True{} : Bool}, Ar.ar_false_true(ha))
    case True{}:
      hb

# a && b == b when b implies a.
def ga_absorb(a: Bool, b: Bool, h: {b == True{} : Bool} -> {a == True{} : Bool})
  -> {Bool.and(a, b) == b : Bool}:
  match a:
    case True{}:
      {==}
    case False{}:
      match b:
        case True{}:
          Empty.absurd({False{} == True{} : Bool}, Ar.ar_false_true(h({==})))
        case False{}:
          {==}

def ga_and_true_r(a: Bool) -> {Bool.and(a, True{}) == a : Bool}:
  match a:
    case True{}:
      {==}
    case False{}:
      {==}

# e + (1 + n) <= B gives e < B and (1 + e) + n <= B.
def ga_lt_add(+e: Nat, +n: Nat, +B: Nat, +h: {Nat.is_le(Nat.add(e, 1n+n), B) == True{} : Bool})
  -> {Nat.is_lt(e, B) == True{} : Bool}:
  +h1 = Ar.ard_rt(Nat.is_le(Nat.add(e, 1n+n), B), Nat.is_le(1n+Nat.add(e, n), B),
    Equal.cong(Nat, Bool, z => Nat.is_le(z, B), Nat.add(e, 1n+n), 1n+Nat.add(e, n), Ar.ar_add_succ_r(e, n)), h)
  +h2 = Ar.ard_tr(Nat.is_lt(Nat.add(e, n), B), Nat.is_le(1n+Nat.add(e, n), B), Ar.ar_lt_succ_le(Nat.add(e, n), B), h1)
  Ar.ar_le_lt_trans(e, Nat.add(e, n), B, Ar.ar_le_add_r(e, n), h2)

def ga_le_step(+e: Nat, +n: Nat, +B: Nat, +h: {Nat.is_le(Nat.add(e, 1n+n), B) == True{} : Bool})
  -> {Nat.is_le(Nat.add(1n+e, n), B) == True{} : Bool}:
  Ar.ard_rt(Nat.is_le(Nat.add(e, 1n+n), B), Nat.is_le(1n+Nat.add(e, n), B),
    Equal.cong(Nat, Bool, z => Nat.is_le(z, B), Nat.add(e, 1n+n), 1n+Nat.add(e, n), Ar.ar_add_succ_r(e, n)), h)

# 20 s + x < 20 K for s < K, x < 20.
def ga_row_lt(+s: Nat, +x: Nat, +K: Nat, +hs: {Nat.is_lt(s, K) == True{} : Bool}, hx: {Nat.is_lt(x, 20n) == True{} : Bool})
  -> {Nat.is_lt(Nat.add(Nat.mul(s, 20n), x), Nat.mul(K, 20n)) == True{} : Bool}:
  +M = Nat.mul(s, 20n)
  +s1 = Ar.ard_tr(Nat.is_lt(Nat.add(M, x), Nat.add(M, 20n)), Nat.is_lt(x, 20n), Ar.ar_lt_add_cancel(M, x, 20n), hx)
  +s2 = Equal.trans(Bool, Nat.is_lt(Nat.add(M, x), Nat.add(20n, M)), Nat.is_lt(Nat.add(M, x), Nat.add(M, 20n)), True{},
    Equal.cong(Nat, Bool, z => Nat.is_lt(Nat.add(M, x), z), Nat.add(20n, M), Nat.add(M, 20n), Ar.ar_add_comm(20n, M)), s1)
  +le = Ar.ar_le_mul_mono(1n+s, K, 20n, Ar.ard_rt(Nat.is_lt(s, K), Nat.is_le(1n+s, K), Ar.ar_lt_succ_le(s, K), hs))
  Ar.ar_lt_le_trans(Nat.add(M, x), Nat.add(20n, M), Nat.mul(K, 20n), s2, le)

# 20 s + x == e < 20 K gives s < K.
def ga_lt_of(+s: Nat, +x: Nat, +K: Nat, +e: Nat, +ho: {Nat.add(Nat.mul(s, 20n), x) == e : Nat},
    +eb: {Nat.is_lt(e, Nat.mul(K, 20n)) == True{} : Bool}) -> {Nat.is_lt(s, K) == True{} : Bool}:
  +M = Nat.mul(s, 20n)
  +le = Ar.ard_rt(Nat.is_le(M, Nat.add(M, x)), Nat.is_le(M, e),
    Equal.cong(Nat, Bool, z => Nat.is_le(M, z), Nat.add(M, x), e, ho), Ar.ar_le_add_r(M, x))
  +lt = Ar.ar_le_lt_trans(M, e, Nat.mul(K, 20n), le, eb)
  Ar.ard_rt(Nat.is_lt(Nat.mul(s, 20n), Nat.mul(K, 20n)), Nat.is_lt(s, K), Ar.ar_mul_lt_cancel(s, K, 20n, {==}), lt)

# The owner of a chunk of token s is 20 s + x for some x (own_k / own_v), so owner e < 20 K puts s below K.
def ga_tok(+part: Nat, +s: Nat, +col: Nat, +c: Nat, +e: Nat, +K: Nat,
    +ho: {Spec.own_loc(c, Impl.Loc{part, s, col}) == e : Nat}, +eb: {Nat.is_lt(e, Nat.mul(K, 20n)) == True{} : Bool})
  -> {Nat.is_lt(s, K) == True{} : Bool}:
  match part:
    case 0n:
      ga_lt_of(s, Nat.add(16n, Nat.div(Nat.sub(col, Nat.mul(c, 32n)), 8n)), K, e, ho, eb)
    case 1n+z:
      match z:
        case 0n:
          ga_lt_of(s, Nat.div(col, 8n), K, e, ho, eb)
        case 1n+w:
          ga_lt_of(s, Nat.add(16n, Nat.div(Nat.sub(col, Nat.mul(c, 32n)), 8n)), K, e, ho, eb)

# the S-independent part of in_read (Spec.in_read's second conjunct)
def ga_r(+c: Nat, +part: Nat, +col: Nat) -> Bool:
  Bool.or(Bool.and(Nat.is_eq(part, 1n), Bool.and(Nat.is_lt(col, 128n), Nat.is_eq(Nat.mod(col, 8n), 0n))),
    Bool.and(Nat.is_eq(part, 2n),
      Bool.and(Nat.is_le(Nat.mul(c, 32n), col),
        Bool.and(Nat.is_lt(Nat.sub(col, Nat.mul(c, 32n)), 32n), Nat.is_eq(Nat.mod(Nat.sub(col, Nat.mul(c, 32n)), 8n), 0n)))))

# into element at 1 + K tokens, e < 20 K: the same element at K tokens.
def ga_in_el(+l: Impl.Loc, +K: Nat, +c: Nat, +e: Nat, +eb: {Nat.is_lt(e, Nat.mul(K, 20n)) == True{} : Bool},
    +h: {Bool.and(Spec.in_read(1n+K, c, l), Nat.is_eq(Spec.own_loc(c, l), e)) == True{} : Bool})
  -> {Bool.and(Spec.in_read(K, c, l), Nat.is_eq(Spec.own_loc(c, l), e)) == True{} : Bool}:
  match l:
    case Impl.Loc{+part, +s, +col}:
      +hh = h
      +L : Impl.Loc = Impl.Loc{part, s, col}
      +hA = Ar.ar_and_l(Spec.in_read(1n+K, c, L), Nat.is_eq(Spec.own_loc(c, L), e), hh)
      +hR = Ar.ar_and_r(Nat.is_lt(s, 1n+K), ga_r(c, part, col), hA)
      +hE = Ar.ar_and_r(Spec.in_read(1n+K, c, L), Nat.is_eq(Spec.own_loc(c, L), e), hh)
      +hO = Ar.ar_eq_true(Spec.own_loc(c, L), e, hE)
      +lt = ga_tok(part, s, col, c, e, K, hO, eb)
      ga_and_i(Spec.in_read(K, c, L), Nat.is_eq(Spec.own_loc(c, L), e),
        ga_and_i(Nat.is_lt(s, K), ga_r(c, part, col), lt, hR), hE)

# into.go over the first n of d + n elements from e at 1 + K tokens, e + n <= 20 K: the n at K tokens.
def ga_in_pre(+n: Nat, +d: Nat, +e: Nat, +K: Nat, +c: Nat,
    +hb: {Nat.is_le(Nat.add(e, n), Nat.mul(K, 20n)) == True{} : Bool},
    +h: {Spec.into.go(Nat.add(d, n), e, 1n+K, c) == True{} : Bool})
  -> {Spec.into.go(n, e, K, c) == True{} : Bool}:
  match n:
    case 0n:
      {==}
    case 1n+m:
      +h2 = Ar.ard_rt(Spec.into.go(Nat.add(d, 1n+m), e, 1n+K, c), Spec.into.go(1n+Nat.add(d, m), e, 1n+K, c),
        Equal.cong(Nat, Bool, z => Spec.into.go(z, e, 1n+K, c), Nat.add(d, 1n+m), 1n+Nat.add(d, m), Ar.ar_add_succ_r(d, m)), h)
      +D = Impl.dst(c, e)
      +el = Ar.ar_and_l(Bool.and(Spec.in_read(1n+K, c, D), Nat.is_eq(Spec.own_loc(c, D), e)),
        Spec.into.go(Nat.add(d, m), 1n+e, 1n+K, c), h2)
      +rest = Ar.ar_and_r(Bool.and(Spec.in_read(1n+K, c, D), Nat.is_eq(Spec.own_loc(c, D), e)),
        Spec.into.go(Nat.add(d, m), 1n+e, 1n+K, c), h2)
      +eb = ga_lt_add(e, m, Nat.mul(K, 20n), hb)
      ga_and_i(Bool.and(Spec.in_read(K, c, D), Nat.is_eq(Spec.own_loc(c, D), e)), Spec.into.go(m, 1n+e, K, c),
        ga_in_el(D, K, c, e, eb, el),
        ga_in_pre(m, d, 1n+e, K, c, ga_le_step(e, m, Nat.mul(K, 20n), hb), rest))

# k chunks kc .. kc + n - 1 (kc + n <= 20) of token s < K: owners active at K, chunk tests unchanged.
def ga_ok(+n: Nat, +kc: Nat, +s: Nat, +K: Nat, +c: Nat, +hs: {Nat.is_lt(s, K) == True{} : Bool},
    +hk: {Nat.is_le(Nat.add(kc, n), 20n) == True{} : Bool}, +h: {Spec.onto_k.go(n, kc, s, 1n+K, c) == True{} : Bool})
  -> {Spec.onto_k.go(n, kc, s, K, c) == True{} : Bool}:
  match n:
    case 0n:
      {==}
    case 1n+m:
      +hh = h
      +o = Spec.own_k(s, kc)
      +Q = Spec.loc_eq(Impl.dst(c, o), Spec.kchunk(s, kc))
      +el = Ar.ar_and_l(Bool.and(Impl.active(1n+K, o), Q), Spec.onto_k.go(m, 1n+kc, s, 1n+K, c), hh)
      +lq = Ar.ar_and_r(Impl.active(1n+K, o), Q, el)
      +rest = Ar.ar_and_r(Bool.and(Impl.active(1n+K, o), Q), Spec.onto_k.go(m, 1n+kc, s, 1n+K, c), hh)
      +act = ga_row_lt(s, kc, K, hs, ga_lt_add(kc, m, 20n, hk))
      ga_and_i(Bool.and(Impl.active(K, o), Q), Spec.onto_k.go(m, 1n+kc, s, K, c),
        ga_and_i(Impl.active(K, o), Q, act, lq),
        ga_ok(m, 1n+kc, s, K, c, hs, ga_le_step(kc, m, 20n, hk), rest))

# v chunks vc .. vc + n - 1 (vc + n <= 4) of token s < K: owners 20 s + 16 + vc active at K.
def ga_ov(+n: Nat, +vc: Nat, +s: Nat, +K: Nat, +c: Nat, +hs: {Nat.is_lt(s, K) == True{} : Bool},
    +hk: {Nat.is_le(Nat.add(vc, n), 4n) == True{} : Bool}, +h: {Spec.onto_v.go(n, vc, s, 1n+K, c) == True{} : Bool})
  -> {Spec.onto_v.go(n, vc, s, K, c) == True{} : Bool}:
  match n:
    case 0n:
      {==}
    case 1n+m:
      +hh = h
      +o = Spec.own_v(s, vc)
      +Q = Spec.loc_eq(Impl.dst(c, o), Spec.vchunk(c, s, vc))
      +el = Ar.ar_and_l(Bool.and(Impl.active(1n+K, o), Q), Spec.onto_v.go(m, 1n+vc, s, 1n+K, c), hh)
      +lq = Ar.ar_and_r(Impl.active(1n+K, o), Q, el)
      +rest = Ar.ar_and_r(Bool.and(Impl.active(1n+K, o), Q), Spec.onto_v.go(m, 1n+vc, s, 1n+K, c), hh)
      +act = ga_row_lt(s, Nat.add(16n, vc), K, hs, ga_lt_add(vc, m, 4n, hk))
      ga_and_i(Bool.and(Impl.active(K, o), Q), Spec.onto_v.go(m, 1n+vc, s, K, c),
        ga_and_i(Impl.active(K, o), Q, act, lq),
        ga_ov(m, 1n+vc, s, K, c, hs, ga_le_step(vc, m, 4n, hk), rest))

# onto.go over the first n of d + n tokens from s at 1 + K tokens, s + n <= K: the n at K tokens.
def ga_on_pre(+n: Nat, +d: Nat, +s: Nat, +K: Nat, +c: Nat, +hb: {Nat.is_le(Nat.add(s, n), K) == True{} : Bool},
    +h: {Spec.onto.go(Nat.add(d, n), s, 1n+K, c) == True{} : Bool})
  -> {Spec.onto.go(n, s, K, c) == True{} : Bool}:
  match n:
    case 0n:
      {==}
    case 1n+m:
      +h2 = Ar.ard_rt(Spec.onto.go(Nat.add(d, 1n+m), s, 1n+K, c), Spec.onto.go(1n+Nat.add(d, m), s, 1n+K, c),
        Equal.cong(Nat, Bool, z => Spec.onto.go(z, s, 1n+K, c), Nat.add(d, 1n+m), 1n+Nat.add(d, m), Ar.ar_add_succ_r(d, m)), h)
      +Ok = Spec.onto_k.go(16n, 0n, s, 1n+K, c)
      +Ov = Spec.onto_v.go(4n, 0n, s, 1n+K, c)
      +blk = Ar.ar_and_l(Bool.and(Ok, Ov), Spec.onto.go(Nat.add(d, m), 1n+s, 1n+K, c), h2)
      +rest = Ar.ar_and_r(Bool.and(Ok, Ov), Spec.onto.go(Nat.add(d, m), 1n+s, 1n+K, c), h2)
      +hs = ga_lt_add(s, m, K, hb)
      +Nk = Spec.onto_k.go(16n, 0n, s, K, c)
      +Nv = Spec.onto_v.go(4n, 0n, s, K, c)
      ga_and_i(Bool.and(Nk, Nv), Spec.onto.go(m, 1n+s, K, c),
        ga_and_i(Nk, Nv, ga_ok(16n, 0n, s, K, c, hs, {==}, Ar.ar_and_l(Ok, Ov, blk)),
          ga_ov(4n, 0n, s, K, c, hs, {==}, Ar.ar_and_r(Ok, Ov, blk))),
        ga_on_pre(m, d, 1n+s, K, c, ga_le_step(s, m, K, hb), rest))

# all_c at 1 + K tokens implies all_c at K tokens.
def ga_allc(+n: Nat, +c: Nat, +K: Nat, +h: {Spec.all_c(n, c, 1n+K) == True{} : Bool})
  -> {Spec.all_c(n, c, K) == True{} : Bool}:
  match n:
    case 0n:
      {==}
    case 1n+m:
      +hh = h
      +I1 = Spec.into(1n+K, c)
      +O1 = Spec.onto(1n+K, c)
      +b = Ar.ar_and_l(Bool.and(I1, O1), Spec.all_c(m, 1n+c, 1n+K), hh)
      +i0 = ga_in_pre(Nat.mul(K, Impl.parts()), Impl.parts(), 0n, K, c, Ar.ar_le_refl(Nat.mul(K, 20n)),
        Ar.ar_and_l(I1, O1, b))
      +o0 = ga_on_pre(K, 1n, 0n, K, c, Ar.ar_le_refl(K), Ar.ar_and_r(I1, O1, b))
      ga_and_i(Bool.and(Spec.into(K, c), Spec.onto(K, c)), Spec.all_c(m, 1n+c, K),
        ga_and_i(Spec.into(K, c), Spec.onto(K, c), i0, o0),
        ga_allc(m, 1n+c, K, Ar.ar_and_r(Bool.and(I1, O1), Spec.all_c(m, 1n+c, 1n+K), hh)))

# all_c at S + d tokens implies all_c at S tokens.
def ga_allc_le(+d: Nat, +S: Nat, +h: {Spec.all_c(4n, 0n, Nat.add(S, d)) == True{} : Bool})
  -> {Spec.all_c(4n, 0n, S) == True{} : Bool}:
  match d:
    case 0n:
      Ar.ard_rt(Spec.all_c(4n, 0n, Nat.add(S, 0n)), Spec.all_c(4n, 0n, S),
        Equal.cong(Nat, Bool, z => Spec.all_c(4n, 0n, z), Nat.add(S, 0n), S, Ar.ar_add_zero_r(S)), h)
    case 1n+k:
      +h1 = Ar.ard_rt(Spec.all_c(4n, 0n, Nat.add(S, 1n+k)), Spec.all_c(4n, 0n, 1n+Nat.add(S, k)),
        Equal.cong(Nat, Bool, z => Spec.all_c(4n, 0n, z), Nat.add(S, 1n+k), 1n+Nat.add(S, k), Ar.ar_add_succ_r(S, k)), h)
      ga_allc_le(k, S, ga_allc(4n, 0n, Nat.add(S, k), h1))

def ga_all(+n: Nat, +S: Nat) -> {Spec.all_S(n, S) == Spec.all_S_ref(n, S) : Bool}:
  match n:
    case 0n:
      {==}
    case 1n+m:
      match m:
        case 0n:
          +A = Spec.all_c(4n, 0n, S)
          Equal.trans(Bool, Spec.all_c(4n, 0n, Nat.add(S, 0n)), A, Bool.and(A, True{}),
            Equal.cong(Nat, Bool, z => Spec.all_c(4n, 0n, z), Nat.add(S, 0n), S, Ar.ar_add_zero_r(S)),
            Equal.sym(Bool, Bool.and(A, True{}), A, ga_and_true_r(A)))
        case 1n+k:
          +A = Spec.all_c(4n, 0n, S)
          +X = Spec.all_c(4n, 0n, Nat.add(1n+S, k))
          +R = Spec.all_S_ref(1n+k, 1n+S)
          +ih = ga_all(1n+k, 1n+S)
          +l1 = Equal.cong(Nat, Bool, z => Spec.all_c(4n, 0n, z), Nat.add(S, 1n+k), 1n+Nat.add(S, k), Ar.ar_add_succ_r(S, k))
          +ab = ga_absorb(A, X, hx => ga_allc_le(1n+k, S,
            Ar.ard_tr(Spec.all_c(4n, 0n, Nat.add(S, 1n+k)), X, l1, hx)))
          +l3 = Equal.cong(Bool, Bool, z => Bool.and(A, z), X, R, ih)
          Equal.trans(Bool, Spec.all_c(4n, 0n, Nat.add(S, 1n+k)), X, Bool.and(A, R), l1,
            Equal.trans(Bool, X, Bool.and(A, X), Bool.and(A, R), Equal.sym(Bool, Bool.and(A, X), X, ab), l3))

def Laws.gather_all_S(n, S):
  ga_all(n, S)

def Laws.gather_widen(b0, b1):
  ({==}, {==})

def Laws.gather_knob():
  ({==}, {==}, {==}, {==}, {==}, {==})

def Laws.chunk_lanes(x, n, xr):
  (Ar.ard_tr(Nat.is_lt(Nat.div(x, 8n), n), Nat.is_lt(x, Nat.mul(n, 8n)), Ar.ar_div_lt(n, x, 8n, {==}), xr),
   Ar.ard_dm_lt(x, 8n, {==}),
   Ar.ard_dm_eq(x, 8n, {==}))

# k: ((sF + (K + P)) + q) == sF + (K + (P + q))
def rg_k(+SF: Nat, +K: Nat, +P: Nat, +q: Nat)
  -> {Nat.add(Nat.add(SF, Nat.add(K, P)), q) == Nat.add(SF, Nat.add(K, Nat.add(P, q))) : Nat}:
  Equal.trans(Nat, Nat.add(Nat.add(SF, Nat.add(K, P)), q), Nat.add(SF, Nat.add(Nat.add(K, P), q)),
    Nat.add(SF, Nat.add(K, Nat.add(P, q))),
    Ar.ar_add_assoc(SF, Nat.add(K, P), q),
    Equal.cong(Nat, Nat, z => Nat.add(SF, z), Nat.add(Nat.add(K, P), q), Nat.add(K, Nat.add(P, q)), Ar.ar_add_assoc(K, P, q)))

# v: ((sF + ((B + C) + P)) + q) == sF + (B + ((C + P) + q))
def rg_v(+SF: Nat, +Bh: Nat, +C: Nat, +P: Nat, +q: Nat)
  -> {Nat.add(Nat.add(SF, Nat.add(Nat.add(Bh, C), P)), q) == Nat.add(SF, Nat.add(Bh, Nat.add(Nat.add(C, P), q))) : Nat}:
  +in1 = Equal.trans(Nat, Nat.add(Nat.add(Nat.add(Bh, C), P), q), Nat.add(Nat.add(Bh, C), Nat.add(P, q)),
    Nat.add(Bh, Nat.add(C, Nat.add(P, q))),
    Ar.ar_add_assoc(Nat.add(Bh, C), P, q), Ar.ar_add_assoc(Bh, C, Nat.add(P, q)))
  +in2 = Equal.cong(Nat, Nat, z => Nat.add(Bh, z), Nat.add(Nat.add(C, P), q), Nat.add(C, Nat.add(P, q)), Ar.ar_add_assoc(C, P, q))
  +inner = Equal.trans(Nat, Nat.add(Nat.add(Nat.add(Bh, C), P), q), Nat.add(Bh, Nat.add(C, Nat.add(P, q))),
    Nat.add(Bh, Nat.add(Nat.add(C, P), q)), in1, Equal.sym(Nat, Nat.add(Bh, Nat.add(Nat.add(C, P), q)), Nat.add(Bh, Nat.add(C, Nat.add(P, q))), in2))
  Equal.trans(Nat, Nat.add(Nat.add(SF, Nat.add(Nat.add(Bh, C), P)), q), Nat.add(SF, Nat.add(Nat.add(Nat.add(Bh, C), P), q)),
    Nat.add(SF, Nat.add(Bh, Nat.add(Nat.add(C, P), q))),
    Ar.ar_add_assoc(SF, Nat.add(Nat.add(Bh, C), P), q),
    Equal.cong(Nat, Nat, z => Nat.add(SF, z), Nat.add(Nat.add(Nat.add(Bh, C), P), q), Nat.add(Bh, Nat.add(Nat.add(C, P), q)), inner))

def rg_by(k: Bool, +F: Nat, +Nk: Nat, +kh: Nat, +h: Nat, +c: Nat, +s: Nat, +p: Nat, +q: Nat)
  -> {Impl.src_by(k, F, Nk, kh, h, c, s, p, q) == Impl.old_src(F, Nk, kh, h, Impl.at(Impl.dst_by(k, c, s, p), q)) : Nat}:
  match k:
    case True{}:
      rg_k(Nat.mul(s, F), Nat.mul(Nat.add(Nk, kh), 128n), Nat.mul(p, 8n), q)
    case False{}:
      rg_v(Nat.mul(s, F), Nat.add(Nat.mul(Nat.mul(2n, Nk), 128n), Nat.mul(h, 128n)), Nat.mul(c, 32n),
        Nat.mul(Nat.sub(p, 16n), 8n), q)

def Laws.gather_src(F, Nk, kh, h, c, e, q):
  rg_by(Impl.is_k(e), F, Nk, kh, h, c, Impl.e_s(e), Impl.e_p(e), q)
