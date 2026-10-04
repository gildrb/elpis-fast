# Proofs of the TREE verify-attention laws (attn_tree_laws.bend) about the Bend model attn_tree.bend
# against the reference attn_tree_spec.bend. No unsafe definitions, axioms, holes or foreign proof
# code. The per-shape laws are discharged by evaluation: `shape_ok` below is a decision procedure
# (a conjunction of every per-shape obligation, computed from the model and the reference, with no
# stored answers), evaluated by the checker over all 5040 legal shapes (all_shapes, `{==}`), and the
# elimination lemmas turn that one computed fact into each quantified law. The tile laws are proven
# generically over Nat with the m16 partition library's arithmetic lemmas.
import Base
import ./attn_tree.bend as Impl
import ./attn_tree_spec.bend as Spec
import ./attn_tree_laws.bend as Laws
import ./attn_stride.bend as Stride
import ./attn_stride_proof.bend as SP
import ./gemm_m16_partition_proof.bend as Ar

# ---------------------------------------------------------------------------------------------
# Bool / Nat helpers

def at_and_l(+a: Bool, -b: Bool, h: {Bool.and(a, b) == True{} : Bool}) -> {a == True{} : Bool}:
  match a:
    case False{}:
      Empty.absurd({False{} == True{} : Bool}, Ar.ar_false_true(h))
    case True{}:
      {==}

def at_and_r(+a: Bool, -b: Bool, h: {Bool.and(a, b) == True{} : Bool}) -> {b == True{} : Bool}:
  match a:
    case False{}:
      Empty.absurd({b == True{} : Bool}, Ar.ar_false_true(h))
    case True{}:
      h

# b from (not a || b) and a
def at_imp(+a: Bool, -b: Bool, h: {Bool.or(Bool.not(a), b) == True{} : Bool}, ha: {a == True{} : Bool})
  -> {b == True{} : Bool}:
  match a:
    case False{}:
      Empty.absurd({b == True{} : Bool}, Ar.ar_false_true(ha))
    case True{}:
      h

def at_eq(+a: Nat, +b: Nat, h: {Nat.is_eq(a, b) == True{} : Bool}) -> {a == b : Nat}:
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      Empty.absurd({0n == 1n+q : Nat}, Ar.ar_false_true(h))
    case 1n+p 0n:
      Empty.absurd({1n+p == 0n : Nat}, Ar.ar_false_true(h))
    case 1n+p 1n+q:
      %at_eq(p, q, h) : {1n+p == 1n+_ : Nat}
      {==}

def at_and_true(+x: Bool) -> {Bool.and(x, True{}) == x : Bool}:
  match x:
    case False{}:
      {==}
    case True{}:
      {==}

# ---------------------------------------------------------------------------------------------
# The per-shape decision procedure (over the TreeDesc fields the kernel reads; the reference's
# depth / ancestors enter through the generic bridge lemmas below). Each conjunction over an index
# i < n is written xs(n, ..) = xs(n - 1, ..) && f(n - 1)

def imp(a: Bool, b: Bool) -> Bool:
  Bool.or(Bool.not(a), b)

# Row r, in-round column j <= depth[r]: the serving pass's path holds r's path byte
def col_ok(+d: Impl.TreeDesc, +r: Nat, +j: Nat) -> Bool:
  imp(Nat.is_le(j, Impl.depth(d, r)), Nat.is_eq(Impl.path(d, Impl.leaf_of(d, r), j), Impl.path(d, r, j)))

def cols(+n: Nat, +d: Impl.TreeDesc, +r: Nat) -> Bool:
  match n:
    case 0n:
      True{}
    case 1n+m:
      Bool.and(cols(m, d, r), col_ok(d, r, m))

# Row r: no entry trap, one pass, the serving leaf at least as deep, the path prefix
def row_ok(+d: Impl.TreeDesc, +r: Nat) -> Bool:
  Bool.and(Impl.entry_ok(8n, d, r),
    Bool.and(Nat.is_eq(Impl.pass_count(Impl.n_leaf(d), d, r), 1n),
      Bool.and(Nat.is_le(Impl.depth(d, r), Impl.depth(d, Impl.leaf_of(d, r))), cols(8n, d, r))))

def rows(+n: Nat, +d: Impl.TreeDesc) -> Bool:
  match n:
    case 0n:
      True{}
    case 1n+m:
      Bool.and(rows(m, d), row_ok(d, m))

def pass_js(+n: Nat, +d: Impl.TreeDesc, +p: Nat) -> Bool:
  match n:
    case 0n:
      True{}
    case 1n+m:
      Bool.and(pass_js(m, d, p), Impl.pass_ok(d, p, m))

def pass_ps(+n: Nat, +d: Impl.TreeDesc) -> Bool:
  match n:
    case 0n:
      True{}
    case 1n+m:
      Bool.and(pass_ps(m, d), pass_js(8n, d, m))

def shape_d(+d: Impl.TreeDesc) -> Bool:
  Bool.and(rows(8n, d), pass_ps(Impl.n_leaf(d), d))

def shape_ok(+ps: List<&2, Nat>) -> Bool:
  shape_d(Spec.desc(ps))

# Every legal shape: q_r < r. Level L enumerates q_L < n and takes the bounds b_{L+1} .. b_7 of the
# levels below as parameters (instantiated 2n .. 7n at the top), so the elimination lemmas below never
# unfold a lower level
def q7s(+n: Nat, +q1: Nat, +q2: Nat, +q3: Nat, +q4: Nat, +q5: Nat, +q6: Nat) -> Bool:
  match n:
    case 0n:
      True{}
    case 1n+m:
      Bool.and(q7s(m, q1, q2, q3, q4, q5, q6), shape_ok(Spec.shape(q1, q2, q3, q4, q5, q6, m)))

def q6s(+n: Nat, +b7: Nat, +q1: Nat, +q2: Nat, +q3: Nat, +q4: Nat, +q5: Nat) -> Bool:
  match n:
    case 0n:
      True{}
    case 1n+m:
      Bool.and(q6s(m, b7, q1, q2, q3, q4, q5), q7s(b7, q1, q2, q3, q4, q5, m))

def q5s(+n: Nat, +b6: Nat, +b7: Nat, +q1: Nat, +q2: Nat, +q3: Nat, +q4: Nat) -> Bool:
  match n:
    case 0n:
      True{}
    case 1n+m:
      Bool.and(q5s(m, b6, b7, q1, q2, q3, q4), q6s(b6, b7, q1, q2, q3, q4, m))

def q4s(+n: Nat, +b5: Nat, +b6: Nat, +b7: Nat, +q1: Nat, +q2: Nat, +q3: Nat) -> Bool:
  match n:
    case 0n:
      True{}
    case 1n+m:
      Bool.and(q4s(m, b5, b6, b7, q1, q2, q3), q5s(b5, b6, b7, q1, q2, q3, m))

def q3s(+n: Nat, +b4: Nat, +b5: Nat, +b6: Nat, +b7: Nat, +q1: Nat, +q2: Nat) -> Bool:
  match n:
    case 0n:
      True{}
    case 1n+m:
      Bool.and(q3s(m, b4, b5, b6, b7, q1, q2), q4s(b4, b5, b6, b7, q1, q2, m))

def q2s(+n: Nat, +b3: Nat, +b4: Nat, +b5: Nat, +b6: Nat, +b7: Nat, +q1: Nat) -> Bool:
  match n:
    case 0n:
      True{}
    case 1n+m:
      Bool.and(q2s(m, b3, b4, b5, b6, b7, q1), q3s(b3, b4, b5, b6, b7, q1, m))

def q1s(+n: Nat, +b2: Nat, +b3: Nat, +b4: Nat, +b5: Nat, +b6: Nat, +b7: Nat) -> Bool:
  match n:
    case 0n:
      True{}
    case 1n+m:
      Bool.and(q1s(m, b2, b3, b4, b5, b6, b7), q2s(b2, b3, b4, b5, b6, b7, m))

# ---------------------------------------------------------------------------------------------
# Elimination: xs(n, ..) == True and t < n give f(t) == True

# Nothing is below 0
def lt_zero(+t: Nat, h: {Nat.is_lt(t, 0n) == True{} : Bool}) -> Empty:
  match t:
    case 0n:
      Ar.ar_false_true(h)
    case 1n+x:
      Ar.ar_false_true(h)

# t < m + 1 and t != m give t < m
def lt_succ_ne(+t: Nat, +m: Nat, lt: {Nat.is_lt(t, 1n+m) == True{} : Bool}, ne: {Nat.is_eq(t, m) == False{} : Bool})
  -> {Nat.is_lt(t, m) == True{} : Bool}:
  match t m:
    case 0n 0n:
      Empty.absurd({Nat.is_lt(0n, 0n) == True{} : Bool}, Ar.ar_false_true(Equal.sym(Bool, True{}, False{}, ne)))
    case 0n 1n+m1:
      {==}
    case 1n+t1 0n:
      Empty.absurd({Nat.is_lt(1n+t1, 0n) == True{} : Bool}, lt_zero(t1, lt))
    case 1n+t1 1n+m1:
      lt_succ_ne(t1, m1, lt, ne)








def cols_pick(+t: Nat, +m: Nat, +d: Impl.TreeDesc, +r: Nat, b: Bool, e: {Nat.is_eq(t, m) == b : Bool},
    lt: {Nat.is_lt(t, 1n+m) == True{} : Bool}, fm: {col_ok(d, r, m) == True{} : Bool},
    ih: {Nat.is_lt(t, m) == True{} : Bool} -> {col_ok(d, r, t) == True{} : Bool}) -> {col_ok(d, r, t) == True{} : Bool}:
  match b:
    case True{}:
      %Equal.sym(Nat, t, m, at_eq(t, m, e)) : {col_ok(d, r, _) == True{} : Bool}
      fm
    case False{}:
      ih(lt_succ_ne(t, m, lt, e))

def cols_go(+n: Nat, +t: Nat, +d: Impl.TreeDesc, +r: Nat, lt: {Nat.is_lt(t, n) == True{} : Bool},
    +h: {cols(n, d, r) == True{} : Bool}) -> {col_ok(d, r, t) == True{} : Bool}:
  match n:
    case 0n:
      Empty.absurd({col_ok(d, r, t) == True{} : Bool}, lt_zero(t, lt))
    case 1n+m:
      cols_pick(t, m, d, r, Nat.is_eq(t, m), {==}, lt, at_and_r(cols(m, d, r), col_ok(d, r, m), h),
        hl => cols_go(m, t, d, r, hl, at_and_l(cols(m, d, r), col_ok(d, r, m), h)))

def rows_pick(+t: Nat, +m: Nat, +d: Impl.TreeDesc, b: Bool, e: {Nat.is_eq(t, m) == b : Bool},
    lt: {Nat.is_lt(t, 1n+m) == True{} : Bool}, fm: {row_ok(d, m) == True{} : Bool},
    ih: {Nat.is_lt(t, m) == True{} : Bool} -> {row_ok(d, t) == True{} : Bool}) -> {row_ok(d, t) == True{} : Bool}:
  match b:
    case True{}:
      %Equal.sym(Nat, t, m, at_eq(t, m, e)) : {row_ok(d, _) == True{} : Bool}
      fm
    case False{}:
      ih(lt_succ_ne(t, m, lt, e))

def rows_go(+n: Nat, +t: Nat, +d: Impl.TreeDesc, lt: {Nat.is_lt(t, n) == True{} : Bool},
    +h: {rows(n, d) == True{} : Bool}) -> {row_ok(d, t) == True{} : Bool}:
  match n:
    case 0n:
      Empty.absurd({row_ok(d, t) == True{} : Bool}, lt_zero(t, lt))
    case 1n+m:
      rows_pick(t, m, d, Nat.is_eq(t, m), {==}, lt, at_and_r(rows(m, d), row_ok(d, m), h),
        hl => rows_go(m, t, d, hl, at_and_l(rows(m, d), row_ok(d, m), h)))

def pass_js_pick(+t: Nat, +m: Nat, +d: Impl.TreeDesc, +p: Nat, b: Bool, e: {Nat.is_eq(t, m) == b : Bool},
    lt: {Nat.is_lt(t, 1n+m) == True{} : Bool}, fm: {Impl.pass_ok(d, p, m) == True{} : Bool},
    ih: {Nat.is_lt(t, m) == True{} : Bool} -> {Impl.pass_ok(d, p, t) == True{} : Bool}) -> {Impl.pass_ok(d, p, t) == True{} : Bool}:
  match b:
    case True{}:
      %Equal.sym(Nat, t, m, at_eq(t, m, e)) : {Impl.pass_ok(d, p, _) == True{} : Bool}
      fm
    case False{}:
      ih(lt_succ_ne(t, m, lt, e))

def pass_js_go(+n: Nat, +t: Nat, +d: Impl.TreeDesc, +p: Nat, lt: {Nat.is_lt(t, n) == True{} : Bool},
    +h: {pass_js(n, d, p) == True{} : Bool}) -> {Impl.pass_ok(d, p, t) == True{} : Bool}:
  match n:
    case 0n:
      Empty.absurd({Impl.pass_ok(d, p, t) == True{} : Bool}, lt_zero(t, lt))
    case 1n+m:
      pass_js_pick(t, m, d, p, Nat.is_eq(t, m), {==}, lt, at_and_r(pass_js(m, d, p), Impl.pass_ok(d, p, m), h),
        hl => pass_js_go(m, t, d, p, hl, at_and_l(pass_js(m, d, p), Impl.pass_ok(d, p, m), h)))

def pass_ps_pick(+t: Nat, +m: Nat, +d: Impl.TreeDesc, b: Bool, e: {Nat.is_eq(t, m) == b : Bool},
    lt: {Nat.is_lt(t, 1n+m) == True{} : Bool}, fm: {pass_js(8n, d, m) == True{} : Bool},
    ih: {Nat.is_lt(t, m) == True{} : Bool} -> {pass_js(8n, d, t) == True{} : Bool}) -> {pass_js(8n, d, t) == True{} : Bool}:
  match b:
    case True{}:
      %Equal.sym(Nat, t, m, at_eq(t, m, e)) : {pass_js(8n, d, _) == True{} : Bool}
      fm
    case False{}:
      ih(lt_succ_ne(t, m, lt, e))

def pass_ps_go(+n: Nat, +t: Nat, +d: Impl.TreeDesc, lt: {Nat.is_lt(t, n) == True{} : Bool},
    +h: {pass_ps(n, d) == True{} : Bool}) -> {pass_js(8n, d, t) == True{} : Bool}:
  match n:
    case 0n:
      Empty.absurd({pass_js(8n, d, t) == True{} : Bool}, lt_zero(t, lt))
    case 1n+m:
      pass_ps_pick(t, m, d, Nat.is_eq(t, m), {==}, lt, at_and_r(pass_ps(m, d), pass_js(8n, d, m), h),
        hl => pass_ps_go(m, t, d, hl, at_and_l(pass_ps(m, d), pass_js(8n, d, m), h)))

def q7s_pick(+t: Nat, +m: Nat, +q1: Nat, +q2: Nat, +q3: Nat, +q4: Nat, +q5: Nat, +q6: Nat, b: Bool, e: {Nat.is_eq(t, m) == b : Bool},
    lt: {Nat.is_lt(t, 1n+m) == True{} : Bool}, fm: {shape_ok(Spec.shape(q1, q2, q3, q4, q5, q6, m)) == True{} : Bool},
    ih: {Nat.is_lt(t, m) == True{} : Bool} -> {shape_ok(Spec.shape(q1, q2, q3, q4, q5, q6, t)) == True{} : Bool}) -> {shape_ok(Spec.shape(q1, q2, q3, q4, q5, q6, t)) == True{} : Bool}:
  match b:
    case True{}:
      %Equal.sym(Nat, t, m, at_eq(t, m, e)) : {shape_ok(Spec.shape(q1, q2, q3, q4, q5, q6, _)) == True{} : Bool}
      fm
    case False{}:
      ih(lt_succ_ne(t, m, lt, e))

def q7s_go(+n: Nat, +t: Nat, +q1: Nat, +q2: Nat, +q3: Nat, +q4: Nat, +q5: Nat, +q6: Nat, lt: {Nat.is_lt(t, n) == True{} : Bool},
    +h: {q7s(n, q1, q2, q3, q4, q5, q6) == True{} : Bool}) -> {shape_ok(Spec.shape(q1, q2, q3, q4, q5, q6, t)) == True{} : Bool}:
  match n:
    case 0n:
      Empty.absurd({shape_ok(Spec.shape(q1, q2, q3, q4, q5, q6, t)) == True{} : Bool}, lt_zero(t, lt))
    case 1n+m:
      q7s_pick(t, m, q1, q2, q3, q4, q5, q6, Nat.is_eq(t, m), {==}, lt, at_and_r(q7s(m, q1, q2, q3, q4, q5, q6), shape_ok(Spec.shape(q1, q2, q3, q4, q5, q6, m)), h),
        hl => q7s_go(m, t, q1, q2, q3, q4, q5, q6, hl, at_and_l(q7s(m, q1, q2, q3, q4, q5, q6), shape_ok(Spec.shape(q1, q2, q3, q4, q5, q6, m)), h)))

def q6s_pick(+t: Nat, +m: Nat, +b7: Nat, +q1: Nat, +q2: Nat, +q3: Nat, +q4: Nat, +q5: Nat, b: Bool, e: {Nat.is_eq(t, m) == b : Bool},
    lt: {Nat.is_lt(t, 1n+m) == True{} : Bool}, fm: {q7s(b7, q1, q2, q3, q4, q5, m) == True{} : Bool},
    ih: {Nat.is_lt(t, m) == True{} : Bool} -> {q7s(b7, q1, q2, q3, q4, q5, t) == True{} : Bool}) -> {q7s(b7, q1, q2, q3, q4, q5, t) == True{} : Bool}:
  match b:
    case True{}:
      %Equal.sym(Nat, t, m, at_eq(t, m, e)) : {q7s(b7, q1, q2, q3, q4, q5, _) == True{} : Bool}
      fm
    case False{}:
      ih(lt_succ_ne(t, m, lt, e))

def q6s_go(+n: Nat, +t: Nat, +b7: Nat, +q1: Nat, +q2: Nat, +q3: Nat, +q4: Nat, +q5: Nat, lt: {Nat.is_lt(t, n) == True{} : Bool},
    +h: {q6s(n, b7, q1, q2, q3, q4, q5) == True{} : Bool}) -> {q7s(b7, q1, q2, q3, q4, q5, t) == True{} : Bool}:
  match n:
    case 0n:
      Empty.absurd({q7s(b7, q1, q2, q3, q4, q5, t) == True{} : Bool}, lt_zero(t, lt))
    case 1n+m:
      q6s_pick(t, m, b7, q1, q2, q3, q4, q5, Nat.is_eq(t, m), {==}, lt, at_and_r(q6s(m, b7, q1, q2, q3, q4, q5), q7s(b7, q1, q2, q3, q4, q5, m), h),
        hl => q6s_go(m, t, b7, q1, q2, q3, q4, q5, hl, at_and_l(q6s(m, b7, q1, q2, q3, q4, q5), q7s(b7, q1, q2, q3, q4, q5, m), h)))

def q5s_pick(+t: Nat, +m: Nat, +b6: Nat, +b7: Nat, +q1: Nat, +q2: Nat, +q3: Nat, +q4: Nat, b: Bool, e: {Nat.is_eq(t, m) == b : Bool},
    lt: {Nat.is_lt(t, 1n+m) == True{} : Bool}, fm: {q6s(b6, b7, q1, q2, q3, q4, m) == True{} : Bool},
    ih: {Nat.is_lt(t, m) == True{} : Bool} -> {q6s(b6, b7, q1, q2, q3, q4, t) == True{} : Bool}) -> {q6s(b6, b7, q1, q2, q3, q4, t) == True{} : Bool}:
  match b:
    case True{}:
      %Equal.sym(Nat, t, m, at_eq(t, m, e)) : {q6s(b6, b7, q1, q2, q3, q4, _) == True{} : Bool}
      fm
    case False{}:
      ih(lt_succ_ne(t, m, lt, e))

def q5s_go(+n: Nat, +t: Nat, +b6: Nat, +b7: Nat, +q1: Nat, +q2: Nat, +q3: Nat, +q4: Nat, lt: {Nat.is_lt(t, n) == True{} : Bool},
    +h: {q5s(n, b6, b7, q1, q2, q3, q4) == True{} : Bool}) -> {q6s(b6, b7, q1, q2, q3, q4, t) == True{} : Bool}:
  match n:
    case 0n:
      Empty.absurd({q6s(b6, b7, q1, q2, q3, q4, t) == True{} : Bool}, lt_zero(t, lt))
    case 1n+m:
      q5s_pick(t, m, b6, b7, q1, q2, q3, q4, Nat.is_eq(t, m), {==}, lt, at_and_r(q5s(m, b6, b7, q1, q2, q3, q4), q6s(b6, b7, q1, q2, q3, q4, m), h),
        hl => q5s_go(m, t, b6, b7, q1, q2, q3, q4, hl, at_and_l(q5s(m, b6, b7, q1, q2, q3, q4), q6s(b6, b7, q1, q2, q3, q4, m), h)))

def q4s_pick(+t: Nat, +m: Nat, +b5: Nat, +b6: Nat, +b7: Nat, +q1: Nat, +q2: Nat, +q3: Nat, b: Bool, e: {Nat.is_eq(t, m) == b : Bool},
    lt: {Nat.is_lt(t, 1n+m) == True{} : Bool}, fm: {q5s(b5, b6, b7, q1, q2, q3, m) == True{} : Bool},
    ih: {Nat.is_lt(t, m) == True{} : Bool} -> {q5s(b5, b6, b7, q1, q2, q3, t) == True{} : Bool}) -> {q5s(b5, b6, b7, q1, q2, q3, t) == True{} : Bool}:
  match b:
    case True{}:
      %Equal.sym(Nat, t, m, at_eq(t, m, e)) : {q5s(b5, b6, b7, q1, q2, q3, _) == True{} : Bool}
      fm
    case False{}:
      ih(lt_succ_ne(t, m, lt, e))

def q4s_go(+n: Nat, +t: Nat, +b5: Nat, +b6: Nat, +b7: Nat, +q1: Nat, +q2: Nat, +q3: Nat, lt: {Nat.is_lt(t, n) == True{} : Bool},
    +h: {q4s(n, b5, b6, b7, q1, q2, q3) == True{} : Bool}) -> {q5s(b5, b6, b7, q1, q2, q3, t) == True{} : Bool}:
  match n:
    case 0n:
      Empty.absurd({q5s(b5, b6, b7, q1, q2, q3, t) == True{} : Bool}, lt_zero(t, lt))
    case 1n+m:
      q4s_pick(t, m, b5, b6, b7, q1, q2, q3, Nat.is_eq(t, m), {==}, lt, at_and_r(q4s(m, b5, b6, b7, q1, q2, q3), q5s(b5, b6, b7, q1, q2, q3, m), h),
        hl => q4s_go(m, t, b5, b6, b7, q1, q2, q3, hl, at_and_l(q4s(m, b5, b6, b7, q1, q2, q3), q5s(b5, b6, b7, q1, q2, q3, m), h)))

def q3s_pick(+t: Nat, +m: Nat, +b4: Nat, +b5: Nat, +b6: Nat, +b7: Nat, +q1: Nat, +q2: Nat, b: Bool, e: {Nat.is_eq(t, m) == b : Bool},
    lt: {Nat.is_lt(t, 1n+m) == True{} : Bool}, fm: {q4s(b4, b5, b6, b7, q1, q2, m) == True{} : Bool},
    ih: {Nat.is_lt(t, m) == True{} : Bool} -> {q4s(b4, b5, b6, b7, q1, q2, t) == True{} : Bool}) -> {q4s(b4, b5, b6, b7, q1, q2, t) == True{} : Bool}:
  match b:
    case True{}:
      %Equal.sym(Nat, t, m, at_eq(t, m, e)) : {q4s(b4, b5, b6, b7, q1, q2, _) == True{} : Bool}
      fm
    case False{}:
      ih(lt_succ_ne(t, m, lt, e))

def q3s_go(+n: Nat, +t: Nat, +b4: Nat, +b5: Nat, +b6: Nat, +b7: Nat, +q1: Nat, +q2: Nat, lt: {Nat.is_lt(t, n) == True{} : Bool},
    +h: {q3s(n, b4, b5, b6, b7, q1, q2) == True{} : Bool}) -> {q4s(b4, b5, b6, b7, q1, q2, t) == True{} : Bool}:
  match n:
    case 0n:
      Empty.absurd({q4s(b4, b5, b6, b7, q1, q2, t) == True{} : Bool}, lt_zero(t, lt))
    case 1n+m:
      q3s_pick(t, m, b4, b5, b6, b7, q1, q2, Nat.is_eq(t, m), {==}, lt, at_and_r(q3s(m, b4, b5, b6, b7, q1, q2), q4s(b4, b5, b6, b7, q1, q2, m), h),
        hl => q3s_go(m, t, b4, b5, b6, b7, q1, q2, hl, at_and_l(q3s(m, b4, b5, b6, b7, q1, q2), q4s(b4, b5, b6, b7, q1, q2, m), h)))

def q2s_pick(+t: Nat, +m: Nat, +b3: Nat, +b4: Nat, +b5: Nat, +b6: Nat, +b7: Nat, +q1: Nat, b: Bool, e: {Nat.is_eq(t, m) == b : Bool},
    lt: {Nat.is_lt(t, 1n+m) == True{} : Bool}, fm: {q3s(b3, b4, b5, b6, b7, q1, m) == True{} : Bool},
    ih: {Nat.is_lt(t, m) == True{} : Bool} -> {q3s(b3, b4, b5, b6, b7, q1, t) == True{} : Bool}) -> {q3s(b3, b4, b5, b6, b7, q1, t) == True{} : Bool}:
  match b:
    case True{}:
      %Equal.sym(Nat, t, m, at_eq(t, m, e)) : {q3s(b3, b4, b5, b6, b7, q1, _) == True{} : Bool}
      fm
    case False{}:
      ih(lt_succ_ne(t, m, lt, e))

def q2s_go(+n: Nat, +t: Nat, +b3: Nat, +b4: Nat, +b5: Nat, +b6: Nat, +b7: Nat, +q1: Nat, lt: {Nat.is_lt(t, n) == True{} : Bool},
    +h: {q2s(n, b3, b4, b5, b6, b7, q1) == True{} : Bool}) -> {q3s(b3, b4, b5, b6, b7, q1, t) == True{} : Bool}:
  match n:
    case 0n:
      Empty.absurd({q3s(b3, b4, b5, b6, b7, q1, t) == True{} : Bool}, lt_zero(t, lt))
    case 1n+m:
      q2s_pick(t, m, b3, b4, b5, b6, b7, q1, Nat.is_eq(t, m), {==}, lt, at_and_r(q2s(m, b3, b4, b5, b6, b7, q1), q3s(b3, b4, b5, b6, b7, q1, m), h),
        hl => q2s_go(m, t, b3, b4, b5, b6, b7, q1, hl, at_and_l(q2s(m, b3, b4, b5, b6, b7, q1), q3s(b3, b4, b5, b6, b7, q1, m), h)))

def q1s_pick(+t: Nat, +m: Nat, +b2: Nat, +b3: Nat, +b4: Nat, +b5: Nat, +b6: Nat, +b7: Nat, b: Bool, e: {Nat.is_eq(t, m) == b : Bool},
    lt: {Nat.is_lt(t, 1n+m) == True{} : Bool}, fm: {q2s(b2, b3, b4, b5, b6, b7, m) == True{} : Bool},
    ih: {Nat.is_lt(t, m) == True{} : Bool} -> {q2s(b2, b3, b4, b5, b6, b7, t) == True{} : Bool}) -> {q2s(b2, b3, b4, b5, b6, b7, t) == True{} : Bool}:
  match b:
    case True{}:
      %Equal.sym(Nat, t, m, at_eq(t, m, e)) : {q2s(b2, b3, b4, b5, b6, b7, _) == True{} : Bool}
      fm
    case False{}:
      ih(lt_succ_ne(t, m, lt, e))

def q1s_go(+n: Nat, +t: Nat, +b2: Nat, +b3: Nat, +b4: Nat, +b5: Nat, +b6: Nat, +b7: Nat, lt: {Nat.is_lt(t, n) == True{} : Bool},
    +h: {q1s(n, b2, b3, b4, b5, b6, b7) == True{} : Bool}) -> {q2s(b2, b3, b4, b5, b6, b7, t) == True{} : Bool}:
  match n:
    case 0n:
      Empty.absurd({q2s(b2, b3, b4, b5, b6, b7, t) == True{} : Bool}, lt_zero(t, lt))
    case 1n+m:
      q1s_pick(t, m, b2, b3, b4, b5, b6, b7, Nat.is_eq(t, m), {==}, lt, at_and_r(q1s(m, b2, b3, b4, b5, b6, b7), q2s(b2, b3, b4, b5, b6, b7, m), h),
        hl => q1s_go(m, t, b2, b3, b4, b5, b6, b7, hl, at_and_l(q1s(m, b2, b3, b4, b5, b6, b7), q2s(b2, b3, b4, b5, b6, b7, m), h)))

# ---------------------------------------------------------------------------------------------
# Every legal shape's decision procedure evaluates to True (all 5040 shapes), in kernel-sized pieces: each sh_ def
# evaluates one q7s group (7 shapes) by `{==}`; the q?s_i introductions glue the groups into
# q1s(1n, 2n, .., 7n) syntactically (the checker compares the glued facts without evaluating them)

def at_and_i(+a: Bool, -b: Bool, ha: {a == True{} : Bool}, hb: {b == True{} : Bool}) -> {Bool.and(a, b) == True{} : Bool}:
  match a:
    case False{}:
      Empty.absurd({Bool.and(False{}, b) == True{} : Bool}, Ar.ar_false_true(ha))
    case True{}:
      hb

def q1s_i(+m: Nat, +b2: Nat, +b3: Nat, +b4: Nat, +b5: Nat, +b6: Nat, +b7: Nat, h1: {q1s(m, b2, b3, b4, b5, b6, b7) == True{} : Bool}, h2: {q2s(b2, b3, b4, b5, b6, b7, m) == True{} : Bool})
  -> {q1s(1n+m, b2, b3, b4, b5, b6, b7) == True{} : Bool}:
  at_and_i(q1s(m, b2, b3, b4, b5, b6, b7), q2s(b2, b3, b4, b5, b6, b7, m), h1, h2)

def q2s_i(+m: Nat, +b3: Nat, +b4: Nat, +b5: Nat, +b6: Nat, +b7: Nat, +q1: Nat, h1: {q2s(m, b3, b4, b5, b6, b7, q1) == True{} : Bool}, h2: {q3s(b3, b4, b5, b6, b7, q1, m) == True{} : Bool})
  -> {q2s(1n+m, b3, b4, b5, b6, b7, q1) == True{} : Bool}:
  at_and_i(q2s(m, b3, b4, b5, b6, b7, q1), q3s(b3, b4, b5, b6, b7, q1, m), h1, h2)

def q3s_i(+m: Nat, +b4: Nat, +b5: Nat, +b6: Nat, +b7: Nat, +q1: Nat, +q2: Nat, h1: {q3s(m, b4, b5, b6, b7, q1, q2) == True{} : Bool}, h2: {q4s(b4, b5, b6, b7, q1, q2, m) == True{} : Bool})
  -> {q3s(1n+m, b4, b5, b6, b7, q1, q2) == True{} : Bool}:
  at_and_i(q3s(m, b4, b5, b6, b7, q1, q2), q4s(b4, b5, b6, b7, q1, q2, m), h1, h2)

def q4s_i(+m: Nat, +b5: Nat, +b6: Nat, +b7: Nat, +q1: Nat, +q2: Nat, +q3: Nat, h1: {q4s(m, b5, b6, b7, q1, q2, q3) == True{} : Bool}, h2: {q5s(b5, b6, b7, q1, q2, q3, m) == True{} : Bool})
  -> {q4s(1n+m, b5, b6, b7, q1, q2, q3) == True{} : Bool}:
  at_and_i(q4s(m, b5, b6, b7, q1, q2, q3), q5s(b5, b6, b7, q1, q2, q3, m), h1, h2)

def q5s_i(+m: Nat, +b6: Nat, +b7: Nat, +q1: Nat, +q2: Nat, +q3: Nat, +q4: Nat, h1: {q5s(m, b6, b7, q1, q2, q3, q4) == True{} : Bool}, h2: {q6s(b6, b7, q1, q2, q3, q4, m) == True{} : Bool})
  -> {q5s(1n+m, b6, b7, q1, q2, q3, q4) == True{} : Bool}:
  at_and_i(q5s(m, b6, b7, q1, q2, q3, q4), q6s(b6, b7, q1, q2, q3, q4, m), h1, h2)

def q6s_i(+m: Nat, +b7: Nat, +q1: Nat, +q2: Nat, +q3: Nat, +q4: Nat, +q5: Nat, h1: {q6s(m, b7, q1, q2, q3, q4, q5) == True{} : Bool}, h2: {q7s(b7, q1, q2, q3, q4, q5, m) == True{} : Bool})
  -> {q6s(1n+m, b7, q1, q2, q3, q4, q5) == True{} : Bool}:
  at_and_i(q6s(m, b7, q1, q2, q3, q4, q5), q7s(b7, q1, q2, q3, q4, q5, m), h1, h2)

def sh_0_0_0_0_0_0() -> {q7s(7n, 0n, 0n, 0n, 0n, 0n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_0_0_0_1() -> {q7s(7n, 0n, 0n, 0n, 0n, 0n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_0_0_0_2() -> {q7s(7n, 0n, 0n, 0n, 0n, 0n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_0_0_0_3() -> {q7s(7n, 0n, 0n, 0n, 0n, 0n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_0_0_0_4() -> {q7s(7n, 0n, 0n, 0n, 0n, 0n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_0_0_0_5() -> {q7s(7n, 0n, 0n, 0n, 0n, 0n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_0_0_1_0() -> {q7s(7n, 0n, 0n, 0n, 0n, 1n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_0_0_1_1() -> {q7s(7n, 0n, 0n, 0n, 0n, 1n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_0_0_1_2() -> {q7s(7n, 0n, 0n, 0n, 0n, 1n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_0_0_1_3() -> {q7s(7n, 0n, 0n, 0n, 0n, 1n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_0_0_1_4() -> {q7s(7n, 0n, 0n, 0n, 0n, 1n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_0_0_1_5() -> {q7s(7n, 0n, 0n, 0n, 0n, 1n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_0_0_2_0() -> {q7s(7n, 0n, 0n, 0n, 0n, 2n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_0_0_2_1() -> {q7s(7n, 0n, 0n, 0n, 0n, 2n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_0_0_2_2() -> {q7s(7n, 0n, 0n, 0n, 0n, 2n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_0_0_2_3() -> {q7s(7n, 0n, 0n, 0n, 0n, 2n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_0_0_2_4() -> {q7s(7n, 0n, 0n, 0n, 0n, 2n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_0_0_2_5() -> {q7s(7n, 0n, 0n, 0n, 0n, 2n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_0_0_3_0() -> {q7s(7n, 0n, 0n, 0n, 0n, 3n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_0_0_3_1() -> {q7s(7n, 0n, 0n, 0n, 0n, 3n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_0_0_3_2() -> {q7s(7n, 0n, 0n, 0n, 0n, 3n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_0_0_3_3() -> {q7s(7n, 0n, 0n, 0n, 0n, 3n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_0_0_3_4() -> {q7s(7n, 0n, 0n, 0n, 0n, 3n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_0_0_3_5() -> {q7s(7n, 0n, 0n, 0n, 0n, 3n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_0_0_4_0() -> {q7s(7n, 0n, 0n, 0n, 0n, 4n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_0_0_4_1() -> {q7s(7n, 0n, 0n, 0n, 0n, 4n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_0_0_4_2() -> {q7s(7n, 0n, 0n, 0n, 0n, 4n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_0_0_4_3() -> {q7s(7n, 0n, 0n, 0n, 0n, 4n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_0_0_4_4() -> {q7s(7n, 0n, 0n, 0n, 0n, 4n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_0_0_4_5() -> {q7s(7n, 0n, 0n, 0n, 0n, 4n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_0_1_0_0() -> {q7s(7n, 0n, 0n, 0n, 1n, 0n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_0_1_0_1() -> {q7s(7n, 0n, 0n, 0n, 1n, 0n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_0_1_0_2() -> {q7s(7n, 0n, 0n, 0n, 1n, 0n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_0_1_0_3() -> {q7s(7n, 0n, 0n, 0n, 1n, 0n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_0_1_0_4() -> {q7s(7n, 0n, 0n, 0n, 1n, 0n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_0_1_0_5() -> {q7s(7n, 0n, 0n, 0n, 1n, 0n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_0_1_1_0() -> {q7s(7n, 0n, 0n, 0n, 1n, 1n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_0_1_1_1() -> {q7s(7n, 0n, 0n, 0n, 1n, 1n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_0_1_1_2() -> {q7s(7n, 0n, 0n, 0n, 1n, 1n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_0_1_1_3() -> {q7s(7n, 0n, 0n, 0n, 1n, 1n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_0_1_1_4() -> {q7s(7n, 0n, 0n, 0n, 1n, 1n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_0_1_1_5() -> {q7s(7n, 0n, 0n, 0n, 1n, 1n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_0_1_2_0() -> {q7s(7n, 0n, 0n, 0n, 1n, 2n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_0_1_2_1() -> {q7s(7n, 0n, 0n, 0n, 1n, 2n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_0_1_2_2() -> {q7s(7n, 0n, 0n, 0n, 1n, 2n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_0_1_2_3() -> {q7s(7n, 0n, 0n, 0n, 1n, 2n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_0_1_2_4() -> {q7s(7n, 0n, 0n, 0n, 1n, 2n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_0_1_2_5() -> {q7s(7n, 0n, 0n, 0n, 1n, 2n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_0_1_3_0() -> {q7s(7n, 0n, 0n, 0n, 1n, 3n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_0_1_3_1() -> {q7s(7n, 0n, 0n, 0n, 1n, 3n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_0_1_3_2() -> {q7s(7n, 0n, 0n, 0n, 1n, 3n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_0_1_3_3() -> {q7s(7n, 0n, 0n, 0n, 1n, 3n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_0_1_3_4() -> {q7s(7n, 0n, 0n, 0n, 1n, 3n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_0_1_3_5() -> {q7s(7n, 0n, 0n, 0n, 1n, 3n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_0_1_4_0() -> {q7s(7n, 0n, 0n, 0n, 1n, 4n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_0_1_4_1() -> {q7s(7n, 0n, 0n, 0n, 1n, 4n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_0_1_4_2() -> {q7s(7n, 0n, 0n, 0n, 1n, 4n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_0_1_4_3() -> {q7s(7n, 0n, 0n, 0n, 1n, 4n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_0_1_4_4() -> {q7s(7n, 0n, 0n, 0n, 1n, 4n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_0_1_4_5() -> {q7s(7n, 0n, 0n, 0n, 1n, 4n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_0_2_0_0() -> {q7s(7n, 0n, 0n, 0n, 2n, 0n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_0_2_0_1() -> {q7s(7n, 0n, 0n, 0n, 2n, 0n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_0_2_0_2() -> {q7s(7n, 0n, 0n, 0n, 2n, 0n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_0_2_0_3() -> {q7s(7n, 0n, 0n, 0n, 2n, 0n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_0_2_0_4() -> {q7s(7n, 0n, 0n, 0n, 2n, 0n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_0_2_0_5() -> {q7s(7n, 0n, 0n, 0n, 2n, 0n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_0_2_1_0() -> {q7s(7n, 0n, 0n, 0n, 2n, 1n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_0_2_1_1() -> {q7s(7n, 0n, 0n, 0n, 2n, 1n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_0_2_1_2() -> {q7s(7n, 0n, 0n, 0n, 2n, 1n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_0_2_1_3() -> {q7s(7n, 0n, 0n, 0n, 2n, 1n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_0_2_1_4() -> {q7s(7n, 0n, 0n, 0n, 2n, 1n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_0_2_1_5() -> {q7s(7n, 0n, 0n, 0n, 2n, 1n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_0_2_2_0() -> {q7s(7n, 0n, 0n, 0n, 2n, 2n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_0_2_2_1() -> {q7s(7n, 0n, 0n, 0n, 2n, 2n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_0_2_2_2() -> {q7s(7n, 0n, 0n, 0n, 2n, 2n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_0_2_2_3() -> {q7s(7n, 0n, 0n, 0n, 2n, 2n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_0_2_2_4() -> {q7s(7n, 0n, 0n, 0n, 2n, 2n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_0_2_2_5() -> {q7s(7n, 0n, 0n, 0n, 2n, 2n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_0_2_3_0() -> {q7s(7n, 0n, 0n, 0n, 2n, 3n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_0_2_3_1() -> {q7s(7n, 0n, 0n, 0n, 2n, 3n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_0_2_3_2() -> {q7s(7n, 0n, 0n, 0n, 2n, 3n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_0_2_3_3() -> {q7s(7n, 0n, 0n, 0n, 2n, 3n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_0_2_3_4() -> {q7s(7n, 0n, 0n, 0n, 2n, 3n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_0_2_3_5() -> {q7s(7n, 0n, 0n, 0n, 2n, 3n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_0_2_4_0() -> {q7s(7n, 0n, 0n, 0n, 2n, 4n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_0_2_4_1() -> {q7s(7n, 0n, 0n, 0n, 2n, 4n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_0_2_4_2() -> {q7s(7n, 0n, 0n, 0n, 2n, 4n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_0_2_4_3() -> {q7s(7n, 0n, 0n, 0n, 2n, 4n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_0_2_4_4() -> {q7s(7n, 0n, 0n, 0n, 2n, 4n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_0_2_4_5() -> {q7s(7n, 0n, 0n, 0n, 2n, 4n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_0_3_0_0() -> {q7s(7n, 0n, 0n, 0n, 3n, 0n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_0_3_0_1() -> {q7s(7n, 0n, 0n, 0n, 3n, 0n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_0_3_0_2() -> {q7s(7n, 0n, 0n, 0n, 3n, 0n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_0_3_0_3() -> {q7s(7n, 0n, 0n, 0n, 3n, 0n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_0_3_0_4() -> {q7s(7n, 0n, 0n, 0n, 3n, 0n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_0_3_0_5() -> {q7s(7n, 0n, 0n, 0n, 3n, 0n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_0_3_1_0() -> {q7s(7n, 0n, 0n, 0n, 3n, 1n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_0_3_1_1() -> {q7s(7n, 0n, 0n, 0n, 3n, 1n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_0_3_1_2() -> {q7s(7n, 0n, 0n, 0n, 3n, 1n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_0_3_1_3() -> {q7s(7n, 0n, 0n, 0n, 3n, 1n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_0_3_1_4() -> {q7s(7n, 0n, 0n, 0n, 3n, 1n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_0_3_1_5() -> {q7s(7n, 0n, 0n, 0n, 3n, 1n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_0_3_2_0() -> {q7s(7n, 0n, 0n, 0n, 3n, 2n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_0_3_2_1() -> {q7s(7n, 0n, 0n, 0n, 3n, 2n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_0_3_2_2() -> {q7s(7n, 0n, 0n, 0n, 3n, 2n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_0_3_2_3() -> {q7s(7n, 0n, 0n, 0n, 3n, 2n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_0_3_2_4() -> {q7s(7n, 0n, 0n, 0n, 3n, 2n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_0_3_2_5() -> {q7s(7n, 0n, 0n, 0n, 3n, 2n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_0_3_3_0() -> {q7s(7n, 0n, 0n, 0n, 3n, 3n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_0_3_3_1() -> {q7s(7n, 0n, 0n, 0n, 3n, 3n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_0_3_3_2() -> {q7s(7n, 0n, 0n, 0n, 3n, 3n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_0_3_3_3() -> {q7s(7n, 0n, 0n, 0n, 3n, 3n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_0_3_3_4() -> {q7s(7n, 0n, 0n, 0n, 3n, 3n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_0_3_3_5() -> {q7s(7n, 0n, 0n, 0n, 3n, 3n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_0_3_4_0() -> {q7s(7n, 0n, 0n, 0n, 3n, 4n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_0_3_4_1() -> {q7s(7n, 0n, 0n, 0n, 3n, 4n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_0_3_4_2() -> {q7s(7n, 0n, 0n, 0n, 3n, 4n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_0_3_4_3() -> {q7s(7n, 0n, 0n, 0n, 3n, 4n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_0_3_4_4() -> {q7s(7n, 0n, 0n, 0n, 3n, 4n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_0_3_4_5() -> {q7s(7n, 0n, 0n, 0n, 3n, 4n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_1_0_0_0() -> {q7s(7n, 0n, 0n, 1n, 0n, 0n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_1_0_0_1() -> {q7s(7n, 0n, 0n, 1n, 0n, 0n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_1_0_0_2() -> {q7s(7n, 0n, 0n, 1n, 0n, 0n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_1_0_0_3() -> {q7s(7n, 0n, 0n, 1n, 0n, 0n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_1_0_0_4() -> {q7s(7n, 0n, 0n, 1n, 0n, 0n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_1_0_0_5() -> {q7s(7n, 0n, 0n, 1n, 0n, 0n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_1_0_1_0() -> {q7s(7n, 0n, 0n, 1n, 0n, 1n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_1_0_1_1() -> {q7s(7n, 0n, 0n, 1n, 0n, 1n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_1_0_1_2() -> {q7s(7n, 0n, 0n, 1n, 0n, 1n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_1_0_1_3() -> {q7s(7n, 0n, 0n, 1n, 0n, 1n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_1_0_1_4() -> {q7s(7n, 0n, 0n, 1n, 0n, 1n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_1_0_1_5() -> {q7s(7n, 0n, 0n, 1n, 0n, 1n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_1_0_2_0() -> {q7s(7n, 0n, 0n, 1n, 0n, 2n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_1_0_2_1() -> {q7s(7n, 0n, 0n, 1n, 0n, 2n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_1_0_2_2() -> {q7s(7n, 0n, 0n, 1n, 0n, 2n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_1_0_2_3() -> {q7s(7n, 0n, 0n, 1n, 0n, 2n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_1_0_2_4() -> {q7s(7n, 0n, 0n, 1n, 0n, 2n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_1_0_2_5() -> {q7s(7n, 0n, 0n, 1n, 0n, 2n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_1_0_3_0() -> {q7s(7n, 0n, 0n, 1n, 0n, 3n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_1_0_3_1() -> {q7s(7n, 0n, 0n, 1n, 0n, 3n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_1_0_3_2() -> {q7s(7n, 0n, 0n, 1n, 0n, 3n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_1_0_3_3() -> {q7s(7n, 0n, 0n, 1n, 0n, 3n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_1_0_3_4() -> {q7s(7n, 0n, 0n, 1n, 0n, 3n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_1_0_3_5() -> {q7s(7n, 0n, 0n, 1n, 0n, 3n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_1_0_4_0() -> {q7s(7n, 0n, 0n, 1n, 0n, 4n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_1_0_4_1() -> {q7s(7n, 0n, 0n, 1n, 0n, 4n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_1_0_4_2() -> {q7s(7n, 0n, 0n, 1n, 0n, 4n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_1_0_4_3() -> {q7s(7n, 0n, 0n, 1n, 0n, 4n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_1_0_4_4() -> {q7s(7n, 0n, 0n, 1n, 0n, 4n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_1_0_4_5() -> {q7s(7n, 0n, 0n, 1n, 0n, 4n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_1_1_0_0() -> {q7s(7n, 0n, 0n, 1n, 1n, 0n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_1_1_0_1() -> {q7s(7n, 0n, 0n, 1n, 1n, 0n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_1_1_0_2() -> {q7s(7n, 0n, 0n, 1n, 1n, 0n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_1_1_0_3() -> {q7s(7n, 0n, 0n, 1n, 1n, 0n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_1_1_0_4() -> {q7s(7n, 0n, 0n, 1n, 1n, 0n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_1_1_0_5() -> {q7s(7n, 0n, 0n, 1n, 1n, 0n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_1_1_1_0() -> {q7s(7n, 0n, 0n, 1n, 1n, 1n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_1_1_1_1() -> {q7s(7n, 0n, 0n, 1n, 1n, 1n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_1_1_1_2() -> {q7s(7n, 0n, 0n, 1n, 1n, 1n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_1_1_1_3() -> {q7s(7n, 0n, 0n, 1n, 1n, 1n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_1_1_1_4() -> {q7s(7n, 0n, 0n, 1n, 1n, 1n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_1_1_1_5() -> {q7s(7n, 0n, 0n, 1n, 1n, 1n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_1_1_2_0() -> {q7s(7n, 0n, 0n, 1n, 1n, 2n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_1_1_2_1() -> {q7s(7n, 0n, 0n, 1n, 1n, 2n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_1_1_2_2() -> {q7s(7n, 0n, 0n, 1n, 1n, 2n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_1_1_2_3() -> {q7s(7n, 0n, 0n, 1n, 1n, 2n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_1_1_2_4() -> {q7s(7n, 0n, 0n, 1n, 1n, 2n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_1_1_2_5() -> {q7s(7n, 0n, 0n, 1n, 1n, 2n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_1_1_3_0() -> {q7s(7n, 0n, 0n, 1n, 1n, 3n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_1_1_3_1() -> {q7s(7n, 0n, 0n, 1n, 1n, 3n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_1_1_3_2() -> {q7s(7n, 0n, 0n, 1n, 1n, 3n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_1_1_3_3() -> {q7s(7n, 0n, 0n, 1n, 1n, 3n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_1_1_3_4() -> {q7s(7n, 0n, 0n, 1n, 1n, 3n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_1_1_3_5() -> {q7s(7n, 0n, 0n, 1n, 1n, 3n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_1_1_4_0() -> {q7s(7n, 0n, 0n, 1n, 1n, 4n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_1_1_4_1() -> {q7s(7n, 0n, 0n, 1n, 1n, 4n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_1_1_4_2() -> {q7s(7n, 0n, 0n, 1n, 1n, 4n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_1_1_4_3() -> {q7s(7n, 0n, 0n, 1n, 1n, 4n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_1_1_4_4() -> {q7s(7n, 0n, 0n, 1n, 1n, 4n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_1_1_4_5() -> {q7s(7n, 0n, 0n, 1n, 1n, 4n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_1_2_0_0() -> {q7s(7n, 0n, 0n, 1n, 2n, 0n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_1_2_0_1() -> {q7s(7n, 0n, 0n, 1n, 2n, 0n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_1_2_0_2() -> {q7s(7n, 0n, 0n, 1n, 2n, 0n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_1_2_0_3() -> {q7s(7n, 0n, 0n, 1n, 2n, 0n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_1_2_0_4() -> {q7s(7n, 0n, 0n, 1n, 2n, 0n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_1_2_0_5() -> {q7s(7n, 0n, 0n, 1n, 2n, 0n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_1_2_1_0() -> {q7s(7n, 0n, 0n, 1n, 2n, 1n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_1_2_1_1() -> {q7s(7n, 0n, 0n, 1n, 2n, 1n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_1_2_1_2() -> {q7s(7n, 0n, 0n, 1n, 2n, 1n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_1_2_1_3() -> {q7s(7n, 0n, 0n, 1n, 2n, 1n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_1_2_1_4() -> {q7s(7n, 0n, 0n, 1n, 2n, 1n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_1_2_1_5() -> {q7s(7n, 0n, 0n, 1n, 2n, 1n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_1_2_2_0() -> {q7s(7n, 0n, 0n, 1n, 2n, 2n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_1_2_2_1() -> {q7s(7n, 0n, 0n, 1n, 2n, 2n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_1_2_2_2() -> {q7s(7n, 0n, 0n, 1n, 2n, 2n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_1_2_2_3() -> {q7s(7n, 0n, 0n, 1n, 2n, 2n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_1_2_2_4() -> {q7s(7n, 0n, 0n, 1n, 2n, 2n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_1_2_2_5() -> {q7s(7n, 0n, 0n, 1n, 2n, 2n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_1_2_3_0() -> {q7s(7n, 0n, 0n, 1n, 2n, 3n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_1_2_3_1() -> {q7s(7n, 0n, 0n, 1n, 2n, 3n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_1_2_3_2() -> {q7s(7n, 0n, 0n, 1n, 2n, 3n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_1_2_3_3() -> {q7s(7n, 0n, 0n, 1n, 2n, 3n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_1_2_3_4() -> {q7s(7n, 0n, 0n, 1n, 2n, 3n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_1_2_3_5() -> {q7s(7n, 0n, 0n, 1n, 2n, 3n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_1_2_4_0() -> {q7s(7n, 0n, 0n, 1n, 2n, 4n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_1_2_4_1() -> {q7s(7n, 0n, 0n, 1n, 2n, 4n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_1_2_4_2() -> {q7s(7n, 0n, 0n, 1n, 2n, 4n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_1_2_4_3() -> {q7s(7n, 0n, 0n, 1n, 2n, 4n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_1_2_4_4() -> {q7s(7n, 0n, 0n, 1n, 2n, 4n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_1_2_4_5() -> {q7s(7n, 0n, 0n, 1n, 2n, 4n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_1_3_0_0() -> {q7s(7n, 0n, 0n, 1n, 3n, 0n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_1_3_0_1() -> {q7s(7n, 0n, 0n, 1n, 3n, 0n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_1_3_0_2() -> {q7s(7n, 0n, 0n, 1n, 3n, 0n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_1_3_0_3() -> {q7s(7n, 0n, 0n, 1n, 3n, 0n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_1_3_0_4() -> {q7s(7n, 0n, 0n, 1n, 3n, 0n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_1_3_0_5() -> {q7s(7n, 0n, 0n, 1n, 3n, 0n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_1_3_1_0() -> {q7s(7n, 0n, 0n, 1n, 3n, 1n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_1_3_1_1() -> {q7s(7n, 0n, 0n, 1n, 3n, 1n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_1_3_1_2() -> {q7s(7n, 0n, 0n, 1n, 3n, 1n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_1_3_1_3() -> {q7s(7n, 0n, 0n, 1n, 3n, 1n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_1_3_1_4() -> {q7s(7n, 0n, 0n, 1n, 3n, 1n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_1_3_1_5() -> {q7s(7n, 0n, 0n, 1n, 3n, 1n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_1_3_2_0() -> {q7s(7n, 0n, 0n, 1n, 3n, 2n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_1_3_2_1() -> {q7s(7n, 0n, 0n, 1n, 3n, 2n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_1_3_2_2() -> {q7s(7n, 0n, 0n, 1n, 3n, 2n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_1_3_2_3() -> {q7s(7n, 0n, 0n, 1n, 3n, 2n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_1_3_2_4() -> {q7s(7n, 0n, 0n, 1n, 3n, 2n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_1_3_2_5() -> {q7s(7n, 0n, 0n, 1n, 3n, 2n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_1_3_3_0() -> {q7s(7n, 0n, 0n, 1n, 3n, 3n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_1_3_3_1() -> {q7s(7n, 0n, 0n, 1n, 3n, 3n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_1_3_3_2() -> {q7s(7n, 0n, 0n, 1n, 3n, 3n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_1_3_3_3() -> {q7s(7n, 0n, 0n, 1n, 3n, 3n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_1_3_3_4() -> {q7s(7n, 0n, 0n, 1n, 3n, 3n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_1_3_3_5() -> {q7s(7n, 0n, 0n, 1n, 3n, 3n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_1_3_4_0() -> {q7s(7n, 0n, 0n, 1n, 3n, 4n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_1_3_4_1() -> {q7s(7n, 0n, 0n, 1n, 3n, 4n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_1_3_4_2() -> {q7s(7n, 0n, 0n, 1n, 3n, 4n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_1_3_4_3() -> {q7s(7n, 0n, 0n, 1n, 3n, 4n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_1_3_4_4() -> {q7s(7n, 0n, 0n, 1n, 3n, 4n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_1_3_4_5() -> {q7s(7n, 0n, 0n, 1n, 3n, 4n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_2_0_0_0() -> {q7s(7n, 0n, 0n, 2n, 0n, 0n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_2_0_0_1() -> {q7s(7n, 0n, 0n, 2n, 0n, 0n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_2_0_0_2() -> {q7s(7n, 0n, 0n, 2n, 0n, 0n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_2_0_0_3() -> {q7s(7n, 0n, 0n, 2n, 0n, 0n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_2_0_0_4() -> {q7s(7n, 0n, 0n, 2n, 0n, 0n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_2_0_0_5() -> {q7s(7n, 0n, 0n, 2n, 0n, 0n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_2_0_1_0() -> {q7s(7n, 0n, 0n, 2n, 0n, 1n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_2_0_1_1() -> {q7s(7n, 0n, 0n, 2n, 0n, 1n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_2_0_1_2() -> {q7s(7n, 0n, 0n, 2n, 0n, 1n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_2_0_1_3() -> {q7s(7n, 0n, 0n, 2n, 0n, 1n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_2_0_1_4() -> {q7s(7n, 0n, 0n, 2n, 0n, 1n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_2_0_1_5() -> {q7s(7n, 0n, 0n, 2n, 0n, 1n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_2_0_2_0() -> {q7s(7n, 0n, 0n, 2n, 0n, 2n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_2_0_2_1() -> {q7s(7n, 0n, 0n, 2n, 0n, 2n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_2_0_2_2() -> {q7s(7n, 0n, 0n, 2n, 0n, 2n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_2_0_2_3() -> {q7s(7n, 0n, 0n, 2n, 0n, 2n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_2_0_2_4() -> {q7s(7n, 0n, 0n, 2n, 0n, 2n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_2_0_2_5() -> {q7s(7n, 0n, 0n, 2n, 0n, 2n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_2_0_3_0() -> {q7s(7n, 0n, 0n, 2n, 0n, 3n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_2_0_3_1() -> {q7s(7n, 0n, 0n, 2n, 0n, 3n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_2_0_3_2() -> {q7s(7n, 0n, 0n, 2n, 0n, 3n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_2_0_3_3() -> {q7s(7n, 0n, 0n, 2n, 0n, 3n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_2_0_3_4() -> {q7s(7n, 0n, 0n, 2n, 0n, 3n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_2_0_3_5() -> {q7s(7n, 0n, 0n, 2n, 0n, 3n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_2_0_4_0() -> {q7s(7n, 0n, 0n, 2n, 0n, 4n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_2_0_4_1() -> {q7s(7n, 0n, 0n, 2n, 0n, 4n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_2_0_4_2() -> {q7s(7n, 0n, 0n, 2n, 0n, 4n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_2_0_4_3() -> {q7s(7n, 0n, 0n, 2n, 0n, 4n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_2_0_4_4() -> {q7s(7n, 0n, 0n, 2n, 0n, 4n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_2_0_4_5() -> {q7s(7n, 0n, 0n, 2n, 0n, 4n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_2_1_0_0() -> {q7s(7n, 0n, 0n, 2n, 1n, 0n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_2_1_0_1() -> {q7s(7n, 0n, 0n, 2n, 1n, 0n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_2_1_0_2() -> {q7s(7n, 0n, 0n, 2n, 1n, 0n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_2_1_0_3() -> {q7s(7n, 0n, 0n, 2n, 1n, 0n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_2_1_0_4() -> {q7s(7n, 0n, 0n, 2n, 1n, 0n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_2_1_0_5() -> {q7s(7n, 0n, 0n, 2n, 1n, 0n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_2_1_1_0() -> {q7s(7n, 0n, 0n, 2n, 1n, 1n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_2_1_1_1() -> {q7s(7n, 0n, 0n, 2n, 1n, 1n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_2_1_1_2() -> {q7s(7n, 0n, 0n, 2n, 1n, 1n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_2_1_1_3() -> {q7s(7n, 0n, 0n, 2n, 1n, 1n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_2_1_1_4() -> {q7s(7n, 0n, 0n, 2n, 1n, 1n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_2_1_1_5() -> {q7s(7n, 0n, 0n, 2n, 1n, 1n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_2_1_2_0() -> {q7s(7n, 0n, 0n, 2n, 1n, 2n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_2_1_2_1() -> {q7s(7n, 0n, 0n, 2n, 1n, 2n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_2_1_2_2() -> {q7s(7n, 0n, 0n, 2n, 1n, 2n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_2_1_2_3() -> {q7s(7n, 0n, 0n, 2n, 1n, 2n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_2_1_2_4() -> {q7s(7n, 0n, 0n, 2n, 1n, 2n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_2_1_2_5() -> {q7s(7n, 0n, 0n, 2n, 1n, 2n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_2_1_3_0() -> {q7s(7n, 0n, 0n, 2n, 1n, 3n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_2_1_3_1() -> {q7s(7n, 0n, 0n, 2n, 1n, 3n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_2_1_3_2() -> {q7s(7n, 0n, 0n, 2n, 1n, 3n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_2_1_3_3() -> {q7s(7n, 0n, 0n, 2n, 1n, 3n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_2_1_3_4() -> {q7s(7n, 0n, 0n, 2n, 1n, 3n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_2_1_3_5() -> {q7s(7n, 0n, 0n, 2n, 1n, 3n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_2_1_4_0() -> {q7s(7n, 0n, 0n, 2n, 1n, 4n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_2_1_4_1() -> {q7s(7n, 0n, 0n, 2n, 1n, 4n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_2_1_4_2() -> {q7s(7n, 0n, 0n, 2n, 1n, 4n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_2_1_4_3() -> {q7s(7n, 0n, 0n, 2n, 1n, 4n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_2_1_4_4() -> {q7s(7n, 0n, 0n, 2n, 1n, 4n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_2_1_4_5() -> {q7s(7n, 0n, 0n, 2n, 1n, 4n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_2_2_0_0() -> {q7s(7n, 0n, 0n, 2n, 2n, 0n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_2_2_0_1() -> {q7s(7n, 0n, 0n, 2n, 2n, 0n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_2_2_0_2() -> {q7s(7n, 0n, 0n, 2n, 2n, 0n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_2_2_0_3() -> {q7s(7n, 0n, 0n, 2n, 2n, 0n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_2_2_0_4() -> {q7s(7n, 0n, 0n, 2n, 2n, 0n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_2_2_0_5() -> {q7s(7n, 0n, 0n, 2n, 2n, 0n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_2_2_1_0() -> {q7s(7n, 0n, 0n, 2n, 2n, 1n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_2_2_1_1() -> {q7s(7n, 0n, 0n, 2n, 2n, 1n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_2_2_1_2() -> {q7s(7n, 0n, 0n, 2n, 2n, 1n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_2_2_1_3() -> {q7s(7n, 0n, 0n, 2n, 2n, 1n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_2_2_1_4() -> {q7s(7n, 0n, 0n, 2n, 2n, 1n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_2_2_1_5() -> {q7s(7n, 0n, 0n, 2n, 2n, 1n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_2_2_2_0() -> {q7s(7n, 0n, 0n, 2n, 2n, 2n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_2_2_2_1() -> {q7s(7n, 0n, 0n, 2n, 2n, 2n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_2_2_2_2() -> {q7s(7n, 0n, 0n, 2n, 2n, 2n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_2_2_2_3() -> {q7s(7n, 0n, 0n, 2n, 2n, 2n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_2_2_2_4() -> {q7s(7n, 0n, 0n, 2n, 2n, 2n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_2_2_2_5() -> {q7s(7n, 0n, 0n, 2n, 2n, 2n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_2_2_3_0() -> {q7s(7n, 0n, 0n, 2n, 2n, 3n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_2_2_3_1() -> {q7s(7n, 0n, 0n, 2n, 2n, 3n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_2_2_3_2() -> {q7s(7n, 0n, 0n, 2n, 2n, 3n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_2_2_3_3() -> {q7s(7n, 0n, 0n, 2n, 2n, 3n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_2_2_3_4() -> {q7s(7n, 0n, 0n, 2n, 2n, 3n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_2_2_3_5() -> {q7s(7n, 0n, 0n, 2n, 2n, 3n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_2_2_4_0() -> {q7s(7n, 0n, 0n, 2n, 2n, 4n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_2_2_4_1() -> {q7s(7n, 0n, 0n, 2n, 2n, 4n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_2_2_4_2() -> {q7s(7n, 0n, 0n, 2n, 2n, 4n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_2_2_4_3() -> {q7s(7n, 0n, 0n, 2n, 2n, 4n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_2_2_4_4() -> {q7s(7n, 0n, 0n, 2n, 2n, 4n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_2_2_4_5() -> {q7s(7n, 0n, 0n, 2n, 2n, 4n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_2_3_0_0() -> {q7s(7n, 0n, 0n, 2n, 3n, 0n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_2_3_0_1() -> {q7s(7n, 0n, 0n, 2n, 3n, 0n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_2_3_0_2() -> {q7s(7n, 0n, 0n, 2n, 3n, 0n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_2_3_0_3() -> {q7s(7n, 0n, 0n, 2n, 3n, 0n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_2_3_0_4() -> {q7s(7n, 0n, 0n, 2n, 3n, 0n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_2_3_0_5() -> {q7s(7n, 0n, 0n, 2n, 3n, 0n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_2_3_1_0() -> {q7s(7n, 0n, 0n, 2n, 3n, 1n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_2_3_1_1() -> {q7s(7n, 0n, 0n, 2n, 3n, 1n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_2_3_1_2() -> {q7s(7n, 0n, 0n, 2n, 3n, 1n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_2_3_1_3() -> {q7s(7n, 0n, 0n, 2n, 3n, 1n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_2_3_1_4() -> {q7s(7n, 0n, 0n, 2n, 3n, 1n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_2_3_1_5() -> {q7s(7n, 0n, 0n, 2n, 3n, 1n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_2_3_2_0() -> {q7s(7n, 0n, 0n, 2n, 3n, 2n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_2_3_2_1() -> {q7s(7n, 0n, 0n, 2n, 3n, 2n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_2_3_2_2() -> {q7s(7n, 0n, 0n, 2n, 3n, 2n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_2_3_2_3() -> {q7s(7n, 0n, 0n, 2n, 3n, 2n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_2_3_2_4() -> {q7s(7n, 0n, 0n, 2n, 3n, 2n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_2_3_2_5() -> {q7s(7n, 0n, 0n, 2n, 3n, 2n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_2_3_3_0() -> {q7s(7n, 0n, 0n, 2n, 3n, 3n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_2_3_3_1() -> {q7s(7n, 0n, 0n, 2n, 3n, 3n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_2_3_3_2() -> {q7s(7n, 0n, 0n, 2n, 3n, 3n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_2_3_3_3() -> {q7s(7n, 0n, 0n, 2n, 3n, 3n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_2_3_3_4() -> {q7s(7n, 0n, 0n, 2n, 3n, 3n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_2_3_3_5() -> {q7s(7n, 0n, 0n, 2n, 3n, 3n, 5n) == True{} : Bool}:
  {==}

def sh_0_0_2_3_4_0() -> {q7s(7n, 0n, 0n, 2n, 3n, 4n, 0n) == True{} : Bool}:
  {==}

def sh_0_0_2_3_4_1() -> {q7s(7n, 0n, 0n, 2n, 3n, 4n, 1n) == True{} : Bool}:
  {==}

def sh_0_0_2_3_4_2() -> {q7s(7n, 0n, 0n, 2n, 3n, 4n, 2n) == True{} : Bool}:
  {==}

def sh_0_0_2_3_4_3() -> {q7s(7n, 0n, 0n, 2n, 3n, 4n, 3n) == True{} : Bool}:
  {==}

def sh_0_0_2_3_4_4() -> {q7s(7n, 0n, 0n, 2n, 3n, 4n, 4n) == True{} : Bool}:
  {==}

def sh_0_0_2_3_4_5() -> {q7s(7n, 0n, 0n, 2n, 3n, 4n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_0_0_0_0() -> {q7s(7n, 0n, 1n, 0n, 0n, 0n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_0_0_0_1() -> {q7s(7n, 0n, 1n, 0n, 0n, 0n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_0_0_0_2() -> {q7s(7n, 0n, 1n, 0n, 0n, 0n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_0_0_0_3() -> {q7s(7n, 0n, 1n, 0n, 0n, 0n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_0_0_0_4() -> {q7s(7n, 0n, 1n, 0n, 0n, 0n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_0_0_0_5() -> {q7s(7n, 0n, 1n, 0n, 0n, 0n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_0_0_1_0() -> {q7s(7n, 0n, 1n, 0n, 0n, 1n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_0_0_1_1() -> {q7s(7n, 0n, 1n, 0n, 0n, 1n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_0_0_1_2() -> {q7s(7n, 0n, 1n, 0n, 0n, 1n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_0_0_1_3() -> {q7s(7n, 0n, 1n, 0n, 0n, 1n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_0_0_1_4() -> {q7s(7n, 0n, 1n, 0n, 0n, 1n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_0_0_1_5() -> {q7s(7n, 0n, 1n, 0n, 0n, 1n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_0_0_2_0() -> {q7s(7n, 0n, 1n, 0n, 0n, 2n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_0_0_2_1() -> {q7s(7n, 0n, 1n, 0n, 0n, 2n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_0_0_2_2() -> {q7s(7n, 0n, 1n, 0n, 0n, 2n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_0_0_2_3() -> {q7s(7n, 0n, 1n, 0n, 0n, 2n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_0_0_2_4() -> {q7s(7n, 0n, 1n, 0n, 0n, 2n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_0_0_2_5() -> {q7s(7n, 0n, 1n, 0n, 0n, 2n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_0_0_3_0() -> {q7s(7n, 0n, 1n, 0n, 0n, 3n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_0_0_3_1() -> {q7s(7n, 0n, 1n, 0n, 0n, 3n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_0_0_3_2() -> {q7s(7n, 0n, 1n, 0n, 0n, 3n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_0_0_3_3() -> {q7s(7n, 0n, 1n, 0n, 0n, 3n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_0_0_3_4() -> {q7s(7n, 0n, 1n, 0n, 0n, 3n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_0_0_3_5() -> {q7s(7n, 0n, 1n, 0n, 0n, 3n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_0_0_4_0() -> {q7s(7n, 0n, 1n, 0n, 0n, 4n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_0_0_4_1() -> {q7s(7n, 0n, 1n, 0n, 0n, 4n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_0_0_4_2() -> {q7s(7n, 0n, 1n, 0n, 0n, 4n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_0_0_4_3() -> {q7s(7n, 0n, 1n, 0n, 0n, 4n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_0_0_4_4() -> {q7s(7n, 0n, 1n, 0n, 0n, 4n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_0_0_4_5() -> {q7s(7n, 0n, 1n, 0n, 0n, 4n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_0_1_0_0() -> {q7s(7n, 0n, 1n, 0n, 1n, 0n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_0_1_0_1() -> {q7s(7n, 0n, 1n, 0n, 1n, 0n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_0_1_0_2() -> {q7s(7n, 0n, 1n, 0n, 1n, 0n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_0_1_0_3() -> {q7s(7n, 0n, 1n, 0n, 1n, 0n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_0_1_0_4() -> {q7s(7n, 0n, 1n, 0n, 1n, 0n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_0_1_0_5() -> {q7s(7n, 0n, 1n, 0n, 1n, 0n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_0_1_1_0() -> {q7s(7n, 0n, 1n, 0n, 1n, 1n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_0_1_1_1() -> {q7s(7n, 0n, 1n, 0n, 1n, 1n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_0_1_1_2() -> {q7s(7n, 0n, 1n, 0n, 1n, 1n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_0_1_1_3() -> {q7s(7n, 0n, 1n, 0n, 1n, 1n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_0_1_1_4() -> {q7s(7n, 0n, 1n, 0n, 1n, 1n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_0_1_1_5() -> {q7s(7n, 0n, 1n, 0n, 1n, 1n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_0_1_2_0() -> {q7s(7n, 0n, 1n, 0n, 1n, 2n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_0_1_2_1() -> {q7s(7n, 0n, 1n, 0n, 1n, 2n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_0_1_2_2() -> {q7s(7n, 0n, 1n, 0n, 1n, 2n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_0_1_2_3() -> {q7s(7n, 0n, 1n, 0n, 1n, 2n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_0_1_2_4() -> {q7s(7n, 0n, 1n, 0n, 1n, 2n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_0_1_2_5() -> {q7s(7n, 0n, 1n, 0n, 1n, 2n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_0_1_3_0() -> {q7s(7n, 0n, 1n, 0n, 1n, 3n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_0_1_3_1() -> {q7s(7n, 0n, 1n, 0n, 1n, 3n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_0_1_3_2() -> {q7s(7n, 0n, 1n, 0n, 1n, 3n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_0_1_3_3() -> {q7s(7n, 0n, 1n, 0n, 1n, 3n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_0_1_3_4() -> {q7s(7n, 0n, 1n, 0n, 1n, 3n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_0_1_3_5() -> {q7s(7n, 0n, 1n, 0n, 1n, 3n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_0_1_4_0() -> {q7s(7n, 0n, 1n, 0n, 1n, 4n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_0_1_4_1() -> {q7s(7n, 0n, 1n, 0n, 1n, 4n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_0_1_4_2() -> {q7s(7n, 0n, 1n, 0n, 1n, 4n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_0_1_4_3() -> {q7s(7n, 0n, 1n, 0n, 1n, 4n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_0_1_4_4() -> {q7s(7n, 0n, 1n, 0n, 1n, 4n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_0_1_4_5() -> {q7s(7n, 0n, 1n, 0n, 1n, 4n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_0_2_0_0() -> {q7s(7n, 0n, 1n, 0n, 2n, 0n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_0_2_0_1() -> {q7s(7n, 0n, 1n, 0n, 2n, 0n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_0_2_0_2() -> {q7s(7n, 0n, 1n, 0n, 2n, 0n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_0_2_0_3() -> {q7s(7n, 0n, 1n, 0n, 2n, 0n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_0_2_0_4() -> {q7s(7n, 0n, 1n, 0n, 2n, 0n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_0_2_0_5() -> {q7s(7n, 0n, 1n, 0n, 2n, 0n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_0_2_1_0() -> {q7s(7n, 0n, 1n, 0n, 2n, 1n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_0_2_1_1() -> {q7s(7n, 0n, 1n, 0n, 2n, 1n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_0_2_1_2() -> {q7s(7n, 0n, 1n, 0n, 2n, 1n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_0_2_1_3() -> {q7s(7n, 0n, 1n, 0n, 2n, 1n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_0_2_1_4() -> {q7s(7n, 0n, 1n, 0n, 2n, 1n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_0_2_1_5() -> {q7s(7n, 0n, 1n, 0n, 2n, 1n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_0_2_2_0() -> {q7s(7n, 0n, 1n, 0n, 2n, 2n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_0_2_2_1() -> {q7s(7n, 0n, 1n, 0n, 2n, 2n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_0_2_2_2() -> {q7s(7n, 0n, 1n, 0n, 2n, 2n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_0_2_2_3() -> {q7s(7n, 0n, 1n, 0n, 2n, 2n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_0_2_2_4() -> {q7s(7n, 0n, 1n, 0n, 2n, 2n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_0_2_2_5() -> {q7s(7n, 0n, 1n, 0n, 2n, 2n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_0_2_3_0() -> {q7s(7n, 0n, 1n, 0n, 2n, 3n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_0_2_3_1() -> {q7s(7n, 0n, 1n, 0n, 2n, 3n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_0_2_3_2() -> {q7s(7n, 0n, 1n, 0n, 2n, 3n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_0_2_3_3() -> {q7s(7n, 0n, 1n, 0n, 2n, 3n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_0_2_3_4() -> {q7s(7n, 0n, 1n, 0n, 2n, 3n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_0_2_3_5() -> {q7s(7n, 0n, 1n, 0n, 2n, 3n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_0_2_4_0() -> {q7s(7n, 0n, 1n, 0n, 2n, 4n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_0_2_4_1() -> {q7s(7n, 0n, 1n, 0n, 2n, 4n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_0_2_4_2() -> {q7s(7n, 0n, 1n, 0n, 2n, 4n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_0_2_4_3() -> {q7s(7n, 0n, 1n, 0n, 2n, 4n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_0_2_4_4() -> {q7s(7n, 0n, 1n, 0n, 2n, 4n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_0_2_4_5() -> {q7s(7n, 0n, 1n, 0n, 2n, 4n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_0_3_0_0() -> {q7s(7n, 0n, 1n, 0n, 3n, 0n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_0_3_0_1() -> {q7s(7n, 0n, 1n, 0n, 3n, 0n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_0_3_0_2() -> {q7s(7n, 0n, 1n, 0n, 3n, 0n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_0_3_0_3() -> {q7s(7n, 0n, 1n, 0n, 3n, 0n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_0_3_0_4() -> {q7s(7n, 0n, 1n, 0n, 3n, 0n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_0_3_0_5() -> {q7s(7n, 0n, 1n, 0n, 3n, 0n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_0_3_1_0() -> {q7s(7n, 0n, 1n, 0n, 3n, 1n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_0_3_1_1() -> {q7s(7n, 0n, 1n, 0n, 3n, 1n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_0_3_1_2() -> {q7s(7n, 0n, 1n, 0n, 3n, 1n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_0_3_1_3() -> {q7s(7n, 0n, 1n, 0n, 3n, 1n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_0_3_1_4() -> {q7s(7n, 0n, 1n, 0n, 3n, 1n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_0_3_1_5() -> {q7s(7n, 0n, 1n, 0n, 3n, 1n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_0_3_2_0() -> {q7s(7n, 0n, 1n, 0n, 3n, 2n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_0_3_2_1() -> {q7s(7n, 0n, 1n, 0n, 3n, 2n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_0_3_2_2() -> {q7s(7n, 0n, 1n, 0n, 3n, 2n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_0_3_2_3() -> {q7s(7n, 0n, 1n, 0n, 3n, 2n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_0_3_2_4() -> {q7s(7n, 0n, 1n, 0n, 3n, 2n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_0_3_2_5() -> {q7s(7n, 0n, 1n, 0n, 3n, 2n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_0_3_3_0() -> {q7s(7n, 0n, 1n, 0n, 3n, 3n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_0_3_3_1() -> {q7s(7n, 0n, 1n, 0n, 3n, 3n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_0_3_3_2() -> {q7s(7n, 0n, 1n, 0n, 3n, 3n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_0_3_3_3() -> {q7s(7n, 0n, 1n, 0n, 3n, 3n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_0_3_3_4() -> {q7s(7n, 0n, 1n, 0n, 3n, 3n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_0_3_3_5() -> {q7s(7n, 0n, 1n, 0n, 3n, 3n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_0_3_4_0() -> {q7s(7n, 0n, 1n, 0n, 3n, 4n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_0_3_4_1() -> {q7s(7n, 0n, 1n, 0n, 3n, 4n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_0_3_4_2() -> {q7s(7n, 0n, 1n, 0n, 3n, 4n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_0_3_4_3() -> {q7s(7n, 0n, 1n, 0n, 3n, 4n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_0_3_4_4() -> {q7s(7n, 0n, 1n, 0n, 3n, 4n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_0_3_4_5() -> {q7s(7n, 0n, 1n, 0n, 3n, 4n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_1_0_0_0() -> {q7s(7n, 0n, 1n, 1n, 0n, 0n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_1_0_0_1() -> {q7s(7n, 0n, 1n, 1n, 0n, 0n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_1_0_0_2() -> {q7s(7n, 0n, 1n, 1n, 0n, 0n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_1_0_0_3() -> {q7s(7n, 0n, 1n, 1n, 0n, 0n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_1_0_0_4() -> {q7s(7n, 0n, 1n, 1n, 0n, 0n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_1_0_0_5() -> {q7s(7n, 0n, 1n, 1n, 0n, 0n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_1_0_1_0() -> {q7s(7n, 0n, 1n, 1n, 0n, 1n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_1_0_1_1() -> {q7s(7n, 0n, 1n, 1n, 0n, 1n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_1_0_1_2() -> {q7s(7n, 0n, 1n, 1n, 0n, 1n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_1_0_1_3() -> {q7s(7n, 0n, 1n, 1n, 0n, 1n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_1_0_1_4() -> {q7s(7n, 0n, 1n, 1n, 0n, 1n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_1_0_1_5() -> {q7s(7n, 0n, 1n, 1n, 0n, 1n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_1_0_2_0() -> {q7s(7n, 0n, 1n, 1n, 0n, 2n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_1_0_2_1() -> {q7s(7n, 0n, 1n, 1n, 0n, 2n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_1_0_2_2() -> {q7s(7n, 0n, 1n, 1n, 0n, 2n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_1_0_2_3() -> {q7s(7n, 0n, 1n, 1n, 0n, 2n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_1_0_2_4() -> {q7s(7n, 0n, 1n, 1n, 0n, 2n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_1_0_2_5() -> {q7s(7n, 0n, 1n, 1n, 0n, 2n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_1_0_3_0() -> {q7s(7n, 0n, 1n, 1n, 0n, 3n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_1_0_3_1() -> {q7s(7n, 0n, 1n, 1n, 0n, 3n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_1_0_3_2() -> {q7s(7n, 0n, 1n, 1n, 0n, 3n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_1_0_3_3() -> {q7s(7n, 0n, 1n, 1n, 0n, 3n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_1_0_3_4() -> {q7s(7n, 0n, 1n, 1n, 0n, 3n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_1_0_3_5() -> {q7s(7n, 0n, 1n, 1n, 0n, 3n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_1_0_4_0() -> {q7s(7n, 0n, 1n, 1n, 0n, 4n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_1_0_4_1() -> {q7s(7n, 0n, 1n, 1n, 0n, 4n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_1_0_4_2() -> {q7s(7n, 0n, 1n, 1n, 0n, 4n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_1_0_4_3() -> {q7s(7n, 0n, 1n, 1n, 0n, 4n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_1_0_4_4() -> {q7s(7n, 0n, 1n, 1n, 0n, 4n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_1_0_4_5() -> {q7s(7n, 0n, 1n, 1n, 0n, 4n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_1_1_0_0() -> {q7s(7n, 0n, 1n, 1n, 1n, 0n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_1_1_0_1() -> {q7s(7n, 0n, 1n, 1n, 1n, 0n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_1_1_0_2() -> {q7s(7n, 0n, 1n, 1n, 1n, 0n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_1_1_0_3() -> {q7s(7n, 0n, 1n, 1n, 1n, 0n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_1_1_0_4() -> {q7s(7n, 0n, 1n, 1n, 1n, 0n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_1_1_0_5() -> {q7s(7n, 0n, 1n, 1n, 1n, 0n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_1_1_1_0() -> {q7s(7n, 0n, 1n, 1n, 1n, 1n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_1_1_1_1() -> {q7s(7n, 0n, 1n, 1n, 1n, 1n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_1_1_1_2() -> {q7s(7n, 0n, 1n, 1n, 1n, 1n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_1_1_1_3() -> {q7s(7n, 0n, 1n, 1n, 1n, 1n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_1_1_1_4() -> {q7s(7n, 0n, 1n, 1n, 1n, 1n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_1_1_1_5() -> {q7s(7n, 0n, 1n, 1n, 1n, 1n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_1_1_2_0() -> {q7s(7n, 0n, 1n, 1n, 1n, 2n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_1_1_2_1() -> {q7s(7n, 0n, 1n, 1n, 1n, 2n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_1_1_2_2() -> {q7s(7n, 0n, 1n, 1n, 1n, 2n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_1_1_2_3() -> {q7s(7n, 0n, 1n, 1n, 1n, 2n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_1_1_2_4() -> {q7s(7n, 0n, 1n, 1n, 1n, 2n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_1_1_2_5() -> {q7s(7n, 0n, 1n, 1n, 1n, 2n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_1_1_3_0() -> {q7s(7n, 0n, 1n, 1n, 1n, 3n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_1_1_3_1() -> {q7s(7n, 0n, 1n, 1n, 1n, 3n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_1_1_3_2() -> {q7s(7n, 0n, 1n, 1n, 1n, 3n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_1_1_3_3() -> {q7s(7n, 0n, 1n, 1n, 1n, 3n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_1_1_3_4() -> {q7s(7n, 0n, 1n, 1n, 1n, 3n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_1_1_3_5() -> {q7s(7n, 0n, 1n, 1n, 1n, 3n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_1_1_4_0() -> {q7s(7n, 0n, 1n, 1n, 1n, 4n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_1_1_4_1() -> {q7s(7n, 0n, 1n, 1n, 1n, 4n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_1_1_4_2() -> {q7s(7n, 0n, 1n, 1n, 1n, 4n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_1_1_4_3() -> {q7s(7n, 0n, 1n, 1n, 1n, 4n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_1_1_4_4() -> {q7s(7n, 0n, 1n, 1n, 1n, 4n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_1_1_4_5() -> {q7s(7n, 0n, 1n, 1n, 1n, 4n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_1_2_0_0() -> {q7s(7n, 0n, 1n, 1n, 2n, 0n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_1_2_0_1() -> {q7s(7n, 0n, 1n, 1n, 2n, 0n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_1_2_0_2() -> {q7s(7n, 0n, 1n, 1n, 2n, 0n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_1_2_0_3() -> {q7s(7n, 0n, 1n, 1n, 2n, 0n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_1_2_0_4() -> {q7s(7n, 0n, 1n, 1n, 2n, 0n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_1_2_0_5() -> {q7s(7n, 0n, 1n, 1n, 2n, 0n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_1_2_1_0() -> {q7s(7n, 0n, 1n, 1n, 2n, 1n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_1_2_1_1() -> {q7s(7n, 0n, 1n, 1n, 2n, 1n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_1_2_1_2() -> {q7s(7n, 0n, 1n, 1n, 2n, 1n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_1_2_1_3() -> {q7s(7n, 0n, 1n, 1n, 2n, 1n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_1_2_1_4() -> {q7s(7n, 0n, 1n, 1n, 2n, 1n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_1_2_1_5() -> {q7s(7n, 0n, 1n, 1n, 2n, 1n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_1_2_2_0() -> {q7s(7n, 0n, 1n, 1n, 2n, 2n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_1_2_2_1() -> {q7s(7n, 0n, 1n, 1n, 2n, 2n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_1_2_2_2() -> {q7s(7n, 0n, 1n, 1n, 2n, 2n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_1_2_2_3() -> {q7s(7n, 0n, 1n, 1n, 2n, 2n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_1_2_2_4() -> {q7s(7n, 0n, 1n, 1n, 2n, 2n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_1_2_2_5() -> {q7s(7n, 0n, 1n, 1n, 2n, 2n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_1_2_3_0() -> {q7s(7n, 0n, 1n, 1n, 2n, 3n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_1_2_3_1() -> {q7s(7n, 0n, 1n, 1n, 2n, 3n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_1_2_3_2() -> {q7s(7n, 0n, 1n, 1n, 2n, 3n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_1_2_3_3() -> {q7s(7n, 0n, 1n, 1n, 2n, 3n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_1_2_3_4() -> {q7s(7n, 0n, 1n, 1n, 2n, 3n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_1_2_3_5() -> {q7s(7n, 0n, 1n, 1n, 2n, 3n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_1_2_4_0() -> {q7s(7n, 0n, 1n, 1n, 2n, 4n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_1_2_4_1() -> {q7s(7n, 0n, 1n, 1n, 2n, 4n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_1_2_4_2() -> {q7s(7n, 0n, 1n, 1n, 2n, 4n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_1_2_4_3() -> {q7s(7n, 0n, 1n, 1n, 2n, 4n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_1_2_4_4() -> {q7s(7n, 0n, 1n, 1n, 2n, 4n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_1_2_4_5() -> {q7s(7n, 0n, 1n, 1n, 2n, 4n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_1_3_0_0() -> {q7s(7n, 0n, 1n, 1n, 3n, 0n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_1_3_0_1() -> {q7s(7n, 0n, 1n, 1n, 3n, 0n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_1_3_0_2() -> {q7s(7n, 0n, 1n, 1n, 3n, 0n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_1_3_0_3() -> {q7s(7n, 0n, 1n, 1n, 3n, 0n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_1_3_0_4() -> {q7s(7n, 0n, 1n, 1n, 3n, 0n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_1_3_0_5() -> {q7s(7n, 0n, 1n, 1n, 3n, 0n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_1_3_1_0() -> {q7s(7n, 0n, 1n, 1n, 3n, 1n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_1_3_1_1() -> {q7s(7n, 0n, 1n, 1n, 3n, 1n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_1_3_1_2() -> {q7s(7n, 0n, 1n, 1n, 3n, 1n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_1_3_1_3() -> {q7s(7n, 0n, 1n, 1n, 3n, 1n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_1_3_1_4() -> {q7s(7n, 0n, 1n, 1n, 3n, 1n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_1_3_1_5() -> {q7s(7n, 0n, 1n, 1n, 3n, 1n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_1_3_2_0() -> {q7s(7n, 0n, 1n, 1n, 3n, 2n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_1_3_2_1() -> {q7s(7n, 0n, 1n, 1n, 3n, 2n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_1_3_2_2() -> {q7s(7n, 0n, 1n, 1n, 3n, 2n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_1_3_2_3() -> {q7s(7n, 0n, 1n, 1n, 3n, 2n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_1_3_2_4() -> {q7s(7n, 0n, 1n, 1n, 3n, 2n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_1_3_2_5() -> {q7s(7n, 0n, 1n, 1n, 3n, 2n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_1_3_3_0() -> {q7s(7n, 0n, 1n, 1n, 3n, 3n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_1_3_3_1() -> {q7s(7n, 0n, 1n, 1n, 3n, 3n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_1_3_3_2() -> {q7s(7n, 0n, 1n, 1n, 3n, 3n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_1_3_3_3() -> {q7s(7n, 0n, 1n, 1n, 3n, 3n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_1_3_3_4() -> {q7s(7n, 0n, 1n, 1n, 3n, 3n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_1_3_3_5() -> {q7s(7n, 0n, 1n, 1n, 3n, 3n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_1_3_4_0() -> {q7s(7n, 0n, 1n, 1n, 3n, 4n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_1_3_4_1() -> {q7s(7n, 0n, 1n, 1n, 3n, 4n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_1_3_4_2() -> {q7s(7n, 0n, 1n, 1n, 3n, 4n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_1_3_4_3() -> {q7s(7n, 0n, 1n, 1n, 3n, 4n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_1_3_4_4() -> {q7s(7n, 0n, 1n, 1n, 3n, 4n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_1_3_4_5() -> {q7s(7n, 0n, 1n, 1n, 3n, 4n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_2_0_0_0() -> {q7s(7n, 0n, 1n, 2n, 0n, 0n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_2_0_0_1() -> {q7s(7n, 0n, 1n, 2n, 0n, 0n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_2_0_0_2() -> {q7s(7n, 0n, 1n, 2n, 0n, 0n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_2_0_0_3() -> {q7s(7n, 0n, 1n, 2n, 0n, 0n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_2_0_0_4() -> {q7s(7n, 0n, 1n, 2n, 0n, 0n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_2_0_0_5() -> {q7s(7n, 0n, 1n, 2n, 0n, 0n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_2_0_1_0() -> {q7s(7n, 0n, 1n, 2n, 0n, 1n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_2_0_1_1() -> {q7s(7n, 0n, 1n, 2n, 0n, 1n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_2_0_1_2() -> {q7s(7n, 0n, 1n, 2n, 0n, 1n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_2_0_1_3() -> {q7s(7n, 0n, 1n, 2n, 0n, 1n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_2_0_1_4() -> {q7s(7n, 0n, 1n, 2n, 0n, 1n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_2_0_1_5() -> {q7s(7n, 0n, 1n, 2n, 0n, 1n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_2_0_2_0() -> {q7s(7n, 0n, 1n, 2n, 0n, 2n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_2_0_2_1() -> {q7s(7n, 0n, 1n, 2n, 0n, 2n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_2_0_2_2() -> {q7s(7n, 0n, 1n, 2n, 0n, 2n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_2_0_2_3() -> {q7s(7n, 0n, 1n, 2n, 0n, 2n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_2_0_2_4() -> {q7s(7n, 0n, 1n, 2n, 0n, 2n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_2_0_2_5() -> {q7s(7n, 0n, 1n, 2n, 0n, 2n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_2_0_3_0() -> {q7s(7n, 0n, 1n, 2n, 0n, 3n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_2_0_3_1() -> {q7s(7n, 0n, 1n, 2n, 0n, 3n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_2_0_3_2() -> {q7s(7n, 0n, 1n, 2n, 0n, 3n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_2_0_3_3() -> {q7s(7n, 0n, 1n, 2n, 0n, 3n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_2_0_3_4() -> {q7s(7n, 0n, 1n, 2n, 0n, 3n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_2_0_3_5() -> {q7s(7n, 0n, 1n, 2n, 0n, 3n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_2_0_4_0() -> {q7s(7n, 0n, 1n, 2n, 0n, 4n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_2_0_4_1() -> {q7s(7n, 0n, 1n, 2n, 0n, 4n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_2_0_4_2() -> {q7s(7n, 0n, 1n, 2n, 0n, 4n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_2_0_4_3() -> {q7s(7n, 0n, 1n, 2n, 0n, 4n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_2_0_4_4() -> {q7s(7n, 0n, 1n, 2n, 0n, 4n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_2_0_4_5() -> {q7s(7n, 0n, 1n, 2n, 0n, 4n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_2_1_0_0() -> {q7s(7n, 0n, 1n, 2n, 1n, 0n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_2_1_0_1() -> {q7s(7n, 0n, 1n, 2n, 1n, 0n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_2_1_0_2() -> {q7s(7n, 0n, 1n, 2n, 1n, 0n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_2_1_0_3() -> {q7s(7n, 0n, 1n, 2n, 1n, 0n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_2_1_0_4() -> {q7s(7n, 0n, 1n, 2n, 1n, 0n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_2_1_0_5() -> {q7s(7n, 0n, 1n, 2n, 1n, 0n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_2_1_1_0() -> {q7s(7n, 0n, 1n, 2n, 1n, 1n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_2_1_1_1() -> {q7s(7n, 0n, 1n, 2n, 1n, 1n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_2_1_1_2() -> {q7s(7n, 0n, 1n, 2n, 1n, 1n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_2_1_1_3() -> {q7s(7n, 0n, 1n, 2n, 1n, 1n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_2_1_1_4() -> {q7s(7n, 0n, 1n, 2n, 1n, 1n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_2_1_1_5() -> {q7s(7n, 0n, 1n, 2n, 1n, 1n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_2_1_2_0() -> {q7s(7n, 0n, 1n, 2n, 1n, 2n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_2_1_2_1() -> {q7s(7n, 0n, 1n, 2n, 1n, 2n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_2_1_2_2() -> {q7s(7n, 0n, 1n, 2n, 1n, 2n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_2_1_2_3() -> {q7s(7n, 0n, 1n, 2n, 1n, 2n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_2_1_2_4() -> {q7s(7n, 0n, 1n, 2n, 1n, 2n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_2_1_2_5() -> {q7s(7n, 0n, 1n, 2n, 1n, 2n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_2_1_3_0() -> {q7s(7n, 0n, 1n, 2n, 1n, 3n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_2_1_3_1() -> {q7s(7n, 0n, 1n, 2n, 1n, 3n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_2_1_3_2() -> {q7s(7n, 0n, 1n, 2n, 1n, 3n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_2_1_3_3() -> {q7s(7n, 0n, 1n, 2n, 1n, 3n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_2_1_3_4() -> {q7s(7n, 0n, 1n, 2n, 1n, 3n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_2_1_3_5() -> {q7s(7n, 0n, 1n, 2n, 1n, 3n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_2_1_4_0() -> {q7s(7n, 0n, 1n, 2n, 1n, 4n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_2_1_4_1() -> {q7s(7n, 0n, 1n, 2n, 1n, 4n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_2_1_4_2() -> {q7s(7n, 0n, 1n, 2n, 1n, 4n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_2_1_4_3() -> {q7s(7n, 0n, 1n, 2n, 1n, 4n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_2_1_4_4() -> {q7s(7n, 0n, 1n, 2n, 1n, 4n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_2_1_4_5() -> {q7s(7n, 0n, 1n, 2n, 1n, 4n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_2_2_0_0() -> {q7s(7n, 0n, 1n, 2n, 2n, 0n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_2_2_0_1() -> {q7s(7n, 0n, 1n, 2n, 2n, 0n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_2_2_0_2() -> {q7s(7n, 0n, 1n, 2n, 2n, 0n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_2_2_0_3() -> {q7s(7n, 0n, 1n, 2n, 2n, 0n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_2_2_0_4() -> {q7s(7n, 0n, 1n, 2n, 2n, 0n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_2_2_0_5() -> {q7s(7n, 0n, 1n, 2n, 2n, 0n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_2_2_1_0() -> {q7s(7n, 0n, 1n, 2n, 2n, 1n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_2_2_1_1() -> {q7s(7n, 0n, 1n, 2n, 2n, 1n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_2_2_1_2() -> {q7s(7n, 0n, 1n, 2n, 2n, 1n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_2_2_1_3() -> {q7s(7n, 0n, 1n, 2n, 2n, 1n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_2_2_1_4() -> {q7s(7n, 0n, 1n, 2n, 2n, 1n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_2_2_1_5() -> {q7s(7n, 0n, 1n, 2n, 2n, 1n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_2_2_2_0() -> {q7s(7n, 0n, 1n, 2n, 2n, 2n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_2_2_2_1() -> {q7s(7n, 0n, 1n, 2n, 2n, 2n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_2_2_2_2() -> {q7s(7n, 0n, 1n, 2n, 2n, 2n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_2_2_2_3() -> {q7s(7n, 0n, 1n, 2n, 2n, 2n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_2_2_2_4() -> {q7s(7n, 0n, 1n, 2n, 2n, 2n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_2_2_2_5() -> {q7s(7n, 0n, 1n, 2n, 2n, 2n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_2_2_3_0() -> {q7s(7n, 0n, 1n, 2n, 2n, 3n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_2_2_3_1() -> {q7s(7n, 0n, 1n, 2n, 2n, 3n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_2_2_3_2() -> {q7s(7n, 0n, 1n, 2n, 2n, 3n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_2_2_3_3() -> {q7s(7n, 0n, 1n, 2n, 2n, 3n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_2_2_3_4() -> {q7s(7n, 0n, 1n, 2n, 2n, 3n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_2_2_3_5() -> {q7s(7n, 0n, 1n, 2n, 2n, 3n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_2_2_4_0() -> {q7s(7n, 0n, 1n, 2n, 2n, 4n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_2_2_4_1() -> {q7s(7n, 0n, 1n, 2n, 2n, 4n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_2_2_4_2() -> {q7s(7n, 0n, 1n, 2n, 2n, 4n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_2_2_4_3() -> {q7s(7n, 0n, 1n, 2n, 2n, 4n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_2_2_4_4() -> {q7s(7n, 0n, 1n, 2n, 2n, 4n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_2_2_4_5() -> {q7s(7n, 0n, 1n, 2n, 2n, 4n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_2_3_0_0() -> {q7s(7n, 0n, 1n, 2n, 3n, 0n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_2_3_0_1() -> {q7s(7n, 0n, 1n, 2n, 3n, 0n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_2_3_0_2() -> {q7s(7n, 0n, 1n, 2n, 3n, 0n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_2_3_0_3() -> {q7s(7n, 0n, 1n, 2n, 3n, 0n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_2_3_0_4() -> {q7s(7n, 0n, 1n, 2n, 3n, 0n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_2_3_0_5() -> {q7s(7n, 0n, 1n, 2n, 3n, 0n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_2_3_1_0() -> {q7s(7n, 0n, 1n, 2n, 3n, 1n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_2_3_1_1() -> {q7s(7n, 0n, 1n, 2n, 3n, 1n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_2_3_1_2() -> {q7s(7n, 0n, 1n, 2n, 3n, 1n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_2_3_1_3() -> {q7s(7n, 0n, 1n, 2n, 3n, 1n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_2_3_1_4() -> {q7s(7n, 0n, 1n, 2n, 3n, 1n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_2_3_1_5() -> {q7s(7n, 0n, 1n, 2n, 3n, 1n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_2_3_2_0() -> {q7s(7n, 0n, 1n, 2n, 3n, 2n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_2_3_2_1() -> {q7s(7n, 0n, 1n, 2n, 3n, 2n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_2_3_2_2() -> {q7s(7n, 0n, 1n, 2n, 3n, 2n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_2_3_2_3() -> {q7s(7n, 0n, 1n, 2n, 3n, 2n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_2_3_2_4() -> {q7s(7n, 0n, 1n, 2n, 3n, 2n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_2_3_2_5() -> {q7s(7n, 0n, 1n, 2n, 3n, 2n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_2_3_3_0() -> {q7s(7n, 0n, 1n, 2n, 3n, 3n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_2_3_3_1() -> {q7s(7n, 0n, 1n, 2n, 3n, 3n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_2_3_3_2() -> {q7s(7n, 0n, 1n, 2n, 3n, 3n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_2_3_3_3() -> {q7s(7n, 0n, 1n, 2n, 3n, 3n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_2_3_3_4() -> {q7s(7n, 0n, 1n, 2n, 3n, 3n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_2_3_3_5() -> {q7s(7n, 0n, 1n, 2n, 3n, 3n, 5n) == True{} : Bool}:
  {==}

def sh_0_1_2_3_4_0() -> {q7s(7n, 0n, 1n, 2n, 3n, 4n, 0n) == True{} : Bool}:
  {==}

def sh_0_1_2_3_4_1() -> {q7s(7n, 0n, 1n, 2n, 3n, 4n, 1n) == True{} : Bool}:
  {==}

def sh_0_1_2_3_4_2() -> {q7s(7n, 0n, 1n, 2n, 3n, 4n, 2n) == True{} : Bool}:
  {==}

def sh_0_1_2_3_4_3() -> {q7s(7n, 0n, 1n, 2n, 3n, 4n, 3n) == True{} : Bool}:
  {==}

def sh_0_1_2_3_4_4() -> {q7s(7n, 0n, 1n, 2n, 3n, 4n, 4n) == True{} : Bool}:
  {==}

def sh_0_1_2_3_4_5() -> {q7s(7n, 0n, 1n, 2n, 3n, 4n, 5n) == True{} : Bool}:
  {==}

def sq5_0_0_0_0() -> {q5s(5n, 6n, 7n, 0n, 0n, 0n, 0n) == True{} : Bool}:
  q5s_i(4n, 6n, 7n, 0n, 0n, 0n, 0n, q5s_i(3n, 6n, 7n, 0n, 0n, 0n, 0n, q5s_i(2n, 6n, 7n, 0n, 0n, 0n, 0n, q5s_i(1n, 6n, 7n, 0n, 0n, 0n, 0n, q5s_i(0n, 6n, 7n, 0n, 0n, 0n, 0n, {==}, q6s_i(5n, 7n, 0n, 0n, 0n, 0n, 0n, q6s_i(4n, 7n, 0n, 0n, 0n, 0n, 0n, q6s_i(3n, 7n, 0n, 0n, 0n, 0n, 0n, q6s_i(2n, 7n, 0n, 0n, 0n, 0n, 0n, q6s_i(1n, 7n, 0n, 0n, 0n, 0n, 0n, q6s_i(0n, 7n, 0n, 0n, 0n, 0n, 0n, {==}, sh_0_0_0_0_0_0()), sh_0_0_0_0_0_1()), sh_0_0_0_0_0_2()), sh_0_0_0_0_0_3()), sh_0_0_0_0_0_4()), sh_0_0_0_0_0_5())), q6s_i(5n, 7n, 0n, 0n, 0n, 0n, 1n, q6s_i(4n, 7n, 0n, 0n, 0n, 0n, 1n, q6s_i(3n, 7n, 0n, 0n, 0n, 0n, 1n, q6s_i(2n, 7n, 0n, 0n, 0n, 0n, 1n, q6s_i(1n, 7n, 0n, 0n, 0n, 0n, 1n, q6s_i(0n, 7n, 0n, 0n, 0n, 0n, 1n, {==}, sh_0_0_0_0_1_0()), sh_0_0_0_0_1_1()), sh_0_0_0_0_1_2()), sh_0_0_0_0_1_3()), sh_0_0_0_0_1_4()), sh_0_0_0_0_1_5())), q6s_i(5n, 7n, 0n, 0n, 0n, 0n, 2n, q6s_i(4n, 7n, 0n, 0n, 0n, 0n, 2n, q6s_i(3n, 7n, 0n, 0n, 0n, 0n, 2n, q6s_i(2n, 7n, 0n, 0n, 0n, 0n, 2n, q6s_i(1n, 7n, 0n, 0n, 0n, 0n, 2n, q6s_i(0n, 7n, 0n, 0n, 0n, 0n, 2n, {==}, sh_0_0_0_0_2_0()), sh_0_0_0_0_2_1()), sh_0_0_0_0_2_2()), sh_0_0_0_0_2_3()), sh_0_0_0_0_2_4()), sh_0_0_0_0_2_5())), q6s_i(5n, 7n, 0n, 0n, 0n, 0n, 3n, q6s_i(4n, 7n, 0n, 0n, 0n, 0n, 3n, q6s_i(3n, 7n, 0n, 0n, 0n, 0n, 3n, q6s_i(2n, 7n, 0n, 0n, 0n, 0n, 3n, q6s_i(1n, 7n, 0n, 0n, 0n, 0n, 3n, q6s_i(0n, 7n, 0n, 0n, 0n, 0n, 3n, {==}, sh_0_0_0_0_3_0()), sh_0_0_0_0_3_1()), sh_0_0_0_0_3_2()), sh_0_0_0_0_3_3()), sh_0_0_0_0_3_4()), sh_0_0_0_0_3_5())), q6s_i(5n, 7n, 0n, 0n, 0n, 0n, 4n, q6s_i(4n, 7n, 0n, 0n, 0n, 0n, 4n, q6s_i(3n, 7n, 0n, 0n, 0n, 0n, 4n, q6s_i(2n, 7n, 0n, 0n, 0n, 0n, 4n, q6s_i(1n, 7n, 0n, 0n, 0n, 0n, 4n, q6s_i(0n, 7n, 0n, 0n, 0n, 0n, 4n, {==}, sh_0_0_0_0_4_0()), sh_0_0_0_0_4_1()), sh_0_0_0_0_4_2()), sh_0_0_0_0_4_3()), sh_0_0_0_0_4_4()), sh_0_0_0_0_4_5()))

def sq5_0_0_0_1() -> {q5s(5n, 6n, 7n, 0n, 0n, 0n, 1n) == True{} : Bool}:
  q5s_i(4n, 6n, 7n, 0n, 0n, 0n, 1n, q5s_i(3n, 6n, 7n, 0n, 0n, 0n, 1n, q5s_i(2n, 6n, 7n, 0n, 0n, 0n, 1n, q5s_i(1n, 6n, 7n, 0n, 0n, 0n, 1n, q5s_i(0n, 6n, 7n, 0n, 0n, 0n, 1n, {==}, q6s_i(5n, 7n, 0n, 0n, 0n, 1n, 0n, q6s_i(4n, 7n, 0n, 0n, 0n, 1n, 0n, q6s_i(3n, 7n, 0n, 0n, 0n, 1n, 0n, q6s_i(2n, 7n, 0n, 0n, 0n, 1n, 0n, q6s_i(1n, 7n, 0n, 0n, 0n, 1n, 0n, q6s_i(0n, 7n, 0n, 0n, 0n, 1n, 0n, {==}, sh_0_0_0_1_0_0()), sh_0_0_0_1_0_1()), sh_0_0_0_1_0_2()), sh_0_0_0_1_0_3()), sh_0_0_0_1_0_4()), sh_0_0_0_1_0_5())), q6s_i(5n, 7n, 0n, 0n, 0n, 1n, 1n, q6s_i(4n, 7n, 0n, 0n, 0n, 1n, 1n, q6s_i(3n, 7n, 0n, 0n, 0n, 1n, 1n, q6s_i(2n, 7n, 0n, 0n, 0n, 1n, 1n, q6s_i(1n, 7n, 0n, 0n, 0n, 1n, 1n, q6s_i(0n, 7n, 0n, 0n, 0n, 1n, 1n, {==}, sh_0_0_0_1_1_0()), sh_0_0_0_1_1_1()), sh_0_0_0_1_1_2()), sh_0_0_0_1_1_3()), sh_0_0_0_1_1_4()), sh_0_0_0_1_1_5())), q6s_i(5n, 7n, 0n, 0n, 0n, 1n, 2n, q6s_i(4n, 7n, 0n, 0n, 0n, 1n, 2n, q6s_i(3n, 7n, 0n, 0n, 0n, 1n, 2n, q6s_i(2n, 7n, 0n, 0n, 0n, 1n, 2n, q6s_i(1n, 7n, 0n, 0n, 0n, 1n, 2n, q6s_i(0n, 7n, 0n, 0n, 0n, 1n, 2n, {==}, sh_0_0_0_1_2_0()), sh_0_0_0_1_2_1()), sh_0_0_0_1_2_2()), sh_0_0_0_1_2_3()), sh_0_0_0_1_2_4()), sh_0_0_0_1_2_5())), q6s_i(5n, 7n, 0n, 0n, 0n, 1n, 3n, q6s_i(4n, 7n, 0n, 0n, 0n, 1n, 3n, q6s_i(3n, 7n, 0n, 0n, 0n, 1n, 3n, q6s_i(2n, 7n, 0n, 0n, 0n, 1n, 3n, q6s_i(1n, 7n, 0n, 0n, 0n, 1n, 3n, q6s_i(0n, 7n, 0n, 0n, 0n, 1n, 3n, {==}, sh_0_0_0_1_3_0()), sh_0_0_0_1_3_1()), sh_0_0_0_1_3_2()), sh_0_0_0_1_3_3()), sh_0_0_0_1_3_4()), sh_0_0_0_1_3_5())), q6s_i(5n, 7n, 0n, 0n, 0n, 1n, 4n, q6s_i(4n, 7n, 0n, 0n, 0n, 1n, 4n, q6s_i(3n, 7n, 0n, 0n, 0n, 1n, 4n, q6s_i(2n, 7n, 0n, 0n, 0n, 1n, 4n, q6s_i(1n, 7n, 0n, 0n, 0n, 1n, 4n, q6s_i(0n, 7n, 0n, 0n, 0n, 1n, 4n, {==}, sh_0_0_0_1_4_0()), sh_0_0_0_1_4_1()), sh_0_0_0_1_4_2()), sh_0_0_0_1_4_3()), sh_0_0_0_1_4_4()), sh_0_0_0_1_4_5()))

def sq5_0_0_0_2() -> {q5s(5n, 6n, 7n, 0n, 0n, 0n, 2n) == True{} : Bool}:
  q5s_i(4n, 6n, 7n, 0n, 0n, 0n, 2n, q5s_i(3n, 6n, 7n, 0n, 0n, 0n, 2n, q5s_i(2n, 6n, 7n, 0n, 0n, 0n, 2n, q5s_i(1n, 6n, 7n, 0n, 0n, 0n, 2n, q5s_i(0n, 6n, 7n, 0n, 0n, 0n, 2n, {==}, q6s_i(5n, 7n, 0n, 0n, 0n, 2n, 0n, q6s_i(4n, 7n, 0n, 0n, 0n, 2n, 0n, q6s_i(3n, 7n, 0n, 0n, 0n, 2n, 0n, q6s_i(2n, 7n, 0n, 0n, 0n, 2n, 0n, q6s_i(1n, 7n, 0n, 0n, 0n, 2n, 0n, q6s_i(0n, 7n, 0n, 0n, 0n, 2n, 0n, {==}, sh_0_0_0_2_0_0()), sh_0_0_0_2_0_1()), sh_0_0_0_2_0_2()), sh_0_0_0_2_0_3()), sh_0_0_0_2_0_4()), sh_0_0_0_2_0_5())), q6s_i(5n, 7n, 0n, 0n, 0n, 2n, 1n, q6s_i(4n, 7n, 0n, 0n, 0n, 2n, 1n, q6s_i(3n, 7n, 0n, 0n, 0n, 2n, 1n, q6s_i(2n, 7n, 0n, 0n, 0n, 2n, 1n, q6s_i(1n, 7n, 0n, 0n, 0n, 2n, 1n, q6s_i(0n, 7n, 0n, 0n, 0n, 2n, 1n, {==}, sh_0_0_0_2_1_0()), sh_0_0_0_2_1_1()), sh_0_0_0_2_1_2()), sh_0_0_0_2_1_3()), sh_0_0_0_2_1_4()), sh_0_0_0_2_1_5())), q6s_i(5n, 7n, 0n, 0n, 0n, 2n, 2n, q6s_i(4n, 7n, 0n, 0n, 0n, 2n, 2n, q6s_i(3n, 7n, 0n, 0n, 0n, 2n, 2n, q6s_i(2n, 7n, 0n, 0n, 0n, 2n, 2n, q6s_i(1n, 7n, 0n, 0n, 0n, 2n, 2n, q6s_i(0n, 7n, 0n, 0n, 0n, 2n, 2n, {==}, sh_0_0_0_2_2_0()), sh_0_0_0_2_2_1()), sh_0_0_0_2_2_2()), sh_0_0_0_2_2_3()), sh_0_0_0_2_2_4()), sh_0_0_0_2_2_5())), q6s_i(5n, 7n, 0n, 0n, 0n, 2n, 3n, q6s_i(4n, 7n, 0n, 0n, 0n, 2n, 3n, q6s_i(3n, 7n, 0n, 0n, 0n, 2n, 3n, q6s_i(2n, 7n, 0n, 0n, 0n, 2n, 3n, q6s_i(1n, 7n, 0n, 0n, 0n, 2n, 3n, q6s_i(0n, 7n, 0n, 0n, 0n, 2n, 3n, {==}, sh_0_0_0_2_3_0()), sh_0_0_0_2_3_1()), sh_0_0_0_2_3_2()), sh_0_0_0_2_3_3()), sh_0_0_0_2_3_4()), sh_0_0_0_2_3_5())), q6s_i(5n, 7n, 0n, 0n, 0n, 2n, 4n, q6s_i(4n, 7n, 0n, 0n, 0n, 2n, 4n, q6s_i(3n, 7n, 0n, 0n, 0n, 2n, 4n, q6s_i(2n, 7n, 0n, 0n, 0n, 2n, 4n, q6s_i(1n, 7n, 0n, 0n, 0n, 2n, 4n, q6s_i(0n, 7n, 0n, 0n, 0n, 2n, 4n, {==}, sh_0_0_0_2_4_0()), sh_0_0_0_2_4_1()), sh_0_0_0_2_4_2()), sh_0_0_0_2_4_3()), sh_0_0_0_2_4_4()), sh_0_0_0_2_4_5()))

def sq5_0_0_0_3() -> {q5s(5n, 6n, 7n, 0n, 0n, 0n, 3n) == True{} : Bool}:
  q5s_i(4n, 6n, 7n, 0n, 0n, 0n, 3n, q5s_i(3n, 6n, 7n, 0n, 0n, 0n, 3n, q5s_i(2n, 6n, 7n, 0n, 0n, 0n, 3n, q5s_i(1n, 6n, 7n, 0n, 0n, 0n, 3n, q5s_i(0n, 6n, 7n, 0n, 0n, 0n, 3n, {==}, q6s_i(5n, 7n, 0n, 0n, 0n, 3n, 0n, q6s_i(4n, 7n, 0n, 0n, 0n, 3n, 0n, q6s_i(3n, 7n, 0n, 0n, 0n, 3n, 0n, q6s_i(2n, 7n, 0n, 0n, 0n, 3n, 0n, q6s_i(1n, 7n, 0n, 0n, 0n, 3n, 0n, q6s_i(0n, 7n, 0n, 0n, 0n, 3n, 0n, {==}, sh_0_0_0_3_0_0()), sh_0_0_0_3_0_1()), sh_0_0_0_3_0_2()), sh_0_0_0_3_0_3()), sh_0_0_0_3_0_4()), sh_0_0_0_3_0_5())), q6s_i(5n, 7n, 0n, 0n, 0n, 3n, 1n, q6s_i(4n, 7n, 0n, 0n, 0n, 3n, 1n, q6s_i(3n, 7n, 0n, 0n, 0n, 3n, 1n, q6s_i(2n, 7n, 0n, 0n, 0n, 3n, 1n, q6s_i(1n, 7n, 0n, 0n, 0n, 3n, 1n, q6s_i(0n, 7n, 0n, 0n, 0n, 3n, 1n, {==}, sh_0_0_0_3_1_0()), sh_0_0_0_3_1_1()), sh_0_0_0_3_1_2()), sh_0_0_0_3_1_3()), sh_0_0_0_3_1_4()), sh_0_0_0_3_1_5())), q6s_i(5n, 7n, 0n, 0n, 0n, 3n, 2n, q6s_i(4n, 7n, 0n, 0n, 0n, 3n, 2n, q6s_i(3n, 7n, 0n, 0n, 0n, 3n, 2n, q6s_i(2n, 7n, 0n, 0n, 0n, 3n, 2n, q6s_i(1n, 7n, 0n, 0n, 0n, 3n, 2n, q6s_i(0n, 7n, 0n, 0n, 0n, 3n, 2n, {==}, sh_0_0_0_3_2_0()), sh_0_0_0_3_2_1()), sh_0_0_0_3_2_2()), sh_0_0_0_3_2_3()), sh_0_0_0_3_2_4()), sh_0_0_0_3_2_5())), q6s_i(5n, 7n, 0n, 0n, 0n, 3n, 3n, q6s_i(4n, 7n, 0n, 0n, 0n, 3n, 3n, q6s_i(3n, 7n, 0n, 0n, 0n, 3n, 3n, q6s_i(2n, 7n, 0n, 0n, 0n, 3n, 3n, q6s_i(1n, 7n, 0n, 0n, 0n, 3n, 3n, q6s_i(0n, 7n, 0n, 0n, 0n, 3n, 3n, {==}, sh_0_0_0_3_3_0()), sh_0_0_0_3_3_1()), sh_0_0_0_3_3_2()), sh_0_0_0_3_3_3()), sh_0_0_0_3_3_4()), sh_0_0_0_3_3_5())), q6s_i(5n, 7n, 0n, 0n, 0n, 3n, 4n, q6s_i(4n, 7n, 0n, 0n, 0n, 3n, 4n, q6s_i(3n, 7n, 0n, 0n, 0n, 3n, 4n, q6s_i(2n, 7n, 0n, 0n, 0n, 3n, 4n, q6s_i(1n, 7n, 0n, 0n, 0n, 3n, 4n, q6s_i(0n, 7n, 0n, 0n, 0n, 3n, 4n, {==}, sh_0_0_0_3_4_0()), sh_0_0_0_3_4_1()), sh_0_0_0_3_4_2()), sh_0_0_0_3_4_3()), sh_0_0_0_3_4_4()), sh_0_0_0_3_4_5()))

def sq5_0_0_1_0() -> {q5s(5n, 6n, 7n, 0n, 0n, 1n, 0n) == True{} : Bool}:
  q5s_i(4n, 6n, 7n, 0n, 0n, 1n, 0n, q5s_i(3n, 6n, 7n, 0n, 0n, 1n, 0n, q5s_i(2n, 6n, 7n, 0n, 0n, 1n, 0n, q5s_i(1n, 6n, 7n, 0n, 0n, 1n, 0n, q5s_i(0n, 6n, 7n, 0n, 0n, 1n, 0n, {==}, q6s_i(5n, 7n, 0n, 0n, 1n, 0n, 0n, q6s_i(4n, 7n, 0n, 0n, 1n, 0n, 0n, q6s_i(3n, 7n, 0n, 0n, 1n, 0n, 0n, q6s_i(2n, 7n, 0n, 0n, 1n, 0n, 0n, q6s_i(1n, 7n, 0n, 0n, 1n, 0n, 0n, q6s_i(0n, 7n, 0n, 0n, 1n, 0n, 0n, {==}, sh_0_0_1_0_0_0()), sh_0_0_1_0_0_1()), sh_0_0_1_0_0_2()), sh_0_0_1_0_0_3()), sh_0_0_1_0_0_4()), sh_0_0_1_0_0_5())), q6s_i(5n, 7n, 0n, 0n, 1n, 0n, 1n, q6s_i(4n, 7n, 0n, 0n, 1n, 0n, 1n, q6s_i(3n, 7n, 0n, 0n, 1n, 0n, 1n, q6s_i(2n, 7n, 0n, 0n, 1n, 0n, 1n, q6s_i(1n, 7n, 0n, 0n, 1n, 0n, 1n, q6s_i(0n, 7n, 0n, 0n, 1n, 0n, 1n, {==}, sh_0_0_1_0_1_0()), sh_0_0_1_0_1_1()), sh_0_0_1_0_1_2()), sh_0_0_1_0_1_3()), sh_0_0_1_0_1_4()), sh_0_0_1_0_1_5())), q6s_i(5n, 7n, 0n, 0n, 1n, 0n, 2n, q6s_i(4n, 7n, 0n, 0n, 1n, 0n, 2n, q6s_i(3n, 7n, 0n, 0n, 1n, 0n, 2n, q6s_i(2n, 7n, 0n, 0n, 1n, 0n, 2n, q6s_i(1n, 7n, 0n, 0n, 1n, 0n, 2n, q6s_i(0n, 7n, 0n, 0n, 1n, 0n, 2n, {==}, sh_0_0_1_0_2_0()), sh_0_0_1_0_2_1()), sh_0_0_1_0_2_2()), sh_0_0_1_0_2_3()), sh_0_0_1_0_2_4()), sh_0_0_1_0_2_5())), q6s_i(5n, 7n, 0n, 0n, 1n, 0n, 3n, q6s_i(4n, 7n, 0n, 0n, 1n, 0n, 3n, q6s_i(3n, 7n, 0n, 0n, 1n, 0n, 3n, q6s_i(2n, 7n, 0n, 0n, 1n, 0n, 3n, q6s_i(1n, 7n, 0n, 0n, 1n, 0n, 3n, q6s_i(0n, 7n, 0n, 0n, 1n, 0n, 3n, {==}, sh_0_0_1_0_3_0()), sh_0_0_1_0_3_1()), sh_0_0_1_0_3_2()), sh_0_0_1_0_3_3()), sh_0_0_1_0_3_4()), sh_0_0_1_0_3_5())), q6s_i(5n, 7n, 0n, 0n, 1n, 0n, 4n, q6s_i(4n, 7n, 0n, 0n, 1n, 0n, 4n, q6s_i(3n, 7n, 0n, 0n, 1n, 0n, 4n, q6s_i(2n, 7n, 0n, 0n, 1n, 0n, 4n, q6s_i(1n, 7n, 0n, 0n, 1n, 0n, 4n, q6s_i(0n, 7n, 0n, 0n, 1n, 0n, 4n, {==}, sh_0_0_1_0_4_0()), sh_0_0_1_0_4_1()), sh_0_0_1_0_4_2()), sh_0_0_1_0_4_3()), sh_0_0_1_0_4_4()), sh_0_0_1_0_4_5()))

def sq5_0_0_1_1() -> {q5s(5n, 6n, 7n, 0n, 0n, 1n, 1n) == True{} : Bool}:
  q5s_i(4n, 6n, 7n, 0n, 0n, 1n, 1n, q5s_i(3n, 6n, 7n, 0n, 0n, 1n, 1n, q5s_i(2n, 6n, 7n, 0n, 0n, 1n, 1n, q5s_i(1n, 6n, 7n, 0n, 0n, 1n, 1n, q5s_i(0n, 6n, 7n, 0n, 0n, 1n, 1n, {==}, q6s_i(5n, 7n, 0n, 0n, 1n, 1n, 0n, q6s_i(4n, 7n, 0n, 0n, 1n, 1n, 0n, q6s_i(3n, 7n, 0n, 0n, 1n, 1n, 0n, q6s_i(2n, 7n, 0n, 0n, 1n, 1n, 0n, q6s_i(1n, 7n, 0n, 0n, 1n, 1n, 0n, q6s_i(0n, 7n, 0n, 0n, 1n, 1n, 0n, {==}, sh_0_0_1_1_0_0()), sh_0_0_1_1_0_1()), sh_0_0_1_1_0_2()), sh_0_0_1_1_0_3()), sh_0_0_1_1_0_4()), sh_0_0_1_1_0_5())), q6s_i(5n, 7n, 0n, 0n, 1n, 1n, 1n, q6s_i(4n, 7n, 0n, 0n, 1n, 1n, 1n, q6s_i(3n, 7n, 0n, 0n, 1n, 1n, 1n, q6s_i(2n, 7n, 0n, 0n, 1n, 1n, 1n, q6s_i(1n, 7n, 0n, 0n, 1n, 1n, 1n, q6s_i(0n, 7n, 0n, 0n, 1n, 1n, 1n, {==}, sh_0_0_1_1_1_0()), sh_0_0_1_1_1_1()), sh_0_0_1_1_1_2()), sh_0_0_1_1_1_3()), sh_0_0_1_1_1_4()), sh_0_0_1_1_1_5())), q6s_i(5n, 7n, 0n, 0n, 1n, 1n, 2n, q6s_i(4n, 7n, 0n, 0n, 1n, 1n, 2n, q6s_i(3n, 7n, 0n, 0n, 1n, 1n, 2n, q6s_i(2n, 7n, 0n, 0n, 1n, 1n, 2n, q6s_i(1n, 7n, 0n, 0n, 1n, 1n, 2n, q6s_i(0n, 7n, 0n, 0n, 1n, 1n, 2n, {==}, sh_0_0_1_1_2_0()), sh_0_0_1_1_2_1()), sh_0_0_1_1_2_2()), sh_0_0_1_1_2_3()), sh_0_0_1_1_2_4()), sh_0_0_1_1_2_5())), q6s_i(5n, 7n, 0n, 0n, 1n, 1n, 3n, q6s_i(4n, 7n, 0n, 0n, 1n, 1n, 3n, q6s_i(3n, 7n, 0n, 0n, 1n, 1n, 3n, q6s_i(2n, 7n, 0n, 0n, 1n, 1n, 3n, q6s_i(1n, 7n, 0n, 0n, 1n, 1n, 3n, q6s_i(0n, 7n, 0n, 0n, 1n, 1n, 3n, {==}, sh_0_0_1_1_3_0()), sh_0_0_1_1_3_1()), sh_0_0_1_1_3_2()), sh_0_0_1_1_3_3()), sh_0_0_1_1_3_4()), sh_0_0_1_1_3_5())), q6s_i(5n, 7n, 0n, 0n, 1n, 1n, 4n, q6s_i(4n, 7n, 0n, 0n, 1n, 1n, 4n, q6s_i(3n, 7n, 0n, 0n, 1n, 1n, 4n, q6s_i(2n, 7n, 0n, 0n, 1n, 1n, 4n, q6s_i(1n, 7n, 0n, 0n, 1n, 1n, 4n, q6s_i(0n, 7n, 0n, 0n, 1n, 1n, 4n, {==}, sh_0_0_1_1_4_0()), sh_0_0_1_1_4_1()), sh_0_0_1_1_4_2()), sh_0_0_1_1_4_3()), sh_0_0_1_1_4_4()), sh_0_0_1_1_4_5()))

def sq5_0_0_1_2() -> {q5s(5n, 6n, 7n, 0n, 0n, 1n, 2n) == True{} : Bool}:
  q5s_i(4n, 6n, 7n, 0n, 0n, 1n, 2n, q5s_i(3n, 6n, 7n, 0n, 0n, 1n, 2n, q5s_i(2n, 6n, 7n, 0n, 0n, 1n, 2n, q5s_i(1n, 6n, 7n, 0n, 0n, 1n, 2n, q5s_i(0n, 6n, 7n, 0n, 0n, 1n, 2n, {==}, q6s_i(5n, 7n, 0n, 0n, 1n, 2n, 0n, q6s_i(4n, 7n, 0n, 0n, 1n, 2n, 0n, q6s_i(3n, 7n, 0n, 0n, 1n, 2n, 0n, q6s_i(2n, 7n, 0n, 0n, 1n, 2n, 0n, q6s_i(1n, 7n, 0n, 0n, 1n, 2n, 0n, q6s_i(0n, 7n, 0n, 0n, 1n, 2n, 0n, {==}, sh_0_0_1_2_0_0()), sh_0_0_1_2_0_1()), sh_0_0_1_2_0_2()), sh_0_0_1_2_0_3()), sh_0_0_1_2_0_4()), sh_0_0_1_2_0_5())), q6s_i(5n, 7n, 0n, 0n, 1n, 2n, 1n, q6s_i(4n, 7n, 0n, 0n, 1n, 2n, 1n, q6s_i(3n, 7n, 0n, 0n, 1n, 2n, 1n, q6s_i(2n, 7n, 0n, 0n, 1n, 2n, 1n, q6s_i(1n, 7n, 0n, 0n, 1n, 2n, 1n, q6s_i(0n, 7n, 0n, 0n, 1n, 2n, 1n, {==}, sh_0_0_1_2_1_0()), sh_0_0_1_2_1_1()), sh_0_0_1_2_1_2()), sh_0_0_1_2_1_3()), sh_0_0_1_2_1_4()), sh_0_0_1_2_1_5())), q6s_i(5n, 7n, 0n, 0n, 1n, 2n, 2n, q6s_i(4n, 7n, 0n, 0n, 1n, 2n, 2n, q6s_i(3n, 7n, 0n, 0n, 1n, 2n, 2n, q6s_i(2n, 7n, 0n, 0n, 1n, 2n, 2n, q6s_i(1n, 7n, 0n, 0n, 1n, 2n, 2n, q6s_i(0n, 7n, 0n, 0n, 1n, 2n, 2n, {==}, sh_0_0_1_2_2_0()), sh_0_0_1_2_2_1()), sh_0_0_1_2_2_2()), sh_0_0_1_2_2_3()), sh_0_0_1_2_2_4()), sh_0_0_1_2_2_5())), q6s_i(5n, 7n, 0n, 0n, 1n, 2n, 3n, q6s_i(4n, 7n, 0n, 0n, 1n, 2n, 3n, q6s_i(3n, 7n, 0n, 0n, 1n, 2n, 3n, q6s_i(2n, 7n, 0n, 0n, 1n, 2n, 3n, q6s_i(1n, 7n, 0n, 0n, 1n, 2n, 3n, q6s_i(0n, 7n, 0n, 0n, 1n, 2n, 3n, {==}, sh_0_0_1_2_3_0()), sh_0_0_1_2_3_1()), sh_0_0_1_2_3_2()), sh_0_0_1_2_3_3()), sh_0_0_1_2_3_4()), sh_0_0_1_2_3_5())), q6s_i(5n, 7n, 0n, 0n, 1n, 2n, 4n, q6s_i(4n, 7n, 0n, 0n, 1n, 2n, 4n, q6s_i(3n, 7n, 0n, 0n, 1n, 2n, 4n, q6s_i(2n, 7n, 0n, 0n, 1n, 2n, 4n, q6s_i(1n, 7n, 0n, 0n, 1n, 2n, 4n, q6s_i(0n, 7n, 0n, 0n, 1n, 2n, 4n, {==}, sh_0_0_1_2_4_0()), sh_0_0_1_2_4_1()), sh_0_0_1_2_4_2()), sh_0_0_1_2_4_3()), sh_0_0_1_2_4_4()), sh_0_0_1_2_4_5()))

def sq5_0_0_1_3() -> {q5s(5n, 6n, 7n, 0n, 0n, 1n, 3n) == True{} : Bool}:
  q5s_i(4n, 6n, 7n, 0n, 0n, 1n, 3n, q5s_i(3n, 6n, 7n, 0n, 0n, 1n, 3n, q5s_i(2n, 6n, 7n, 0n, 0n, 1n, 3n, q5s_i(1n, 6n, 7n, 0n, 0n, 1n, 3n, q5s_i(0n, 6n, 7n, 0n, 0n, 1n, 3n, {==}, q6s_i(5n, 7n, 0n, 0n, 1n, 3n, 0n, q6s_i(4n, 7n, 0n, 0n, 1n, 3n, 0n, q6s_i(3n, 7n, 0n, 0n, 1n, 3n, 0n, q6s_i(2n, 7n, 0n, 0n, 1n, 3n, 0n, q6s_i(1n, 7n, 0n, 0n, 1n, 3n, 0n, q6s_i(0n, 7n, 0n, 0n, 1n, 3n, 0n, {==}, sh_0_0_1_3_0_0()), sh_0_0_1_3_0_1()), sh_0_0_1_3_0_2()), sh_0_0_1_3_0_3()), sh_0_0_1_3_0_4()), sh_0_0_1_3_0_5())), q6s_i(5n, 7n, 0n, 0n, 1n, 3n, 1n, q6s_i(4n, 7n, 0n, 0n, 1n, 3n, 1n, q6s_i(3n, 7n, 0n, 0n, 1n, 3n, 1n, q6s_i(2n, 7n, 0n, 0n, 1n, 3n, 1n, q6s_i(1n, 7n, 0n, 0n, 1n, 3n, 1n, q6s_i(0n, 7n, 0n, 0n, 1n, 3n, 1n, {==}, sh_0_0_1_3_1_0()), sh_0_0_1_3_1_1()), sh_0_0_1_3_1_2()), sh_0_0_1_3_1_3()), sh_0_0_1_3_1_4()), sh_0_0_1_3_1_5())), q6s_i(5n, 7n, 0n, 0n, 1n, 3n, 2n, q6s_i(4n, 7n, 0n, 0n, 1n, 3n, 2n, q6s_i(3n, 7n, 0n, 0n, 1n, 3n, 2n, q6s_i(2n, 7n, 0n, 0n, 1n, 3n, 2n, q6s_i(1n, 7n, 0n, 0n, 1n, 3n, 2n, q6s_i(0n, 7n, 0n, 0n, 1n, 3n, 2n, {==}, sh_0_0_1_3_2_0()), sh_0_0_1_3_2_1()), sh_0_0_1_3_2_2()), sh_0_0_1_3_2_3()), sh_0_0_1_3_2_4()), sh_0_0_1_3_2_5())), q6s_i(5n, 7n, 0n, 0n, 1n, 3n, 3n, q6s_i(4n, 7n, 0n, 0n, 1n, 3n, 3n, q6s_i(3n, 7n, 0n, 0n, 1n, 3n, 3n, q6s_i(2n, 7n, 0n, 0n, 1n, 3n, 3n, q6s_i(1n, 7n, 0n, 0n, 1n, 3n, 3n, q6s_i(0n, 7n, 0n, 0n, 1n, 3n, 3n, {==}, sh_0_0_1_3_3_0()), sh_0_0_1_3_3_1()), sh_0_0_1_3_3_2()), sh_0_0_1_3_3_3()), sh_0_0_1_3_3_4()), sh_0_0_1_3_3_5())), q6s_i(5n, 7n, 0n, 0n, 1n, 3n, 4n, q6s_i(4n, 7n, 0n, 0n, 1n, 3n, 4n, q6s_i(3n, 7n, 0n, 0n, 1n, 3n, 4n, q6s_i(2n, 7n, 0n, 0n, 1n, 3n, 4n, q6s_i(1n, 7n, 0n, 0n, 1n, 3n, 4n, q6s_i(0n, 7n, 0n, 0n, 1n, 3n, 4n, {==}, sh_0_0_1_3_4_0()), sh_0_0_1_3_4_1()), sh_0_0_1_3_4_2()), sh_0_0_1_3_4_3()), sh_0_0_1_3_4_4()), sh_0_0_1_3_4_5()))

def sq5_0_0_2_0() -> {q5s(5n, 6n, 7n, 0n, 0n, 2n, 0n) == True{} : Bool}:
  q5s_i(4n, 6n, 7n, 0n, 0n, 2n, 0n, q5s_i(3n, 6n, 7n, 0n, 0n, 2n, 0n, q5s_i(2n, 6n, 7n, 0n, 0n, 2n, 0n, q5s_i(1n, 6n, 7n, 0n, 0n, 2n, 0n, q5s_i(0n, 6n, 7n, 0n, 0n, 2n, 0n, {==}, q6s_i(5n, 7n, 0n, 0n, 2n, 0n, 0n, q6s_i(4n, 7n, 0n, 0n, 2n, 0n, 0n, q6s_i(3n, 7n, 0n, 0n, 2n, 0n, 0n, q6s_i(2n, 7n, 0n, 0n, 2n, 0n, 0n, q6s_i(1n, 7n, 0n, 0n, 2n, 0n, 0n, q6s_i(0n, 7n, 0n, 0n, 2n, 0n, 0n, {==}, sh_0_0_2_0_0_0()), sh_0_0_2_0_0_1()), sh_0_0_2_0_0_2()), sh_0_0_2_0_0_3()), sh_0_0_2_0_0_4()), sh_0_0_2_0_0_5())), q6s_i(5n, 7n, 0n, 0n, 2n, 0n, 1n, q6s_i(4n, 7n, 0n, 0n, 2n, 0n, 1n, q6s_i(3n, 7n, 0n, 0n, 2n, 0n, 1n, q6s_i(2n, 7n, 0n, 0n, 2n, 0n, 1n, q6s_i(1n, 7n, 0n, 0n, 2n, 0n, 1n, q6s_i(0n, 7n, 0n, 0n, 2n, 0n, 1n, {==}, sh_0_0_2_0_1_0()), sh_0_0_2_0_1_1()), sh_0_0_2_0_1_2()), sh_0_0_2_0_1_3()), sh_0_0_2_0_1_4()), sh_0_0_2_0_1_5())), q6s_i(5n, 7n, 0n, 0n, 2n, 0n, 2n, q6s_i(4n, 7n, 0n, 0n, 2n, 0n, 2n, q6s_i(3n, 7n, 0n, 0n, 2n, 0n, 2n, q6s_i(2n, 7n, 0n, 0n, 2n, 0n, 2n, q6s_i(1n, 7n, 0n, 0n, 2n, 0n, 2n, q6s_i(0n, 7n, 0n, 0n, 2n, 0n, 2n, {==}, sh_0_0_2_0_2_0()), sh_0_0_2_0_2_1()), sh_0_0_2_0_2_2()), sh_0_0_2_0_2_3()), sh_0_0_2_0_2_4()), sh_0_0_2_0_2_5())), q6s_i(5n, 7n, 0n, 0n, 2n, 0n, 3n, q6s_i(4n, 7n, 0n, 0n, 2n, 0n, 3n, q6s_i(3n, 7n, 0n, 0n, 2n, 0n, 3n, q6s_i(2n, 7n, 0n, 0n, 2n, 0n, 3n, q6s_i(1n, 7n, 0n, 0n, 2n, 0n, 3n, q6s_i(0n, 7n, 0n, 0n, 2n, 0n, 3n, {==}, sh_0_0_2_0_3_0()), sh_0_0_2_0_3_1()), sh_0_0_2_0_3_2()), sh_0_0_2_0_3_3()), sh_0_0_2_0_3_4()), sh_0_0_2_0_3_5())), q6s_i(5n, 7n, 0n, 0n, 2n, 0n, 4n, q6s_i(4n, 7n, 0n, 0n, 2n, 0n, 4n, q6s_i(3n, 7n, 0n, 0n, 2n, 0n, 4n, q6s_i(2n, 7n, 0n, 0n, 2n, 0n, 4n, q6s_i(1n, 7n, 0n, 0n, 2n, 0n, 4n, q6s_i(0n, 7n, 0n, 0n, 2n, 0n, 4n, {==}, sh_0_0_2_0_4_0()), sh_0_0_2_0_4_1()), sh_0_0_2_0_4_2()), sh_0_0_2_0_4_3()), sh_0_0_2_0_4_4()), sh_0_0_2_0_4_5()))

def sq5_0_0_2_1() -> {q5s(5n, 6n, 7n, 0n, 0n, 2n, 1n) == True{} : Bool}:
  q5s_i(4n, 6n, 7n, 0n, 0n, 2n, 1n, q5s_i(3n, 6n, 7n, 0n, 0n, 2n, 1n, q5s_i(2n, 6n, 7n, 0n, 0n, 2n, 1n, q5s_i(1n, 6n, 7n, 0n, 0n, 2n, 1n, q5s_i(0n, 6n, 7n, 0n, 0n, 2n, 1n, {==}, q6s_i(5n, 7n, 0n, 0n, 2n, 1n, 0n, q6s_i(4n, 7n, 0n, 0n, 2n, 1n, 0n, q6s_i(3n, 7n, 0n, 0n, 2n, 1n, 0n, q6s_i(2n, 7n, 0n, 0n, 2n, 1n, 0n, q6s_i(1n, 7n, 0n, 0n, 2n, 1n, 0n, q6s_i(0n, 7n, 0n, 0n, 2n, 1n, 0n, {==}, sh_0_0_2_1_0_0()), sh_0_0_2_1_0_1()), sh_0_0_2_1_0_2()), sh_0_0_2_1_0_3()), sh_0_0_2_1_0_4()), sh_0_0_2_1_0_5())), q6s_i(5n, 7n, 0n, 0n, 2n, 1n, 1n, q6s_i(4n, 7n, 0n, 0n, 2n, 1n, 1n, q6s_i(3n, 7n, 0n, 0n, 2n, 1n, 1n, q6s_i(2n, 7n, 0n, 0n, 2n, 1n, 1n, q6s_i(1n, 7n, 0n, 0n, 2n, 1n, 1n, q6s_i(0n, 7n, 0n, 0n, 2n, 1n, 1n, {==}, sh_0_0_2_1_1_0()), sh_0_0_2_1_1_1()), sh_0_0_2_1_1_2()), sh_0_0_2_1_1_3()), sh_0_0_2_1_1_4()), sh_0_0_2_1_1_5())), q6s_i(5n, 7n, 0n, 0n, 2n, 1n, 2n, q6s_i(4n, 7n, 0n, 0n, 2n, 1n, 2n, q6s_i(3n, 7n, 0n, 0n, 2n, 1n, 2n, q6s_i(2n, 7n, 0n, 0n, 2n, 1n, 2n, q6s_i(1n, 7n, 0n, 0n, 2n, 1n, 2n, q6s_i(0n, 7n, 0n, 0n, 2n, 1n, 2n, {==}, sh_0_0_2_1_2_0()), sh_0_0_2_1_2_1()), sh_0_0_2_1_2_2()), sh_0_0_2_1_2_3()), sh_0_0_2_1_2_4()), sh_0_0_2_1_2_5())), q6s_i(5n, 7n, 0n, 0n, 2n, 1n, 3n, q6s_i(4n, 7n, 0n, 0n, 2n, 1n, 3n, q6s_i(3n, 7n, 0n, 0n, 2n, 1n, 3n, q6s_i(2n, 7n, 0n, 0n, 2n, 1n, 3n, q6s_i(1n, 7n, 0n, 0n, 2n, 1n, 3n, q6s_i(0n, 7n, 0n, 0n, 2n, 1n, 3n, {==}, sh_0_0_2_1_3_0()), sh_0_0_2_1_3_1()), sh_0_0_2_1_3_2()), sh_0_0_2_1_3_3()), sh_0_0_2_1_3_4()), sh_0_0_2_1_3_5())), q6s_i(5n, 7n, 0n, 0n, 2n, 1n, 4n, q6s_i(4n, 7n, 0n, 0n, 2n, 1n, 4n, q6s_i(3n, 7n, 0n, 0n, 2n, 1n, 4n, q6s_i(2n, 7n, 0n, 0n, 2n, 1n, 4n, q6s_i(1n, 7n, 0n, 0n, 2n, 1n, 4n, q6s_i(0n, 7n, 0n, 0n, 2n, 1n, 4n, {==}, sh_0_0_2_1_4_0()), sh_0_0_2_1_4_1()), sh_0_0_2_1_4_2()), sh_0_0_2_1_4_3()), sh_0_0_2_1_4_4()), sh_0_0_2_1_4_5()))

def sq5_0_0_2_2() -> {q5s(5n, 6n, 7n, 0n, 0n, 2n, 2n) == True{} : Bool}:
  q5s_i(4n, 6n, 7n, 0n, 0n, 2n, 2n, q5s_i(3n, 6n, 7n, 0n, 0n, 2n, 2n, q5s_i(2n, 6n, 7n, 0n, 0n, 2n, 2n, q5s_i(1n, 6n, 7n, 0n, 0n, 2n, 2n, q5s_i(0n, 6n, 7n, 0n, 0n, 2n, 2n, {==}, q6s_i(5n, 7n, 0n, 0n, 2n, 2n, 0n, q6s_i(4n, 7n, 0n, 0n, 2n, 2n, 0n, q6s_i(3n, 7n, 0n, 0n, 2n, 2n, 0n, q6s_i(2n, 7n, 0n, 0n, 2n, 2n, 0n, q6s_i(1n, 7n, 0n, 0n, 2n, 2n, 0n, q6s_i(0n, 7n, 0n, 0n, 2n, 2n, 0n, {==}, sh_0_0_2_2_0_0()), sh_0_0_2_2_0_1()), sh_0_0_2_2_0_2()), sh_0_0_2_2_0_3()), sh_0_0_2_2_0_4()), sh_0_0_2_2_0_5())), q6s_i(5n, 7n, 0n, 0n, 2n, 2n, 1n, q6s_i(4n, 7n, 0n, 0n, 2n, 2n, 1n, q6s_i(3n, 7n, 0n, 0n, 2n, 2n, 1n, q6s_i(2n, 7n, 0n, 0n, 2n, 2n, 1n, q6s_i(1n, 7n, 0n, 0n, 2n, 2n, 1n, q6s_i(0n, 7n, 0n, 0n, 2n, 2n, 1n, {==}, sh_0_0_2_2_1_0()), sh_0_0_2_2_1_1()), sh_0_0_2_2_1_2()), sh_0_0_2_2_1_3()), sh_0_0_2_2_1_4()), sh_0_0_2_2_1_5())), q6s_i(5n, 7n, 0n, 0n, 2n, 2n, 2n, q6s_i(4n, 7n, 0n, 0n, 2n, 2n, 2n, q6s_i(3n, 7n, 0n, 0n, 2n, 2n, 2n, q6s_i(2n, 7n, 0n, 0n, 2n, 2n, 2n, q6s_i(1n, 7n, 0n, 0n, 2n, 2n, 2n, q6s_i(0n, 7n, 0n, 0n, 2n, 2n, 2n, {==}, sh_0_0_2_2_2_0()), sh_0_0_2_2_2_1()), sh_0_0_2_2_2_2()), sh_0_0_2_2_2_3()), sh_0_0_2_2_2_4()), sh_0_0_2_2_2_5())), q6s_i(5n, 7n, 0n, 0n, 2n, 2n, 3n, q6s_i(4n, 7n, 0n, 0n, 2n, 2n, 3n, q6s_i(3n, 7n, 0n, 0n, 2n, 2n, 3n, q6s_i(2n, 7n, 0n, 0n, 2n, 2n, 3n, q6s_i(1n, 7n, 0n, 0n, 2n, 2n, 3n, q6s_i(0n, 7n, 0n, 0n, 2n, 2n, 3n, {==}, sh_0_0_2_2_3_0()), sh_0_0_2_2_3_1()), sh_0_0_2_2_3_2()), sh_0_0_2_2_3_3()), sh_0_0_2_2_3_4()), sh_0_0_2_2_3_5())), q6s_i(5n, 7n, 0n, 0n, 2n, 2n, 4n, q6s_i(4n, 7n, 0n, 0n, 2n, 2n, 4n, q6s_i(3n, 7n, 0n, 0n, 2n, 2n, 4n, q6s_i(2n, 7n, 0n, 0n, 2n, 2n, 4n, q6s_i(1n, 7n, 0n, 0n, 2n, 2n, 4n, q6s_i(0n, 7n, 0n, 0n, 2n, 2n, 4n, {==}, sh_0_0_2_2_4_0()), sh_0_0_2_2_4_1()), sh_0_0_2_2_4_2()), sh_0_0_2_2_4_3()), sh_0_0_2_2_4_4()), sh_0_0_2_2_4_5()))

def sq5_0_0_2_3() -> {q5s(5n, 6n, 7n, 0n, 0n, 2n, 3n) == True{} : Bool}:
  q5s_i(4n, 6n, 7n, 0n, 0n, 2n, 3n, q5s_i(3n, 6n, 7n, 0n, 0n, 2n, 3n, q5s_i(2n, 6n, 7n, 0n, 0n, 2n, 3n, q5s_i(1n, 6n, 7n, 0n, 0n, 2n, 3n, q5s_i(0n, 6n, 7n, 0n, 0n, 2n, 3n, {==}, q6s_i(5n, 7n, 0n, 0n, 2n, 3n, 0n, q6s_i(4n, 7n, 0n, 0n, 2n, 3n, 0n, q6s_i(3n, 7n, 0n, 0n, 2n, 3n, 0n, q6s_i(2n, 7n, 0n, 0n, 2n, 3n, 0n, q6s_i(1n, 7n, 0n, 0n, 2n, 3n, 0n, q6s_i(0n, 7n, 0n, 0n, 2n, 3n, 0n, {==}, sh_0_0_2_3_0_0()), sh_0_0_2_3_0_1()), sh_0_0_2_3_0_2()), sh_0_0_2_3_0_3()), sh_0_0_2_3_0_4()), sh_0_0_2_3_0_5())), q6s_i(5n, 7n, 0n, 0n, 2n, 3n, 1n, q6s_i(4n, 7n, 0n, 0n, 2n, 3n, 1n, q6s_i(3n, 7n, 0n, 0n, 2n, 3n, 1n, q6s_i(2n, 7n, 0n, 0n, 2n, 3n, 1n, q6s_i(1n, 7n, 0n, 0n, 2n, 3n, 1n, q6s_i(0n, 7n, 0n, 0n, 2n, 3n, 1n, {==}, sh_0_0_2_3_1_0()), sh_0_0_2_3_1_1()), sh_0_0_2_3_1_2()), sh_0_0_2_3_1_3()), sh_0_0_2_3_1_4()), sh_0_0_2_3_1_5())), q6s_i(5n, 7n, 0n, 0n, 2n, 3n, 2n, q6s_i(4n, 7n, 0n, 0n, 2n, 3n, 2n, q6s_i(3n, 7n, 0n, 0n, 2n, 3n, 2n, q6s_i(2n, 7n, 0n, 0n, 2n, 3n, 2n, q6s_i(1n, 7n, 0n, 0n, 2n, 3n, 2n, q6s_i(0n, 7n, 0n, 0n, 2n, 3n, 2n, {==}, sh_0_0_2_3_2_0()), sh_0_0_2_3_2_1()), sh_0_0_2_3_2_2()), sh_0_0_2_3_2_3()), sh_0_0_2_3_2_4()), sh_0_0_2_3_2_5())), q6s_i(5n, 7n, 0n, 0n, 2n, 3n, 3n, q6s_i(4n, 7n, 0n, 0n, 2n, 3n, 3n, q6s_i(3n, 7n, 0n, 0n, 2n, 3n, 3n, q6s_i(2n, 7n, 0n, 0n, 2n, 3n, 3n, q6s_i(1n, 7n, 0n, 0n, 2n, 3n, 3n, q6s_i(0n, 7n, 0n, 0n, 2n, 3n, 3n, {==}, sh_0_0_2_3_3_0()), sh_0_0_2_3_3_1()), sh_0_0_2_3_3_2()), sh_0_0_2_3_3_3()), sh_0_0_2_3_3_4()), sh_0_0_2_3_3_5())), q6s_i(5n, 7n, 0n, 0n, 2n, 3n, 4n, q6s_i(4n, 7n, 0n, 0n, 2n, 3n, 4n, q6s_i(3n, 7n, 0n, 0n, 2n, 3n, 4n, q6s_i(2n, 7n, 0n, 0n, 2n, 3n, 4n, q6s_i(1n, 7n, 0n, 0n, 2n, 3n, 4n, q6s_i(0n, 7n, 0n, 0n, 2n, 3n, 4n, {==}, sh_0_0_2_3_4_0()), sh_0_0_2_3_4_1()), sh_0_0_2_3_4_2()), sh_0_0_2_3_4_3()), sh_0_0_2_3_4_4()), sh_0_0_2_3_4_5()))

def sq5_0_1_0_0() -> {q5s(5n, 6n, 7n, 0n, 1n, 0n, 0n) == True{} : Bool}:
  q5s_i(4n, 6n, 7n, 0n, 1n, 0n, 0n, q5s_i(3n, 6n, 7n, 0n, 1n, 0n, 0n, q5s_i(2n, 6n, 7n, 0n, 1n, 0n, 0n, q5s_i(1n, 6n, 7n, 0n, 1n, 0n, 0n, q5s_i(0n, 6n, 7n, 0n, 1n, 0n, 0n, {==}, q6s_i(5n, 7n, 0n, 1n, 0n, 0n, 0n, q6s_i(4n, 7n, 0n, 1n, 0n, 0n, 0n, q6s_i(3n, 7n, 0n, 1n, 0n, 0n, 0n, q6s_i(2n, 7n, 0n, 1n, 0n, 0n, 0n, q6s_i(1n, 7n, 0n, 1n, 0n, 0n, 0n, q6s_i(0n, 7n, 0n, 1n, 0n, 0n, 0n, {==}, sh_0_1_0_0_0_0()), sh_0_1_0_0_0_1()), sh_0_1_0_0_0_2()), sh_0_1_0_0_0_3()), sh_0_1_0_0_0_4()), sh_0_1_0_0_0_5())), q6s_i(5n, 7n, 0n, 1n, 0n, 0n, 1n, q6s_i(4n, 7n, 0n, 1n, 0n, 0n, 1n, q6s_i(3n, 7n, 0n, 1n, 0n, 0n, 1n, q6s_i(2n, 7n, 0n, 1n, 0n, 0n, 1n, q6s_i(1n, 7n, 0n, 1n, 0n, 0n, 1n, q6s_i(0n, 7n, 0n, 1n, 0n, 0n, 1n, {==}, sh_0_1_0_0_1_0()), sh_0_1_0_0_1_1()), sh_0_1_0_0_1_2()), sh_0_1_0_0_1_3()), sh_0_1_0_0_1_4()), sh_0_1_0_0_1_5())), q6s_i(5n, 7n, 0n, 1n, 0n, 0n, 2n, q6s_i(4n, 7n, 0n, 1n, 0n, 0n, 2n, q6s_i(3n, 7n, 0n, 1n, 0n, 0n, 2n, q6s_i(2n, 7n, 0n, 1n, 0n, 0n, 2n, q6s_i(1n, 7n, 0n, 1n, 0n, 0n, 2n, q6s_i(0n, 7n, 0n, 1n, 0n, 0n, 2n, {==}, sh_0_1_0_0_2_0()), sh_0_1_0_0_2_1()), sh_0_1_0_0_2_2()), sh_0_1_0_0_2_3()), sh_0_1_0_0_2_4()), sh_0_1_0_0_2_5())), q6s_i(5n, 7n, 0n, 1n, 0n, 0n, 3n, q6s_i(4n, 7n, 0n, 1n, 0n, 0n, 3n, q6s_i(3n, 7n, 0n, 1n, 0n, 0n, 3n, q6s_i(2n, 7n, 0n, 1n, 0n, 0n, 3n, q6s_i(1n, 7n, 0n, 1n, 0n, 0n, 3n, q6s_i(0n, 7n, 0n, 1n, 0n, 0n, 3n, {==}, sh_0_1_0_0_3_0()), sh_0_1_0_0_3_1()), sh_0_1_0_0_3_2()), sh_0_1_0_0_3_3()), sh_0_1_0_0_3_4()), sh_0_1_0_0_3_5())), q6s_i(5n, 7n, 0n, 1n, 0n, 0n, 4n, q6s_i(4n, 7n, 0n, 1n, 0n, 0n, 4n, q6s_i(3n, 7n, 0n, 1n, 0n, 0n, 4n, q6s_i(2n, 7n, 0n, 1n, 0n, 0n, 4n, q6s_i(1n, 7n, 0n, 1n, 0n, 0n, 4n, q6s_i(0n, 7n, 0n, 1n, 0n, 0n, 4n, {==}, sh_0_1_0_0_4_0()), sh_0_1_0_0_4_1()), sh_0_1_0_0_4_2()), sh_0_1_0_0_4_3()), sh_0_1_0_0_4_4()), sh_0_1_0_0_4_5()))

def sq5_0_1_0_1() -> {q5s(5n, 6n, 7n, 0n, 1n, 0n, 1n) == True{} : Bool}:
  q5s_i(4n, 6n, 7n, 0n, 1n, 0n, 1n, q5s_i(3n, 6n, 7n, 0n, 1n, 0n, 1n, q5s_i(2n, 6n, 7n, 0n, 1n, 0n, 1n, q5s_i(1n, 6n, 7n, 0n, 1n, 0n, 1n, q5s_i(0n, 6n, 7n, 0n, 1n, 0n, 1n, {==}, q6s_i(5n, 7n, 0n, 1n, 0n, 1n, 0n, q6s_i(4n, 7n, 0n, 1n, 0n, 1n, 0n, q6s_i(3n, 7n, 0n, 1n, 0n, 1n, 0n, q6s_i(2n, 7n, 0n, 1n, 0n, 1n, 0n, q6s_i(1n, 7n, 0n, 1n, 0n, 1n, 0n, q6s_i(0n, 7n, 0n, 1n, 0n, 1n, 0n, {==}, sh_0_1_0_1_0_0()), sh_0_1_0_1_0_1()), sh_0_1_0_1_0_2()), sh_0_1_0_1_0_3()), sh_0_1_0_1_0_4()), sh_0_1_0_1_0_5())), q6s_i(5n, 7n, 0n, 1n, 0n, 1n, 1n, q6s_i(4n, 7n, 0n, 1n, 0n, 1n, 1n, q6s_i(3n, 7n, 0n, 1n, 0n, 1n, 1n, q6s_i(2n, 7n, 0n, 1n, 0n, 1n, 1n, q6s_i(1n, 7n, 0n, 1n, 0n, 1n, 1n, q6s_i(0n, 7n, 0n, 1n, 0n, 1n, 1n, {==}, sh_0_1_0_1_1_0()), sh_0_1_0_1_1_1()), sh_0_1_0_1_1_2()), sh_0_1_0_1_1_3()), sh_0_1_0_1_1_4()), sh_0_1_0_1_1_5())), q6s_i(5n, 7n, 0n, 1n, 0n, 1n, 2n, q6s_i(4n, 7n, 0n, 1n, 0n, 1n, 2n, q6s_i(3n, 7n, 0n, 1n, 0n, 1n, 2n, q6s_i(2n, 7n, 0n, 1n, 0n, 1n, 2n, q6s_i(1n, 7n, 0n, 1n, 0n, 1n, 2n, q6s_i(0n, 7n, 0n, 1n, 0n, 1n, 2n, {==}, sh_0_1_0_1_2_0()), sh_0_1_0_1_2_1()), sh_0_1_0_1_2_2()), sh_0_1_0_1_2_3()), sh_0_1_0_1_2_4()), sh_0_1_0_1_2_5())), q6s_i(5n, 7n, 0n, 1n, 0n, 1n, 3n, q6s_i(4n, 7n, 0n, 1n, 0n, 1n, 3n, q6s_i(3n, 7n, 0n, 1n, 0n, 1n, 3n, q6s_i(2n, 7n, 0n, 1n, 0n, 1n, 3n, q6s_i(1n, 7n, 0n, 1n, 0n, 1n, 3n, q6s_i(0n, 7n, 0n, 1n, 0n, 1n, 3n, {==}, sh_0_1_0_1_3_0()), sh_0_1_0_1_3_1()), sh_0_1_0_1_3_2()), sh_0_1_0_1_3_3()), sh_0_1_0_1_3_4()), sh_0_1_0_1_3_5())), q6s_i(5n, 7n, 0n, 1n, 0n, 1n, 4n, q6s_i(4n, 7n, 0n, 1n, 0n, 1n, 4n, q6s_i(3n, 7n, 0n, 1n, 0n, 1n, 4n, q6s_i(2n, 7n, 0n, 1n, 0n, 1n, 4n, q6s_i(1n, 7n, 0n, 1n, 0n, 1n, 4n, q6s_i(0n, 7n, 0n, 1n, 0n, 1n, 4n, {==}, sh_0_1_0_1_4_0()), sh_0_1_0_1_4_1()), sh_0_1_0_1_4_2()), sh_0_1_0_1_4_3()), sh_0_1_0_1_4_4()), sh_0_1_0_1_4_5()))

def sq5_0_1_0_2() -> {q5s(5n, 6n, 7n, 0n, 1n, 0n, 2n) == True{} : Bool}:
  q5s_i(4n, 6n, 7n, 0n, 1n, 0n, 2n, q5s_i(3n, 6n, 7n, 0n, 1n, 0n, 2n, q5s_i(2n, 6n, 7n, 0n, 1n, 0n, 2n, q5s_i(1n, 6n, 7n, 0n, 1n, 0n, 2n, q5s_i(0n, 6n, 7n, 0n, 1n, 0n, 2n, {==}, q6s_i(5n, 7n, 0n, 1n, 0n, 2n, 0n, q6s_i(4n, 7n, 0n, 1n, 0n, 2n, 0n, q6s_i(3n, 7n, 0n, 1n, 0n, 2n, 0n, q6s_i(2n, 7n, 0n, 1n, 0n, 2n, 0n, q6s_i(1n, 7n, 0n, 1n, 0n, 2n, 0n, q6s_i(0n, 7n, 0n, 1n, 0n, 2n, 0n, {==}, sh_0_1_0_2_0_0()), sh_0_1_0_2_0_1()), sh_0_1_0_2_0_2()), sh_0_1_0_2_0_3()), sh_0_1_0_2_0_4()), sh_0_1_0_2_0_5())), q6s_i(5n, 7n, 0n, 1n, 0n, 2n, 1n, q6s_i(4n, 7n, 0n, 1n, 0n, 2n, 1n, q6s_i(3n, 7n, 0n, 1n, 0n, 2n, 1n, q6s_i(2n, 7n, 0n, 1n, 0n, 2n, 1n, q6s_i(1n, 7n, 0n, 1n, 0n, 2n, 1n, q6s_i(0n, 7n, 0n, 1n, 0n, 2n, 1n, {==}, sh_0_1_0_2_1_0()), sh_0_1_0_2_1_1()), sh_0_1_0_2_1_2()), sh_0_1_0_2_1_3()), sh_0_1_0_2_1_4()), sh_0_1_0_2_1_5())), q6s_i(5n, 7n, 0n, 1n, 0n, 2n, 2n, q6s_i(4n, 7n, 0n, 1n, 0n, 2n, 2n, q6s_i(3n, 7n, 0n, 1n, 0n, 2n, 2n, q6s_i(2n, 7n, 0n, 1n, 0n, 2n, 2n, q6s_i(1n, 7n, 0n, 1n, 0n, 2n, 2n, q6s_i(0n, 7n, 0n, 1n, 0n, 2n, 2n, {==}, sh_0_1_0_2_2_0()), sh_0_1_0_2_2_1()), sh_0_1_0_2_2_2()), sh_0_1_0_2_2_3()), sh_0_1_0_2_2_4()), sh_0_1_0_2_2_5())), q6s_i(5n, 7n, 0n, 1n, 0n, 2n, 3n, q6s_i(4n, 7n, 0n, 1n, 0n, 2n, 3n, q6s_i(3n, 7n, 0n, 1n, 0n, 2n, 3n, q6s_i(2n, 7n, 0n, 1n, 0n, 2n, 3n, q6s_i(1n, 7n, 0n, 1n, 0n, 2n, 3n, q6s_i(0n, 7n, 0n, 1n, 0n, 2n, 3n, {==}, sh_0_1_0_2_3_0()), sh_0_1_0_2_3_1()), sh_0_1_0_2_3_2()), sh_0_1_0_2_3_3()), sh_0_1_0_2_3_4()), sh_0_1_0_2_3_5())), q6s_i(5n, 7n, 0n, 1n, 0n, 2n, 4n, q6s_i(4n, 7n, 0n, 1n, 0n, 2n, 4n, q6s_i(3n, 7n, 0n, 1n, 0n, 2n, 4n, q6s_i(2n, 7n, 0n, 1n, 0n, 2n, 4n, q6s_i(1n, 7n, 0n, 1n, 0n, 2n, 4n, q6s_i(0n, 7n, 0n, 1n, 0n, 2n, 4n, {==}, sh_0_1_0_2_4_0()), sh_0_1_0_2_4_1()), sh_0_1_0_2_4_2()), sh_0_1_0_2_4_3()), sh_0_1_0_2_4_4()), sh_0_1_0_2_4_5()))

def sq5_0_1_0_3() -> {q5s(5n, 6n, 7n, 0n, 1n, 0n, 3n) == True{} : Bool}:
  q5s_i(4n, 6n, 7n, 0n, 1n, 0n, 3n, q5s_i(3n, 6n, 7n, 0n, 1n, 0n, 3n, q5s_i(2n, 6n, 7n, 0n, 1n, 0n, 3n, q5s_i(1n, 6n, 7n, 0n, 1n, 0n, 3n, q5s_i(0n, 6n, 7n, 0n, 1n, 0n, 3n, {==}, q6s_i(5n, 7n, 0n, 1n, 0n, 3n, 0n, q6s_i(4n, 7n, 0n, 1n, 0n, 3n, 0n, q6s_i(3n, 7n, 0n, 1n, 0n, 3n, 0n, q6s_i(2n, 7n, 0n, 1n, 0n, 3n, 0n, q6s_i(1n, 7n, 0n, 1n, 0n, 3n, 0n, q6s_i(0n, 7n, 0n, 1n, 0n, 3n, 0n, {==}, sh_0_1_0_3_0_0()), sh_0_1_0_3_0_1()), sh_0_1_0_3_0_2()), sh_0_1_0_3_0_3()), sh_0_1_0_3_0_4()), sh_0_1_0_3_0_5())), q6s_i(5n, 7n, 0n, 1n, 0n, 3n, 1n, q6s_i(4n, 7n, 0n, 1n, 0n, 3n, 1n, q6s_i(3n, 7n, 0n, 1n, 0n, 3n, 1n, q6s_i(2n, 7n, 0n, 1n, 0n, 3n, 1n, q6s_i(1n, 7n, 0n, 1n, 0n, 3n, 1n, q6s_i(0n, 7n, 0n, 1n, 0n, 3n, 1n, {==}, sh_0_1_0_3_1_0()), sh_0_1_0_3_1_1()), sh_0_1_0_3_1_2()), sh_0_1_0_3_1_3()), sh_0_1_0_3_1_4()), sh_0_1_0_3_1_5())), q6s_i(5n, 7n, 0n, 1n, 0n, 3n, 2n, q6s_i(4n, 7n, 0n, 1n, 0n, 3n, 2n, q6s_i(3n, 7n, 0n, 1n, 0n, 3n, 2n, q6s_i(2n, 7n, 0n, 1n, 0n, 3n, 2n, q6s_i(1n, 7n, 0n, 1n, 0n, 3n, 2n, q6s_i(0n, 7n, 0n, 1n, 0n, 3n, 2n, {==}, sh_0_1_0_3_2_0()), sh_0_1_0_3_2_1()), sh_0_1_0_3_2_2()), sh_0_1_0_3_2_3()), sh_0_1_0_3_2_4()), sh_0_1_0_3_2_5())), q6s_i(5n, 7n, 0n, 1n, 0n, 3n, 3n, q6s_i(4n, 7n, 0n, 1n, 0n, 3n, 3n, q6s_i(3n, 7n, 0n, 1n, 0n, 3n, 3n, q6s_i(2n, 7n, 0n, 1n, 0n, 3n, 3n, q6s_i(1n, 7n, 0n, 1n, 0n, 3n, 3n, q6s_i(0n, 7n, 0n, 1n, 0n, 3n, 3n, {==}, sh_0_1_0_3_3_0()), sh_0_1_0_3_3_1()), sh_0_1_0_3_3_2()), sh_0_1_0_3_3_3()), sh_0_1_0_3_3_4()), sh_0_1_0_3_3_5())), q6s_i(5n, 7n, 0n, 1n, 0n, 3n, 4n, q6s_i(4n, 7n, 0n, 1n, 0n, 3n, 4n, q6s_i(3n, 7n, 0n, 1n, 0n, 3n, 4n, q6s_i(2n, 7n, 0n, 1n, 0n, 3n, 4n, q6s_i(1n, 7n, 0n, 1n, 0n, 3n, 4n, q6s_i(0n, 7n, 0n, 1n, 0n, 3n, 4n, {==}, sh_0_1_0_3_4_0()), sh_0_1_0_3_4_1()), sh_0_1_0_3_4_2()), sh_0_1_0_3_4_3()), sh_0_1_0_3_4_4()), sh_0_1_0_3_4_5()))

def sq5_0_1_1_0() -> {q5s(5n, 6n, 7n, 0n, 1n, 1n, 0n) == True{} : Bool}:
  q5s_i(4n, 6n, 7n, 0n, 1n, 1n, 0n, q5s_i(3n, 6n, 7n, 0n, 1n, 1n, 0n, q5s_i(2n, 6n, 7n, 0n, 1n, 1n, 0n, q5s_i(1n, 6n, 7n, 0n, 1n, 1n, 0n, q5s_i(0n, 6n, 7n, 0n, 1n, 1n, 0n, {==}, q6s_i(5n, 7n, 0n, 1n, 1n, 0n, 0n, q6s_i(4n, 7n, 0n, 1n, 1n, 0n, 0n, q6s_i(3n, 7n, 0n, 1n, 1n, 0n, 0n, q6s_i(2n, 7n, 0n, 1n, 1n, 0n, 0n, q6s_i(1n, 7n, 0n, 1n, 1n, 0n, 0n, q6s_i(0n, 7n, 0n, 1n, 1n, 0n, 0n, {==}, sh_0_1_1_0_0_0()), sh_0_1_1_0_0_1()), sh_0_1_1_0_0_2()), sh_0_1_1_0_0_3()), sh_0_1_1_0_0_4()), sh_0_1_1_0_0_5())), q6s_i(5n, 7n, 0n, 1n, 1n, 0n, 1n, q6s_i(4n, 7n, 0n, 1n, 1n, 0n, 1n, q6s_i(3n, 7n, 0n, 1n, 1n, 0n, 1n, q6s_i(2n, 7n, 0n, 1n, 1n, 0n, 1n, q6s_i(1n, 7n, 0n, 1n, 1n, 0n, 1n, q6s_i(0n, 7n, 0n, 1n, 1n, 0n, 1n, {==}, sh_0_1_1_0_1_0()), sh_0_1_1_0_1_1()), sh_0_1_1_0_1_2()), sh_0_1_1_0_1_3()), sh_0_1_1_0_1_4()), sh_0_1_1_0_1_5())), q6s_i(5n, 7n, 0n, 1n, 1n, 0n, 2n, q6s_i(4n, 7n, 0n, 1n, 1n, 0n, 2n, q6s_i(3n, 7n, 0n, 1n, 1n, 0n, 2n, q6s_i(2n, 7n, 0n, 1n, 1n, 0n, 2n, q6s_i(1n, 7n, 0n, 1n, 1n, 0n, 2n, q6s_i(0n, 7n, 0n, 1n, 1n, 0n, 2n, {==}, sh_0_1_1_0_2_0()), sh_0_1_1_0_2_1()), sh_0_1_1_0_2_2()), sh_0_1_1_0_2_3()), sh_0_1_1_0_2_4()), sh_0_1_1_0_2_5())), q6s_i(5n, 7n, 0n, 1n, 1n, 0n, 3n, q6s_i(4n, 7n, 0n, 1n, 1n, 0n, 3n, q6s_i(3n, 7n, 0n, 1n, 1n, 0n, 3n, q6s_i(2n, 7n, 0n, 1n, 1n, 0n, 3n, q6s_i(1n, 7n, 0n, 1n, 1n, 0n, 3n, q6s_i(0n, 7n, 0n, 1n, 1n, 0n, 3n, {==}, sh_0_1_1_0_3_0()), sh_0_1_1_0_3_1()), sh_0_1_1_0_3_2()), sh_0_1_1_0_3_3()), sh_0_1_1_0_3_4()), sh_0_1_1_0_3_5())), q6s_i(5n, 7n, 0n, 1n, 1n, 0n, 4n, q6s_i(4n, 7n, 0n, 1n, 1n, 0n, 4n, q6s_i(3n, 7n, 0n, 1n, 1n, 0n, 4n, q6s_i(2n, 7n, 0n, 1n, 1n, 0n, 4n, q6s_i(1n, 7n, 0n, 1n, 1n, 0n, 4n, q6s_i(0n, 7n, 0n, 1n, 1n, 0n, 4n, {==}, sh_0_1_1_0_4_0()), sh_0_1_1_0_4_1()), sh_0_1_1_0_4_2()), sh_0_1_1_0_4_3()), sh_0_1_1_0_4_4()), sh_0_1_1_0_4_5()))

def sq5_0_1_1_1() -> {q5s(5n, 6n, 7n, 0n, 1n, 1n, 1n) == True{} : Bool}:
  q5s_i(4n, 6n, 7n, 0n, 1n, 1n, 1n, q5s_i(3n, 6n, 7n, 0n, 1n, 1n, 1n, q5s_i(2n, 6n, 7n, 0n, 1n, 1n, 1n, q5s_i(1n, 6n, 7n, 0n, 1n, 1n, 1n, q5s_i(0n, 6n, 7n, 0n, 1n, 1n, 1n, {==}, q6s_i(5n, 7n, 0n, 1n, 1n, 1n, 0n, q6s_i(4n, 7n, 0n, 1n, 1n, 1n, 0n, q6s_i(3n, 7n, 0n, 1n, 1n, 1n, 0n, q6s_i(2n, 7n, 0n, 1n, 1n, 1n, 0n, q6s_i(1n, 7n, 0n, 1n, 1n, 1n, 0n, q6s_i(0n, 7n, 0n, 1n, 1n, 1n, 0n, {==}, sh_0_1_1_1_0_0()), sh_0_1_1_1_0_1()), sh_0_1_1_1_0_2()), sh_0_1_1_1_0_3()), sh_0_1_1_1_0_4()), sh_0_1_1_1_0_5())), q6s_i(5n, 7n, 0n, 1n, 1n, 1n, 1n, q6s_i(4n, 7n, 0n, 1n, 1n, 1n, 1n, q6s_i(3n, 7n, 0n, 1n, 1n, 1n, 1n, q6s_i(2n, 7n, 0n, 1n, 1n, 1n, 1n, q6s_i(1n, 7n, 0n, 1n, 1n, 1n, 1n, q6s_i(0n, 7n, 0n, 1n, 1n, 1n, 1n, {==}, sh_0_1_1_1_1_0()), sh_0_1_1_1_1_1()), sh_0_1_1_1_1_2()), sh_0_1_1_1_1_3()), sh_0_1_1_1_1_4()), sh_0_1_1_1_1_5())), q6s_i(5n, 7n, 0n, 1n, 1n, 1n, 2n, q6s_i(4n, 7n, 0n, 1n, 1n, 1n, 2n, q6s_i(3n, 7n, 0n, 1n, 1n, 1n, 2n, q6s_i(2n, 7n, 0n, 1n, 1n, 1n, 2n, q6s_i(1n, 7n, 0n, 1n, 1n, 1n, 2n, q6s_i(0n, 7n, 0n, 1n, 1n, 1n, 2n, {==}, sh_0_1_1_1_2_0()), sh_0_1_1_1_2_1()), sh_0_1_1_1_2_2()), sh_0_1_1_1_2_3()), sh_0_1_1_1_2_4()), sh_0_1_1_1_2_5())), q6s_i(5n, 7n, 0n, 1n, 1n, 1n, 3n, q6s_i(4n, 7n, 0n, 1n, 1n, 1n, 3n, q6s_i(3n, 7n, 0n, 1n, 1n, 1n, 3n, q6s_i(2n, 7n, 0n, 1n, 1n, 1n, 3n, q6s_i(1n, 7n, 0n, 1n, 1n, 1n, 3n, q6s_i(0n, 7n, 0n, 1n, 1n, 1n, 3n, {==}, sh_0_1_1_1_3_0()), sh_0_1_1_1_3_1()), sh_0_1_1_1_3_2()), sh_0_1_1_1_3_3()), sh_0_1_1_1_3_4()), sh_0_1_1_1_3_5())), q6s_i(5n, 7n, 0n, 1n, 1n, 1n, 4n, q6s_i(4n, 7n, 0n, 1n, 1n, 1n, 4n, q6s_i(3n, 7n, 0n, 1n, 1n, 1n, 4n, q6s_i(2n, 7n, 0n, 1n, 1n, 1n, 4n, q6s_i(1n, 7n, 0n, 1n, 1n, 1n, 4n, q6s_i(0n, 7n, 0n, 1n, 1n, 1n, 4n, {==}, sh_0_1_1_1_4_0()), sh_0_1_1_1_4_1()), sh_0_1_1_1_4_2()), sh_0_1_1_1_4_3()), sh_0_1_1_1_4_4()), sh_0_1_1_1_4_5()))

def sq5_0_1_1_2() -> {q5s(5n, 6n, 7n, 0n, 1n, 1n, 2n) == True{} : Bool}:
  q5s_i(4n, 6n, 7n, 0n, 1n, 1n, 2n, q5s_i(3n, 6n, 7n, 0n, 1n, 1n, 2n, q5s_i(2n, 6n, 7n, 0n, 1n, 1n, 2n, q5s_i(1n, 6n, 7n, 0n, 1n, 1n, 2n, q5s_i(0n, 6n, 7n, 0n, 1n, 1n, 2n, {==}, q6s_i(5n, 7n, 0n, 1n, 1n, 2n, 0n, q6s_i(4n, 7n, 0n, 1n, 1n, 2n, 0n, q6s_i(3n, 7n, 0n, 1n, 1n, 2n, 0n, q6s_i(2n, 7n, 0n, 1n, 1n, 2n, 0n, q6s_i(1n, 7n, 0n, 1n, 1n, 2n, 0n, q6s_i(0n, 7n, 0n, 1n, 1n, 2n, 0n, {==}, sh_0_1_1_2_0_0()), sh_0_1_1_2_0_1()), sh_0_1_1_2_0_2()), sh_0_1_1_2_0_3()), sh_0_1_1_2_0_4()), sh_0_1_1_2_0_5())), q6s_i(5n, 7n, 0n, 1n, 1n, 2n, 1n, q6s_i(4n, 7n, 0n, 1n, 1n, 2n, 1n, q6s_i(3n, 7n, 0n, 1n, 1n, 2n, 1n, q6s_i(2n, 7n, 0n, 1n, 1n, 2n, 1n, q6s_i(1n, 7n, 0n, 1n, 1n, 2n, 1n, q6s_i(0n, 7n, 0n, 1n, 1n, 2n, 1n, {==}, sh_0_1_1_2_1_0()), sh_0_1_1_2_1_1()), sh_0_1_1_2_1_2()), sh_0_1_1_2_1_3()), sh_0_1_1_2_1_4()), sh_0_1_1_2_1_5())), q6s_i(5n, 7n, 0n, 1n, 1n, 2n, 2n, q6s_i(4n, 7n, 0n, 1n, 1n, 2n, 2n, q6s_i(3n, 7n, 0n, 1n, 1n, 2n, 2n, q6s_i(2n, 7n, 0n, 1n, 1n, 2n, 2n, q6s_i(1n, 7n, 0n, 1n, 1n, 2n, 2n, q6s_i(0n, 7n, 0n, 1n, 1n, 2n, 2n, {==}, sh_0_1_1_2_2_0()), sh_0_1_1_2_2_1()), sh_0_1_1_2_2_2()), sh_0_1_1_2_2_3()), sh_0_1_1_2_2_4()), sh_0_1_1_2_2_5())), q6s_i(5n, 7n, 0n, 1n, 1n, 2n, 3n, q6s_i(4n, 7n, 0n, 1n, 1n, 2n, 3n, q6s_i(3n, 7n, 0n, 1n, 1n, 2n, 3n, q6s_i(2n, 7n, 0n, 1n, 1n, 2n, 3n, q6s_i(1n, 7n, 0n, 1n, 1n, 2n, 3n, q6s_i(0n, 7n, 0n, 1n, 1n, 2n, 3n, {==}, sh_0_1_1_2_3_0()), sh_0_1_1_2_3_1()), sh_0_1_1_2_3_2()), sh_0_1_1_2_3_3()), sh_0_1_1_2_3_4()), sh_0_1_1_2_3_5())), q6s_i(5n, 7n, 0n, 1n, 1n, 2n, 4n, q6s_i(4n, 7n, 0n, 1n, 1n, 2n, 4n, q6s_i(3n, 7n, 0n, 1n, 1n, 2n, 4n, q6s_i(2n, 7n, 0n, 1n, 1n, 2n, 4n, q6s_i(1n, 7n, 0n, 1n, 1n, 2n, 4n, q6s_i(0n, 7n, 0n, 1n, 1n, 2n, 4n, {==}, sh_0_1_1_2_4_0()), sh_0_1_1_2_4_1()), sh_0_1_1_2_4_2()), sh_0_1_1_2_4_3()), sh_0_1_1_2_4_4()), sh_0_1_1_2_4_5()))

def sq5_0_1_1_3() -> {q5s(5n, 6n, 7n, 0n, 1n, 1n, 3n) == True{} : Bool}:
  q5s_i(4n, 6n, 7n, 0n, 1n, 1n, 3n, q5s_i(3n, 6n, 7n, 0n, 1n, 1n, 3n, q5s_i(2n, 6n, 7n, 0n, 1n, 1n, 3n, q5s_i(1n, 6n, 7n, 0n, 1n, 1n, 3n, q5s_i(0n, 6n, 7n, 0n, 1n, 1n, 3n, {==}, q6s_i(5n, 7n, 0n, 1n, 1n, 3n, 0n, q6s_i(4n, 7n, 0n, 1n, 1n, 3n, 0n, q6s_i(3n, 7n, 0n, 1n, 1n, 3n, 0n, q6s_i(2n, 7n, 0n, 1n, 1n, 3n, 0n, q6s_i(1n, 7n, 0n, 1n, 1n, 3n, 0n, q6s_i(0n, 7n, 0n, 1n, 1n, 3n, 0n, {==}, sh_0_1_1_3_0_0()), sh_0_1_1_3_0_1()), sh_0_1_1_3_0_2()), sh_0_1_1_3_0_3()), sh_0_1_1_3_0_4()), sh_0_1_1_3_0_5())), q6s_i(5n, 7n, 0n, 1n, 1n, 3n, 1n, q6s_i(4n, 7n, 0n, 1n, 1n, 3n, 1n, q6s_i(3n, 7n, 0n, 1n, 1n, 3n, 1n, q6s_i(2n, 7n, 0n, 1n, 1n, 3n, 1n, q6s_i(1n, 7n, 0n, 1n, 1n, 3n, 1n, q6s_i(0n, 7n, 0n, 1n, 1n, 3n, 1n, {==}, sh_0_1_1_3_1_0()), sh_0_1_1_3_1_1()), sh_0_1_1_3_1_2()), sh_0_1_1_3_1_3()), sh_0_1_1_3_1_4()), sh_0_1_1_3_1_5())), q6s_i(5n, 7n, 0n, 1n, 1n, 3n, 2n, q6s_i(4n, 7n, 0n, 1n, 1n, 3n, 2n, q6s_i(3n, 7n, 0n, 1n, 1n, 3n, 2n, q6s_i(2n, 7n, 0n, 1n, 1n, 3n, 2n, q6s_i(1n, 7n, 0n, 1n, 1n, 3n, 2n, q6s_i(0n, 7n, 0n, 1n, 1n, 3n, 2n, {==}, sh_0_1_1_3_2_0()), sh_0_1_1_3_2_1()), sh_0_1_1_3_2_2()), sh_0_1_1_3_2_3()), sh_0_1_1_3_2_4()), sh_0_1_1_3_2_5())), q6s_i(5n, 7n, 0n, 1n, 1n, 3n, 3n, q6s_i(4n, 7n, 0n, 1n, 1n, 3n, 3n, q6s_i(3n, 7n, 0n, 1n, 1n, 3n, 3n, q6s_i(2n, 7n, 0n, 1n, 1n, 3n, 3n, q6s_i(1n, 7n, 0n, 1n, 1n, 3n, 3n, q6s_i(0n, 7n, 0n, 1n, 1n, 3n, 3n, {==}, sh_0_1_1_3_3_0()), sh_0_1_1_3_3_1()), sh_0_1_1_3_3_2()), sh_0_1_1_3_3_3()), sh_0_1_1_3_3_4()), sh_0_1_1_3_3_5())), q6s_i(5n, 7n, 0n, 1n, 1n, 3n, 4n, q6s_i(4n, 7n, 0n, 1n, 1n, 3n, 4n, q6s_i(3n, 7n, 0n, 1n, 1n, 3n, 4n, q6s_i(2n, 7n, 0n, 1n, 1n, 3n, 4n, q6s_i(1n, 7n, 0n, 1n, 1n, 3n, 4n, q6s_i(0n, 7n, 0n, 1n, 1n, 3n, 4n, {==}, sh_0_1_1_3_4_0()), sh_0_1_1_3_4_1()), sh_0_1_1_3_4_2()), sh_0_1_1_3_4_3()), sh_0_1_1_3_4_4()), sh_0_1_1_3_4_5()))

def sq5_0_1_2_0() -> {q5s(5n, 6n, 7n, 0n, 1n, 2n, 0n) == True{} : Bool}:
  q5s_i(4n, 6n, 7n, 0n, 1n, 2n, 0n, q5s_i(3n, 6n, 7n, 0n, 1n, 2n, 0n, q5s_i(2n, 6n, 7n, 0n, 1n, 2n, 0n, q5s_i(1n, 6n, 7n, 0n, 1n, 2n, 0n, q5s_i(0n, 6n, 7n, 0n, 1n, 2n, 0n, {==}, q6s_i(5n, 7n, 0n, 1n, 2n, 0n, 0n, q6s_i(4n, 7n, 0n, 1n, 2n, 0n, 0n, q6s_i(3n, 7n, 0n, 1n, 2n, 0n, 0n, q6s_i(2n, 7n, 0n, 1n, 2n, 0n, 0n, q6s_i(1n, 7n, 0n, 1n, 2n, 0n, 0n, q6s_i(0n, 7n, 0n, 1n, 2n, 0n, 0n, {==}, sh_0_1_2_0_0_0()), sh_0_1_2_0_0_1()), sh_0_1_2_0_0_2()), sh_0_1_2_0_0_3()), sh_0_1_2_0_0_4()), sh_0_1_2_0_0_5())), q6s_i(5n, 7n, 0n, 1n, 2n, 0n, 1n, q6s_i(4n, 7n, 0n, 1n, 2n, 0n, 1n, q6s_i(3n, 7n, 0n, 1n, 2n, 0n, 1n, q6s_i(2n, 7n, 0n, 1n, 2n, 0n, 1n, q6s_i(1n, 7n, 0n, 1n, 2n, 0n, 1n, q6s_i(0n, 7n, 0n, 1n, 2n, 0n, 1n, {==}, sh_0_1_2_0_1_0()), sh_0_1_2_0_1_1()), sh_0_1_2_0_1_2()), sh_0_1_2_0_1_3()), sh_0_1_2_0_1_4()), sh_0_1_2_0_1_5())), q6s_i(5n, 7n, 0n, 1n, 2n, 0n, 2n, q6s_i(4n, 7n, 0n, 1n, 2n, 0n, 2n, q6s_i(3n, 7n, 0n, 1n, 2n, 0n, 2n, q6s_i(2n, 7n, 0n, 1n, 2n, 0n, 2n, q6s_i(1n, 7n, 0n, 1n, 2n, 0n, 2n, q6s_i(0n, 7n, 0n, 1n, 2n, 0n, 2n, {==}, sh_0_1_2_0_2_0()), sh_0_1_2_0_2_1()), sh_0_1_2_0_2_2()), sh_0_1_2_0_2_3()), sh_0_1_2_0_2_4()), sh_0_1_2_0_2_5())), q6s_i(5n, 7n, 0n, 1n, 2n, 0n, 3n, q6s_i(4n, 7n, 0n, 1n, 2n, 0n, 3n, q6s_i(3n, 7n, 0n, 1n, 2n, 0n, 3n, q6s_i(2n, 7n, 0n, 1n, 2n, 0n, 3n, q6s_i(1n, 7n, 0n, 1n, 2n, 0n, 3n, q6s_i(0n, 7n, 0n, 1n, 2n, 0n, 3n, {==}, sh_0_1_2_0_3_0()), sh_0_1_2_0_3_1()), sh_0_1_2_0_3_2()), sh_0_1_2_0_3_3()), sh_0_1_2_0_3_4()), sh_0_1_2_0_3_5())), q6s_i(5n, 7n, 0n, 1n, 2n, 0n, 4n, q6s_i(4n, 7n, 0n, 1n, 2n, 0n, 4n, q6s_i(3n, 7n, 0n, 1n, 2n, 0n, 4n, q6s_i(2n, 7n, 0n, 1n, 2n, 0n, 4n, q6s_i(1n, 7n, 0n, 1n, 2n, 0n, 4n, q6s_i(0n, 7n, 0n, 1n, 2n, 0n, 4n, {==}, sh_0_1_2_0_4_0()), sh_0_1_2_0_4_1()), sh_0_1_2_0_4_2()), sh_0_1_2_0_4_3()), sh_0_1_2_0_4_4()), sh_0_1_2_0_4_5()))

def sq5_0_1_2_1() -> {q5s(5n, 6n, 7n, 0n, 1n, 2n, 1n) == True{} : Bool}:
  q5s_i(4n, 6n, 7n, 0n, 1n, 2n, 1n, q5s_i(3n, 6n, 7n, 0n, 1n, 2n, 1n, q5s_i(2n, 6n, 7n, 0n, 1n, 2n, 1n, q5s_i(1n, 6n, 7n, 0n, 1n, 2n, 1n, q5s_i(0n, 6n, 7n, 0n, 1n, 2n, 1n, {==}, q6s_i(5n, 7n, 0n, 1n, 2n, 1n, 0n, q6s_i(4n, 7n, 0n, 1n, 2n, 1n, 0n, q6s_i(3n, 7n, 0n, 1n, 2n, 1n, 0n, q6s_i(2n, 7n, 0n, 1n, 2n, 1n, 0n, q6s_i(1n, 7n, 0n, 1n, 2n, 1n, 0n, q6s_i(0n, 7n, 0n, 1n, 2n, 1n, 0n, {==}, sh_0_1_2_1_0_0()), sh_0_1_2_1_0_1()), sh_0_1_2_1_0_2()), sh_0_1_2_1_0_3()), sh_0_1_2_1_0_4()), sh_0_1_2_1_0_5())), q6s_i(5n, 7n, 0n, 1n, 2n, 1n, 1n, q6s_i(4n, 7n, 0n, 1n, 2n, 1n, 1n, q6s_i(3n, 7n, 0n, 1n, 2n, 1n, 1n, q6s_i(2n, 7n, 0n, 1n, 2n, 1n, 1n, q6s_i(1n, 7n, 0n, 1n, 2n, 1n, 1n, q6s_i(0n, 7n, 0n, 1n, 2n, 1n, 1n, {==}, sh_0_1_2_1_1_0()), sh_0_1_2_1_1_1()), sh_0_1_2_1_1_2()), sh_0_1_2_1_1_3()), sh_0_1_2_1_1_4()), sh_0_1_2_1_1_5())), q6s_i(5n, 7n, 0n, 1n, 2n, 1n, 2n, q6s_i(4n, 7n, 0n, 1n, 2n, 1n, 2n, q6s_i(3n, 7n, 0n, 1n, 2n, 1n, 2n, q6s_i(2n, 7n, 0n, 1n, 2n, 1n, 2n, q6s_i(1n, 7n, 0n, 1n, 2n, 1n, 2n, q6s_i(0n, 7n, 0n, 1n, 2n, 1n, 2n, {==}, sh_0_1_2_1_2_0()), sh_0_1_2_1_2_1()), sh_0_1_2_1_2_2()), sh_0_1_2_1_2_3()), sh_0_1_2_1_2_4()), sh_0_1_2_1_2_5())), q6s_i(5n, 7n, 0n, 1n, 2n, 1n, 3n, q6s_i(4n, 7n, 0n, 1n, 2n, 1n, 3n, q6s_i(3n, 7n, 0n, 1n, 2n, 1n, 3n, q6s_i(2n, 7n, 0n, 1n, 2n, 1n, 3n, q6s_i(1n, 7n, 0n, 1n, 2n, 1n, 3n, q6s_i(0n, 7n, 0n, 1n, 2n, 1n, 3n, {==}, sh_0_1_2_1_3_0()), sh_0_1_2_1_3_1()), sh_0_1_2_1_3_2()), sh_0_1_2_1_3_3()), sh_0_1_2_1_3_4()), sh_0_1_2_1_3_5())), q6s_i(5n, 7n, 0n, 1n, 2n, 1n, 4n, q6s_i(4n, 7n, 0n, 1n, 2n, 1n, 4n, q6s_i(3n, 7n, 0n, 1n, 2n, 1n, 4n, q6s_i(2n, 7n, 0n, 1n, 2n, 1n, 4n, q6s_i(1n, 7n, 0n, 1n, 2n, 1n, 4n, q6s_i(0n, 7n, 0n, 1n, 2n, 1n, 4n, {==}, sh_0_1_2_1_4_0()), sh_0_1_2_1_4_1()), sh_0_1_2_1_4_2()), sh_0_1_2_1_4_3()), sh_0_1_2_1_4_4()), sh_0_1_2_1_4_5()))

def sq5_0_1_2_2() -> {q5s(5n, 6n, 7n, 0n, 1n, 2n, 2n) == True{} : Bool}:
  q5s_i(4n, 6n, 7n, 0n, 1n, 2n, 2n, q5s_i(3n, 6n, 7n, 0n, 1n, 2n, 2n, q5s_i(2n, 6n, 7n, 0n, 1n, 2n, 2n, q5s_i(1n, 6n, 7n, 0n, 1n, 2n, 2n, q5s_i(0n, 6n, 7n, 0n, 1n, 2n, 2n, {==}, q6s_i(5n, 7n, 0n, 1n, 2n, 2n, 0n, q6s_i(4n, 7n, 0n, 1n, 2n, 2n, 0n, q6s_i(3n, 7n, 0n, 1n, 2n, 2n, 0n, q6s_i(2n, 7n, 0n, 1n, 2n, 2n, 0n, q6s_i(1n, 7n, 0n, 1n, 2n, 2n, 0n, q6s_i(0n, 7n, 0n, 1n, 2n, 2n, 0n, {==}, sh_0_1_2_2_0_0()), sh_0_1_2_2_0_1()), sh_0_1_2_2_0_2()), sh_0_1_2_2_0_3()), sh_0_1_2_2_0_4()), sh_0_1_2_2_0_5())), q6s_i(5n, 7n, 0n, 1n, 2n, 2n, 1n, q6s_i(4n, 7n, 0n, 1n, 2n, 2n, 1n, q6s_i(3n, 7n, 0n, 1n, 2n, 2n, 1n, q6s_i(2n, 7n, 0n, 1n, 2n, 2n, 1n, q6s_i(1n, 7n, 0n, 1n, 2n, 2n, 1n, q6s_i(0n, 7n, 0n, 1n, 2n, 2n, 1n, {==}, sh_0_1_2_2_1_0()), sh_0_1_2_2_1_1()), sh_0_1_2_2_1_2()), sh_0_1_2_2_1_3()), sh_0_1_2_2_1_4()), sh_0_1_2_2_1_5())), q6s_i(5n, 7n, 0n, 1n, 2n, 2n, 2n, q6s_i(4n, 7n, 0n, 1n, 2n, 2n, 2n, q6s_i(3n, 7n, 0n, 1n, 2n, 2n, 2n, q6s_i(2n, 7n, 0n, 1n, 2n, 2n, 2n, q6s_i(1n, 7n, 0n, 1n, 2n, 2n, 2n, q6s_i(0n, 7n, 0n, 1n, 2n, 2n, 2n, {==}, sh_0_1_2_2_2_0()), sh_0_1_2_2_2_1()), sh_0_1_2_2_2_2()), sh_0_1_2_2_2_3()), sh_0_1_2_2_2_4()), sh_0_1_2_2_2_5())), q6s_i(5n, 7n, 0n, 1n, 2n, 2n, 3n, q6s_i(4n, 7n, 0n, 1n, 2n, 2n, 3n, q6s_i(3n, 7n, 0n, 1n, 2n, 2n, 3n, q6s_i(2n, 7n, 0n, 1n, 2n, 2n, 3n, q6s_i(1n, 7n, 0n, 1n, 2n, 2n, 3n, q6s_i(0n, 7n, 0n, 1n, 2n, 2n, 3n, {==}, sh_0_1_2_2_3_0()), sh_0_1_2_2_3_1()), sh_0_1_2_2_3_2()), sh_0_1_2_2_3_3()), sh_0_1_2_2_3_4()), sh_0_1_2_2_3_5())), q6s_i(5n, 7n, 0n, 1n, 2n, 2n, 4n, q6s_i(4n, 7n, 0n, 1n, 2n, 2n, 4n, q6s_i(3n, 7n, 0n, 1n, 2n, 2n, 4n, q6s_i(2n, 7n, 0n, 1n, 2n, 2n, 4n, q6s_i(1n, 7n, 0n, 1n, 2n, 2n, 4n, q6s_i(0n, 7n, 0n, 1n, 2n, 2n, 4n, {==}, sh_0_1_2_2_4_0()), sh_0_1_2_2_4_1()), sh_0_1_2_2_4_2()), sh_0_1_2_2_4_3()), sh_0_1_2_2_4_4()), sh_0_1_2_2_4_5()))

def sq5_0_1_2_3() -> {q5s(5n, 6n, 7n, 0n, 1n, 2n, 3n) == True{} : Bool}:
  q5s_i(4n, 6n, 7n, 0n, 1n, 2n, 3n, q5s_i(3n, 6n, 7n, 0n, 1n, 2n, 3n, q5s_i(2n, 6n, 7n, 0n, 1n, 2n, 3n, q5s_i(1n, 6n, 7n, 0n, 1n, 2n, 3n, q5s_i(0n, 6n, 7n, 0n, 1n, 2n, 3n, {==}, q6s_i(5n, 7n, 0n, 1n, 2n, 3n, 0n, q6s_i(4n, 7n, 0n, 1n, 2n, 3n, 0n, q6s_i(3n, 7n, 0n, 1n, 2n, 3n, 0n, q6s_i(2n, 7n, 0n, 1n, 2n, 3n, 0n, q6s_i(1n, 7n, 0n, 1n, 2n, 3n, 0n, q6s_i(0n, 7n, 0n, 1n, 2n, 3n, 0n, {==}, sh_0_1_2_3_0_0()), sh_0_1_2_3_0_1()), sh_0_1_2_3_0_2()), sh_0_1_2_3_0_3()), sh_0_1_2_3_0_4()), sh_0_1_2_3_0_5())), q6s_i(5n, 7n, 0n, 1n, 2n, 3n, 1n, q6s_i(4n, 7n, 0n, 1n, 2n, 3n, 1n, q6s_i(3n, 7n, 0n, 1n, 2n, 3n, 1n, q6s_i(2n, 7n, 0n, 1n, 2n, 3n, 1n, q6s_i(1n, 7n, 0n, 1n, 2n, 3n, 1n, q6s_i(0n, 7n, 0n, 1n, 2n, 3n, 1n, {==}, sh_0_1_2_3_1_0()), sh_0_1_2_3_1_1()), sh_0_1_2_3_1_2()), sh_0_1_2_3_1_3()), sh_0_1_2_3_1_4()), sh_0_1_2_3_1_5())), q6s_i(5n, 7n, 0n, 1n, 2n, 3n, 2n, q6s_i(4n, 7n, 0n, 1n, 2n, 3n, 2n, q6s_i(3n, 7n, 0n, 1n, 2n, 3n, 2n, q6s_i(2n, 7n, 0n, 1n, 2n, 3n, 2n, q6s_i(1n, 7n, 0n, 1n, 2n, 3n, 2n, q6s_i(0n, 7n, 0n, 1n, 2n, 3n, 2n, {==}, sh_0_1_2_3_2_0()), sh_0_1_2_3_2_1()), sh_0_1_2_3_2_2()), sh_0_1_2_3_2_3()), sh_0_1_2_3_2_4()), sh_0_1_2_3_2_5())), q6s_i(5n, 7n, 0n, 1n, 2n, 3n, 3n, q6s_i(4n, 7n, 0n, 1n, 2n, 3n, 3n, q6s_i(3n, 7n, 0n, 1n, 2n, 3n, 3n, q6s_i(2n, 7n, 0n, 1n, 2n, 3n, 3n, q6s_i(1n, 7n, 0n, 1n, 2n, 3n, 3n, q6s_i(0n, 7n, 0n, 1n, 2n, 3n, 3n, {==}, sh_0_1_2_3_3_0()), sh_0_1_2_3_3_1()), sh_0_1_2_3_3_2()), sh_0_1_2_3_3_3()), sh_0_1_2_3_3_4()), sh_0_1_2_3_3_5())), q6s_i(5n, 7n, 0n, 1n, 2n, 3n, 4n, q6s_i(4n, 7n, 0n, 1n, 2n, 3n, 4n, q6s_i(3n, 7n, 0n, 1n, 2n, 3n, 4n, q6s_i(2n, 7n, 0n, 1n, 2n, 3n, 4n, q6s_i(1n, 7n, 0n, 1n, 2n, 3n, 4n, q6s_i(0n, 7n, 0n, 1n, 2n, 3n, 4n, {==}, sh_0_1_2_3_4_0()), sh_0_1_2_3_4_1()), sh_0_1_2_3_4_2()), sh_0_1_2_3_4_3()), sh_0_1_2_3_4_4()), sh_0_1_2_3_4_5()))

def shape_all() -> {q1s(1n, 2n, 3n, 4n, 5n, 6n, 7n) == True{} : Bool}:
  q1s_i(0n, 2n, 3n, 4n, 5n, 6n, 7n, {==}, q2s_i(1n, 3n, 4n, 5n, 6n, 7n, 0n, q2s_i(0n, 3n, 4n, 5n, 6n, 7n, 0n, {==}, q3s_i(2n, 4n, 5n, 6n, 7n, 0n, 0n, q3s_i(1n, 4n, 5n, 6n, 7n, 0n, 0n, q3s_i(0n, 4n, 5n, 6n, 7n, 0n, 0n, {==}, q4s_i(3n, 5n, 6n, 7n, 0n, 0n, 0n, q4s_i(2n, 5n, 6n, 7n, 0n, 0n, 0n, q4s_i(1n, 5n, 6n, 7n, 0n, 0n, 0n, q4s_i(0n, 5n, 6n, 7n, 0n, 0n, 0n, {==}, sq5_0_0_0_0()), sq5_0_0_0_1()), sq5_0_0_0_2()), sq5_0_0_0_3())), q4s_i(3n, 5n, 6n, 7n, 0n, 0n, 1n, q4s_i(2n, 5n, 6n, 7n, 0n, 0n, 1n, q4s_i(1n, 5n, 6n, 7n, 0n, 0n, 1n, q4s_i(0n, 5n, 6n, 7n, 0n, 0n, 1n, {==}, sq5_0_0_1_0()), sq5_0_0_1_1()), sq5_0_0_1_2()), sq5_0_0_1_3())), q4s_i(3n, 5n, 6n, 7n, 0n, 0n, 2n, q4s_i(2n, 5n, 6n, 7n, 0n, 0n, 2n, q4s_i(1n, 5n, 6n, 7n, 0n, 0n, 2n, q4s_i(0n, 5n, 6n, 7n, 0n, 0n, 2n, {==}, sq5_0_0_2_0()), sq5_0_0_2_1()), sq5_0_0_2_2()), sq5_0_0_2_3()))), q3s_i(2n, 4n, 5n, 6n, 7n, 0n, 1n, q3s_i(1n, 4n, 5n, 6n, 7n, 0n, 1n, q3s_i(0n, 4n, 5n, 6n, 7n, 0n, 1n, {==}, q4s_i(3n, 5n, 6n, 7n, 0n, 1n, 0n, q4s_i(2n, 5n, 6n, 7n, 0n, 1n, 0n, q4s_i(1n, 5n, 6n, 7n, 0n, 1n, 0n, q4s_i(0n, 5n, 6n, 7n, 0n, 1n, 0n, {==}, sq5_0_1_0_0()), sq5_0_1_0_1()), sq5_0_1_0_2()), sq5_0_1_0_3())), q4s_i(3n, 5n, 6n, 7n, 0n, 1n, 1n, q4s_i(2n, 5n, 6n, 7n, 0n, 1n, 1n, q4s_i(1n, 5n, 6n, 7n, 0n, 1n, 1n, q4s_i(0n, 5n, 6n, 7n, 0n, 1n, 1n, {==}, sq5_0_1_1_0()), sq5_0_1_1_1()), sq5_0_1_1_2()), sq5_0_1_1_3())), q4s_i(3n, 5n, 6n, 7n, 0n, 1n, 2n, q4s_i(2n, 5n, 6n, 7n, 0n, 1n, 2n, q4s_i(1n, 5n, 6n, 7n, 0n, 1n, 2n, q4s_i(0n, 5n, 6n, 7n, 0n, 1n, 2n, {==}, sq5_0_1_2_0()), sq5_0_1_2_1()), sq5_0_1_2_2()), sq5_0_1_2_3()))))


# Down the enumeration levels, with the level bounds b2 .. b7 kept opaque (no unfolding)
def shape_chain(+q1: Nat, +q2: Nat, +q3: Nat, +q4: Nat, +q5: Nat, +q6: Nat, +q7: Nat, +b2: Nat, +b3: Nat, +b4: Nat, +b5: Nat, +b6: Nat, +b7: Nat,
    h1: {Nat.is_lt(q1, 1n) == True{} : Bool}, h2: {Nat.is_lt(q2, b2) == True{} : Bool}, h3: {Nat.is_lt(q3, b3) == True{} : Bool}, h4: {Nat.is_lt(q4, b4) == True{} : Bool}, h5: {Nat.is_lt(q5, b5) == True{} : Bool}, h6: {Nat.is_lt(q6, b6) == True{} : Bool}, h7: {Nat.is_lt(q7, b7) == True{} : Bool},
    all: {q1s(1n, b2, b3, b4, b5, b6, b7) == True{} : Bool}) -> {shape_ok(Spec.shape(q1, q2, q3, q4, q5, q6, q7)) == True{} : Bool}:
  +s1 = q1s_go(1n, q1, b2, b3, b4, b5, b6, b7, h1, all)
  +s2 = q2s_go(b2, q2, b3, b4, b5, b6, b7, q1, h2, s1)
  +s3 = q3s_go(b3, q3, b4, b5, b6, b7, q1, q2, h3, s2)
  +s4 = q4s_go(b4, q4, b5, b6, b7, q1, q2, q3, h4, s3)
  +s5 = q5s_go(b5, q5, b6, b7, q1, q2, q3, q4, h5, s4)
  +s6 = q6s_go(b6, q6, b7, q1, q2, q3, q4, q5, h6, s5)
  +s7 = q7s_go(b7, q7, q1, q2, q3, q4, q5, q6, h7, s6)
  s7

# The legal shape's decision procedure holds
def shape_true(+q1: Nat, +q2: Nat, +q3: Nat, +q4: Nat, +q5: Nat, +q6: Nat, +q7: Nat, +legal: {Spec.legal(q1, q2, q3, q4, q5, q6, q7) == True{} : Bool}) -> {shape_ok(Spec.shape(q1, q2, q3, q4, q5, q6, q7)) == True{} : Bool}:
  +l0 = legal
  +h1 = at_and_l(Nat.is_lt(q1, 1n), Bool.and(Nat.is_lt(q2, 2n), Bool.and(Nat.is_lt(q3, 3n), Bool.and(Nat.is_lt(q4, 4n), Bool.and(Nat.is_lt(q5, 5n), Bool.and(Nat.is_lt(q6, 6n), Nat.is_lt(q7, 7n)))))), l0)
  +l1 = at_and_r(Nat.is_lt(q1, 1n), Bool.and(Nat.is_lt(q2, 2n), Bool.and(Nat.is_lt(q3, 3n), Bool.and(Nat.is_lt(q4, 4n), Bool.and(Nat.is_lt(q5, 5n), Bool.and(Nat.is_lt(q6, 6n), Nat.is_lt(q7, 7n)))))), l0)
  +h2 = at_and_l(Nat.is_lt(q2, 2n), Bool.and(Nat.is_lt(q3, 3n), Bool.and(Nat.is_lt(q4, 4n), Bool.and(Nat.is_lt(q5, 5n), Bool.and(Nat.is_lt(q6, 6n), Nat.is_lt(q7, 7n))))), l1)
  +l2 = at_and_r(Nat.is_lt(q2, 2n), Bool.and(Nat.is_lt(q3, 3n), Bool.and(Nat.is_lt(q4, 4n), Bool.and(Nat.is_lt(q5, 5n), Bool.and(Nat.is_lt(q6, 6n), Nat.is_lt(q7, 7n))))), l1)
  +h3 = at_and_l(Nat.is_lt(q3, 3n), Bool.and(Nat.is_lt(q4, 4n), Bool.and(Nat.is_lt(q5, 5n), Bool.and(Nat.is_lt(q6, 6n), Nat.is_lt(q7, 7n)))), l2)
  +l3 = at_and_r(Nat.is_lt(q3, 3n), Bool.and(Nat.is_lt(q4, 4n), Bool.and(Nat.is_lt(q5, 5n), Bool.and(Nat.is_lt(q6, 6n), Nat.is_lt(q7, 7n)))), l2)
  +h4 = at_and_l(Nat.is_lt(q4, 4n), Bool.and(Nat.is_lt(q5, 5n), Bool.and(Nat.is_lt(q6, 6n), Nat.is_lt(q7, 7n))), l3)
  +l4 = at_and_r(Nat.is_lt(q4, 4n), Bool.and(Nat.is_lt(q5, 5n), Bool.and(Nat.is_lt(q6, 6n), Nat.is_lt(q7, 7n))), l3)
  +h5 = at_and_l(Nat.is_lt(q5, 5n), Bool.and(Nat.is_lt(q6, 6n), Nat.is_lt(q7, 7n)), l4)
  +l5 = at_and_r(Nat.is_lt(q5, 5n), Bool.and(Nat.is_lt(q6, 6n), Nat.is_lt(q7, 7n)), l4)
  +h6 = at_and_l(Nat.is_lt(q6, 6n), Nat.is_lt(q7, 7n), l5)
  +l6 = at_and_r(Nat.is_lt(q6, 6n), Nat.is_lt(q7, 7n), l5)
  +h7 = l6
  shape_chain(q1, q2, q3, q4, q5, q6, q7, 2n, 3n, 4n, 5n, 6n, 7n, h1, h2, h3, h4, h5, h6, h7, shape_all())

# Row r's conjuncts
def row_true(+q1: Nat, +q2: Nat, +q3: Nat, +q4: Nat, +q5: Nat, +q6: Nat, +q7: Nat, +legal: {Spec.legal(q1, q2, q3, q4, q5, q6, q7) == True{} : Bool}, +r: Nat, +hr: {Nat.is_lt(r, 8n) == True{} : Bool}) -> {row_ok(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r) == True{} : Bool}:
  rows_go(8n, r, Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), hr, at_and_l(rows(8n, Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7))), pass_ps(Impl.n_leaf(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7))), Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7))), shape_true(q1, q2, q3, q4, q5, q6, q7, legal)))

def entry_true(+q1: Nat, +q2: Nat, +q3: Nat, +q4: Nat, +q5: Nat, +q6: Nat, +q7: Nat, +legal: {Spec.legal(q1, q2, q3, q4, q5, q6, q7) == True{} : Bool}, +r: Nat, +hr: {Nat.is_lt(r, 8n) == True{} : Bool}) -> {Impl.entry_ok(8n, Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r) == True{} : Bool}:
  at_and_l(Impl.entry_ok(8n, Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), Bool.and(Nat.is_eq(Impl.pass_count(Impl.n_leaf(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7))), Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), 1n), Bool.and(Nat.is_le(Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), Impl.leaf_of(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r))), cols(8n, Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r))), row_true(q1, q2, q3, q4, q5, q6, q7, legal, r, hr))

def once_true(+q1: Nat, +q2: Nat, +q3: Nat, +q4: Nat, +q5: Nat, +q6: Nat, +q7: Nat, +legal: {Spec.legal(q1, q2, q3, q4, q5, q6, q7) == True{} : Bool}, +r: Nat, +hr: {Nat.is_lt(r, 8n) == True{} : Bool}) -> {Nat.is_eq(Impl.pass_count(Impl.n_leaf(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7))), Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), 1n) == True{} : Bool}:
  at_and_l(Nat.is_eq(Impl.pass_count(Impl.n_leaf(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7))), Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), 1n), Bool.and(Nat.is_le(Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), Impl.leaf_of(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r))), cols(8n, Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r)), at_and_r(Impl.entry_ok(8n, Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), Bool.and(Nat.is_eq(Impl.pass_count(Impl.n_leaf(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7))), Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), 1n), Bool.and(Nat.is_le(Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), Impl.leaf_of(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r))), cols(8n, Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r))), row_true(q1, q2, q3, q4, q5, q6, q7, legal, r, hr)))

def dl_true(+q1: Nat, +q2: Nat, +q3: Nat, +q4: Nat, +q5: Nat, +q6: Nat, +q7: Nat, +legal: {Spec.legal(q1, q2, q3, q4, q5, q6, q7) == True{} : Bool}, +r: Nat, +hr: {Nat.is_lt(r, 8n) == True{} : Bool}) -> {Nat.is_le(Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), Impl.leaf_of(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r))) == True{} : Bool}:
  at_and_l(Nat.is_le(Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), Impl.leaf_of(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r))), cols(8n, Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), at_and_r(Nat.is_eq(Impl.pass_count(Impl.n_leaf(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7))), Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), 1n), Bool.and(Nat.is_le(Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), Impl.leaf_of(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r))), cols(8n, Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r)), at_and_r(Impl.entry_ok(8n, Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), Bool.and(Nat.is_eq(Impl.pass_count(Impl.n_leaf(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7))), Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), 1n), Bool.and(Nat.is_le(Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), Impl.leaf_of(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r))), cols(8n, Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r))), row_true(q1, q2, q3, q4, q5, q6, q7, legal, r, hr))))

def cols_true(+q1: Nat, +q2: Nat, +q3: Nat, +q4: Nat, +q5: Nat, +q6: Nat, +q7: Nat, +legal: {Spec.legal(q1, q2, q3, q4, q5, q6, q7) == True{} : Bool}, +r: Nat, +hr: {Nat.is_lt(r, 8n) == True{} : Bool}) -> {cols(8n, Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r) == True{} : Bool}:
  at_and_r(Nat.is_le(Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), Impl.leaf_of(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r))), cols(8n, Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), at_and_r(Nat.is_eq(Impl.pass_count(Impl.n_leaf(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7))), Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), 1n), Bool.and(Nat.is_le(Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), Impl.leaf_of(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r))), cols(8n, Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r)), at_and_r(Impl.entry_ok(8n, Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), Bool.and(Nat.is_eq(Impl.pass_count(Impl.n_leaf(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7))), Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), 1n), Bool.and(Nat.is_le(Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), Impl.leaf_of(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r))), cols(8n, Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r))), row_true(q1, q2, q3, q4, q5, q6, q7, legal, r, hr))))

def pass_once_eq(+q1: Nat, +q2: Nat, +q3: Nat, +q4: Nat, +q5: Nat, +q6: Nat, +q7: Nat, +legal: {Spec.legal(q1, q2, q3, q4, q5, q6, q7) == True{} : Bool}, +r: Nat, +hr: {Nat.is_lt(r, 8n) == True{} : Bool}) -> {Impl.pass_count(Impl.n_leaf(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7))), Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r) == 1n : Nat}:
  at_eq(Impl.pass_count(Impl.n_leaf(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7))), Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), 1n, once_true(q1, q2, q3, q4, q5, q6, q7, legal, r, hr))

# entry_ok's last conjunct: depth[r] < 8 (the header bytes are the literal 0x54, 1, 8)
def depth_lt(+q1: Nat, +q2: Nat, +q3: Nat, +q4: Nat, +q5: Nat, +q6: Nat, +q7: Nat, +legal: {Spec.legal(q1, q2, q3, q4, q5, q6, q7) == True{} : Bool}, +r: Nat, +hr: {Nat.is_lt(r, 8n) == True{} : Bool}) -> {Nat.is_lt(Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), 8n) == True{} : Bool}:
  at_and_r(Nat.is_lt(Impl.n_leaf(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7))), 8n), Nat.is_lt(Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), 8n),
    at_and_r(Nat.is_le(1n, Impl.n_leaf(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)))), Bool.and(Nat.is_lt(Impl.n_leaf(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7))), 8n), Nat.is_lt(Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), 8n)), entry_true(q1, q2, q3, q4, q5, q6, q7, legal, r, hr)))

# ---------------------------------------------------------------------------------------------
# Bridge: the TreeDesc fields the reference builds are its depth and ancestors (generic in ps)

def br_depths(+i: Nat, +n: Nat, +ps: List<&2, Nat>, +k: Nat, lt: {Nat.is_lt(i, n) == True{} : Bool})
  -> {Impl.at(Spec.depths(n, ps, k), i) == Spec.depth(ps, Nat.add(k, i)) : Nat}:
  match i n:
    case 0n 0n:
      Empty.absurd({Impl.at(Spec.depths(0n, ps, k), 0n) == Spec.depth(ps, Nat.add(k, 0n)) : Nat}, Ar.ar_false_true(lt))
    case 0n 1n+m:
      %Equal.sym(Nat, Nat.add(k, 0n), k, Ar.ar_add_zero_r(k)) : {Spec.depth(ps, k) == Spec.depth(ps, _) : Nat}
      {==}
    case 1n+s 0n:
      Empty.absurd({Impl.at(Spec.depths(0n, ps, k), 1n+s) == Spec.depth(ps, Nat.add(k, 1n+s)) : Nat}, lt_zero(1n+s, lt))
    case 1n+s 1n+m:
      %Equal.sym(Nat, Nat.add(k, 1n+s), 1n+Nat.add(k, s), Ar.ar_add_succ_r(k, s)) : {Impl.at(Spec.depths(m, ps, 1n+k), s) == Spec.depth(ps, _) : Nat}
      br_depths(s, m, ps, 1n+k, lt)

def br_depth(+ps: List<&2, Nat>, +r: Nat, lt: {Nat.is_lt(r, 8n) == True{} : Bool})
  -> {Impl.depth(Spec.desc(ps), r) == Spec.depth(ps, r) : Nat}:
  br_depths(r, 8n, ps, 0n, lt)

def br_rows(+i: Nat, +n: Nat, +ps: List<&2, Nat>, +k: Nat, lt: {Nat.is_lt(i, n) == True{} : Bool})
  -> {Impl.row(Spec.paths(n, ps, k), i) == Spec.path_row(8n, ps, Nat.add(k, i), Spec.depth(ps, Nat.add(k, i)), 0n) : List<&2, Nat>}:
  match i n:
    case 0n 0n:
      Empty.absurd({Impl.row(Spec.paths(0n, ps, k), 0n) == Spec.path_row(8n, ps, Nat.add(k, 0n), Spec.depth(ps, Nat.add(k, 0n)), 0n) : List<&2, Nat>}, Ar.ar_false_true(lt))
    case 0n 1n+m:
      %Equal.sym(Nat, Nat.add(k, 0n), k, Ar.ar_add_zero_r(k)) : {Spec.path_row(8n, ps, k, Spec.depth(ps, k), 0n) == Spec.path_row(8n, ps, _, Spec.depth(ps, _), 0n) : List<&2, Nat>}
      {==}
    case 1n+s 0n:
      Empty.absurd({Impl.row(Spec.paths(0n, ps, k), 1n+s) == Spec.path_row(8n, ps, Nat.add(k, 1n+s), Spec.depth(ps, Nat.add(k, 1n+s)), 0n) : List<&2, Nat>}, lt_zero(1n+s, lt))
    case 1n+s 1n+m:
      %Equal.sym(Nat, Nat.add(k, 1n+s), 1n+Nat.add(k, s), Ar.ar_add_succ_r(k, s)) : {Impl.row(Spec.paths(m, ps, 1n+k), s) == Spec.path_row(8n, ps, _, Spec.depth(ps, _), 0n) : List<&2, Nat>}
      br_rows(s, m, ps, 1n+k, lt)

def br_prow(+i: Nat, +n: Nat, +ps: List<&2, Nat>, +r: Nat, +dr: Nat, +k: Nat, lt: {Nat.is_lt(i, n) == True{} : Bool})
  -> {Impl.at(Spec.path_row(n, ps, r, dr, k), i) == Spec.pick(Nat.is_le(Nat.add(k, i), dr), Spec.anc_d(ps, r, dr, Nat.add(k, i)), 255n) : Nat}:
  match i n:
    case 0n 0n:
      Empty.absurd({Impl.at(Spec.path_row(0n, ps, r, dr, k), 0n) == Spec.pick(Nat.is_le(Nat.add(k, 0n), dr), Spec.anc_d(ps, r, dr, Nat.add(k, 0n)), 255n) : Nat}, Ar.ar_false_true(lt))
    case 0n 1n+m:
      %Equal.sym(Nat, Nat.add(k, 0n), k, Ar.ar_add_zero_r(k)) : {Spec.pick(Nat.is_le(k, dr), Spec.anc_d(ps, r, dr, k), 255n) == Spec.pick(Nat.is_le(_, dr), Spec.anc_d(ps, r, dr, _), 255n) : Nat}
      {==}
    case 1n+s 0n:
      Empty.absurd({Impl.at(Spec.path_row(0n, ps, r, dr, k), 1n+s) == Spec.pick(Nat.is_le(Nat.add(k, 1n+s), dr), Spec.anc_d(ps, r, dr, Nat.add(k, 1n+s)), 255n) : Nat}, lt_zero(1n+s, lt))
    case 1n+s 1n+m:
      %Equal.sym(Nat, Nat.add(k, 1n+s), 1n+Nat.add(k, s), Ar.ar_add_succ_r(k, s)) : {Impl.at(Spec.path_row(m, ps, r, dr, 1n+k), s) == Spec.pick(Nat.is_le(_, dr), Spec.anc_d(ps, r, dr, _), 255n) : Nat}
      br_prow(s, m, ps, r, dr, 1n+k, lt)

def spick_true(b: Bool, +x: Nat, +y: Nat, h: {b == True{} : Bool}) -> {Spec.pick(b, x, y) == x : Nat}:
  match b:
    case False{}:
      Empty.absurd({Spec.pick(False{}, x, y) == x : Nat}, Ar.ar_false_true(h))
    case True{}:
      {==}

def ipick_true(b: Bool, +x: Nat, +y: Nat, h: {b == True{} : Bool}) -> {Impl.pick(b, x, y) == x : Nat}:
  match b:
    case False{}:
      Empty.absurd({Impl.pick(False{}, x, y) == x : Nat}, Ar.ar_false_true(h))
    case True{}:
      {==}

# path[r][j] of the reference's TreeDesc is r's depth-j ancestor (j <= depth(r))
def br_path(+ps: List<&2, Nat>, +r: Nat, +j: Nat, hr: {Nat.is_lt(r, 8n) == True{} : Bool},
    hj: {Nat.is_lt(j, 8n) == True{} : Bool}, hle: {Nat.is_le(j, Spec.depth(ps, r)) == True{} : Bool})
  -> {Impl.path(Spec.desc(ps), r, j) == Spec.anc(ps, r, j) : Nat}:
  Equal.trans(Nat, Impl.at(Impl.row(Spec.paths(8n, ps, 0n), r), j),
    Impl.at(Spec.path_row(8n, ps, r, Spec.depth(ps, r), 0n), j), Spec.anc(ps, r, j),
    Equal.cong(List<&2, Nat>, Nat, t => Impl.at(t, j), Impl.row(Spec.paths(8n, ps, 0n), r),
      Spec.path_row(8n, ps, r, Spec.depth(ps, r), 0n), br_rows(r, 8n, ps, 0n, hr)),
    Equal.trans(Nat, Impl.at(Spec.path_row(8n, ps, r, Spec.depth(ps, r), 0n), j),
      Spec.pick(Nat.is_le(j, Spec.depth(ps, r)), Spec.anc_d(ps, r, Spec.depth(ps, r), j), 255n), Spec.anc(ps, r, j),
      br_prow(j, 8n, ps, r, Spec.depth(ps, r), 0n, hj),
      spick_true(Nat.is_le(j, Spec.depth(ps, r)), Spec.anc_d(ps, r, Spec.depth(ps, r), j), 255n, hle)))

# ---------------------------------------------------------------------------------------------
# The pass loop's hits: a position the row's causal bound admits enters once per admitting pass,
# one past it never

def add_cong(-a: Nat, -b: Nat, -c: Nat, -e: Nat, h1: {a == b : Nat}, h2: {c == e : Nat})
  -> {Nat.add(a, c) == Nat.add(b, e) : Nat}:
  %h1 : {Nat.add(a, c) == Nat.add(_, e) : Nat}
  %h2 : {Nat.add(a, c) == Nat.add(a, _) : Nat}
  {==}

def bit_and_t(+x: Bool, -y: Bool, h: {y == True{} : Bool}) -> {Impl.bit(Bool.and(x, y)) == Impl.bit(x) : Nat}:
  match x:
    case False{}:
      {==}
    case True{}:
      %Equal.sym(Bool, y, True{}, h) : {Impl.bit(_) == 1n : Nat}
      {==}

def bit_and_f(+x: Bool, -y: Bool, h: {y == False{} : Bool}) -> {Impl.bit(Bool.and(x, y)) == 0n : Nat}:
  match x:
    case False{}:
      {==}
    case True{}:
      %Equal.sym(Bool, y, False{}, h) : {Impl.bit(_) == 0n : Nat}
      {==}

def hits_le(+k: Nat, +d: Impl.TreeDesc, +r: Nat, +tok: Nat, +qa: Nat, +h: {Nat.is_le(tok, qa) == True{} : Bool})
  -> {Impl.pass_hits(k, d, r, tok, qa) == Impl.pass_count(k, d, r) : Nat}:
  match k:
    case 0n:
      {==}
    case 1n+p:
      add_cong(Impl.pass_hits(p, d, r, tok, qa), Impl.pass_count(p, d, r),
        Impl.bit(Impl.sees(True{}, Impl.leaf_of(d, r), Impl.leaf(d, p), tok, qa)),
        Impl.bit(Impl.in_pass(True{}, Impl.leaf_of(d, r), Impl.leaf(d, p))),
        hits_le(p, d, r, tok, qa, h),
        bit_and_t(Impl.in_pass(True{}, Impl.leaf_of(d, r), Impl.leaf(d, p)), Nat.is_le(tok, qa), h))

def hits_gt(+k: Nat, +d: Impl.TreeDesc, +r: Nat, +tok: Nat, +qa: Nat, +h: {Nat.is_le(tok, qa) == False{} : Bool})
  -> {Impl.pass_hits(k, d, r, tok, qa) == 0n : Nat}:
  match k:
    case 0n:
      {==}
    case 1n+p:
      add_cong(Impl.pass_hits(p, d, r, tok, qa), 0n,
        Impl.bit(Impl.sees(True{}, Impl.leaf_of(d, r), Impl.leaf(d, p), tok, qa)), 0n,
        hits_gt(p, d, r, tok, qa, h),
        bit_and_f(Impl.in_pass(True{}, Impl.leaf_of(d, r), Impl.leaf(d, p)), Nat.is_le(tok, qa), h))

# ---------------------------------------------------------------------------------------------
# Laws

def Laws.entry_ok(q1, q2, q3, q4, q5, q6, q7, legal, r, hr):
  +legal = legal
  +r = r
  +hr = hr
  entry_true(q1, q2, q3, q4, q5, q6, q7, legal, r, hr)

def Laws.pass_once(q1, q2, q3, q4, q5, q6, q7, legal, r, hr):
  +legal = legal
  +r = r
  +hr = hr
  pass_once_eq(q1, q2, q3, q4, q5, q6, q7, legal, r, hr)

def Laws.pass_ok(q1, q2, q3, q4, q5, q6, q7, legal, p, j, hp, hj):
  +legal = legal
  +p = p
  +j = j
  +hp = hp
  +hj = hj
  pass_js_go(8n, j, Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), p, hj, pass_ps_go(Impl.n_leaf(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7))), p, Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), hp, at_and_r(rows(8n, Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7))), pass_ps(Impl.n_leaf(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7))), Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7))), shape_true(q1, q2, q3, q4, q5, q6, q7, legal))))

# The serving pass's path is r's path, for j <= depth[r] (kernel form of the bound)
def prefix_path(+q1: Nat, +q2: Nat, +q3: Nat, +q4: Nat, +q5: Nat, +q6: Nat, +q7: Nat, +legal: {Spec.legal(q1, q2, q3, q4, q5, q6, q7) == True{} : Bool}, +r: Nat, +hr: {Nat.is_lt(r, 8n) == True{} : Bool}, +j: Nat, +hle: {Nat.is_le(j, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r)) == True{} : Bool})
  -> {Impl.path(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), Impl.leaf_of(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), j) == Spec.anc(Spec.shape(q1, q2, q3, q4, q5, q6, q7), r, j) : Nat}:
  +b1 = br_depth(Spec.shape(q1, q2, q3, q4, q5, q6, q7), r, hr)
  +hj8 = Ar.ar_le_lt_trans(j, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), 8n, hle, depth_lt(q1, q2, q3, q4, q5, q6, q7, legal, r, hr))
  +col = cols_go(8n, j, Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r, hj8, cols_true(q1, q2, q3, q4, q5, q6, q7, legal, r, hr))
  +e1 = at_eq(Impl.path(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), Impl.leaf_of(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), j), Impl.path(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r, j),
    at_imp(Nat.is_le(j, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r)), Nat.is_eq(Impl.path(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), Impl.leaf_of(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), j), Impl.path(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r, j)), col, hle))
  +hs = Equal.trans(Bool, Nat.is_le(j, Spec.depth(Spec.shape(q1, q2, q3, q4, q5, q6, q7), r)), Nat.is_le(j, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r)), True{},
    Equal.cong(Nat, Bool, t => Nat.is_le(j, t), Spec.depth(Spec.shape(q1, q2, q3, q4, q5, q6, q7), r), Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), Equal.sym(Nat, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), Spec.depth(Spec.shape(q1, q2, q3, q4, q5, q6, q7), r), b1)), hle)
  Equal.trans(Nat, Impl.path(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), Impl.leaf_of(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), j), Impl.path(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r, j), Spec.anc(Spec.shape(q1, q2, q3, q4, q5, q6, q7), r, j), e1, br_path(Spec.shape(q1, q2, q3, q4, q5, q6, q7), r, j, hr, hj8, hs))

def Laws.path_prefix(q1, q2, q3, q4, q5, q6, q7, legal, r, j, hr, hj):
  +legal = legal
  +r = r
  +j = j
  +hr = hr
  +hj = hj
  +b1 = br_depth(Spec.shape(q1, q2, q3, q4, q5, q6, q7), r, hr)
  +h = hj
  +hk = Equal.trans(Bool, Nat.is_le(j, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r)), Nat.is_le(j, Spec.depth(Spec.shape(q1, q2, q3, q4, q5, q6, q7), r)), True{},
    Equal.cong(Nat, Bool, t => Nat.is_le(j, t), Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), Spec.depth(Spec.shape(q1, q2, q3, q4, q5, q6, q7), r), b1), h)
  +c1 = dl_true(q1, q2, q3, q4, q5, q6, q7, legal, r, hr)
  ((%b1 : {Nat.is_le(_, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), Impl.leaf_of(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r))) == True{} : Bool}
    c1), prefix_path(q1, q2, q3, q4, q5, q6, q7, legal, r, hr, j, hk))

def sees_once_case(+q1: Nat, +q2: Nat, +q3: Nat, +q4: Nat, +q5: Nat, +q6: Nat, +q7: Nat, +legal: {Spec.legal(q1, q2, q3, q4, q5, q6, q7) == True{} : Bool}, +r: Nat, +hr: {Nat.is_lt(r, 8n) == True{} : Bool}, +base: Nat, +j: Nat, b: Bool, +e: {Nat.is_le(j, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r)) == b : Bool})
  -> {Impl.pass_hits(Impl.n_leaf(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7))), Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r, Nat.add(base, j), Impl.q_abs(base, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r))) ==
    Spec.once(Spec.chain_sees(j, Spec.depth(Spec.shape(q1, q2, q3, q4, q5, q6, q7), r))) : Nat}:
  match b:
    case True{}:
      +b1 = br_depth(Spec.shape(q1, q2, q3, q4, q5, q6, q7), r, hr)
      %b1 : {Impl.pass_hits(Impl.n_leaf(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7))), Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r, Nat.add(base, j), Nat.add(base, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r))) == Spec.once(Nat.is_le(j, _)) : Nat}
      %Equal.sym(Bool, Nat.is_le(j, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r)), True{}, e) : {Impl.pass_hits(Impl.n_leaf(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7))), Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r, Nat.add(base, j), Nat.add(base, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r))) == Spec.once(_) : Nat}
      Equal.trans(Nat, Impl.pass_hits(Impl.n_leaf(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7))), Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r, Nat.add(base, j), Nat.add(base, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r))), Impl.pass_count(Impl.n_leaf(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7))), Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), 1n,
        hits_le(Impl.n_leaf(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7))), Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r, Nat.add(base, j), Nat.add(base, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r)),
          Equal.trans(Bool, Nat.is_le(Nat.add(base, j), Nat.add(base, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r))), Nat.is_le(j, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r)), True{}, Ar.ar_le_add_cancel(base, j, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r)), e)),
        pass_once_eq(q1, q2, q3, q4, q5, q6, q7, legal, r, hr))
    case False{}:
      +b1 = br_depth(Spec.shape(q1, q2, q3, q4, q5, q6, q7), r, hr)
      %b1 : {Impl.pass_hits(Impl.n_leaf(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7))), Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r, Nat.add(base, j), Nat.add(base, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r))) == Spec.once(Nat.is_le(j, _)) : Nat}
      %Equal.sym(Bool, Nat.is_le(j, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r)), False{}, e) : {Impl.pass_hits(Impl.n_leaf(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7))), Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r, Nat.add(base, j), Nat.add(base, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r))) == Spec.once(_) : Nat}
      hits_gt(Impl.n_leaf(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7))), Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r, Nat.add(base, j), Nat.add(base, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r)),
        Equal.trans(Bool, Nat.is_le(Nat.add(base, j), Nat.add(base, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r))), Nat.is_le(j, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r)), False{}, Ar.ar_le_add_cancel(base, j, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r)), e))

def Laws.sees_once(q1, q2, q3, q4, q5, q6, q7, legal, base, r, j, hr, hj):
  +legal = legal
  +base = base
  +r = r
  +j = j
  +hr = hr
  +hj = hj
  sees_once_case(q1, q2, q3, q4, q5, q6, q7, legal, r, hr, base, j, Nat.is_le(j, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r)), {==})

def Laws.sees_path(q1, q2, q3, q4, q5, q6, q7, legal, base, r, p, j, hr, hp, hj, admitted):
  +legal = legal
  +base = base
  +r = r
  +p = p
  +j = j
  +hr = hr
  +hp = hp
  +hj = hj
  +adm = admitted
  +eqlf = at_eq(Impl.leaf_of(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), Impl.leaf(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), p), at_and_l(Nat.is_eq(Impl.leaf_of(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), Impl.leaf(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), p)), Nat.is_le(Nat.add(base, j), Nat.add(base, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r))), adm))
  +lej = Equal.trans(Bool, Nat.is_le(j, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r)), Nat.is_le(Nat.add(base, j), Nat.add(base, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r))), True{},
    Equal.sym(Bool, Nat.is_le(Nat.add(base, j), Nat.add(base, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r))), Nat.is_le(j, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r)), Ar.ar_le_add_cancel(base, j, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r))),
    at_and_r(Nat.is_eq(Impl.leaf_of(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), Impl.leaf(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), p)), Nat.is_le(Nat.add(base, j), Nat.add(base, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r))), adm))
  +le2 = Ar.ar_le_trans(j, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), Impl.leaf_of(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r)), lej, dl_true(q1, q2, q3, q4, q5, q6, q7, legal, r, hr))
  %eqlf : {Impl.pick(Nat.is_le(j, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), _)), Impl.path(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), _, j), j) == Spec.anc(Spec.shape(q1, q2, q3, q4, q5, q6, q7), r, j) : Nat}
  Equal.trans(Nat, Impl.pick(Nat.is_le(j, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), Impl.leaf_of(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r))), Impl.path(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), Impl.leaf_of(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), j), j), Impl.path(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), Impl.leaf_of(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), j), Spec.anc(Spec.shape(q1, q2, q3, q4, q5, q6, q7), r, j),
    ipick_true(Nat.is_le(j, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), Impl.leaf_of(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r))), Impl.path(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), Impl.leaf_of(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), j), j, le2),
    prefix_path(q1, q2, q3, q4, q5, q6, q7, legal, r, hr, j, lej))

def Laws.context_once(q1, q2, q3, q4, q5, q6, q7, legal, base, tok, r, hr, ctx):
  +legal = legal
  +base = base
  +tok = tok
  +r = r
  +hr = hr
  Equal.trans(Nat, Impl.pass_hits(Impl.n_leaf(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7))), Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r, tok, Nat.add(base, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r))), Impl.pass_count(Impl.n_leaf(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7))), Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r), 1n,
    hits_le(Impl.n_leaf(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7))), Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r, tok, Nat.add(base, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r)), Ar.ar_le_trans(tok, base, Nat.add(base, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r)), ctx, Ar.ar_le_add_r(base, Impl.depth(Spec.desc(Spec.shape(q1, q2, q3, q4, q5, q6, q7)), r)))),
    pass_once_eq(q1, q2, q3, q4, q5, q6, q7, legal, r, hr))

# CHAIN rows / columns, by cases on r, j < 8
def chain_depth(+r: Nat, hr: {Nat.is_lt(r, 8n) == True{} : Bool}) -> {Impl.depth(Spec.desc(Spec.chain()), r) == r : Nat}:
  match r:
    case 0n:
      {==}
    case 1n+r1:
      match r1:
        case 0n:
          {==}
        case 1n+r2:
          match r2:
            case 0n:
              {==}
            case 1n+r3:
              match r3:
                case 0n:
                  {==}
                case 1n+r4:
                  match r4:
                    case 0n:
                      {==}
                    case 1n+r5:
                      match r5:
                        case 0n:
                          {==}
                        case 1n+r6:
                          match r6:
                            case 0n:
                              {==}
                            case 1n+r7:
                              match r7:
                                case 0n:
                                  {==}
                                case 1n+r8:
                                  Empty.absurd({Impl.depth(Spec.desc(Spec.chain()), 8n+r8) == 8n+r8 : Nat}, lt_zero(r8, hr))

def chain_in(+r: Nat, hr: {Nat.is_lt(r, 8n) == True{} : Bool})
  -> {Impl.in_pass(True{}, Impl.leaf_of(Spec.desc(Spec.chain()), r), Impl.leaf(Spec.desc(Spec.chain()), 0n)) == True{} : Bool}:
  match r:
    case 0n:
      {==}
    case 1n+r1:
      match r1:
        case 0n:
          {==}
        case 1n+r2:
          match r2:
            case 0n:
              {==}
            case 1n+r3:
              match r3:
                case 0n:
                  {==}
                case 1n+r4:
                  match r4:
                    case 0n:
                      {==}
                    case 1n+r5:
                      match r5:
                        case 0n:
                          {==}
                        case 1n+r6:
                          match r6:
                            case 0n:
                              {==}
                            case 1n+r7:
                              match r7:
                                case 0n:
                                  {==}
                                case 1n+r8:
                                  Empty.absurd({Impl.in_pass(True{}, Impl.leaf_of(Spec.desc(Spec.chain()), 8n+r8), Impl.leaf(Spec.desc(Spec.chain()), 0n)) == True{} : Bool}, lt_zero(r8, hr))

def chain_col(+j: Nat, hj: {Nat.is_lt(j, 8n) == True{} : Bool}) -> {Impl.slot(Spec.desc(Spec.chain()), 0n, j) == j : Nat}:
  match j:
    case 0n:
      {==}
    case 1n+j1:
      match j1:
        case 0n:
          {==}
        case 1n+j2:
          match j2:
            case 0n:
              {==}
            case 1n+j3:
              match j3:
                case 0n:
                  {==}
                case 1n+j4:
                  match j4:
                    case 0n:
                      {==}
                    case 1n+j5:
                      match j5:
                        case 0n:
                          {==}
                        case 1n+j6:
                          match j6:
                            case 0n:
                              {==}
                            case 1n+j7:
                              match j7:
                                case 0n:
                                  {==}
                                case 1n+j8:
                                  Empty.absurd({Impl.slot(Spec.desc(Spec.chain()), 0n, 8n+j8) == 8n+j8 : Nat}, lt_zero(j8, hj))

def Laws.chain(base, r, j, hr, hj):
  +base = base
  +r = r
  +j = j
  +hr = hr
  +hj = hj
  +h = hr
  ({==}, (chain_in(r, h), (chain_col(j, hj),
    (%Equal.sym(Nat, Nat.sub(Nat.add(base, 8n), 8n), base, SP.sp_qabs(base, 8n)) : {Nat.add(base, Impl.depth(Spec.desc(Spec.chain()), r)) == Nat.add(_, r) : Nat}
     %Equal.sym(Nat, Impl.depth(Spec.desc(Spec.chain()), r), r, chain_depth(r, h)) : {Nat.add(base, _) == Nat.add(base, r) : Nat}
     {==}))))

# ---------------------------------------------------------------------------------------------
# Tail tiles (generic in base)

def pos64() -> {Nat.is_lt(0n, 64n) == True{} : Bool}:
  {==}

# lo = 0, x = 1 + x1 <= hi = 1 + y1 <= 1: x = hi = 1
def between1(+x1: Nat, +y1: Nat, h2: {Nat.is_le(1n+x1, 1n+y1) == True{} : Bool}, h3: {Nat.is_le(1n+y1, 1n) == True{} : Bool})
  -> {Bool.or(Nat.is_eq(1n+x1, 0n), Nat.is_eq(1n+x1, 1n+y1)) == True{} : Bool}:
  match x1 y1:
    case 0n 0n:
      {==}
    case 0n 1n+y2:
      Empty.absurd({Bool.or(Nat.is_eq(1n, 0n), Nat.is_eq(1n, 2n+y2)) == True{} : Bool}, Ar.ar_false_true(h3))
    case 1n+x2 0n:
      Empty.absurd({Bool.or(Nat.is_eq(2n+x2, 0n), Nat.is_eq(2n+x2, 1n)) == True{} : Bool}, Ar.ar_false_true(h2))
    case 1n+x2 1n+y2:
      Empty.absurd({Bool.or(Nat.is_eq(2n+x2, 0n), Nat.is_eq(2n+x2, 2n+y2)) == True{} : Bool}, Ar.ar_false_true(h3))

# lo <= x <= hi <= lo + 1: x is lo or hi
def between(+lo: Nat, +x: Nat, +hi: Nat, h1: {Nat.is_le(lo, x) == True{} : Bool},
    h2: {Nat.is_le(x, hi) == True{} : Bool}, h3: {Nat.is_le(hi, 1n+lo) == True{} : Bool})
  -> {Bool.or(Nat.is_eq(x, lo), Nat.is_eq(x, hi)) == True{} : Bool}:
  match lo x hi:
    case 0n 0n _:
      {==}
    case 0n 1n+x1 0n:
      Empty.absurd({Bool.or(Nat.is_eq(1n+x1, 0n), Nat.is_eq(1n+x1, 0n)) == True{} : Bool}, Ar.ar_false_true(h2))
    case 0n 1n+x1 1n+y1:
      between1(x1, y1, h2, h3)
    case 1n+l 0n _:
      Empty.absurd({Bool.or(Nat.is_eq(0n, 1n+l), Nat.is_eq(0n, hi)) == True{} : Bool}, Ar.ar_false_true(h1))
    case 1n+l 1n+x1 0n:
      Empty.absurd({Bool.or(Nat.is_eq(1n+x1, 1n+l), Nat.is_eq(1n+x1, 0n)) == True{} : Bool}, Ar.ar_false_true(h2))
    case 1n+l 1n+x1 1n+y1:
      between(l, x1, y1, h1, h2, h3)

# T1 <= T0 + 1
def hi_le(+base: Nat) -> {Nat.is_le(Impl.tail_hi(base), 1n+Impl.tail_lo(base)) == True{} : Bool}:
  +le = Ar.ar_div_mono(Nat.add(base, 7n), Nat.add(base, 64n), 64n, pos64(),
    Equal.trans(Bool, Nat.is_le(Nat.add(base, 7n), Nat.add(base, 64n)), Nat.is_le(7n, 64n), True{}, Ar.ar_le_add_cancel(base, 7n, 64n), {==}))
  %Ar.ar_div_add_self(base, 64n, pos64()) : {Nat.is_le(Nat.div(Nat.add(base, 7n), 64n), _) == True{} : Bool}
  le

def Laws.tail_cover(base, j, hj):
  +base = base
  +j = j
  +hj = hj
  +j7 = Equal.trans(Bool, Nat.is_le(j, 7n), Nat.is_lt(j, 8n), True{}, Ar.p1_le_lt_succ(j, 7n), hj)
  between(Nat.div(base, 64n), Nat.div(Nat.add(base, j), 64n), Nat.div(Nat.add(base, 7n), 64n),
    Ar.ar_div_mono(base, Nat.add(base, j), 64n, pos64(), Ar.ar_le_add_r(base, j)),
    Ar.ar_div_mono(Nat.add(base, j), Nat.add(base, 7n), 64n, pos64(),
      Equal.trans(Bool, Nat.is_le(Nat.add(base, j), Nat.add(base, 7n)), Nat.is_le(j, 7n), True{}, Ar.ar_le_add_cancel(base, j, 7n), j7)),
    hi_le(base))

def Laws.tail_last(base):
  +base = base
  +e1 = Equal.trans(Nat, Nat.sub(Nat.add(Nat.add(base, 8n), 64n), 1n), Nat.sub(Nat.add(base, 72n), 1n), Nat.add(base, 71n),
    Equal.cong(Nat, Nat, t => Nat.sub(t, 1n), Nat.add(Nat.add(base, 8n), 64n), Nat.add(base, 72n), Ar.ar_add_assoc(base, 8n, 64n)),
    SP.sp_pred(base, 71n))
  (%Equal.sym(Nat, Nat.sub(Nat.add(Nat.add(base, 8n), 64n), 1n), Nat.add(base, 71n), e1) : {1n+Nat.div(Nat.add(base, 7n), 64n) == Nat.div(_, 64n) : Nat}
   %Ar.ar_add_assoc(base, 7n, 64n) : {1n+Nat.div(Nat.add(base, 7n), 64n) == Nat.div(_, 64n) : Nat}
   Equal.sym(Nat, Nat.div(Nat.add(Nat.add(base, 7n), 64n), 64n), 1n+Nat.div(Nat.add(base, 7n), 64n), Ar.ar_div_add_self(Nat.add(base, 7n), 64n, pos64())),
   (hi_le(base), Ar.ar_div_mono(base, Nat.add(base, 7n), 64n, pos64(), Ar.ar_le_add_r(base, 7n))))

def Laws.context_tiles(base, tile, below):
  +base = base
  +tile = tile
  +h = Equal.trans(Bool, Nat.is_le(1n+tile, Nat.div(base, 64n)), Nat.is_lt(tile, Nat.div(base, 64n)), True{},
    Equal.sym(Bool, Nat.is_lt(tile, Nat.div(base, 64n)), Nat.is_le(1n+tile, Nat.div(base, 64n)), Ar.ar_lt_succ_le(tile, Nat.div(base, 64n))), below)
  %Equal.sym(Nat, Nat.add(Nat.mul(tile, 64n), 64n), Nat.add(64n, Nat.mul(tile, 64n)), Ar.ar_add_comm(Nat.mul(tile, 64n), 64n)) : {Nat.is_le(_, base) == True{} : Bool}
  Ar.ar_le_trans(Nat.mul(1n+tile, 64n), Nat.mul(Nat.div(base, 64n), 64n), base,
    Ar.ar_le_mul_mono(1n+tile, Nat.div(base, 64n), 64n, h), Ar.ard_le_mul_div(base, 64n, pos64()))

