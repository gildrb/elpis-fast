# Proofs of the m16 GEMM (patch 2001 v2) work-partition laws (gemm_m16_partition_laws.bend)
# about the Bend model gemm_m16_partition.bend. Structural evidence only: no unsafe
# definitions, axioms, holes, foreign proof code or answer tables. Every def precedes its uses.
import Base
import ./gemm_m16_partition.bend as Impl
import ./gemm_m16_partition_spec.bend as Spec
import ./gemm_m16_partition_laws.bend as Laws

# ==== Nat / Bool arithmetic lemmas ====

def ar_Truth(value: Bool) -> Type:
  match value:
    case False{}:
      Empty
    case True{}:
      Unit

# False == True is absurd.
def ar_false_true(equal: {False{} == True{} : Bool}) -> Empty:
  %Equal.sym(Bool, False{}, True{}, equal) : ar_Truth(_)
  Unit{}

law ar_add_zero_r:
  for +a: Nat
  {Nat.add(a, 0n) == a : Nat}

def ar_add_zero_r(a):
  match a:
    case 0n:
      {==}
    case 1n+p:
      %ar_add_zero_r(p) : {1n+Nat.add(p, 0n) == 1n+_ : Nat}
      {==}

law ar_add_succ_r:
  for +a: Nat
  for +b: Nat
  {Nat.add(a, 1n+b) == 1n+Nat.add(a, b) : Nat}

def ar_add_succ_r(a, b):
  match a:
    case 0n:
      {==}
    case 1n+p:
      %ar_add_succ_r(p, b) : {1n+Nat.add(p, 1n+b) == 1n+_ : Nat}
      {==}

law ar_add_comm:
  for +a: Nat
  for +b: Nat
  {Nat.add(a, b) == Nat.add(b, a) : Nat}

def ar_add_comm(a, b):
  match a:
    case 0n:
      Equal.sym(Nat, Nat.add(b, 0n), b, ar_add_zero_r(b))
    case 1n+p:
      %Equal.sym(Nat, Nat.add(b, 1n+p), 1n+Nat.add(b, p), ar_add_succ_r(b, p)) : {1n+Nat.add(p, b) == _ : Nat}
      %ar_add_comm(p, b) : {1n+Nat.add(p, b) == 1n+_ : Nat}
      {==}

law ar_add_assoc:
  for +a: Nat
  for +b: Nat
  for +c: Nat
  {Nat.add(Nat.add(a, b), c) == Nat.add(a, Nat.add(b, c)) : Nat}

def ar_add_assoc(a, b, c):
  match a:
    case 0n:
      {==}
    case 1n+p:
      %ar_add_assoc(p, b, c) : {1n+Nat.add(Nat.add(p, b), c) == 1n+_ : Nat}
      {==}

law ar_mul_zero_r:
  for +a: Nat
  {Nat.mul(a, 0n) == 0n : Nat}

def ar_mul_zero_r(a):
  match a:
    case 0n:
      {==}
    case 1n+p:
      ar_mul_zero_r(p)

# a + (b + c) == b + (a + c).
def ar_h_add_swap(+a: Nat, +b: Nat, +c: Nat)
  -> {Nat.add(a, Nat.add(b, c)) == Nat.add(b, Nat.add(a, c)) : Nat}:
  %ar_add_assoc(b, a, c) : {Nat.add(a, Nat.add(b, c)) == _ : Nat}
  %ar_add_comm(a, b) : {Nat.add(a, Nat.add(b, c)) == Nat.add(_, c) : Nat}
  Equal.sym(Nat, Nat.add(Nat.add(a, b), c), Nat.add(a, Nat.add(b, c)), ar_add_assoc(a, b, c))

law ar_mul_succ_r:
  for +a: Nat
  for +b: Nat
  {Nat.mul(a, 1n+b) == Nat.add(a, Nat.mul(a, b)) : Nat}

def ar_mul_succ_r(a, b):
  match a:
    case 0n:
      {==}
    case 1n+p:
      %Equal.sym(Nat, Nat.mul(p, 1n+b), Nat.add(p, Nat.mul(p, b)), ar_mul_succ_r(p, b)) : {1n+Nat.add(b, _) == 1n+Nat.add(p, Nat.add(b, Nat.mul(p, b))) : Nat}
      %ar_h_add_swap(b, p, Nat.mul(p, b)) : {1n+Nat.add(b, Nat.add(p, Nat.mul(p, b))) == 1n+_ : Nat}
      {==}

law ar_mul_comm:
  for +a: Nat
  for +b: Nat
  {Nat.mul(a, b) == Nat.mul(b, a) : Nat}

def ar_mul_comm(a, b):
  match a:
    case 0n:
      Equal.sym(Nat, Nat.mul(b, 0n), 0n, ar_mul_zero_r(b))
    case 1n+p:
      %Equal.sym(Nat, Nat.mul(b, 1n+p), Nat.add(b, Nat.mul(b, p)), ar_mul_succ_r(b, p)) : {Nat.add(b, Nat.mul(p, b)) == _ : Nat}
      %ar_mul_comm(p, b) : {Nat.add(b, Nat.mul(p, b)) == Nat.add(b, _) : Nat}
      {==}

law ar_mul_add_l:
  for +a: Nat
  for +b: Nat
  for +c: Nat
  {Nat.mul(Nat.add(a, b), c) == Nat.add(Nat.mul(a, c), Nat.mul(b, c)) : Nat}

def ar_mul_add_l(a, b, c):
  match a:
    case 0n:
      {==}
    case 1n+p:
      %Equal.sym(Nat, Nat.mul(Nat.add(p, b), c), Nat.add(Nat.mul(p, c), Nat.mul(b, c)), ar_mul_add_l(p, b, c)) : {Nat.add(c, _) == Nat.add(Nat.add(c, Nat.mul(p, c)), Nat.mul(b, c)) : Nat}
      Equal.sym(Nat, Nat.add(Nat.add(c, Nat.mul(p, c)), Nat.mul(b, c)), Nat.add(c, Nat.add(Nat.mul(p, c), Nat.mul(b, c))), ar_add_assoc(c, Nat.mul(p, c), Nat.mul(b, c)))

law ar_mul_assoc:
  for +a: Nat
  for +b: Nat
  for +c: Nat
  {Nat.mul(Nat.mul(a, b), c) == Nat.mul(a, Nat.mul(b, c)) : Nat}

def ar_mul_assoc(a, b, c):
  match a:
    case 0n:
      {==}
    case 1n+p:
      %ar_mul_assoc(p, b, c) : {Nat.mul(Nat.add(b, Nat.mul(p, b)), c) == Nat.add(Nat.mul(b, c), _) : Nat}
      ar_mul_add_l(b, Nat.mul(p, b), c)

law ar_le_refl:
  for +a: Nat
  {Nat.is_le(a, a) == True{} : Bool}

def ar_le_refl(a):
  match a:
    case 0n:
      {==}
    case 1n+p:
      ar_le_refl(p)

law ar_lt_not_le:
  for +a: Nat
  for +b: Nat
  {Nat.is_lt(a, b) == Bool.not(Nat.is_le(b, a)) : Bool}

def ar_lt_not_le(a, b):
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      {==}
    case 1n+p 0n:
      {==}
    case 1n+p 1n+q:
      ar_lt_not_le(p, q)

law ar_gt_flip:
  for +a: Nat
  for +b: Nat
  {Nat.is_gt(a, b) == Nat.is_lt(b, a) : Bool}

def ar_gt_flip(a, b):
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      {==}
    case 1n+p 0n:
      {==}
    case 1n+p 1n+q:
      ar_gt_flip(p, q)

law ar_ge_flip:
  for +a: Nat
  for +b: Nat
  {Nat.is_ge(a, b) == Nat.is_le(b, a) : Bool}

def ar_ge_flip(a, b):
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      {==}
    case 1n+p 0n:
      {==}
    case 1n+p 1n+q:
      ar_ge_flip(p, q)

def ar_h_le_zero(n: Nat) -> {Nat.is_le(0n, n) == True{} : Bool}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      {==}

law ar_lt_succ_le:
  for +a: Nat
  for +b: Nat
  {Nat.is_lt(a, b) == Nat.is_le(1n+a, b) : Bool}

def ar_lt_succ_le(a, b):
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      Equal.sym(Bool, Nat.is_le(0n, q), True{}, ar_h_le_zero(q))
    case 1n+p 0n:
      {==}
    case 1n+p 1n+q:
      ar_lt_succ_le(p, q)

law ar_le_trans:
  for +a: Nat
  for +b: Nat
  for +c: Nat
  for ab: {Nat.is_le(a, b) == True{} : Bool}
  for bc: {Nat.is_le(b, c) == True{} : Bool}
  {Nat.is_le(a, c) == True{} : Bool}

def ar_le_trans(a, b, c, ab, bc):
  match a b c:
    case 0n _ _:
      ar_h_le_zero(c)
    case 1n+p 0n _:
      Empty.absurd({Nat.is_le(1n+p, c) == True{} : Bool}, ar_false_true(ab))
    case 1n+p 1n+q 0n:
      Empty.absurd({Nat.is_le(1n+p, 0n) == True{} : Bool}, ar_false_true(bc))
    case 1n+p 1n+q 1n+r:
      ar_le_trans(p, q, r, ab, bc)

law ar_le_antisym:
  for +a: Nat
  for +b: Nat
  for ab: {Nat.is_le(a, b) == True{} : Bool}
  for ba: {Nat.is_le(b, a) == True{} : Bool}
  {a == b : Nat}

def ar_le_antisym(a, b, ab, ba):
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      Empty.absurd({0n == 1n+q : Nat}, ar_false_true(ba))
    case 1n+p 0n:
      Empty.absurd({1n+p == 0n : Nat}, ar_false_true(ab))
    case 1n+p 1n+q:
      %ar_le_antisym(p, q, ab, ba) : {1n+p == 1n+_ : Nat}
      {==}

law ar_le_add_cancel:
  for +c: Nat
  for +a: Nat
  for +b: Nat
  {Nat.is_le(Nat.add(c, a), Nat.add(c, b)) == Nat.is_le(a, b) : Bool}

def ar_le_add_cancel(c, a, b):
  match c:
    case 0n:
      {==}
    case 1n+p:
      ar_le_add_cancel(p, a, b)

law ar_lt_add_cancel:
  for +c: Nat
  for +a: Nat
  for +b: Nat
  {Nat.is_lt(Nat.add(c, a), Nat.add(c, b)) == Nat.is_lt(a, b) : Bool}

def ar_lt_add_cancel(c, a, b):
  match c:
    case 0n:
      {==}
    case 1n+p:
      ar_lt_add_cancel(p, a, b)

law ar_le_add_r:
  for +a: Nat
  for +b: Nat
  {Nat.is_le(a, Nat.add(a, b)) == True{} : Bool}

def ar_le_add_r(a, b):
  match a:
    case 0n:
      ar_h_le_zero(b)
    case 1n+p:
      ar_le_add_r(p, b)

law ar_sub_add:
  for +a: Nat
  for +b: Nat
  for ba: {Nat.is_le(b, a) == True{} : Bool}
  {Nat.add(b, Nat.sub(a, b)) == a : Nat}

def ar_sub_add(a, b, ba):
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      Empty.absurd({Nat.add(1n+q, Nat.sub(0n, 1n+q)) == 0n : Nat}, ar_false_true(ba))
    case 1n+p 0n:
      {==}
    case 1n+p 1n+q:
      %ar_sub_add(p, q, ba) : {1n+Nat.add(q, Nat.sub(p, q)) == 1n+_ : Nat}
      {==}

law ar_add_sub:
  for +a: Nat
  for +b: Nat
  {Nat.sub(Nat.add(a, b), a) == b : Nat}

def ar_add_sub(a, b):
  match a:
    case 0n:
      match b:
        case 0n:
          {==}
        case 1n+q:
          {==}
    case 1n+p:
      ar_add_sub(p, b)

law ar_le_mul_mono:
  for +a: Nat
  for +b: Nat
  for +c: Nat
  for ab: {Nat.is_le(a, b) == True{} : Bool}
  {Nat.is_le(Nat.mul(a, c), Nat.mul(b, c)) == True{} : Bool}

def ar_le_mul_mono(a, b, c, ab):
  match a b:
    case 0n _:
      ar_h_le_zero(Nat.mul(b, c))
    case 1n+p 0n:
      Empty.absurd({Nat.is_le(Nat.mul(1n+p, c), 0n) == True{} : Bool}, ar_false_true(ab))
    case 1n+p 1n+q:
      Equal.trans(Bool, Nat.is_le(Nat.add(c, Nat.mul(p, c)), Nat.add(c, Nat.mul(q, c))),
        Nat.is_le(Nat.mul(p, c), Nat.mul(q, c)), True{},
        ar_le_add_cancel(c, Nat.mul(p, c), Nat.mul(q, c)), ar_le_mul_mono(p, q, c, ab))

# ---- constructor and Bool facts ----

def ar_true_false(equal: {True{} == False{} : Bool}) -> Empty:
  %equal : ar_Truth(_)
  Unit{}

def ar_h_Positive(value: Nat) -> Type:
  match value:
    case 0n:
      Unit
    case 1n+p:
      Empty

def ar_h_pred(value: Nat) -> Nat:
  match value:
    case 0n:
      0n
    case 1n+p:
      p

def ar_h_succ(value: Nat) -> Nat:
  1n+value

def ar_zero_succ(-n: Nat, equal: {0n == 1n+n : Nat}) -> Empty:
  %equal : ar_h_Positive(_)
  Unit{}

def ar_succ_inj(-a: Nat, -b: Nat, equal: {1n+a == 1n+b : Nat}) -> {a == b : Nat}:
  Equal.cong(Nat, Nat, ar_h_pred, 1n+a, 1n+b, equal)

# Boolean extensionality from both implications.
def ar_bool_ext(a: Bool, b: Bool) -> ({a == True{} : Bool} -> {b == True{} : Bool}) ->
    ({b == True{} : Bool} -> {a == True{} : Bool}) -> {a == b : Bool}:
  match a b:
    case False{} False{}:
      f => g => {==}
    case False{} True{}:
      f => g => Empty.absurd({False{} == True{} : Bool}, ar_false_true(g({==})))
    case True{} False{}:
      f => g => Empty.absurd({True{} == False{} : Bool}, ar_false_true(f({==})))
    case True{} True{}:
      f => g => {==}

def ar_and_l(a: Bool, -b: Bool, both: {Bool.and(a, b) == True{} : Bool})
  -> {a == True{} : Bool}:
  match a:
    case False{}:
      Empty.absurd({False{} == True{} : Bool}, ar_false_true(both))
    case True{}:
      {==}

def ar_and_r(a: Bool, b: Bool, both: {Bool.and(a, b) == True{} : Bool})
  -> {b == True{} : Bool}:
  match a:
    case False{}:
      Empty.absurd({b == True{} : Bool}, ar_false_true(both))
    case True{}:
      both

def ar_and_intro(-a: Bool, -b: Bool, left: {a == True{} : Bool}, right: {b == True{} : Bool})
  -> {Bool.and(a, b) == True{} : Bool}:
  %Equal.sym(Bool, a, True{}, left) : {Bool.and(_, b) == True{} : Bool}
  right

def ar_false_ne(-a: Bool, no: {a == False{} : Bool}, yes: {a == True{} : Bool}) -> Empty:
  ar_false_true(Equal.trans(Bool, False{}, a, True{}, Equal.sym(Bool, a, False{}, no), yes))

# A Bool that is not True is False.
def ar_not_true(a: Bool, no: {a == True{} : Bool} -> Empty) -> {a == False{} : Bool}:
  match a:
    case False{}:
      {==}
    case True{}:
      Empty.absurd({True{} == False{} : Bool}, no({==}))

law ar_sub_zero_r:
  for +a: Nat
  {Nat.sub(a, 0n) == a : Nat}

def ar_sub_zero_r(a):
  match a:
    case 0n:
      {==}
    case 1n+p:
      {==}

law ar_mul_one_l:
  for +a: Nat
  {Nat.mul(1n, a) == a : Nat}

def ar_mul_one_l(a):
  ar_add_zero_r(a)

law ar_lt_irrefl:
  for +a: Nat
  {Nat.is_lt(a, a) == False{} : Bool}

def ar_lt_irrefl(a):
  match a:
    case 0n:
      {==}
    case 1n+p:
      ar_lt_irrefl(p)

law ar_eq_true:
  for +a: Nat
  for +b: Nat
  for equal: {Nat.is_eq(a, b) == True{} : Bool}
  {a == b : Nat}

def ar_eq_true(a, b, equal):
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      Empty.absurd({0n == 1n+q : Nat}, ar_false_true(equal))
    case 1n+p 0n:
      Empty.absurd({1n+p == 0n : Nat}, ar_false_true(equal))
    case 1n+p 1n+q:
      %ar_eq_true(p, q, equal) : {1n+p == 1n+_ : Nat}
      {==}

law ar_eq_refl:
  for +a: Nat
  {Nat.is_eq(a, a) == True{} : Bool}

def ar_eq_refl(a):
  match a:
    case 0n:
      {==}
    case 1n+p:
      ar_eq_refl(p)

# is_eq is False exactly when the values differ.
law ar_eq_false:
  for +a: Nat
  for +b: Nat
  for differ: {a == b : Nat} -> Empty
  {Nat.is_eq(a, b) == False{} : Bool}

def ar_eq_false(a, b, differ):
  match a b:
    case 0n 0n:
      Empty.absurd({Nat.is_eq(0n, 0n) == False{} : Bool}, differ({==}))
    case 0n 1n+q:
      {==}
    case 1n+p 0n:
      {==}
    case 1n+p 1n+q:
      ar_eq_false(p, q, e => differ(Equal.cong(Nat, Nat, ar_h_succ, p, q, e)))

law ar_lt_trans:
  for +a: Nat
  for +b: Nat
  for +c: Nat
  for ab: {Nat.is_lt(a, b) == True{} : Bool}
  for bc: {Nat.is_lt(b, c) == True{} : Bool}
  {Nat.is_lt(a, c) == True{} : Bool}

def ar_lt_trans(a, b, c, ab, bc):
  match a b c:
    case 0n 0n _:
      Empty.absurd({Nat.is_lt(0n, c) == True{} : Bool}, ar_false_true(ab))
    case 0n 1n+q 0n:
      Empty.absurd({Nat.is_lt(0n, 0n) == True{} : Bool}, ar_false_true(bc))
    case 0n 1n+q 1n+r:
      {==}
    case 1n+p 0n _:
      Empty.absurd({Nat.is_lt(1n+p, c) == True{} : Bool}, ar_false_true(ab))
    case 1n+p 1n+q 0n:
      Empty.absurd({Nat.is_lt(1n+p, 0n) == True{} : Bool}, ar_false_true(bc))
    case 1n+p 1n+q 1n+r:
      ar_lt_trans(p, q, r, ab, bc)

law ar_le_lt_trans:
  for +a: Nat
  for +b: Nat
  for +c: Nat
  for ab: {Nat.is_le(a, b) == True{} : Bool}
  for bc: {Nat.is_lt(b, c) == True{} : Bool}
  {Nat.is_lt(a, c) == True{} : Bool}

def ar_le_lt_trans(a, b, c, ab, bc):
  match a b c:
    case 0n 0n 0n:
      Empty.absurd({Nat.is_lt(0n, 0n) == True{} : Bool}, ar_false_true(bc))
    case 0n 0n 1n+r:
      {==}
    case 0n 1n+q 0n:
      Empty.absurd({Nat.is_lt(0n, 0n) == True{} : Bool}, ar_false_true(bc))
    case 0n 1n+q 1n+r:
      {==}
    case 1n+p 0n _:
      Empty.absurd({Nat.is_lt(1n+p, c) == True{} : Bool}, ar_false_true(ab))
    case 1n+p 1n+q 0n:
      Empty.absurd({Nat.is_lt(1n+p, 0n) == True{} : Bool}, ar_false_true(bc))
    case 1n+p 1n+q 1n+r:
      ar_le_lt_trans(p, q, r, ab, bc)

law ar_lt_le_trans:
  for +a: Nat
  for +b: Nat
  for +c: Nat
  for ab: {Nat.is_lt(a, b) == True{} : Bool}
  for bc: {Nat.is_le(b, c) == True{} : Bool}
  {Nat.is_lt(a, c) == True{} : Bool}

def ar_lt_le_trans(a, b, c, ab, bc):
  match a b c:
    case 0n 0n _:
      Empty.absurd({Nat.is_lt(0n, c) == True{} : Bool}, ar_false_true(ab))
    case 0n 1n+q 0n:
      Empty.absurd({Nat.is_lt(0n, 0n) == True{} : Bool}, ar_false_true(bc))
    case 0n 1n+q 1n+r:
      {==}
    case 1n+p 0n _:
      Empty.absurd({Nat.is_lt(1n+p, c) == True{} : Bool}, ar_false_true(ab))
    case 1n+p 1n+q 0n:
      Empty.absurd({Nat.is_lt(1n+p, 0n) == True{} : Bool}, ar_false_true(bc))
    case 1n+p 1n+q 1n+r:
      ar_lt_le_trans(p, q, r, ab, bc)

law ar_lt_le:
  for +a: Nat
  for +b: Nat
  for ab: {Nat.is_lt(a, b) == True{} : Bool}
  {Nat.is_le(a, b) == True{} : Bool}

def ar_lt_le(a, b, ab):
  match a b:
    case 0n 0n:
      Empty.absurd({Nat.is_le(0n, 0n) == True{} : Bool}, ar_false_true(ab))
    case 0n 1n+q:
      {==}
    case 1n+p 0n:
      Empty.absurd({Nat.is_le(1n+p, 0n) == True{} : Bool}, ar_false_true(ab))
    case 1n+p 1n+q:
      ar_lt_le(p, q, ab)

def ar_h_le_succ(n: Nat) -> {Nat.is_le(n, 1n+n) == True{} : Bool}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      ar_h_le_succ(p)

law ar_le_add_l:
  for +a: Nat
  for +b: Nat
  {Nat.is_le(b, Nat.add(a, b)) == True{} : Bool}

def ar_le_add_l(a, b):
  match a:
    case 0n:
      ar_le_refl(b)
    case 1n+p:
      ar_le_trans(b, Nat.add(p, b), 1n+Nat.add(p, b), ar_le_add_l(p, b), ar_h_le_succ(Nat.add(p, b)))

def ar_h_lt_zero(n: Nat) -> {Nat.is_lt(n, 0n) == False{} : Bool}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      {==}

# a - b < c iff a < b + c, for c >= 1 (false at c = 0 when a < b).
law ar_sub_lt_iff:
  for +a: Nat
  for +b: Nat
  for +c: Nat
  for pos: {Nat.is_lt(0n, c) == True{} : Bool}
  {Nat.is_lt(Nat.sub(a, b), c) == Nat.is_lt(a, Nat.add(b, c)) : Bool}

def ar_sub_lt_iff(a, b, c, pos):
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      pos
    case 1n+p 0n:
      {==}
    case 1n+p 1n+q:
      ar_sub_lt_iff(p, q, c, pos)

# c < a - b iff b + c < a (no side condition on Nat).
law ar_lt_sub_iff:
  for +a: Nat
  for +b: Nat
  for +c: Nat
  {Nat.is_lt(c, Nat.sub(a, b)) == Nat.is_lt(Nat.add(b, c), a) : Bool}

def ar_lt_sub_iff(a, b, c):
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      ar_h_lt_zero(c)
    case 1n+p 0n:
      {==}
    case 1n+p 1n+q:
      ar_lt_sub_iff(p, q, c)

law ar_le_sub_iff:
  for +a: Nat
  for +b: Nat
  for +c: Nat
  {Nat.is_le(Nat.sub(a, b), c) == Nat.is_le(a, Nat.add(b, c)) : Bool}

def ar_le_sub_iff(a, b, c):
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      ar_h_le_zero(c)
    case 1n+p 0n:
      {==}
    case 1n+p 1n+q:
      ar_le_sub_iff(p, q, c)

law ar_mul_le_cancel:
  for +a: Nat
  for +b: Nat
  for +c: Nat
  for pos: {Nat.is_lt(0n, c) == True{} : Bool}
  {Nat.is_le(Nat.mul(a, c), Nat.mul(b, c)) == Nat.is_le(a, b) : Bool}

def ar_mul_le_cancel(a, b, c, pos):
  match a b:
    case 0n _:
      Equal.trans(Bool, Nat.is_le(0n, Nat.mul(b, c)), True{}, Nat.is_le(0n, b),
        ar_h_le_zero(Nat.mul(b, c)), Equal.sym(Bool, Nat.is_le(0n, b), True{}, ar_h_le_zero(b)))
    case 1n+p 0n:
      match c:
        case 0n:
          Empty.absurd({Nat.is_le(Nat.mul(1n+p, 0n), 0n) == False{} : Bool}, ar_false_true(pos))
        case 1n+r:
          {==}
    case 1n+p 1n+q:
      Equal.trans(Bool, Nat.is_le(Nat.add(c, Nat.mul(p, c)), Nat.add(c, Nat.mul(q, c))),
        Nat.is_le(Nat.mul(p, c), Nat.mul(q, c)), Nat.is_le(p, q),
        ar_le_add_cancel(c, Nat.mul(p, c), Nat.mul(q, c)), ar_mul_le_cancel(p, q, c, pos))

law ar_mul_lt_cancel:
  for +a: Nat
  for +b: Nat
  for +c: Nat
  for pos: {Nat.is_lt(0n, c) == True{} : Bool}
  {Nat.is_lt(Nat.mul(a, c), Nat.mul(b, c)) == Nat.is_lt(a, b) : Bool}

def ar_mul_lt_cancel(a, b, c, pos):
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      match c:
        case 0n:
          Empty.absurd({Nat.is_lt(0n, Nat.mul(1n+q, 0n)) == True{} : Bool}, ar_false_true(pos))
        case 1n+r:
          {==}
    case 1n+p 0n:
      ar_h_lt_zero(Nat.mul(1n+p, c))
    case 1n+p 1n+q:
      Equal.trans(Bool, Nat.is_lt(Nat.add(c, Nat.mul(p, c)), Nat.add(c, Nat.mul(q, c))),
        Nat.is_lt(Nat.mul(p, c), Nat.mul(q, c)), Nat.is_lt(p, q),
        ar_lt_add_cancel(c, Nat.mul(p, c), Nat.mul(q, c)), ar_mul_lt_cancel(p, q, c, pos))

def ar_h_and_false_r(x: Bool) -> {False{} == Bool.and(x, False{}) : Bool}:
  match x:
    case False{}:
      {==}
    case True{}:
      {==}

law ar_min_lt:
  for +c: Nat
  for +a: Nat
  for +b: Nat
  {Nat.is_lt(c, Nat.min(a, b)) == Bool.and(Nat.is_lt(c, a), Nat.is_lt(c, b)) : Bool}

def ar_min_lt(c, a, b):
  match c a b:
    case 0n 0n _:
      {==}
    case 0n 1n+p 0n:
      {==}
    case 0n 1n+p 1n+q:
      {==}
    case 1n+r 0n _:
      {==}
    case 1n+r 1n+p 0n:
      ar_h_and_false_r(Nat.is_lt(r, p))
    case 1n+r 1n+p 1n+q:
      ar_min_lt(r, p, q)

law ar_min_left:
  for +a: Nat
  for +b: Nat
  for ab: {Nat.is_le(a, b) == True{} : Bool}
  {Nat.min(a, b) == a : Nat}

def ar_min_left(a, b, ab):
  match a b:
    case 0n _:
      {==}
    case 1n+p 0n:
      Empty.absurd({0n == 1n+p : Nat}, ar_false_true(ab))
    case 1n+p 1n+q:
      %Equal.sym(Nat, Nat.min(p, q), p, ar_min_left(p, q, ab)) : {1n+_ == 1n+p : Nat}
      {==}

# ==== Floor division lemmas (via the Nat.divmod.go invariant) ====

def ard_rt(-a: Bool, -b: Bool, e: {a == b : Bool}, h: {a == True{} : Bool})
  -> {b == True{} : Bool}:
  Equal.trans(Bool, b, a, True{}, Equal.sym(Bool, a, b, e), h)

def ard_tr(-a: Bool, -b: Bool, e: {a == b : Bool}, h: {b == True{} : Bool})
  -> {a == True{} : Bool}:
  Equal.trans(Bool, a, b, True{}, e, h)

# Projections of a proof pair.
def ard_fst(-A: Type, -B: Type, p: A & B) -> A:
  (x, y) = p
  x

def ard_snd(-A: Type, -B: Type, p: A & B) -> B:
  (x, y) = p
  y

# m < m is absurd.
def ard_lt_self(+m: Nat, h: {Nat.is_lt(m, m) == True{} : Bool}) -> Empty:
  ar_false_true(Equal.trans(Bool, False{}, Nat.is_lt(m, m), True{},
    Equal.sym(Bool, Nat.is_lt(m, m), False{}, ar_lt_irrefl(m)), h))

# r < b forces 0 < b.
def ard_pos(r: Nat, +b: Nat, small: {Nat.is_lt(r, b) == True{} : Bool})
  -> {Nat.is_lt(0n, b) == True{} : Bool}:
  match r:
    case 0n:
      small
    case 1n+p:
      ar_lt_trans(0n, 1n+p, b, {==}, small)

# r + m == bp gives r <= bp.
def ard_le_of_add(+r: Nat, +m: Nat, -bp: Nat, inv: {Nat.add(r, m) == bp : Nat})
  -> {Nat.is_le(r, bp) == True{} : Bool}:
  %inv : {Nat.is_le(r, _) == True{} : Bool}
  ar_le_add_r(r, m)

# 1 + ((y + x) + n) == x + (z + (1 + n)) when z == y.
def ard_shuffle(+x: Nat, -y: Nat, +z: Nat, +n: Nat, e: {z == y : Nat})
  -> {1n+Nat.add(Nat.add(y, x), n) == Nat.add(x, Nat.add(z, 1n+n)) : Nat}:
  %e : {1n+Nat.add(Nat.add(_, x), n) == Nat.add(x, Nat.add(z, 1n+n)) : Nat}
  %Equal.sym(Nat, Nat.add(z, 1n+n), 1n+Nat.add(z, n), ar_add_succ_r(z, n)) : {1n+Nat.add(Nat.add(z, x), n) == Nat.add(x, _) : Nat}
  %Equal.sym(Nat, Nat.add(x, 1n+Nat.add(z, n)), 1n+Nat.add(x, Nat.add(z, n)), ar_add_succ_r(x, Nat.add(z, n))) : {1n+Nat.add(Nat.add(z, x), n) == _ : Nat}
  %ar_add_comm(x, z) : {1n+Nat.add(_, n) == 1n+Nat.add(x, Nat.add(z, n)) : Nat}
  %Equal.sym(Nat, Nat.add(Nat.add(x, z), n), Nat.add(x, Nat.add(z, n)), ar_add_assoc(x, z, n)) : {1n+_ == 1n+Nat.add(x, Nat.add(z, n)) : Nat}
  {==}

# Invariant of Nat.divmod.go with divisor b = 1 + bp: when r + m == bp,
# q * b + r2 == d * b + (r + n).
def ard_go_eq(+n: Nat, +m: Nat, +d: Nat, +r: Nat, +bp: Nat, inv: {Nat.add(r, m) == bp : Nat})
  -> {Nat.add(Nat.mul(Pair.fst(Nat, Nat, Nat.divmod.go(n, m, d, r)), 1n+bp), Pair.snd(Nat, Nat, Nat.divmod.go(n, m, d, r))) == Nat.add(Nat.mul(d, 1n+bp), Nat.add(r, n)) : Nat}:
  match n:
    case 0n:
      %ar_add_zero_r(r) : {Nat.add(Nat.mul(d, 1n+bp), _) == Nat.add(Nat.mul(d, 1n+bp), Nat.add(r, 0n)) : Nat}
      {==}
    case 1n+np:
      match m:
        case 0n:
          +inv2 = Equal.trans(Nat, r, Nat.add(r, 0n), bp,
            Equal.sym(Nat, Nat.add(r, 0n), r, ar_add_zero_r(r)), inv)
          %ard_shuffle(Nat.mul(d, 1n+bp), bp, r, np, inv2) : {Nat.add(Nat.mul(Pair.fst(Nat, Nat, Nat.divmod.go(np, r, 1n+d, 0n)), 1n+bp), Pair.snd(Nat, Nat, Nat.divmod.go(np, r, 1n+d, 0n))) == _ : Nat}
          ard_go_eq(np, r, 1n+d, 0n, bp, inv2)
        case 1n+mp:
          %Equal.sym(Nat, Nat.add(r, 1n+np), 1n+Nat.add(r, np), ar_add_succ_r(r, np)) : {Nat.add(Nat.mul(Pair.fst(Nat, Nat, Nat.divmod.go(np, mp, d, 1n+r)), 1n+bp), Pair.snd(Nat, Nat, Nat.divmod.go(np, mp, d, 1n+r))) == Nat.add(Nat.mul(d, 1n+bp), _) : Nat}
          ard_go_eq(np, mp, d, 1n+r, bp,
            Equal.trans(Nat, 1n+Nat.add(r, mp), Nat.add(r, 1n+mp), bp,
              Equal.sym(Nat, Nat.add(r, 1n+mp), 1n+Nat.add(r, mp), ar_add_succ_r(r, mp)), inv))

# The remainder of Nat.divmod.go stays below b = 1 + bp.
def ard_go_lt(+n: Nat, +m: Nat, +d: Nat, +r: Nat, +bp: Nat, inv: {Nat.add(r, m) == bp : Nat})
  -> {Nat.is_lt(Pair.snd(Nat, Nat, Nat.divmod.go(n, m, d, r)), 1n+bp) == True{} : Bool}:
  match n:
    case 0n:
      Equal.trans(Bool, Nat.is_lt(r, 1n+bp), Nat.is_le(r, bp), True{},
        ar_lt_succ_le(r, 1n+bp), ard_le_of_add(r, m, bp, inv))
    case 1n+np:
      match m:
        case 0n:
          ard_go_lt(np, r, 1n+d, 0n, bp, Equal.trans(Nat, r, Nat.add(r, 0n), bp,
            Equal.sym(Nat, Nat.add(r, 0n), r, ar_add_zero_r(r)), inv))
        case 1n+mp:
          ard_go_lt(np, mp, d, 1n+r, bp,
            Equal.trans(Nat, 1n+Nat.add(r, mp), Nat.add(r, 1n+mp), bp,
              Equal.sym(Nat, Nat.add(r, 1n+mp), 1n+Nat.add(r, mp), ar_add_succ_r(r, mp)), inv))

def ard_le_big(+r: Nat, +m: Nat, +x: Nat, +y: Nat, eq: {r == Nat.add(Nat.add(m, x), y) : Nat})
  -> {Nat.is_le(m, r) == True{} : Bool}:
  %Equal.sym(Nat, r, Nat.add(Nat.add(m, x), y), eq) : {Nat.is_le(m, _) == True{} : Bool}
  ar_le_trans(m, Nat.add(m, x), Nat.add(Nat.add(m, x), y),
    ar_le_add_r(m, x), ar_le_add_r(Nat.add(m, x), y))

# r == (m + x) + y with r < m is absurd.
def ard_big(+r: Nat, +m: Nat, +x: Nat, +y: Nat, small: {Nat.is_lt(r, m) == True{} : Bool},
    eq: {r == Nat.add(Nat.add(m, x), y) : Nat}) -> Empty:
  ard_lt_self(m, ar_le_lt_trans(m, r, m, ard_le_big(r, m, x, y, eq), small))

# (m + x) + r1 == (m + y) + r2 gives x + r1 == y + r2.
def ard_cancel(+m: Nat, +x: Nat, +r1: Nat, +y: Nat, +r2: Nat,
    same: {Nat.add(Nat.add(m, x), r1) == Nat.add(Nat.add(m, y), r2) : Nat})
  -> {Nat.add(x, r1) == Nat.add(y, r2) : Nat}:
  %ar_add_sub(m, Nat.add(x, r1)) : {_ == Nat.add(y, r2) : Nat}
  %ar_add_sub(m, Nat.add(y, r2)) : {Nat.sub(Nat.add(m, Nat.add(x, r1)), m) == _ : Nat}
  %ar_add_assoc(m, x, r1) : {Nat.sub(_, m) == Nat.sub(Nat.add(m, Nat.add(y, r2)), m) : Nat}
  %ar_add_assoc(m, y, r2) : {Nat.sub(Nat.add(Nat.add(m, x), r1), m) == Nat.sub(_, m) : Nat}
  Equal.cong(Nat, Nat, t => Nat.sub(t, m), Nat.add(Nat.add(m, x), r1), Nat.add(Nat.add(m, y), r2), same)

# (b + x) + r == (x + r) + b.
def ard_swap(+x: Nat, +r: Nat, +b: Nat)
  -> {Nat.add(Nat.add(b, x), r) == Nat.add(Nat.add(x, r), b) : Nat}:
  Equal.trans(Nat, Nat.add(Nat.add(b, x), r), Nat.add(b, Nat.add(x, r)), Nat.add(Nat.add(x, r), b),
    ar_add_assoc(b, x, r), ar_add_comm(b, Nat.add(x, r)))

law ar_divmod:
  for +a: Nat
  for +b: Nat
  for pos: {Nat.is_lt(0n, b) == True{} : Bool}
  {Nat.add(Nat.mul(Nat.div(a, b), b), Nat.mod(a, b)) == a : Nat} &
  {Nat.is_lt(Nat.mod(a, b), b) == True{} : Bool}

def ar_divmod(a, b, pos):
  match b:
    case 0n:
      Empty.absurd({Nat.add(Nat.mul(Nat.div(a, 0n), 0n), Nat.mod(a, 0n)) == a : Nat} &
        {Nat.is_lt(Nat.mod(a, 0n), 0n) == True{} : Bool}, ar_false_true(pos))
    case 1n+bp:
      (ard_go_eq(a, bp, 0n, 0n, bp, {==}), ard_go_lt(a, bp, 0n, 0n, bp, {==}))

# Lift a (q, r) equality pair through a successor on q.
def ard_succ_pair(-p1: Nat, -p2: Nat, -r1: Nat, -r2: Nat, h: {p1 == p2 : Nat} & {r1 == r2 : Nat})
  -> {1n+p1 == 1n+p2 : Nat} & {r1 == r2 : Nat}:
  (x, y) = h
  (Equal.cong(Nat, Nat, t => 1n+t, p1, p2, x), y)

# q1 * m + r1 == q2 * m + r2 with r1, r2 < m forces q1 == q2 and r1 == r2.
law ar_divmod_inj:
  for +q1: Nat
  for +r1: Nat
  for +q2: Nat
  for +r2: Nat
  for +m: Nat
  for small1: {Nat.is_lt(r1, m) == True{} : Bool}
  for small2: {Nat.is_lt(r2, m) == True{} : Bool}
  for same: {Nat.add(Nat.mul(q1, m), r1) == Nat.add(Nat.mul(q2, m), r2) : Nat}
  {q1 == q2 : Nat} & {r1 == r2 : Nat}

def ar_divmod_inj(q1, r1, q2, r2, m, small1, small2, same):
  match q1 q2:
    case 0n 0n:
      ({==}, same)
    case 0n 1n+p2:
      Empty.absurd({0n == 1n+p2 : Nat} & {r1 == r2 : Nat},
        ard_big(r1, m, Nat.mul(p2, m), r2, small1, same))
    case 1n+p1 0n:
      Empty.absurd({1n+p1 == 0n : Nat} & {r1 == r2 : Nat},
        ard_big(r2, m, Nat.mul(p1, m), r1, small2,
          Equal.sym(Nat, Nat.add(Nat.add(m, Nat.mul(p1, m)), r1), r2, same)))
    case 1n+p1 1n+p2:
      ard_succ_pair(p1, p2, r1, r2, ar_divmod_inj(p1, r1, p2, r2, m, small1, small2,
        ard_cancel(m, Nat.mul(p1, m), r1, Nat.mul(p2, m), r2, same)))

# Components of ar_divmod.
def ard_dm_eq(+a: Nat, +b: Nat, pos: {Nat.is_lt(0n, b) == True{} : Bool})
  -> {Nat.add(Nat.mul(Nat.div(a, b), b), Nat.mod(a, b)) == a : Nat}:
  ard_fst({Nat.add(Nat.mul(Nat.div(a, b), b), Nat.mod(a, b)) == a : Nat},
    {Nat.is_lt(Nat.mod(a, b), b) == True{} : Bool}, ar_divmod(a, b, pos))

def ard_dm_lt(+a: Nat, +b: Nat, pos: {Nat.is_lt(0n, b) == True{} : Bool})
  -> {Nat.is_lt(Nat.mod(a, b), b) == True{} : Bool}:
  ard_snd({Nat.add(Nat.mul(Nat.div(a, b), b), Nat.mod(a, b)) == a : Nat},
    {Nat.is_lt(Nat.mod(a, b), b) == True{} : Bool}, ar_divmod(a, b, pos))

# (a / b) * b <= a.
def ard_le_mul_div(+a: Nat, +b: Nat, pos: {Nat.is_lt(0n, b) == True{} : Bool})
  -> {Nat.is_le(Nat.mul(Nat.div(a, b), b), a) == True{} : Bool}:
  %ard_dm_eq(a, b, pos) : {Nat.is_le(Nat.mul(Nat.div(a, b), b), _) == True{} : Bool}
  ar_le_add_r(Nat.mul(Nat.div(a, b), b), Nat.mod(a, b))

# a < (a / b + 1) * b.
def ard_lt_next(+a: Nat, +b: Nat, +pos: {Nat.is_lt(0n, b) == True{} : Bool})
  -> {Nat.is_lt(a, Nat.mul(1n+Nat.div(a, b), b)) == True{} : Bool}:
  %ar_add_comm(Nat.mul(Nat.div(a, b), b), b) : {Nat.is_lt(a, _) == True{} : Bool}
  %ard_dm_eq(a, b, pos) : {Nat.is_lt(_, Nat.add(Nat.mul(Nat.div(a, b), b), b)) == True{} : Bool}
  ard_tr(Nat.is_lt(Nat.add(Nat.mul(Nat.div(a, b), b), Nat.mod(a, b)), Nat.add(Nat.mul(Nat.div(a, b), b), b)),
    Nat.is_lt(Nat.mod(a, b), b),
    ar_lt_add_cancel(Nat.mul(Nat.div(a, b), b), Nat.mod(a, b), b), ard_dm_lt(a, b, pos))

def ard_div_le_fwd(+q: Nat, +a: Nat, +b: Nat, +pos: {Nat.is_lt(0n, b) == True{} : Bool},
    h: {Nat.is_le(Nat.mul(q, b), a) == True{} : Bool})
  -> {Nat.is_le(q, Nat.div(a, b)) == True{} : Bool}:
  ard_rt(Nat.is_lt(q, 1n+Nat.div(a, b)), Nat.is_le(q, Nat.div(a, b)),
    ar_lt_succ_le(q, 1n+Nat.div(a, b)),
    ard_rt(Nat.is_lt(Nat.mul(q, b), Nat.mul(1n+Nat.div(a, b), b)), Nat.is_lt(q, 1n+Nat.div(a, b)),
      ar_mul_lt_cancel(q, 1n+Nat.div(a, b), b, pos),
      ar_le_lt_trans(Nat.mul(q, b), a, Nat.mul(1n+Nat.div(a, b), b), h,
        ard_lt_next(a, b, pos))))

def ard_div_le_bwd(+q: Nat, +a: Nat, +b: Nat, pos: {Nat.is_lt(0n, b) == True{} : Bool},
    h: {Nat.is_le(q, Nat.div(a, b)) == True{} : Bool})
  -> {Nat.is_le(Nat.mul(q, b), a) == True{} : Bool}:
  ar_le_trans(Nat.mul(q, b), Nat.mul(Nat.div(a, b), b), a,
    ar_le_mul_mono(q, Nat.div(a, b), b, h), ard_le_mul_div(a, b, pos))

# Galois connection of floor division: q * b <= a iff q <= a / b (b >= 1).
law ar_div_le:
  for +q: Nat
  for +a: Nat
  for +b: Nat
  for pos: {Nat.is_lt(0n, b) == True{} : Bool}
  {Nat.is_le(Nat.mul(q, b), a) == Nat.is_le(q, Nat.div(a, b)) : Bool}

def ar_div_le(q, a, b, pos):
  +p = pos
  ar_bool_ext(Nat.is_le(Nat.mul(q, b), a), Nat.is_le(q, Nat.div(a, b)),
    h => ard_div_le_fwd(q, a, b, p, h), h => ard_div_le_bwd(q, a, b, p, h))

law ar_div_unique:
  for +q: Nat
  for +r: Nat
  for +b: Nat
  for small: {Nat.is_lt(r, b) == True{} : Bool}
  {Nat.div(Nat.add(Nat.mul(q, b), r), b) == q : Nat} &
  {Nat.mod(Nat.add(Nat.mul(q, b), r), b) == r : Nat}

def ar_div_unique(q, r, b, small):
  +s = small
  +ps = ard_pos(r, b, s)
  ar_divmod_inj(Nat.div(Nat.add(Nat.mul(q, b), r), b), Nat.mod(Nat.add(Nat.mul(q, b), r), b),
    q, r, b, ard_dm_lt(Nat.add(Nat.mul(q, b), r), b, ps), s,
    ard_dm_eq(Nat.add(Nat.mul(q, b), r), b, ps))

# Dual Galois connection of floor division: a / b < q iff a < q * b (b >= 1).
law ar_div_lt:
  for +q: Nat
  for +a: Nat
  for +b: Nat
  for pos: {Nat.is_lt(0n, b) == True{} : Bool}
  {Nat.is_lt(Nat.div(a, b), q) == Nat.is_lt(a, Nat.mul(q, b)) : Bool}

def ar_div_lt(q, a, b, pos):
  %Equal.sym(Bool, Nat.is_lt(Nat.div(a, b), q), Bool.not(Nat.is_le(q, Nat.div(a, b))), ar_lt_not_le(Nat.div(a, b), q)) : {_ == Nat.is_lt(a, Nat.mul(q, b)) : Bool}
  %ar_div_le(q, a, b, pos) : {Bool.not(_) == Nat.is_lt(a, Nat.mul(q, b)) : Bool}
  Equal.sym(Bool, Nat.is_lt(a, Nat.mul(q, b)), Bool.not(Nat.is_le(Nat.mul(q, b), a)),
    ar_lt_not_le(a, Nat.mul(q, b)))

law ar_div_add_self:
  for +a: Nat
  for +b: Nat
  for pos: {Nat.is_lt(0n, b) == True{} : Bool}
  {Nat.div(Nat.add(a, b), b) == 1n+Nat.div(a, b) : Nat}

def ar_div_add_self(a, b, pos):
  +p = pos
  %ard_dm_eq(a, b, p) : {Nat.div(Nat.add(_, b), b) == 1n+Nat.div(a, b) : Nat}
  %ard_swap(Nat.mul(Nat.div(a, b), b), Nat.mod(a, b), b) : {Nat.div(_, b) == 1n+Nat.div(a, b) : Nat}
  ard_fst({Nat.div(Nat.add(Nat.mul(1n+Nat.div(a, b), b), Nat.mod(a, b)), b) == 1n+Nat.div(a, b) : Nat},
    {Nat.mod(Nat.add(Nat.mul(1n+Nat.div(a, b), b), Nat.mod(a, b)), b) == Nat.mod(a, b) : Nat},
    ar_div_unique(1n+Nat.div(a, b), Nat.mod(a, b), b, ard_dm_lt(a, b, p)))

law ar_div_mono:
  for +a: Nat
  for +c: Nat
  for +b: Nat
  for pos: {Nat.is_lt(0n, b) == True{} : Bool}
  for ac: {Nat.is_le(a, c) == True{} : Bool}
  {Nat.is_le(Nat.div(a, b), Nat.div(c, b)) == True{} : Bool}

def ar_div_mono(a, c, b, pos, ac):
  +p = pos
  ard_rt(Nat.is_le(Nat.mul(Nat.div(a, b), b), c), Nat.is_le(Nat.div(a, b), Nat.div(c, b)),
    ar_div_le(Nat.div(a, b), c, b, p),
    ar_le_trans(Nat.mul(Nat.div(a, b), b), a, c, ard_le_mul_div(a, b, p), ac))

# ==== Core owner / start lemmas ====

def p1_rt(-a: Bool, -b: Bool, e: {a == b : Bool}, h: {a == True{} : Bool})
  -> {b == True{} : Bool}:
  Equal.trans(Bool, b, a, True{}, Equal.sym(Bool, a, b, e), h)

def p1_not_not(z: Bool) -> {Bool.not(Bool.not(z)) == z : Bool}:
  match z:
    case False{}:
      {==}
    case True{}:
      {==}

# s <= x iff s < x + 1.
def p1_le_lt_succ(+s: Nat, +x: Nat) -> {Nat.is_le(s, x) == Nat.is_lt(s, 1n+x) : Bool}:
  Equal.sym(Bool, Nat.is_lt(s, 1n+x), Nat.is_le(1n+s, 1n+x), ar_lt_succ_le(s, 1n+x))

# s <= m - 1 iff s < m, for m >= 1.
def p1_le_pred(+s: Nat, +m: Nat, pos: {Nat.is_le(1n, m) == True{} : Bool})
  -> {Nat.is_le(s, Nat.sub(m, 1n)) == Nat.is_lt(s, m) : Bool}:
  %ar_sub_add(m, 1n, pos) : {Nat.is_le(s, Nat.sub(m, 1n)) == Nat.is_lt(s, _) : Bool}
  Equal.sym(Bool, Nat.is_lt(s, 1n+Nat.sub(m, 1n)), Nat.is_le(1n+s, 1n+Nat.sub(m, 1n)),
    ar_lt_succ_le(s, 1n+Nat.sub(m, 1n)))

# 0 < (y + 1) * G for G >= 1.
def p1_mul_pos(+y: Nat, +G: Nat, gpos: {Nat.is_lt(0n, G) == True{} : Bool})
  -> {Nat.is_lt(0n, Nat.mul(1n+y, G)) == True{} : Bool}:
  ar_lt_le_trans(0n, G, Nat.mul(1n+y, G), gpos, ar_le_add_r(G, Nat.mul(y, G)))

# start(b) <= y iff b * I < (y + 1) * G.
def p1_start_le(+b: Nat, +y: Nat, +I: Nat, +G: Nat, gpos: {Nat.is_lt(0n, G) == True{} : Bool})
  -> {Nat.is_le(Impl.start(b, I, G), y) == Nat.is_lt(Nat.mul(b, I), Nat.mul(1n+y, G)) : Bool}:
  Equal.trans(Bool, Nat.is_le(Impl.start(b, I, G), y), Nat.is_lt(Impl.start(b, I, G), 1n+y),
    Nat.is_lt(Nat.mul(b, I), Nat.mul(1n+y, G)),
    p1_le_lt_succ(Impl.start(b, I, G), y),
    ar_div_lt(1n+y, Nat.mul(b, I), G, gpos))

# x < start(b) iff (x + 1) * G <= b * I.
def p1_lt_start(+x: Nat, +b: Nat, +I: Nat, +G: Nat, gpos: {Nat.is_lt(0n, G) == True{} : Bool})
  -> {Nat.is_lt(x, Impl.start(b, I, G)) == Nat.is_le(Nat.mul(1n+x, G), Nat.mul(b, I)) : Bool}:
  Equal.trans(Bool, Nat.is_lt(x, Impl.start(b, I, G)), Nat.is_le(1n+x, Impl.start(b, I, G)),
    Nat.is_le(Nat.mul(1n+x, G), Nat.mul(b, I)),
    ar_lt_succ_le(x, Impl.start(b, I, G)),
    Equal.sym(Bool, Nat.is_le(Nat.mul(1n+x, G), Nat.mul(b, I)),
      Nat.is_le(1n+x, Impl.start(b, I, G)),
      ar_div_le(1n+x, Nat.mul(b, I), G, gpos)))

# D - 1 <= b iff D < b + 2.
def p1_core_a(+D: Nat, +b: Nat)
  -> {Nat.is_le(Nat.sub(D, 1n), b) == Nat.is_lt(D, 2n+b) : Bool}:
  Equal.trans(Bool, Nat.is_le(Nat.sub(D, 1n), b), Nat.is_le(D, 1n+b), Nat.is_lt(D, 2n+b),
    ar_le_sub_iff(D, 1n, b), p1_le_lt_succ(D, 1n+b))

# A - 1 < M iff A <= M, for M >= 1.
def p1_sub1_lt(+A: Nat, +M: Nat, mpos: {Nat.is_lt(0n, M) == True{} : Bool})
  -> {Nat.is_lt(Nat.sub(A, 1n), M) == Nat.is_le(A, M) : Bool}:
  Equal.trans(Bool, Nat.is_lt(Nat.sub(A, 1n), M), Nat.is_lt(A, 1n+M), Nat.is_le(A, M),
    ar_sub_lt_iff(A, 1n, M, mpos),
    Equal.sym(Bool, Nat.is_le(A, M), Nat.is_lt(A, 1n+M), p1_le_lt_succ(A, M)))

# N + I <= I + P iff N <= P.
def p1_core_c(+N: Nat, +I: Nat, +P: Nat)
  -> {Nat.is_le(Nat.add(N, I), Nat.add(I, P)) == Nat.is_le(N, P) : Bool}:
  %ar_add_comm(I, N) : {Nat.is_le(_, Nat.add(I, P)) == Nat.is_le(N, P) : Bool}
  ar_le_add_cancel(I, N, P)

# ((N + I - 1) / I) - 1 <= b iff N <= (b + 1) * I   (I >= 1).
def p1_owner_core(+N: Nat, +I: Nat, +b: Nat, +ipos: {Nat.is_lt(0n, I) == True{} : Bool})
  -> {Nat.is_le(Nat.sub(Nat.div(Nat.sub(Nat.add(N, I), 1n), I), 1n), b) ==
    Nat.is_le(N, Nat.mul(1n+b, I)) : Bool}:
  Equal.trans(Bool, Nat.is_le(Nat.sub(Nat.div(Nat.sub(Nat.add(N, I), 1n), I), 1n), b),
    Nat.is_lt(Nat.div(Nat.sub(Nat.add(N, I), 1n), I), 2n+b),
    Nat.is_le(N, Nat.mul(1n+b, I)),
    p1_core_a(Nat.div(Nat.sub(Nat.add(N, I), 1n), I), b),
    Equal.trans(Bool, Nat.is_lt(Nat.div(Nat.sub(Nat.add(N, I), 1n), I), 2n+b),
      Nat.is_lt(Nat.sub(Nat.add(N, I), 1n), Nat.mul(2n+b, I)),
      Nat.is_le(N, Nat.mul(1n+b, I)),
      ar_div_lt(2n+b, Nat.sub(Nat.add(N, I), 1n), I, ipos),
      Equal.trans(Bool, Nat.is_lt(Nat.sub(Nat.add(N, I), 1n), Nat.mul(2n+b, I)),
        Nat.is_le(Nat.add(N, I), Nat.mul(2n+b, I)),
        Nat.is_le(N, Nat.mul(1n+b, I)),
        p1_sub1_lt(Nat.add(N, I), Nat.mul(2n+b, I), p1_mul_pos(1n+b, I, ipos)),
        p1_core_c(N, I, Nat.mul(1n+b, I)))))

# ---- core lemmas ----

# owns(b, x) iff start(b) <= x < start(b + 1)   (G >= 1; any I)
def p1_owns_range(+b: Nat, +x: Nat, +I: Nat, +G: Nat,
    gpos: {Nat.is_lt(0n, G) == True{} : Bool})
  -> {Spec.owns(b, x, I, G) == Bool.and(Nat.is_le(Impl.start(b, I, G), x),
    Nat.is_lt(x, Impl.start(1n+b, I, G))) : Bool}:
  +gp = gpos
  %Equal.sym(Bool, Nat.is_le(Impl.start(b, I, G), x), Nat.is_lt(Nat.mul(b, I), Nat.mul(1n+x, G)), p1_start_le(b, x, I, G, gp)) : {Spec.owns(b, x, I, G) == Bool.and(_, Nat.is_lt(x, Impl.start(1n+b, I, G))) : Bool}
  %Equal.sym(Bool, Nat.is_lt(x, Impl.start(1n+b, I, G)), Nat.is_le(Nat.mul(1n+x, G), Nat.mul(1n+b, I)), p1_lt_start(x, 1n+b, I, G, gp)) : {Spec.owns(b, x, I, G) == Bool.and(Nat.is_lt(Nat.mul(b, I), Nat.mul(1n+x, G)), _) : Bool}
  {==}

# owner(x) <= b iff x < start(b + 1)   (I, G >= 1)
def p1_owner_le(+x: Nat, +b: Nat, +I: Nat, +G: Nat,
    ipos: {Nat.is_lt(0n, I) == True{} : Bool},
    gpos: {Nat.is_lt(0n, G) == True{} : Bool})
  -> {Nat.is_le(Impl.owner(x, I, G), b) == Nat.is_lt(x, Impl.start(1n+b, I, G)) : Bool}:
  Equal.trans(Bool, Nat.is_le(Impl.owner(x, I, G), b),
    Nat.is_le(Nat.mul(1n+x, G), Nat.mul(1n+b, I)),
    Nat.is_lt(x, Impl.start(1n+b, I, G)),
    p1_owner_core(Nat.mul(1n+x, G), I, b, ipos),
    Equal.sym(Bool, Nat.is_lt(x, Impl.start(1n+b, I, G)),
      Nat.is_le(Nat.mul(1n+x, G), Nat.mul(1n+b, I)),
      p1_lt_start(x, 1n+b, I, G, gpos)))

# c + 1 <= owner(y) iff start(c + 1) <= y   (I, G >= 1)
def p1_le_owner_succ(+c: Nat, +y: Nat, +I: Nat, +G: Nat,
    ipos: {Nat.is_lt(0n, I) == True{} : Bool},
    gpos: {Nat.is_lt(0n, G) == True{} : Bool})
  -> {Nat.is_le(1n+c, Impl.owner(y, I, G)) == Nat.is_le(Impl.start(1n+c, I, G), y) : Bool}:
  %ar_lt_succ_le(c, Impl.owner(y, I, G)) : {_ == Nat.is_le(Impl.start(1n+c, I, G), y) : Bool}
  %Equal.sym(Bool, Nat.is_lt(c, Impl.owner(y, I, G)), Bool.not(Nat.is_le(Impl.owner(y, I, G), c)), ar_lt_not_le(c, Impl.owner(y, I, G))) : {_ == Nat.is_le(Impl.start(1n+c, I, G), y) : Bool}
  %Equal.sym(Bool, Nat.is_le(Impl.owner(y, I, G), c), Nat.is_lt(y, Impl.start(1n+c, I, G)), p1_owner_le(y, c, I, G, ipos, gpos)) : {Bool.not(_) == Nat.is_le(Impl.start(1n+c, I, G), y) : Bool}
  %Equal.sym(Bool, Nat.is_lt(y, Impl.start(1n+c, I, G)), Bool.not(Nat.is_le(Impl.start(1n+c, I, G), y)), ar_lt_not_le(y, Impl.start(1n+c, I, G))) : {Bool.not(_) == Nat.is_le(Impl.start(1n+c, I, G), y) : Bool}
  p1_not_not(Nat.is_le(Impl.start(1n+c, I, G), y))

# 0 <= owner(y) and start(0) <= y   (G >= 1)
def p1_le_owner_zero(+y: Nat, +I: Nat, +G: Nat, +gpos: {Nat.is_lt(0n, G) == True{} : Bool})
  -> {Nat.is_le(0n, Impl.owner(y, I, G)) == Nat.is_le(Impl.start(0n, I, G), y) : Bool}:
  Equal.trans(Bool, Nat.is_le(0n, Impl.owner(y, I, G)), True{},
    Nat.is_le(Impl.start(0n, I, G), y),
    ar_le_add_r(0n, Impl.owner(y, I, G)),
    Equal.sym(Bool, Nat.is_le(Impl.start(0n, I, G), y), True{},
      Equal.trans(Bool, Nat.is_le(Impl.start(0n, I, G), y),
        Nat.is_lt(Nat.mul(0n, I), Nat.mul(1n+y, G)), True{},
        p1_start_le(0n, y, I, G, gpos), p1_mul_pos(y, G, gpos))))

def p1_le_owner_go(b: Nat, +y: Nat, +I: Nat, +G: Nat,
    ipos: {Nat.is_lt(0n, I) == True{} : Bool},
    gpos: {Nat.is_lt(0n, G) == True{} : Bool})
  -> {Nat.is_le(b, Impl.owner(y, I, G)) == Nat.is_le(Impl.start(b, I, G), y) : Bool}:
  match b:
    case 0n:
      p1_le_owner_zero(y, I, G, gpos)
    case 1n+c:
      p1_le_owner_succ(c, y, I, G, ipos, gpos)

# b <= owner(y) iff start(b) <= y   (I, G >= 1)
def p1_le_owner(+b: Nat, +y: Nat, +I: Nat, +G: Nat,
    ipos: {Nat.is_lt(0n, I) == True{} : Bool},
    gpos: {Nat.is_lt(0n, G) == True{} : Bool})
  -> {Nat.is_le(b, Impl.owner(y, I, G)) == Nat.is_le(Impl.start(b, I, G), y) : Bool}:
  p1_le_owner_go(b, y, I, G, ipos, gpos)

# owner(x) is in the grid for x < I   (G >= 1)
def p1_owner_lt(+x: Nat, +I: Nat, +G: Nat,
    gpos: {Nat.is_lt(0n, G) == True{} : Bool},
    inside: {Nat.is_lt(x, I) == True{} : Bool})
  -> {Nat.is_lt(Impl.owner(x, I, G), G) == True{} : Bool}:
  +gp = gpos
  +lt1 = p1_rt(Nat.is_lt(x, I), Nat.is_le(1n+x, I), ar_lt_succ_le(x, I), inside)
  +g1 = p1_rt(Nat.is_lt(0n, G), Nat.is_le(1n, G), ar_lt_succ_le(0n, G), gp)
  +ipos = ar_lt_le_trans(0n, 1n+x, I, {==}, lt1)
  %p1_le_pred(Impl.owner(x, I, G), G, g1) : {_ == True{} : Bool}
  %Equal.sym(Bool, Nat.is_le(Impl.owner(x, I, G), Nat.sub(G, 1n)), Nat.is_lt(x, Impl.start(1n+Nat.sub(G, 1n), I, G)), p1_owner_le(x, Nat.sub(G, 1n), I, G, ipos, gp)) : {_ == True{} : Bool}
  %Equal.sym(Nat, Nat.add(1n, Nat.sub(G, 1n)), G, ar_sub_add(G, 1n, g1)) : {Nat.is_lt(x, Nat.div(Nat.mul(_, I), G)) == True{} : Bool}
  %Equal.sym(Bool, Nat.is_lt(x, Nat.div(Nat.mul(G, I), G)), Nat.is_le(Nat.mul(1n+x, G), Nat.mul(G, I)), p1_lt_start(x, G, I, G, gp)) : {_ == True{} : Bool}
  %ar_mul_comm(I, G) : {Nat.is_le(Nat.mul(1n+x, G), _) == True{} : Bool}
  ar_le_mul_mono(1n+x, I, G, lt1)

# ==== Group touch = block range meets group range ====

def p1t_os(left: Nat, +x: Nat, +s: Nat, +e: Nat) -> Bool:
  match left:
    case 0n:
      False{}
    case 1n+r:
      Bool.or(Bool.and(Nat.is_le(s, x), Nat.is_lt(x, e)), p1t_os(r, 1n+x, s, e))

# The closed form of p1t_os(1 + l, x, s, e): [s, e) is non-empty and meets [x, x + 1 + l).
def p1t_rng(+l: Nat, +x: Nat, +s: Nat, +e: Nat) -> Bool:
  Bool.and(Nat.is_lt(s, e), Bool.and(Nat.is_lt(s, Nat.add(x, 1n+l)), Nat.is_lt(x, e)))

# owns_some agrees with p1t_os at s = start(b), e = start(b + 1).
def p1t_os_eq(left: Nat, +x: Nat, +b: Nat, +I: Nat, +G: Nat,
    +gpos: {Nat.is_lt(0n, G) == True{} : Bool})
  -> {Spec.owns_some(left, x, b, I, G) ==
    p1t_os(left, x, Impl.start(b, I, G), Impl.start(1n+b, I, G)) : Bool}:
  match left:
    case 0n:
      {==}
    case 1n+r:
      %p1t_os_eq(r, 1n+x, b, I, G, gpos) : {Bool.or(Spec.owns(b, x, I, G), Spec.owns_some(r, 1n+x, b, I, G)) == Bool.or(Bool.and(Nat.is_le(Impl.start(b, I, G), x), Nat.is_lt(x, Impl.start(1n+b, I, G))), _) : Bool}
      %p1_owns_range(b, x, I, G, gpos) : {Bool.or(Spec.owns(b, x, I, G), Spec.owns_some(r, 1n+x, b, I, G)) == Bool.or(_, Spec.owns_some(r, 1n+x, b, I, G)) : Bool}
      {==}

def p1t_and_false_r(a: Bool) -> {False{} == Bool.and(a, False{}) : Bool}:
  match a:
    case False{}:
      {==}
    case True{}:
      {==}

def p1t_lt0(x: Nat) -> {Nat.is_lt(x, 0n) == False{} : Bool}:
  match x:
    case 0n:
      {==}
    case 1n+p:
      {==}

def p1t_lt0_and(s: Nat, -c: Bool) -> {False{} == Bool.and(Nat.is_lt(s, 0n), c) : Bool}:
  match s:
    case 0n:
      {==}
    case 1n+p:
      {==}

# not (x < e) gives not (x + 1 < e).
def p1t_ltf(x: Nat, e: Nat, h: {Nat.is_lt(x, e) == False{} : Bool})
  -> {Nat.is_lt(1n+x, e) == False{} : Bool}:
  match x e:
    case 0n 0n:
      {==}
    case 0n 1n+ep:
      Empty.absurd({Nat.is_lt(1n, 1n+ep) == False{} : Bool},
        ar_false_true(Equal.sym(Bool, True{}, False{}, h)))
    case 1n+xp 0n:
      {==}
    case 1n+xp 1n+ep:
      p1t_ltf(xp, ep, h)

# Past the end of the range nothing is owned.
def p1t_past(left: Nat, +x: Nat, +s: Nat, +e: Nat, +h: {Nat.is_lt(x, e) == False{} : Bool})
  -> {p1t_os(left, x, s, e) == False{} : Bool}:
  match left:
    case 0n:
      {==}
    case 1n+r:
      %Equal.sym(Bool, Nat.is_lt(x, e), False{}, h) : {Bool.or(Bool.and(Nat.is_le(s, x), _), p1t_os(r, 1n+x, s, e)) == False{} : Bool}
      %p1t_and_false_r(Nat.is_le(s, x)) : {Bool.or(_, p1t_os(r, 1n+x, s, e)) == False{} : Bool}
      p1t_past(r, 1n+x, s, e, p1t_ltf(x, e, h))

# Shifting x, s and e together changes nothing.
def p1t_shift(left: Nat, +x: Nat, +s: Nat, +e: Nat)
  -> {p1t_os(left, 1n+x, 1n+s, 1n+e) == p1t_os(left, x, s, e) : Bool}:
  match left:
    case 0n:
      {==}
    case 1n+r:
      %p1t_shift(r, 1n+x, s, e) : {Bool.or(Bool.and(Nat.is_le(s, x), Nat.is_lt(x, e)), p1t_os(r, 2n+x, 1n+s, 1n+e)) == Bool.or(Bool.and(Nat.is_le(s, x), Nat.is_lt(x, e)), _) : Bool}
      {==}

# With s = 0: the head decides, since once x >= e nothing later is owned.
def p1t_absorb(+l: Nat, +x: Nat, +e: Nat, v: Bool, h: {Nat.is_lt(x, e) == v : Bool})
  -> {Bool.or(v, p1t_os(l, 1n+x, 0n, e)) == v : Bool}:
  match v:
    case True{}:
      {==}
    case False{}:
      p1t_past(l, 1n+x, 0n, e, p1t_ltf(x, e, h))

# With x = 0 the last conjunct 0 < e is redundant next to s < e.
def p1t_fix(s: Nat, e: Nat, -d: Bool)
  -> {Bool.and(Nat.is_lt(s, e), Bool.and(d, Nat.is_lt(0n, e))) ==
    Bool.and(Nat.is_lt(s, e), Bool.and(d, True{})) : Bool}:
  match s e:
    case 0n 0n:
      {==}
    case 0n 1n+ep:
      {==}
    case 1n+sp 0n:
      {==}
    case 1n+sp 1n+ep:
      {==}

# Main characterization, by recursion on s (shifting x, s, e down together).
def p1t_main(+s: Nat, +e: Nat, +x: Nat, +l: Nat)
  -> {p1t_os(1n+l, x, s, e) == p1t_rng(l, x, s, e) : Bool}:
  match s e x l:
    case 0n 0n xv lv:
      p1t_past(1n+lv, xv, 0n, 0n, p1t_lt0(xv))
    case 1n+sp 0n xv lv:
      p1t_past(1n+lv, xv, 1n+sp, 0n, p1t_lt0(xv))
    case 0n 1n+ep 0n lv:
      {==}
    case 0n 1n+ep 1n+xp lv:
      p1t_absorb(lv, 1n+xp, 1n+ep, Nat.is_lt(1n+xp, 1n+ep), {==})
    case 1n+sp 1n+ep 0n 0n:
      %Equal.sym(Bool, Nat.is_lt(sp, 0n), False{}, p1t_lt0(sp)) : {False{} == Bool.and(Nat.is_lt(sp, ep), Bool.and(_, True{})) : Bool}
      p1t_and_false_r(Nat.is_lt(sp, ep))
    case 1n+sp 1n+ep 0n 1n+lp:
      Equal.trans(Bool, p1t_os(1n+lp, 1n, 1n+sp, 1n+ep),
        p1t_os(1n+lp, 0n, sp, ep),
        Bool.and(Nat.is_lt(sp, ep), Bool.and(Nat.is_lt(sp, 1n+lp), True{})),
        p1t_shift(1n+lp, 0n, sp, ep),
        Equal.trans(Bool, p1t_os(1n+lp, 0n, sp, ep), p1t_rng(lp, 0n, sp, ep),
          Bool.and(Nat.is_lt(sp, ep), Bool.and(Nat.is_lt(sp, 1n+lp), True{})),
          p1t_main(sp, ep, 0n, lp), p1t_fix(sp, ep, Nat.is_lt(sp, 1n+lp))))
    case 1n+sp 1n+ep 1n+xp lv:
      Equal.trans(Bool, p1t_os(1n+lv, 1n+xp, 1n+sp, 1n+ep), p1t_os(1n+lv, xp, sp, ep),
        p1t_rng(lv, 1n+xp, 1n+sp, 1n+ep),
        p1t_shift(1n+lv, xp, sp, ep), p1t_main(sp, ep, xp, lv))

# Block b touches group g iff its range [start(b), start(b + 1)) is non-empty and meets
# [g * KT, (g + 1) * KT)   (G >= 1; any KT, I)
def p1_touches_range(+b: Nat, +g: Nat, +KT: Nat, +I: Nat, +G: Nat,
    gpos: {Nat.is_lt(0n, G) == True{} : Bool})
  -> {Spec.touches(b, g, KT, I, G) ==
    Bool.and(Nat.is_lt(Impl.start(b, I, G), Impl.start(1n+b, I, G)),
      Bool.and(Nat.is_lt(Impl.start(b, I, G), Nat.mul(1n+g, KT)),
        Nat.is_lt(Nat.mul(g, KT), Impl.start(1n+b, I, G)))) : Bool}:
  match KT:
    case 0n:
      %Equal.sym(Nat, Nat.mul(g, 0n), 0n, ar_mul_zero_r(g)) : {False{} == Bool.and(Nat.is_lt(Impl.start(b, I, G), Impl.start(1n+b, I, G)), Bool.and(Nat.is_lt(Impl.start(b, I, G), _), Nat.is_lt(_, Impl.start(1n+b, I, G)))) : Bool}
      %Equal.sym(Bool, Nat.is_lt(Impl.start(b, I, G), 0n), False{}, p1t_lt0(Impl.start(b, I, G))) : {False{} == Bool.and(Nat.is_lt(Impl.start(b, I, G), Impl.start(1n+b, I, G)), Bool.and(_, Nat.is_lt(0n, Impl.start(1n+b, I, G)))) : Bool}
      p1t_and_false_r(Nat.is_lt(Impl.start(b, I, G), Impl.start(1n+b, I, G)))
    case 1n+k:
      %ar_add_comm(Nat.mul(g, 1n+k), 1n+k) : {Spec.touches(b, g, 1n+k, I, G) == Bool.and(Nat.is_lt(Impl.start(b, I, G), Impl.start(1n+b, I, G)), Bool.and(Nat.is_lt(Impl.start(b, I, G), _), Nat.is_lt(Nat.mul(g, 1n+k), Impl.start(1n+b, I, G)))) : Bool}
      Equal.trans(Bool, Spec.owns_some(1n+k, Nat.mul(g, 1n+k), b, I, G),
        p1t_os(1n+k, Nat.mul(g, 1n+k), Impl.start(b, I, G), Impl.start(1n+b, I, G)),
        p1t_rng(k, Nat.mul(g, 1n+k), Impl.start(b, I, G), Impl.start(1n+b, I, G)),
        p1t_os_eq(1n+k, Nat.mul(g, 1n+k), b, I, G, gpos),
        p1t_main(Impl.start(b, I, G), Impl.start(1n+b, I, G), Nat.mul(g, 1n+k), k))

# ==== Laws owners_exactly_one, range_is_owned, owner_owns ====

def p1l_rt(-a: Bool, -b: Bool, e: {a == b : Bool}, h: {a == True{} : Bool})
  -> {b == True{} : Bool}:
  Equal.trans(Bool, b, a, True{}, Equal.sym(Bool, a, b, e), h)

def p1l_fst(-A: Type, -B: Type, p: A & B) -> A:
  match p:
    case (a, b):
      a

def p1l_zero_le(x: Nat) -> {Nat.is_le(0n, x) == True{} : Bool}:
  match x:
    case 0n:
      {==}
    case 1n+p:
      {==}

def p1l_lt_zero(x: Nat) -> {Nat.is_lt(x, 0n) == False{} : Bool}:
  match x:
    case 0n:
      {==}
    case 1n+p:
      {==}

def p1l_not_true(x: Bool, h: {Bool.not(x) == True{} : Bool}) -> {x == False{} : Bool}:
  match x:
    case True{}:
      Empty.absurd({True{} == False{} : Bool}, ar_false_true(h))
    case False{}:
      {==}

def p1l_not_false(x: Bool, h: {Bool.not(x) == False{} : Bool}) -> {x == True{} : Bool}:
  match x:
    case True{}:
      {==}
    case False{}:
      Empty.absurd({False{} == True{} : Bool}, ar_true_false(h))

# ---- range_is_owned ----

def Laws.range_is_owned(mats, n, k, G, b, x, grid):
  Equal.sym(Bool, Spec.owns(b, x, Impl.total(mats, n, k), G),
    Bool.and(Nat.is_le(Impl.start(b, Impl.total(mats, n, k), G), x),
      Nat.is_lt(x, Impl.start(1n+b, Impl.total(mats, n, k), G))),
    p1_owns_range(b, x, Impl.total(mats, n, k), G, grid))

# ---- owner_owns ----

def p1l_owner_owns(+x: Nat, +I: Nat, +G: Nat,
    +gpos: {Nat.is_lt(0n, G) == True{} : Bool},
    +inside: {Nat.is_lt(x, I) == True{} : Bool})
  -> {Spec.owns(Impl.owner(x, I, G), x, I, G) == True{} : Bool}:
  +ipos = ar_le_lt_trans(0n, x, I, p1l_zero_le(x), inside)
  +o = Impl.owner(x, I, G)
  %Equal.sym(Bool, Spec.owns(o, x, I, G), Bool.and(Nat.is_le(Impl.start(o, I, G), x), Nat.is_lt(x, Impl.start(1n+o, I, G))), p1_owns_range(o, x, I, G, gpos)) : {_ == True{} : Bool}
  ar_and_intro(Nat.is_le(Impl.start(o, I, G), x), Nat.is_lt(x, Impl.start(1n+o, I, G)),
    p1l_rt(Nat.is_le(o, o), Nat.is_le(Impl.start(o, I, G), x),
      p1_le_owner(o, x, I, G, ipos, gpos), ar_le_refl(o)),
    p1l_rt(Nat.is_le(o, o), Nat.is_lt(x, Impl.start(1n+o, I, G)),
      p1_owner_le(x, o, I, G, ipos, gpos), ar_le_refl(o)))

def Laws.owner_owns(mats, n, k, G, x, grid, inside):
  +gp = grid
  +ins = inside
  (p1l_owner_owns(x, Impl.total(mats, n, k), G, gp, ins),
    p1_owner_lt(x, Impl.total(mats, n, k), G, gp, ins))

# ---- owners_exactly_one ----

# One telescoping step: [x < s0] + [s0 <= x < s1] == [x < s1] when s0 <= s1.
def p1l_bit_step_go(a: Bool, +x: Nat, +s0: Nat, +s1: Nat,
    mono: {Nat.is_le(s0, s1) == True{} : Bool},
    +e: {Nat.is_lt(x, s0) == a : Bool})
  -> {Nat.add(Spec.bit(a), Spec.bit(Bool.and(Nat.is_le(s0, x), Nat.is_lt(x, s1)))) ==
    Spec.bit(Nat.is_lt(x, s1)) : Nat}:
  match a:
    case True{}:
      +nle = p1l_not_true(Nat.is_le(s0, x),
        p1l_rt(Nat.is_lt(x, s0), Bool.not(Nat.is_le(s0, x)), ar_lt_not_le(x, s0), e))
      +lt1 = ar_lt_le_trans(x, s0, s1, e, mono)
      %Equal.sym(Bool, Nat.is_le(s0, x), False{}, nle) : {Nat.add(1n, Spec.bit(Bool.and(_, Nat.is_lt(x, s1)))) == Spec.bit(Nat.is_lt(x, s1)) : Nat}
      %Equal.sym(Bool, Nat.is_lt(x, s1), True{}, lt1) : {Nat.add(1n, Spec.bit(Bool.and(False{}, Nat.is_lt(x, s1)))) == Spec.bit(_) : Nat}
      {==}
    case False{}:
      +le = p1l_not_false(Nat.is_le(s0, x),
        Equal.trans(Bool, Bool.not(Nat.is_le(s0, x)), Nat.is_lt(x, s0), False{},
          Equal.sym(Bool, Nat.is_lt(x, s0), Bool.not(Nat.is_le(s0, x)),
            ar_lt_not_le(x, s0)),
          e))
      %Equal.sym(Bool, Nat.is_le(s0, x), True{}, le) : {Nat.add(0n, Spec.bit(Bool.and(_, Nat.is_lt(x, s1)))) == Spec.bit(Nat.is_lt(x, s1)) : Nat}
      {==}

# owners(c, x) == [x < start(c)] for every c (G >= 1; any I).
def p1l_owners_start(+c: Nat, +x: Nat, +I: Nat, +G: Nat,
    +gpos: {Nat.is_lt(0n, G) == True{} : Bool})
  -> {Spec.owners(c, x, I, G) == Spec.bit(Nat.is_lt(x, Impl.start(c, I, G))) : Nat}:
  match c:
    case 0n:
      %Equal.sym(Nat, Nat.div(0n, G), 0n, p1l_fst({Nat.div(0n, G) == 0n : Nat}, {Nat.mod(0n, G) == 0n : Nat}, ar_div_unique(0n, 0n, G, gpos))) : {0n == Spec.bit(Nat.is_lt(x, _)) : Nat}
      %Equal.sym(Bool, Nat.is_lt(x, 0n), False{}, p1l_lt_zero(x)) : {0n == Spec.bit(_) : Nat}
      {==}
    case 1n+b:
      %Equal.sym(Nat, Spec.owners(b, x, I, G), Spec.bit(Nat.is_lt(x, Impl.start(b, I, G))), p1l_owners_start(b, x, I, G, gpos)) : {Nat.add(_, Spec.bit(Spec.owns(b, x, I, G))) == Spec.bit(Nat.is_lt(x, Impl.start(1n+b, I, G))) : Nat}
      %Equal.sym(Bool, Spec.owns(b, x, I, G), Bool.and(Nat.is_le(Impl.start(b, I, G), x), Nat.is_lt(x, Impl.start(1n+b, I, G))), p1_owns_range(b, x, I, G, gpos)) : {Nat.add(Spec.bit(Nat.is_lt(x, Impl.start(b, I, G))), Spec.bit(_)) == Spec.bit(Nat.is_lt(x, Impl.start(1n+b, I, G))) : Nat}
      p1l_bit_step_go(Nat.is_lt(x, Impl.start(b, I, G)), x, Impl.start(b, I, G),
        Impl.start(1n+b, I, G),
        ar_div_mono(Nat.mul(b, I), Nat.mul(1n+b, I), G, gpos,
          ar_le_add_l(I, Nat.mul(b, I))),
        {==})

# start(G) = G * I / G = I.
def p1l_start_full(+I: Nat, +G: Nat, gpos: {Nat.is_lt(0n, G) == True{} : Bool})
  -> {Impl.start(G, I, G) == I : Nat}:
  %ar_mul_comm(I, G) : {Nat.div(_, G) == I : Nat}
  %ar_add_zero_r(Nat.mul(I, G)) : {Nat.div(_, G) == I : Nat}
  p1l_fst({Nat.div(Nat.add(Nat.mul(I, G), 0n), G) == I : Nat},
    {Nat.mod(Nat.add(Nat.mul(I, G), 0n), G) == 0n : Nat},
    ar_div_unique(I, 0n, G, gpos))

def Laws.owners_exactly_one(mats, n, k, G, x, grid, inside):
  +gp = grid
  %Equal.sym(Nat, Spec.owners(G, x, Impl.total(mats, n, k), G), Spec.bit(Nat.is_lt(x, Impl.start(G, Impl.total(mats, n, k), G))), p1l_owners_start(G, x, Impl.total(mats, n, k), G, gp)) : {_ == 1n : Nat}
  %Equal.sym(Nat, Impl.start(G, Impl.total(mats, n, k), G), Impl.total(mats, n, k), p1l_start_full(Impl.total(mats, n, k), G, gp)) : {Spec.bit(Nat.is_lt(x, _)) == 1n : Nat}
  %Equal.sym(Bool, Nat.is_lt(x, Impl.total(mats, n, k)), True{}, inside) : {Spec.bit(_) == 1n : Nat}
  {==}

# ---- shared helpers (used by later parts) ----

# KT = k / 16 >= 8 > 0 for k >= 128.
def p1l_kt_pos(+k: Nat, shaped: {Nat.is_le(128n, k) == True{} : Bool})
  -> {Nat.is_lt(0n, Impl.kt(k)) == True{} : Bool}:
  ar_lt_le_trans(0n, 8n, Impl.kt(k), {==},
    p1l_rt(Nat.is_le(Nat.mul(8n, 16n), k), Nat.is_le(8n, Nat.div(k, 16n)),
      ar_div_le(8n, k, 16n, {==}), shaped))

# s <= m - 1 iff s < m, for m >= 1.
def p1l_le_pred(+s: Nat, +m: Nat, pos: {Nat.is_le(1n, m) == True{} : Bool})
  -> {Nat.is_le(s, Nat.sub(m, 1n)) == Nat.is_lt(s, m) : Bool}:
  %ar_sub_add(m, 1n, pos) : {Nat.is_le(s, Nat.sub(m, 1n)) == Nat.is_lt(s, _) : Bool}
  Equal.sym(Bool, Nat.is_lt(s, 1n+Nat.sub(m, 1n)), Nat.is_le(1n+s, 1n+Nat.sub(m, 1n)),
    ar_lt_succ_le(s, 1n+Nat.sub(m, 1n)))

# ==== Laws contributors_contiguous, slot_in_range, slot_injective ====

def p2_rt(-a: Bool, -b: Bool, e: {a == b : Bool}, h: {a == True{} : Bool})
  -> {b == True{} : Bool}:
  Equal.trans(Bool, b, a, True{}, Equal.sym(Bool, a, b, e), h)

def p2_tr(-a: Bool, -b: Bool, e: {a == b : Bool}, h: {b == True{} : Bool})
  -> {a == True{} : Bool}:
  Equal.trans(Bool, a, b, True{}, e, h)

def p2_and_true_l(a: Bool, -b: Bool, h: {a == True{} : Bool})
  -> {Bool.and(a, b) == b : Bool}:
  match a:
    case False{}:
      Empty.absurd({False{} == b : Bool}, ar_false_true(h))
    case True{}:
      {==}

def p2_and_comm(a: Bool, b: Bool) -> {Bool.and(a, b) == Bool.and(b, a) : Bool}:
  match a b:
    case False{} False{}:
      {==}
    case False{} True{}:
      {==}
    case True{} False{}:
      {==}
    case True{} True{}:
      {==}

def p2_not_false(x: Bool, h: {Bool.not(x) == False{} : Bool}) -> {x == True{} : Bool}:
  match x:
    case True{}:
      {==}
    case False{}:
      Empty.absurd({False{} == True{} : Bool}, ar_true_false(h))

# ---- Nat helpers ----

def p2_lt_succ(a: Nat) -> {Nat.is_lt(a, 1n+a) == True{} : Bool}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      p2_lt_succ(p)

# KT = k / 16 >= 8 > 0 for k >= 128.
def p2_kt_pos(+k: Nat, shaped: {Nat.is_le(128n, k) == True{} : Bool})
  -> {Nat.is_lt(0n, Impl.kt(k)) == True{} : Bool}:
  ar_lt_le_trans(0n, 8n, Impl.kt(k), {==},
    p2_rt(Nat.is_le(Nat.mul(8n, 16n), k), Nat.is_le(8n, Nat.div(k, 16n)),
      ar_div_le(8n, k, 16n, {==}), shaped))

def p2_m_pos(+g: Nat, +KT: Nat, kpos: {Nat.is_lt(0n, KT) == True{} : Bool})
  -> {Nat.is_le(1n, Nat.mul(1n+g, KT)) == True{} : Bool}:
  ar_le_trans(1n, KT, Nat.mul(1n+g, KT),
    p2_rt(Nat.is_lt(0n, KT), Nat.is_le(1n, KT), ar_lt_succ_le(0n, KT), kpos),
    ar_le_add_r(KT, Nat.mul(g, KT)))

# g < Gr gives (g + 1) * m <= Gr * m.
def p2_m_le(+g: Nat, +Gr: Nat, +KT: Nat, group: {Nat.is_lt(g, Gr) == True{} : Bool})
  -> {Nat.is_le(Nat.mul(1n+g, KT), Nat.mul(Gr, KT)) == True{} : Bool}:
  ar_le_mul_mono(1n+g, Gr, KT,
    p2_rt(Nat.is_lt(g, Gr), Nat.is_le(1n+g, Gr), ar_lt_succ_le(g, Gr), group))

# s <= m - 1 iff s < m, for m >= 1.
def p2_le_pred(+s: Nat, +m: Nat, pos: {Nat.is_le(1n, m) == True{} : Bool})
  -> {Nat.is_le(s, Nat.sub(m, 1n)) == Nat.is_lt(s, m) : Bool}:
  %ar_sub_add(m, 1n, pos) : {Nat.is_le(s, Nat.sub(m, 1n)) == Nat.is_lt(s, _) : Bool}
  Equal.sym(Bool, Nat.is_lt(s, 1n+Nat.sub(m, 1n)), Nat.is_le(1n+s, 1n+Nat.sub(m, 1n)),
    ar_lt_succ_le(s, 1n+Nat.sub(m, 1n)))

def p2_gk_lt_m(+g: Nat, +KT: Nat, kpos: {Nat.is_lt(0n, KT) == True{} : Bool})
  -> {Nat.is_lt(Nat.mul(g, KT), Nat.mul(1n+g, KT)) == True{} : Bool}:
  p2_tr(Nat.is_lt(Nat.mul(g, KT), Nat.mul(1n+g, KT)), Nat.is_lt(g, 1n+g),
    ar_mul_lt_cancel(g, 1n+g, KT, kpos), p2_lt_succ(g))

def p2_gk_le_xhi(+g: Nat, +KT: Nat, +kpos: {Nat.is_lt(0n, KT) == True{} : Bool})
  -> {Nat.is_le(Nat.mul(g, KT), Nat.sub(Nat.mul(1n+g, KT), 1n)) == True{} : Bool}:
  p2_tr(Nat.is_le(Nat.mul(g, KT), Nat.sub(Nat.mul(1n+g, KT), 1n)),
    Nat.is_lt(Nat.mul(g, KT), Nat.mul(1n+g, KT)),
    p2_le_pred(Nat.mul(g, KT), Nat.mul(1n+g, KT), p2_m_pos(g, KT, kpos)),
    p2_gk_lt_m(g, KT, kpos))

# (g + 1) * KT - 1 < I when (g + 1) * KT <= I.
def p2_xhi_lt(+g: Nat, +KT: Nat, +I: Nat, kpos: {Nat.is_lt(0n, KT) == True{} : Bool},
    mle: {Nat.is_le(Nat.mul(1n+g, KT), I) == True{} : Bool})
  -> {Nat.is_lt(Nat.sub(Nat.mul(1n+g, KT), 1n), I) == True{} : Bool}:
  ar_lt_le_trans(Nat.sub(Nat.mul(1n+g, KT), 1n), Nat.mul(1n+g, KT), I,
    p2_tr(Nat.is_lt(Nat.sub(Nat.mul(1n+g, KT), 1n), Nat.mul(1n+g, KT)),
      Nat.is_le(Nat.sub(Nat.mul(1n+g, KT), 1n), Nat.sub(Nat.mul(1n+g, KT), 1n)),
      Equal.sym(Bool,
        Nat.is_le(Nat.sub(Nat.mul(1n+g, KT), 1n), Nat.sub(Nat.mul(1n+g, KT), 1n)),
        Nat.is_lt(Nat.sub(Nat.mul(1n+g, KT), 1n), Nat.mul(1n+g, KT)),
        p2_le_pred(Nat.sub(Nat.mul(1n+g, KT), 1n), Nat.mul(1n+g, KT), p2_m_pos(g, KT, kpos))),
      ar_le_refl(Nat.sub(Nat.mul(1n+g, KT), 1n))),
    mle)

# b * I + G <= (b + 1) * I when G <= I.
def p2_ne_step(+b: Nat, +I: Nat, +G: Nat, fits: {Nat.is_le(G, I) == True{} : Bool})
  -> {Nat.is_le(Nat.add(Nat.mul(b, I), G), Nat.mul(1n+b, I)) == True{} : Bool}:
  %ar_add_comm(Nat.mul(b, I), I) : {Nat.is_le(Nat.add(Nat.mul(b, I), G), _) == True{} : Bool}
  p2_rt(Nat.is_le(G, I), Nat.is_le(Nat.add(Nat.mul(b, I), G), Nat.add(Nat.mul(b, I), I)),
    Equal.sym(Bool, Nat.is_le(Nat.add(Nat.mul(b, I), G), Nat.add(Nat.mul(b, I), I)),
      Nat.is_le(G, I), ar_le_add_cancel(Nat.mul(b, I), G, I)),
    fits)

# Under G <= I every block range is non-empty: start(b) < start(b + 1).
def p2_nonempty(+b: Nat, +I: Nat, +G: Nat, +gpos: {Nat.is_lt(0n, G) == True{} : Bool},
    fits: {Nat.is_le(G, I) == True{} : Bool})
  -> {Nat.is_lt(Impl.start(b, I, G), Impl.start(1n+b, I, G)) == True{} : Bool}:
  %Equal.sym(Bool, Nat.is_lt(Impl.start(b, I, G), Impl.start(1n+b, I, G)), Nat.is_le(1n+Impl.start(b, I, G), Impl.start(1n+b, I, G)), ar_lt_succ_le(Impl.start(b, I, G), Impl.start(1n+b, I, G))) : {_ == True{} : Bool}
  %ar_div_add_self(Nat.mul(b, I), G, gpos) : {Nat.is_le(_, Impl.start(1n+b, I, G)) == True{} : Bool}
  ar_div_mono(Nat.add(Nat.mul(b, I), G), Nat.mul(1n+b, I), G, gpos,
    p2_ne_step(b, I, G, fits))

# ---- contributors_contiguous, generic in KT, I, G ----

def p2_contig1(+b: Nat, +g: Nat, +KT: Nat, +I: Nat, +G: Nat,
    +gpos: {Nat.is_lt(0n, G) == True{} : Bool},
    +fits: {Nat.is_le(G, I) == True{} : Bool},
    kpos: {Nat.is_lt(0n, KT) == True{} : Bool})
  -> {Spec.touches(b, g, KT, I, G) ==
    Bool.and(Nat.is_le(Impl.owner(Nat.mul(g, KT), I, G), b),
      Nat.is_le(b, Impl.owner(Nat.sub(Nat.mul(1n+g, KT), 1n), I, G))) : Bool}:
  +ipos = ar_lt_le_trans(0n, G, I, gpos, fits)
  %Equal.sym(Bool, Nat.is_le(Impl.owner(Nat.mul(g, KT), I, G), b), Nat.is_lt(Nat.mul(g, KT), Impl.start(1n+b, I, G)), p1_owner_le(Nat.mul(g, KT), b, I, G, ipos, gpos)) : {Spec.touches(b, g, KT, I, G) == Bool.and(_, Nat.is_le(b, Impl.owner(Nat.sub(Nat.mul(1n+g, KT), 1n), I, G))) : Bool}
  %Equal.trans(Bool, Nat.is_lt(Impl.start(b, I, G), Nat.mul(1n+g, KT)), Nat.is_le(Impl.start(b, I, G), Nat.sub(Nat.mul(1n+g, KT), 1n)), Nat.is_le(b, Impl.owner(Nat.sub(Nat.mul(1n+g, KT), 1n), I, G)), Equal.sym(Bool, Nat.is_le(Impl.start(b, I, G), Nat.sub(Nat.mul(1n+g, KT), 1n)), Nat.is_lt(Impl.start(b, I, G), Nat.mul(1n+g, KT)), p2_le_pred(Impl.start(b, I, G), Nat.mul(1n+g, KT), p2_m_pos(g, KT, kpos))), Equal.sym(Bool, Nat.is_le(b, Impl.owner(Nat.sub(Nat.mul(1n+g, KT), 1n), I, G)), Nat.is_le(Impl.start(b, I, G), Nat.sub(Nat.mul(1n+g, KT), 1n)), p1_le_owner(b, Nat.sub(Nat.mul(1n+g, KT), 1n), I, G, ipos, gpos))) : {Spec.touches(b, g, KT, I, G) == Bool.and(Nat.is_lt(Nat.mul(g, KT), Impl.start(1n+b, I, G)), _) : Bool}
  %p2_and_comm(Nat.is_lt(Impl.start(b, I, G), Nat.mul(1n+g, KT)), Nat.is_lt(Nat.mul(g, KT), Impl.start(1n+b, I, G))) : {Spec.touches(b, g, KT, I, G) == _ : Bool}
  Equal.trans(Bool, Spec.touches(b, g, KT, I, G),
    Bool.and(Nat.is_lt(Impl.start(b, I, G), Impl.start(1n+b, I, G)),
      Bool.and(Nat.is_lt(Impl.start(b, I, G), Nat.mul(1n+g, KT)),
        Nat.is_lt(Nat.mul(g, KT), Impl.start(1n+b, I, G)))),
    Bool.and(Nat.is_lt(Impl.start(b, I, G), Nat.mul(1n+g, KT)),
      Nat.is_lt(Nat.mul(g, KT), Impl.start(1n+b, I, G))),
    p1_touches_range(b, g, KT, I, G, gpos),
    p2_and_true_l(Nat.is_lt(Impl.start(b, I, G), Impl.start(1n+b, I, G)),
      Bool.and(Nat.is_lt(Impl.start(b, I, G), Nat.mul(1n+g, KT)),
        Nat.is_lt(Nat.mul(g, KT), Impl.start(1n+b, I, G))),
      p2_nonempty(b, I, G, gpos, fits)))

def p2_lo_le_hi(+g: Nat, +KT: Nat, +I: Nat, +G: Nat,
    +gpos: {Nat.is_lt(0n, G) == True{} : Bool},
    fits: {Nat.is_le(G, I) == True{} : Bool},
    kpos: {Nat.is_lt(0n, KT) == True{} : Bool})
  -> {Nat.is_le(Impl.owner(Nat.mul(g, KT), I, G),
    Impl.owner(Nat.sub(Nat.mul(1n+g, KT), 1n), I, G)) == True{} : Bool}:
  +ipos = ar_lt_le_trans(0n, G, I, gpos, fits)
  p2_rt(Nat.is_lt(Nat.mul(g, KT), Impl.start(1n+Impl.owner(Nat.sub(Nat.mul(1n+g, KT), 1n), I, G), I, G)),
    Nat.is_le(Impl.owner(Nat.mul(g, KT), I, G), Impl.owner(Nat.sub(Nat.mul(1n+g, KT), 1n), I, G)),
    Equal.sym(Bool,
      Nat.is_le(Impl.owner(Nat.mul(g, KT), I, G), Impl.owner(Nat.sub(Nat.mul(1n+g, KT), 1n), I, G)),
      Nat.is_lt(Nat.mul(g, KT), Impl.start(1n+Impl.owner(Nat.sub(Nat.mul(1n+g, KT), 1n), I, G), I, G)),
      p1_owner_le(Nat.mul(g, KT), Impl.owner(Nat.sub(Nat.mul(1n+g, KT), 1n), I, G), I, G, ipos, gpos)),
    ar_le_lt_trans(Nat.mul(g, KT), Nat.sub(Nat.mul(1n+g, KT), 1n),
      Impl.start(1n+Impl.owner(Nat.sub(Nat.mul(1n+g, KT), 1n), I, G), I, G),
      p2_gk_le_xhi(g, KT, kpos),
      p2_rt(Nat.is_le(Impl.owner(Nat.sub(Nat.mul(1n+g, KT), 1n), I, G), Impl.owner(Nat.sub(Nat.mul(1n+g, KT), 1n), I, G)),
        Nat.is_lt(Nat.sub(Nat.mul(1n+g, KT), 1n), Impl.start(1n+Impl.owner(Nat.sub(Nat.mul(1n+g, KT), 1n), I, G), I, G)),
        p1_owner_le(Nat.sub(Nat.mul(1n+g, KT), 1n), Impl.owner(Nat.sub(Nat.mul(1n+g, KT), 1n), I, G), I, G, ipos, gpos),
        ar_le_refl(Impl.owner(Nat.sub(Nat.mul(1n+g, KT), 1n), I, G)))))

def p2_hi_lt(+g: Nat, +KT: Nat, +I: Nat, +G: Nat,
    gpos: {Nat.is_lt(0n, G) == True{} : Bool},
    kpos: {Nat.is_lt(0n, KT) == True{} : Bool},
    mle: {Nat.is_le(Nat.mul(1n+g, KT), I) == True{} : Bool})
  -> {Nat.is_lt(Impl.owner(Nat.sub(Nat.mul(1n+g, KT), 1n), I, G), G) == True{} : Bool}:
  p1_owner_lt(Nat.sub(Nat.mul(1n+g, KT), 1n), I, G, gpos, p2_xhi_lt(g, KT, I, kpos, mle))

def Laws.contributors_contiguous(mats, n, k, G, g, b, grid, shaped, fits, group):
  +gp = grid
  +fp = fits
  +kp = p2_kt_pos(k, shaped)
  (p2_contig1(b, g, Impl.kt(k), Impl.total(mats, n, k), G, gp, fp, kp),
    (p2_lo_le_hi(g, Impl.kt(k), Impl.total(mats, n, k), G, gp, fp, kp),
      p2_hi_lt(g, Impl.kt(k), Impl.total(mats, n, k), G, gp, kp,
        p2_m_le(g, Impl.groups(mats, n), Impl.kt(k), group))))

# ---- slot_in_range / slot_injective, generic in KT, I, G ----

# x <= y gives x < y + 1.
def p2_lt_add1(+x: Nat, +y: Nat, h: {Nat.is_le(x, y) == True{} : Bool})
  -> {Nat.is_lt(x, Nat.add(y, 1n)) == True{} : Bool}:
  %Equal.sym(Nat, Nat.add(y, 1n), Nat.add(1n, y), ar_add_comm(y, 1n)) : {Nat.is_lt(x, _) == True{} : Bool}
  p2_tr(Nat.is_lt(x, 1n+y), Nat.is_le(1n+x, 1n+y), ar_lt_succ_le(x, 1n+y), h)

def p2_le_hi(+b: Nat, +lo: Nat, +hi: Nat, lohi: {Nat.is_le(lo, hi) == True{} : Bool},
    bhi: {Nat.is_le(b, hi) == True{} : Bool})
  -> {Nat.is_le(b, Nat.add(lo, Nat.sub(hi, lo))) == True{} : Bool}:
  %Equal.sym(Nat, Nat.add(lo, Nat.sub(hi, lo)), hi, ar_sub_add(hi, lo, lohi)) : {Nat.is_le(b, _) == True{} : Bool}
  bhi

# lo <= b <= hi gives b - lo <= hi - lo.
def p2_off_le(+b: Nat, +lo: Nat, +hi: Nat, lob: {Nat.is_le(lo, b) == True{} : Bool},
    +bhi: {Nat.is_le(b, hi) == True{} : Bool})
  -> {Nat.is_le(Nat.sub(b, lo), Nat.sub(hi, lo)) == True{} : Bool}:
  p2_tr(Nat.is_le(Nat.sub(b, lo), Nat.sub(hi, lo)), Nat.is_le(b, Nat.add(lo, Nat.sub(hi, lo))),
    ar_le_sub_iff(b, lo, Nat.sub(hi, lo)),
    p2_le_hi(b, lo, hi, ar_le_trans(lo, b, hi, lob, bhi), bhi))

# A contributor's slot b - b_lo is below nc.
def p2_off_lt_nc(+b: Nat, +g: Nat, +KT: Nat, +I: Nat, +G: Nat,
    lob: {Nat.is_le(Impl.b_lo(g, KT, I, G), b) == True{} : Bool},
    bhi: {Nat.is_le(b, Impl.b_hi(g, KT, I, G)) == True{} : Bool})
  -> {Nat.is_lt(Nat.sub(b, Impl.b_lo(g, KT, I, G)), Impl.nc(g, KT, I, G)) == True{} : Bool}:
  p2_lt_add1(Nat.sub(b, Impl.b_lo(g, KT, I, G)), Nat.sub(Impl.b_hi(g, KT, I, G), Impl.b_lo(g, KT, I, G)),
    p2_off_le(b, Impl.b_lo(g, KT, I, G), Impl.b_hi(g, KT, I, G), lob, bhi))

# max_contrib.pick keeps the larger of n and mc.
def p2_pick_n_go(bigger: Bool, +n: Nat, +mc: Nat, e: {Nat.is_gt(n, mc) == bigger : Bool})
  -> {Nat.is_le(n, Impl.max_contrib.pick(bigger, n, mc)) == True{} : Bool}:
  match bigger:
    case True{}:
      ar_le_refl(n)
    case False{}:
      p2_not_false(Nat.is_le(n, mc),
        Equal.trans(Bool, Bool.not(Nat.is_le(n, mc)), Nat.is_gt(n, mc), False{},
          Equal.sym(Bool, Nat.is_gt(n, mc), Bool.not(Nat.is_le(n, mc)),
            Equal.trans(Bool, Nat.is_gt(n, mc), Nat.is_lt(mc, n), Bool.not(Nat.is_le(n, mc)),
              ar_gt_flip(n, mc), ar_lt_not_le(mc, n))),
          e))

def p2_pick_mc_go(bigger: Bool, +n: Nat, +mc: Nat, e: {Nat.is_gt(n, mc) == bigger : Bool})
  -> {Nat.is_le(mc, Impl.max_contrib.pick(bigger, n, mc)) == True{} : Bool}:
  match bigger:
    case True{}:
      ar_lt_le(mc, n,
        Equal.trans(Bool, Nat.is_lt(mc, n), Nat.is_gt(n, mc), True{},
          Equal.sym(Bool, Nat.is_gt(n, mc), Nat.is_lt(mc, n), ar_gt_flip(n, mc)), e))
    case False{}:
      ar_le_refl(mc)

# The running maximum never drops below its accumulator.
def p2_mc_go_mc(+left: Nat, +g0: Nat, +KT: Nat, +I: Nat, +G: Nat, +mc: Nat)
  -> {Nat.is_le(mc, Impl.max_contrib.go(left, g0, KT, I, G, mc)) == True{} : Bool}:
  match left:
    case 0n:
      ar_le_refl(mc)
    case 1n+rest:
      ar_le_trans(mc, Impl.max_contrib.pick(Nat.is_gt(Impl.nc(g0, KT, I, G), mc), Impl.nc(g0, KT, I, G), mc),
        Impl.max_contrib.go(rest, 1n+g0, KT, I, G, Impl.max_contrib.pick(Nat.is_gt(Impl.nc(g0, KT, I, G), mc), Impl.nc(g0, KT, I, G), mc)),
        p2_pick_mc_go(Nat.is_gt(Impl.nc(g0, KT, I, G), mc), Impl.nc(g0, KT, I, G), mc, {==}),
        p2_mc_go_mc(rest, 1n+g0, KT, I, G, Impl.max_contrib.pick(Nat.is_gt(Impl.nc(g0, KT, I, G), mc), Impl.nc(g0, KT, I, G), mc)))

# The running maximum over groups [g0, g0 + left) bounds nc(d + g0) for every d < left.
def p2_mc_go_nc(+d: Nat, +left: Nat, +g0: Nat, +KT: Nat, +I: Nat, +G: Nat, +mc: Nat,
    small: {Nat.is_lt(d, left) == True{} : Bool})
  -> {Nat.is_le(Impl.nc(Nat.add(d, g0), KT, I, G), Impl.max_contrib.go(left, g0, KT, I, G, mc))
    == True{} : Bool}:
  match d left:
    case 0n 0n:
      Empty.absurd({Nat.is_le(Impl.nc(g0, KT, I, G), mc) == True{} : Bool}, ar_false_true(small))
    case 1n+dp 0n:
      Empty.absurd({Nat.is_le(Impl.nc(1n+Nat.add(dp, g0), KT, I, G), mc) == True{} : Bool},
        ar_false_true(small))
    case 0n 1n+rest:
      ar_le_trans(Impl.nc(g0, KT, I, G), Impl.max_contrib.pick(Nat.is_gt(Impl.nc(g0, KT, I, G), mc), Impl.nc(g0, KT, I, G), mc),
        Impl.max_contrib.go(rest, 1n+g0, KT, I, G, Impl.max_contrib.pick(Nat.is_gt(Impl.nc(g0, KT, I, G), mc), Impl.nc(g0, KT, I, G), mc)),
        p2_pick_n_go(Nat.is_gt(Impl.nc(g0, KT, I, G), mc), Impl.nc(g0, KT, I, G), mc, {==}),
        p2_mc_go_mc(rest, 1n+g0, KT, I, G, Impl.max_contrib.pick(Nat.is_gt(Impl.nc(g0, KT, I, G), mc), Impl.nc(g0, KT, I, G), mc)))
    case 1n+dp 1n+rest:
      %ar_add_succ_r(dp, g0) : {Nat.is_le(Impl.nc(_, KT, I, G), Impl.max_contrib.go(rest, 1n+g0, KT, I, G, Impl.max_contrib.pick(Nat.is_gt(Impl.nc(g0, KT, I, G), mc), Impl.nc(g0, KT, I, G), mc))) == True{} : Bool}
      p2_mc_go_nc(dp, rest, 1n+g0, KT, I, G, Impl.max_contrib.pick(Nat.is_gt(Impl.nc(g0, KT, I, G), mc), Impl.nc(g0, KT, I, G), mc), small)

# nc(g) <= max_contrib for every group g < Gr.
def p2_nc_le_mc(+g: Nat, +Gr: Nat, +KT: Nat, +I: Nat, +G: Nat,
    group: {Nat.is_lt(g, Gr) == True{} : Bool})
  -> {Nat.is_le(Impl.nc(g, KT, I, G), Impl.max_contrib(Gr, KT, I, G)) == True{} : Bool}:
  %ar_add_zero_r(g) : {Nat.is_le(Impl.nc(_, KT, I, G), Impl.max_contrib(Gr, KT, I, G)) == True{} : Bool}
  p2_mc_go_nc(g, Gr, 0n, KT, I, G, 1n, group)

# g * mc + r < (g + 1) * mc when r < mc.
def p2_addr_step(+g: Nat, +mc: Nat, +r: Nat, rlt: {Nat.is_lt(r, mc) == True{} : Bool})
  -> {Nat.is_lt(Nat.add(Nat.mul(g, mc), r), Nat.mul(1n+g, mc)) == True{} : Bool}:
  %ar_add_comm(Nat.mul(g, mc), mc) : {Nat.is_lt(Nat.add(Nat.mul(g, mc), r), _) == True{} : Bool}
  p2_rt(Nat.is_lt(r, mc), Nat.is_lt(Nat.add(Nat.mul(g, mc), r), Nat.add(Nat.mul(g, mc), mc)),
    Equal.sym(Bool, Nat.is_lt(Nat.add(Nat.mul(g, mc), r), Nat.add(Nat.mul(g, mc), mc)),
      Nat.is_lt(r, mc), ar_lt_add_cancel(Nat.mul(g, mc), r, mc)),
    rlt)

def p2_addr_lt(+g: Nat, +Gr: Nat, +mc: Nat, +r: Nat, rlt: {Nat.is_lt(r, mc) == True{} : Bool},
    group: {Nat.is_lt(g, Gr) == True{} : Bool})
  -> {Nat.is_lt(Nat.add(Nat.mul(g, mc), r), Nat.mul(Gr, mc)) == True{} : Bool}:
  ar_lt_le_trans(Nat.add(Nat.mul(g, mc), r), Nat.mul(1n+g, mc), Nat.mul(Gr, mc),
    p2_addr_step(g, mc, r, rlt), p2_m_le(g, Gr, mc, group))

# A touching block lies in [b_lo(g), b_hi(g)].
def p2_touch_and(+b: Nat, +g: Nat, +KT: Nat, +I: Nat, +G: Nat,
    +gpos: {Nat.is_lt(0n, G) == True{} : Bool},
    +fits: {Nat.is_le(G, I) == True{} : Bool},
    kpos: {Nat.is_lt(0n, KT) == True{} : Bool},
    touch: {Spec.touches(b, g, KT, I, G) == True{} : Bool})
  -> {Bool.and(Nat.is_le(Impl.b_lo(g, KT, I, G), b), Nat.is_le(b, Impl.b_hi(g, KT, I, G))) == True{} : Bool}:
  p2_rt(Spec.touches(b, g, KT, I, G),
    Bool.and(Nat.is_le(Impl.b_lo(g, KT, I, G), b), Nat.is_le(b, Impl.b_hi(g, KT, I, G))),
    p2_contig1(b, g, KT, I, G, gpos, fits, kpos), touch)

def p2_lo_le_b(+b: Nat, +g: Nat, +KT: Nat, +I: Nat, +G: Nat,
    +gpos: {Nat.is_lt(0n, G) == True{} : Bool},
    +fits: {Nat.is_le(G, I) == True{} : Bool},
    kpos: {Nat.is_lt(0n, KT) == True{} : Bool},
    touch: {Spec.touches(b, g, KT, I, G) == True{} : Bool})
  -> {Nat.is_le(Impl.b_lo(g, KT, I, G), b) == True{} : Bool}:
  ar_and_l(Nat.is_le(Impl.b_lo(g, KT, I, G), b), Nat.is_le(b, Impl.b_hi(g, KT, I, G)),
    p2_touch_and(b, g, KT, I, G, gpos, fits, kpos, touch))

def p2_b_le_hi(+b: Nat, +g: Nat, +KT: Nat, +I: Nat, +G: Nat,
    +gpos: {Nat.is_lt(0n, G) == True{} : Bool},
    +fits: {Nat.is_le(G, I) == True{} : Bool},
    kpos: {Nat.is_lt(0n, KT) == True{} : Bool},
    touch: {Spec.touches(b, g, KT, I, G) == True{} : Bool})
  -> {Nat.is_le(b, Impl.b_hi(g, KT, I, G)) == True{} : Bool}:
  ar_and_r(Nat.is_le(Impl.b_lo(g, KT, I, G), b), Nat.is_le(b, Impl.b_hi(g, KT, I, G)),
    p2_touch_and(b, g, KT, I, G, gpos, fits, kpos, touch))

# A contributor's slot b - b_lo(g) is below max_contrib.
def p2_off_lt_mc(+b: Nat, +g: Nat, +Gr: Nat, +KT: Nat, +I: Nat, +G: Nat,
    +gpos: {Nat.is_lt(0n, G) == True{} : Bool},
    +fits: {Nat.is_le(G, I) == True{} : Bool},
    +kpos: {Nat.is_lt(0n, KT) == True{} : Bool},
    +touch: {Spec.touches(b, g, KT, I, G) == True{} : Bool},
    group: {Nat.is_lt(g, Gr) == True{} : Bool})
  -> {Nat.is_lt(Nat.sub(b, Impl.b_lo(g, KT, I, G)), Impl.max_contrib(Gr, KT, I, G)) == True{} : Bool}:
  ar_lt_le_trans(Nat.sub(b, Impl.b_lo(g, KT, I, G)), Impl.nc(g, KT, I, G), Impl.max_contrib(Gr, KT, I, G),
    p2_off_lt_nc(b, g, KT, I, G, p2_lo_le_b(b, g, KT, I, G, gpos, fits, kpos, touch),
      p2_b_le_hi(b, g, KT, I, G, gpos, fits, kpos, touch)),
    p2_nc_le_mc(g, Gr, KT, I, G, group))

# Equal slots, both above their b_lo, name the same block.
def p2_same_block(+b: Nat, +c: Nat, +g: Nat, +h: Nat, +KT: Nat, +I: Nat, +G: Nat,
    gh: {g == h : Nat},
    rr: {Nat.sub(b, Impl.b_lo(g, KT, I, G)) == Nat.sub(c, Impl.b_lo(h, KT, I, G)) : Nat},
    lob: {Nat.is_le(Impl.b_lo(g, KT, I, G), b) == True{} : Bool},
    loc: {Nat.is_le(Impl.b_lo(h, KT, I, G), c) == True{} : Bool})
  -> {b == c : Nat}:
  %ar_sub_add(b, Impl.b_lo(g, KT, I, G), lob) : {_ == c : Nat}
  %Equal.sym(Nat, Nat.sub(b, Impl.b_lo(g, KT, I, G)), Nat.sub(c, Impl.b_lo(h, KT, I, G)), rr) : {Nat.add(Impl.b_lo(g, KT, I, G), _) == c : Nat}
  %Equal.sym(Nat, g, h, gh) : {Nat.add(Impl.b_lo(_, KT, I, G), Nat.sub(c, Impl.b_lo(h, KT, I, G))) == c : Nat}
  ar_sub_add(c, Impl.b_lo(h, KT, I, G), loc)

def p2_inj_finish(+b: Nat, +c: Nat, +g: Nat, +h: Nat, +KT: Nat, +I: Nat, +G: Nat,
    both: {g == h : Nat} & {Nat.sub(b, Impl.b_lo(g, KT, I, G)) == Nat.sub(c, Impl.b_lo(h, KT, I, G)) : Nat},
    lob: {Nat.is_le(Impl.b_lo(g, KT, I, G), b) == True{} : Bool},
    loc: {Nat.is_le(Impl.b_lo(h, KT, I, G), c) == True{} : Bool})
  -> {g == h : Nat} & {b == c : Nat}:
  (gh, rr) = both
  +ghp = gh
  (ghp, p2_same_block(b, c, g, h, KT, I, G, ghp, rr, lob, loc))

def Laws.slot_in_range(mats, n, k, G, g, b, grid, shaped, fits, group, contributor):
  +gp = grid
  +fp = fits
  +gr = group
  +tc = contributor
  +kp = p2_kt_pos(k, shaped)
  +off = p2_off_lt_nc(b, g, Impl.kt(k), Impl.total(mats, n, k), G,
    p2_lo_le_b(b, g, Impl.kt(k), Impl.total(mats, n, k), G, gp, fp, kp, tc),
    p2_b_le_hi(b, g, Impl.kt(k), Impl.total(mats, n, k), G, gp, fp, kp, tc))
  +ncle = p2_nc_le_mc(g, Impl.groups(mats, n), Impl.kt(k), Impl.total(mats, n, k), G, gr)
  (off, (ncle, p2_addr_lt(g, Impl.groups(mats, n), Impl.max_contrib(Impl.groups(mats, n), Impl.kt(k), Impl.total(mats, n, k), G), Nat.sub(b, Impl.b_lo(g, Impl.kt(k), Impl.total(mats, n, k), G)),
    ar_lt_le_trans(Nat.sub(b, Impl.b_lo(g, Impl.kt(k), Impl.total(mats, n, k), G)), Impl.nc(g, Impl.kt(k), Impl.total(mats, n, k), G), Impl.max_contrib(Impl.groups(mats, n), Impl.kt(k), Impl.total(mats, n, k), G), off, ncle), gr)))

def Laws.slot_injective(mats, n, k, G, g, b, h, c, grid, shaped, fits, group_g, group_h,
    touch_b, touch_c, same):
  +gp = grid
  +fp = fits
  +tb = touch_b
  +tc = touch_c
  +kp = p2_kt_pos(k, shaped)
  p2_inj_finish(b, c, g, h, Impl.kt(k), Impl.total(mats, n, k), G,
    ar_divmod_inj(g, Nat.sub(b, Impl.b_lo(g, Impl.kt(k), Impl.total(mats, n, k), G)), h, Nat.sub(c, Impl.b_lo(h, Impl.kt(k), Impl.total(mats, n, k), G)), Impl.max_contrib(Impl.groups(mats, n), Impl.kt(k), Impl.total(mats, n, k), G),
      p2_off_lt_mc(b, g, Impl.groups(mats, n), Impl.kt(k), Impl.total(mats, n, k), G, gp, fp, kp, tb, group_g),
      p2_off_lt_mc(c, h, Impl.groups(mats, n), Impl.kt(k), Impl.total(mats, n, k), G, gp, fp, kp, tc, group_h),
      same),
    p2_lo_le_b(b, g, Impl.kt(k), Impl.total(mats, n, k), G, gp, fp, kp, tb),
    p2_lo_le_b(c, h, Impl.kt(k), Impl.total(mats, n, k), G, gp, fp, kp, tc))

# ==== Law issue_cursor ====

def p3_ge2(q: Nat, h: {Nat.is_lt(2n+q, 2n) == True{} : Bool}) -> Empty:
  match q:
    case 0n:
      ar_false_true(h)
    case 1n+r:
      ar_false_true(h)

def p3_ne_aux(+x: Nat, -y: Nat, same: {x == y : Nat}) -> {True{} == Nat.is_eq(x, y) : Bool}:
  %same : {True{} == Nat.is_eq(x, _) : Bool}
  Equal.sym(Bool, Nat.is_eq(x, x), True{}, ar_eq_refl(x))

# Equal Nats are is_eq.
def p3_ne(+x: Nat, -y: Nat, same: {x == y : Nat},
    diff: {Nat.is_eq(x, y) == False{} : Bool}) -> Empty:
  ar_true_false(Equal.trans(Bool, True{}, Nat.is_eq(x, y), False{},
    p3_ne_aux(x, y, same), diff))

def p3_not_t(-b: Bool, h: {b == True{} : Bool}) -> {Bool.not(b) == False{} : Bool}:
  %Equal.sym(Bool, b, True{}, h) : {Bool.not(_) == False{} : Bool}
  {==}

def p3_not_f(b: Bool, h: {Bool.not(b) == False{} : Bool}) -> {b == True{} : Bool}:
  match b:
    case False{}:
      Empty.absurd({False{} == True{} : Bool}, ar_true_false(h))
    case True{}:
      {==}

def p3_lt_zero(j: Nat) -> {Nat.is_lt(j, 0n) == False{} : Bool}:
  match j:
    case 0n:
      {==}
    case 1n+p:
      {==}

def p3_lt_succ(n: Nat) -> {Nat.is_lt(n, 1n+n) == True{} : Bool}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      p3_lt_succ(p)


# ---- law issue_cursor ----

def p3_fst(-A: Type, -B: Type, p: A & B) -> A:
  (x, y) = p
  x

# a <= b and a != b give a < b.
def p3_le_ne_lt(a: Nat, b: Nat, le: {Nat.is_le(a, b) == True{} : Bool},
    ne: {Nat.is_eq(a, b) == False{} : Bool}) -> {Nat.is_lt(a, b) == True{} : Bool}:
  match a b:
    case 0n 0n:
      Empty.absurd({Nat.is_lt(0n, 0n) == True{} : Bool}, ar_true_false(ne))
    case 0n 1n+bp:
      {==}
    case 1n+ap 0n:
      Empty.absurd({Nat.is_lt(1n+ap, 0n) == True{} : Bool}, ar_false_true(le))
    case 1n+ap 1n+bp:
      p3_le_ne_lt(ap, bp, le, ne)

# decode(q * KT + r) == Cursor{q, r} for r < KT.
def p3_dec_qr(+q: Nat, +r: Nat, +KT: Nat, small: {Nat.is_lt(r, KT) == True{} : Bool})
  -> {Impl.decode(Nat.add(Nat.mul(q, KT), r), KT) == Impl.Cursor{q, r} : Impl.Cursor}:
  +dq = p3_fst({Nat.div(Nat.add(Nat.mul(q, KT), r), KT) == q : Nat},
    {Nat.mod(Nat.add(Nat.mul(q, KT), r), KT) == r : Nat},
    ar_div_unique(q, r, KT, small))
  %Equal.sym(Nat, Nat.div(Nat.add(Nat.mul(q, KT), r), KT), q, dq)
    : {Impl.Cursor{_, Nat.sub(Nat.add(Nat.mul(q, KT), r),
        Nat.mul(Nat.div(Nat.add(Nat.mul(q, KT), r), KT), KT))} == Impl.Cursor{q, r} : Impl.Cursor}
  %Equal.sym(Nat, Nat.div(Nat.add(Nat.mul(q, KT), r), KT), q, dq)
    : {Impl.Cursor{q, Nat.sub(Nat.add(Nat.mul(q, KT), r), Nat.mul(_, KT))} ==
      Impl.Cursor{q, r} : Impl.Cursor}
  %Equal.sym(Nat, Nat.sub(Nat.add(Nat.mul(q, KT), r), Nat.mul(q, KT)), r,
    ar_add_sub(Nat.mul(q, KT), r))
    : {Impl.Cursor{q, _} == Impl.Cursor{q, r} : Impl.Cursor}
  {==}

# One issue from Cursor{q, r} (r < KT) lands on decode(q * KT + r + 1).
def p3_step_pick(w: Bool, +q: Nat, +r: Nat, +KT: Nat,
    small: {Nat.is_lt(r, KT) == True{} : Bool}, pos: {Nat.is_lt(0n, KT) == True{} : Bool},
    hb: {Nat.is_eq(1n+r, KT) == w : Bool})
  -> {Impl.advance.pick(w, q, 1n+r) ==
    Impl.decode(1n+Nat.add(Nat.mul(q, KT), r), KT) : Impl.Cursor}:
  match w:
    case True{}:
      +eqk = ar_eq_true(1n+r, KT, hb)
      %ar_add_succ_r(Nat.mul(q, KT), r)
        : {Impl.Cursor{1n+q, 0n} == Impl.decode(_, KT) : Impl.Cursor}
      %Equal.sym(Nat, 1n+r, KT, eqk)
        : {Impl.Cursor{1n+q, 0n} == Impl.decode(Nat.add(Nat.mul(q, KT), _), KT) : Impl.Cursor}
      %ar_add_comm(KT, Nat.mul(q, KT))
        : {Impl.Cursor{1n+q, 0n} == Impl.decode(_, KT) : Impl.Cursor}
      %ar_add_zero_r(Nat.add(KT, Nat.mul(q, KT)))
        : {Impl.Cursor{1n+q, 0n} == Impl.decode(_, KT) : Impl.Cursor}
      Equal.sym(Impl.Cursor, Impl.decode(Nat.add(Nat.mul(1n+q, KT), 0n), KT),
        Impl.Cursor{1n+q, 0n}, p3_dec_qr(1n+q, 0n, KT, pos))
    case False{}:
      %ar_add_succ_r(Nat.mul(q, KT), r)
        : {Impl.Cursor{q, 1n+r} == Impl.decode(_, KT) : Impl.Cursor}
      Equal.sym(Impl.Cursor, Impl.decode(Nat.add(Nat.mul(q, KT), 1n+r), KT),
        Impl.Cursor{q, 1n+r},
        p3_dec_qr(q, 1n+r, KT,
          p3_le_ne_lt(1n+r, KT,
            Equal.trans(Bool, Nat.is_le(1n+r, KT), Nat.is_lt(r, KT), True{},
              Equal.sym(Bool, Nat.is_lt(r, KT), Nat.is_le(1n+r, KT), ar_lt_succ_le(r, KT)),
              small),
            hb)))

# The decoded remainder it - (it / KT) * KT is below KT.
def p3_rem_lt(+it: Nat, +KT: Nat, +pos: {Nat.is_lt(0n, KT) == True{} : Bool})
  -> {Nat.is_lt(Nat.sub(it, Nat.mul(Nat.div(it, KT), KT)), KT) == True{} : Bool}:
  %Equal.sym(Bool, Nat.is_lt(Nat.sub(it, Nat.mul(Nat.div(it, KT), KT)), KT),
    Nat.is_lt(it, Nat.add(Nat.mul(Nat.div(it, KT), KT), KT)),
    ar_sub_lt_iff(it, Nat.mul(Nat.div(it, KT), KT), KT, pos)) : {_ == True{} : Bool}
  %ar_add_comm(KT, Nat.mul(Nat.div(it, KT), KT)) : {Nat.is_lt(it, _) == True{} : Bool}
  Equal.trans(Bool, Nat.is_lt(it, Nat.mul(1n+Nat.div(it, KT), KT)),
    Nat.is_lt(Nat.div(it, KT), 1n+Nat.div(it, KT)), True{},
    Equal.sym(Bool, Nat.is_lt(Nat.div(it, KT), 1n+Nat.div(it, KT)),
      Nat.is_lt(it, Nat.mul(1n+Nat.div(it, KT), KT)),
      ar_div_lt(1n+Nat.div(it, KT), it, KT, pos)),
    p3_lt_succ(Nat.div(it, KT)))

# One issue advances the decoded cursor of it to the decoded cursor of it + 1.
def p3_step(+it: Nat, +KT: Nat, +pos: {Nat.is_lt(0n, KT) == True{} : Bool})
  -> {Impl.advance(Impl.decode(it, KT), KT) == Impl.decode(1n+it, KT) : Impl.Cursor}:
  +le = Equal.trans(Bool, Nat.is_le(Nat.mul(Nat.div(it, KT), KT), it),
    Nat.is_le(Nat.div(it, KT), Nat.div(it, KT)), True{},
    ar_div_le(Nat.div(it, KT), it, KT, pos), ar_le_refl(Nat.div(it, KT)))
  %ar_sub_add(it, Nat.mul(Nat.div(it, KT), KT), le)
    : {Impl.advance(Impl.decode(it, KT), KT) == Impl.decode(1n+_, KT) : Impl.Cursor}
  p3_step_pick(Nat.is_eq(1n+Nat.sub(it, Nat.mul(Nat.div(it, KT), KT)), KT),
    Nat.div(it, KT), Nat.sub(it, Nat.mul(Nat.div(it, KT), KT)), KT,
    p3_rem_lt(it, KT, pos), pos, {==})

def p3_issue(+j: Nat, +s: Nat, +KT: Nat, +tiles: {Nat.is_lt(0n, KT) == True{} : Bool})
  -> {Impl.issued(j, s, KT) == Impl.decode(Nat.add(s, j), KT) : Impl.Cursor}:
  match j:
    case 0n:
      %Equal.sym(Nat, Nat.add(s, 0n), s, ar_add_zero_r(s))
        : {Impl.decode(s, KT) == Impl.decode(_, KT) : Impl.Cursor}
      {==}
    case 1n+i:
      %Equal.sym(Impl.Cursor, Impl.issued(i, s, KT), Impl.decode(Nat.add(s, i), KT),
        p3_issue(i, s, KT, tiles))
        : {Impl.advance(_, KT) == Impl.decode(Nat.add(s, 1n+i), KT) : Impl.Cursor}
      %Equal.sym(Nat, Nat.add(s, 1n+i), 1n+Nat.add(s, i), ar_add_succ_r(s, i))
        : {Impl.advance(Impl.decode(Nat.add(s, i), KT), KT) == Impl.decode(_, KT) : Impl.Cursor}
      p3_step(Nat.add(s, i), KT, tiles)

def Laws.issue_cursor(s, j, KT, tiles):
  +t = tiles
  p3_issue(j, s, KT, t)

# ==== Law reduction_order ====

def p4_ne(+x: Nat, -y: Nat, same: {x == y : Nat},
    diff: {Nat.is_eq(x, y) == False{} : Bool}) -> Empty:
  ar_true_false(Equal.trans(Bool, True{}, Nat.is_eq(x, x), False{},
    Equal.sym(Bool, Nat.is_eq(x, x), True{}, ar_eq_refl(x)),
    Equal.trans(Bool, Nat.is_eq(x, x), Nat.is_eq(x, y), False{},
      Equal.cong(Nat, Bool, z => Nat.is_eq(x, z), x, y, same), diff)))

# ---- workspace store / slot_at ----

def p4_store_len(ws: List<&2, Spec.Sum>, i: Nat, v: Spec.Sum)
  -> {List.length(&2, Spec.Sum, Impl.store(ws, i, v)) == List.length(&2, Spec.Sum, ws) : Nat}:
  match ws:
    case Nil{}:
      {==}
    case w <> rest:
      match i:
        case 0n:
          {==}
        case 1n+j:
          %p4_store_len(rest, j, v) : {1n+List.length(&2, Spec.Sum, Impl.store(rest, j, v)) == 1n+_ : Nat}
          {==}

def p4_ss_eq(ws: List<&2, Spec.Sum>, i: Nat, v: Spec.Sum,
    h: {Nat.is_lt(i, List.length(&2, Spec.Sum, ws)) == True{} : Bool})
  -> {Impl.slot_at(Impl.store(ws, i, v), i) == v : Spec.Sum}:
  match ws:
    case Nil{}:
      match i:
        case 0n:
          Empty.absurd({Impl.slot_at(Nil{}, 0n) == v : Spec.Sum}, ar_false_true(h))
        case 1n+j:
          Empty.absurd({Impl.slot_at(Nil{}, 1n+j) == v : Spec.Sum}, ar_false_true(h))
    case w <> rest:
      match i:
        case 0n:
          {==}
        case 1n+j:
          p4_ss_eq(rest, j, v, h)

def p4_ss_ne(ws: List<&2, Spec.Sum>, i: Nat, j: Nat, v: Spec.Sum,
    ne: {Nat.is_eq(i, j) == False{} : Bool})
  -> {Impl.slot_at(Impl.store(ws, i, v), j) == Impl.slot_at(ws, j) : Spec.Sum}:
  match ws:
    case Nil{}:
      {==}
    case w <> rest:
      match i j:
        case 0n 0n:
          Empty.absurd({Impl.slot_at(v <> rest, 0n) == w : Spec.Sum}, ar_true_false(ne))
        case 0n 1n+q:
          {==}
        case 1n+p 0n:
          {==}
        case 1n+p 1n+q:
          p4_ss_ne(rest, p, q, v, ne)

# ---- arrive: every arrived block's slot holds its partial ----

# Every element of the order is >= L.
def p4_all_ge(order: List<&2, Nat>, +L: Nat) -> Bool:
  match order:
    case Nil{}:
      True{}
    case +a <> rest:
      Bool.and(Nat.is_le(L, a), p4_all_ge(rest, L))

def p4_sub_inj(+a: Nat, +b: Nat, +L: Nat, hLa: {Nat.is_le(L, a) == True{} : Bool},
    hLb: {Nat.is_le(L, b) == True{} : Bool}, e: {Nat.sub(a, L) == Nat.sub(b, L) : Nat})
  -> {a == b : Nat}:
  Equal.trans(Nat, a, Nat.add(L, Nat.sub(a, L)), b,
    Equal.sym(Nat, Nat.add(L, Nat.sub(a, L)), a, ar_sub_add(a, L, hLa)),
    Equal.trans(Nat, Nat.add(L, Nat.sub(a, L)), Nat.add(L, Nat.sub(b, L)), b,
      Equal.cong(Nat, Nat, z => Nat.add(L, z), Nat.sub(a, L), Nat.sub(b, L), e),
      ar_sub_add(b, L, hLb)))

# Block a == b stores Part{b} into slot b - L.
def p4_slot_hit(+a: Nat, +b: Nat, +L: Nat, +ws: List<&2, Spec.Sum>, e: {b == a : Nat},
    hlen: {Nat.is_lt(Nat.sub(b, L), List.length(&2, Spec.Sum, ws)) == True{} : Bool})
  -> {Impl.slot_at(Impl.store(ws, Nat.sub(a, L), Spec.Part{a}), Nat.sub(b, L)) ==
    Spec.Part{b} : Spec.Sum}:
  %e : {Impl.slot_at(Impl.store(ws, Nat.sub(_, L), Spec.Part{_}), Nat.sub(b, L)) == Spec.Part{b} : Spec.Sum}
  p4_ss_eq(ws, Nat.sub(b, L), Spec.Part{b}, hlen)

# A store by block a (L <= a) keeps slot b - L holding Part{b}.
def p4_keep_slot(x: Bool, +a: Nat, +b: Nat, +L: Nat, +ws: List<&2, Spec.Sum>,
    hx: {Nat.is_eq(a, b) == x : Bool},
    hLa: {Nat.is_le(L, a) == True{} : Bool}, hLb: {Nat.is_le(L, b) == True{} : Bool},
    hlen: {Nat.is_lt(Nat.sub(b, L), List.length(&2, Spec.Sum, ws)) == True{} : Bool},
    hs: {Impl.slot_at(ws, Nat.sub(b, L)) == Spec.Part{b} : Spec.Sum})
  -> {Impl.slot_at(Impl.store(ws, Nat.sub(a, L), Spec.Part{a}), Nat.sub(b, L)) ==
    Spec.Part{b} : Spec.Sum}:
  match x:
    case True{}:
      p4_slot_hit(a, b, L, ws, Equal.sym(Nat, a, b, ar_eq_true(a, b, hx)), hlen)
    case False{}:
      Equal.trans(Spec.Sum,
        Impl.slot_at(Impl.store(ws, Nat.sub(a, L), Spec.Part{a}), Nat.sub(b, L)),
        Impl.slot_at(ws, Nat.sub(b, L)), Spec.Part{b},
        p4_ss_ne(ws, Nat.sub(a, L), Nat.sub(b, L), Spec.Part{a},
          ar_eq_false(Nat.sub(a, L), Nat.sub(b, L),
            e => p4_ne(a, b, p4_sub_inj(a, b, L, hLa, hLb, e), hx))),
        hs)

def p4_len_step(+ws: List<&2, Spec.Sum>, +i: Nat, +v: Spec.Sum, +j: Nat,
    hlen: {Nat.is_lt(j, List.length(&2, Spec.Sum, ws)) == True{} : Bool})
  -> {Nat.is_lt(j, List.length(&2, Spec.Sum, Impl.store(ws, i, v))) == True{} : Bool}:
  Equal.trans(Bool, Nat.is_lt(j, List.length(&2, Spec.Sum, Impl.store(ws, i, v))),
    Nat.is_lt(j, List.length(&2, Spec.Sum, ws)), True{},
    Equal.cong(Nat, Bool, z => Nat.is_lt(j, z),
      List.length(&2, Spec.Sum, Impl.store(ws, i, v)), List.length(&2, Spec.Sum, ws),
      p4_store_len(ws, i, v)),
    hlen)

# Once slot b - L holds Part{b}, later arrivals (all >= L) keep it.
def p4_keep(+order: List<&2, Nat>, +ws: List<&2, Spec.Sum>, +L: Nat, +b: Nat,
    +hL: {Nat.is_le(L, b) == True{} : Bool},
    +hlen: {Nat.is_lt(Nat.sub(b, L), List.length(&2, Spec.Sum, ws)) == True{} : Bool},
    +hall: {p4_all_ge(order, L) == True{} : Bool},
    hs: {Impl.slot_at(ws, Nat.sub(b, L)) == Spec.Part{b} : Spec.Sum})
  -> {Impl.slot_at(Impl.arrive(order, ws, L), Nat.sub(b, L)) == Spec.Part{b} : Spec.Sum}:
  match order:
    case Nil{}:
      hs
    case +a <> rest:
      p4_keep(rest, Impl.store(ws, Nat.sub(a, L), Spec.Part{a}), L, b, hL,
        p4_len_step(ws, Nat.sub(a, L), Spec.Part{a}, Nat.sub(b, L), hlen),
        ar_and_r(Nat.is_le(L, a), p4_all_ge(rest, L), hall),
        p4_keep_slot(Nat.is_eq(a, b), a, b, L, ws, {==},
          ar_and_l(Nat.is_le(L, a), p4_all_ge(rest, L), hall), hL, hlen, hs))

def p4_hit_step(x: Bool, +a: Nat, +b: Nat, +L: Nat, +ws: List<&2, Spec.Sum>,
    -rest: List<&2, Nat>, hx: {Nat.is_eq(a, b) == x : Bool},
    hlen: {Nat.is_lt(Nat.sub(b, L), List.length(&2, Spec.Sum, ws)) == True{} : Bool},
    harr: {Bool.or(x, Spec.arrived(rest, b)) == True{} : Bool},
    ih: {Spec.arrived(rest, b) == True{} : Bool} ->
      {Impl.slot_at(Impl.arrive(rest, Impl.store(ws, Nat.sub(a, L), Spec.Part{a}), L),
        Nat.sub(b, L)) == Spec.Part{b} : Spec.Sum},
    kp: {Impl.slot_at(Impl.store(ws, Nat.sub(a, L), Spec.Part{a}), Nat.sub(b, L)) ==
        Spec.Part{b} : Spec.Sum} ->
      {Impl.slot_at(Impl.arrive(rest, Impl.store(ws, Nat.sub(a, L), Spec.Part{a}), L),
        Nat.sub(b, L)) == Spec.Part{b} : Spec.Sum})
  -> {Impl.slot_at(Impl.arrive(rest, Impl.store(ws, Nat.sub(a, L), Spec.Part{a}), L),
    Nat.sub(b, L)) == Spec.Part{b} : Spec.Sum}:
  match x:
    case True{}:
      kp(p4_slot_hit(a, b, L, ws, Equal.sym(Nat, a, b, ar_eq_true(a, b, hx)), hlen))
    case False{}:
      ih(harr)

# An arrived block b (L <= b, slot in range) ends with Part{b} in slot b - L.
def p4_hit(+order: List<&2, Nat>, +ws: List<&2, Spec.Sum>, +L: Nat, +b: Nat,
    +hL: {Nat.is_le(L, b) == True{} : Bool},
    +hlen: {Nat.is_lt(Nat.sub(b, L), List.length(&2, Spec.Sum, ws)) == True{} : Bool},
    +hall: {p4_all_ge(order, L) == True{} : Bool},
    harr: {Spec.arrived(order, b) == True{} : Bool})
  -> {Impl.slot_at(Impl.arrive(order, ws, L), Nat.sub(b, L)) == Spec.Part{b} : Spec.Sum}:
  match order:
    case Nil{}:
      Empty.absurd({Impl.slot_at(ws, Nat.sub(b, L)) == Spec.Part{b} : Spec.Sum},
        ar_false_true(harr))
    case +a <> rest:
      +ws2 = Impl.store(ws, Nat.sub(a, L), Spec.Part{a})
      +hlen2 = p4_len_step(ws, Nat.sub(a, L), Spec.Part{a}, Nat.sub(b, L), hlen)
      +hall2 = ar_and_r(Nat.is_le(L, a), p4_all_ge(rest, L), hall)
      p4_hit_step(Nat.is_eq(a, b), a, b, L, ws, rest, {==}, hlen, harr,
        h => p4_hit(rest, ws2, L, b, hL, hlen2, hall2, h),
        s => p4_keep(rest, ws2, L, b, hL, hlen2, hall2, s))

# ---- ascending fold over the contiguous range [L, H] ----

def p4_le_succ(b: Nat) -> {Nat.is_le(b, 1n+b) == True{} : Bool}:
  match b:
    case 0n:
      {==}
    case 1n+p:
      p4_le_succ(p)

def p4_lt_succ(b: Nat) -> {Nat.is_lt(b, 1n+b) == True{} : Bool}:
  match b:
    case 0n:
      {==}
    case 1n+p:
      p4_lt_succ(p)

def p4_not_t_f(x: Bool, h: {Bool.not(x) == True{} : Bool}) -> {x == False{} : Bool}:
  match x:
    case True{}:
      Empty.absurd({True{} == False{} : Bool}, ar_false_true(h))
    case False{}:
      {==}

def p4_and_f(x: Bool) -> {Bool.and(x, False{}) == False{} : Bool}:
  match x:
    case True{}:
      {==}
    case False{}:
      {==}

# Block b lies in [L, H].
def p4_in(+L: Nat, +H: Nat, +b: Nat) -> Bool:
  Bool.and(Nat.is_le(L, b), Nat.is_le(b, H))

# fold_touching with touches(b) replaced by b in [L, H].
def p4_rng(left: Nat, +b: Nat, +L: Nat, +H: Nat, acc: Maybe<&2, Spec.Sum>)
  -> Maybe<&2, Spec.Sum>:
  match left:
    case 0n:
      acc
    case 1n+rest:
      p4_rng(rest, 1n+b, L, H, Spec.fold_step(p4_in(L, H, b), b, acc))

# v + Part{b} + Part{b + 1} + ... (left terms), summed left to right.
def p4_asc(left: Nat, +b: Nat, v: Spec.Sum) -> Spec.Sum:
  match left:
    case 0n:
      v
    case 1n+rest:
      p4_asc(rest, 1n+b, Spec.Add{v, Spec.Part{b}})

def p4_split(+a: Nat, +r: Nat, +b: Nat, +L: Nat, +H: Nat, +acc: Maybe<&2, Spec.Sum>)
  -> {p4_rng(Nat.add(a, r), b, L, H, acc) ==
    p4_rng(r, Nat.add(a, b), L, H, p4_rng(a, b, L, H, acc)) : Maybe<&2, Spec.Sum>}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      Equal.trans(Maybe<&2, Spec.Sum>,
        p4_rng(Nat.add(p, r), 1n+b, L, H, Spec.fold_step(p4_in(L, H, b), b, acc)),
        p4_rng(r, Nat.add(p, 1n+b), L, H,
          p4_rng(p, 1n+b, L, H, Spec.fold_step(p4_in(L, H, b), b, acc))),
        p4_rng(r, 1n+Nat.add(p, b), L, H,
          p4_rng(p, 1n+b, L, H, Spec.fold_step(p4_in(L, H, b), b, acc))),
        p4_split(p, r, 1n+b, L, H, Spec.fold_step(p4_in(L, H, b), b, acc)),
        Equal.cong(Nat, Maybe<&2, Spec.Sum>,
          z => p4_rng(r, z, L, H,
            p4_rng(p, 1n+b, L, H, Spec.fold_step(p4_in(L, H, b), b, acc))),
          Nat.add(p, 1n+b), 1n+Nat.add(p, b), ar_add_succ_r(p, b)))

# Blocks b .. b + a - 1 all below L leave acc unchanged.
def p4_pre(+a: Nat, +b: Nat, +L: Nat, +H: Nat, +acc: Maybe<&2, Spec.Sum>,
    +h: {Nat.is_le(Nat.add(a, b), L) == True{} : Bool})
  -> {p4_rng(a, b, L, H, acc) == acc : Maybe<&2, Spec.Sum>}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      +hlt = Equal.trans(Bool, Nat.is_lt(b, L), Nat.is_le(1n+b, L), True{},
        ar_lt_succ_le(b, L),
        ar_le_trans(1n+b, 1n+Nat.add(p, b), L, ar_le_add_l(p, b), h))
      +hf = p4_not_t_f(Nat.is_le(L, b),
        Equal.trans(Bool, Bool.not(Nat.is_le(L, b)), Nat.is_lt(b, L), True{},
          Equal.sym(Bool, Nat.is_lt(b, L), Bool.not(Nat.is_le(L, b)),
            ar_lt_not_le(b, L)),
          hlt))
      +hin = Equal.cong(Bool, Bool, z => Bool.and(z, Nat.is_le(b, H)), Nat.is_le(L, b),
        False{}, hf)
      %Equal.sym(Bool, p4_in(L, H, b), False{}, hin) : {p4_rng(p, 1n+b, L, H, Spec.fold_step(_, b, acc)) == acc : Maybe<&2, Spec.Sum>}
      p4_pre(p, 1n+b, L, H, acc,
        Equal.trans(Bool, Nat.is_le(Nat.add(p, 1n+b), L), Nat.is_le(1n+Nat.add(p, b), L),
          True{},
          Equal.cong(Nat, Bool, z => Nat.is_le(z, L), Nat.add(p, 1n+b), 1n+Nat.add(p, b),
            ar_add_succ_r(p, b)),
          h))

# Blocks past H leave acc unchanged.
def p4_post(+a: Nat, +b: Nat, +L: Nat, +H: Nat, +acc: Maybe<&2, Spec.Sum>,
    +h: {Nat.is_lt(H, b) == True{} : Bool})
  -> {p4_rng(a, b, L, H, acc) == acc : Maybe<&2, Spec.Sum>}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      +hf = p4_not_t_f(Nat.is_le(b, H),
        Equal.trans(Bool, Bool.not(Nat.is_le(b, H)), Nat.is_lt(H, b), True{},
          Equal.sym(Bool, Nat.is_lt(H, b), Bool.not(Nat.is_le(b, H)),
            ar_lt_not_le(H, b)),
          h))
      +hin = Equal.trans(Bool, p4_in(L, H, b), Bool.and(Nat.is_le(L, b), False{}), False{},
        Equal.cong(Bool, Bool, z => Bool.and(Nat.is_le(L, b), z), Nat.is_le(b, H), False{},
          hf),
        p4_and_f(Nat.is_le(L, b)))
      %Equal.sym(Bool, p4_in(L, H, b), False{}, hin) : {p4_rng(p, 1n+b, L, H, Spec.fold_step(_, b, acc)) == acc : Maybe<&2, Spec.Sum>}
      p4_post(p, 1n+b, L, H, acc, ar_lt_trans(H, b, 1n+b, h, p4_lt_succ(b)))

# Blocks b .. b + a - 1 all inside [L, H] extend Some{v} one partial at a time.
def p4_mid(+a: Nat, +b: Nat, +L: Nat, +H: Nat, +v: Spec.Sum,
    +hL: {Nat.is_le(L, b) == True{} : Bool},
    +hH: {Nat.is_le(Nat.add(a, b), 1n+H) == True{} : Bool})
  -> {p4_rng(a, b, L, H, Some{v}) == Some{p4_asc(a, b, v)} : Maybe<&2, Spec.Sum>}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      +hin = ar_and_intro(Nat.is_le(L, b), Nat.is_le(b, H), hL,
        ar_le_trans(b, Nat.add(p, b), H, ar_le_add_l(p, b), hH))
      %Equal.sym(Bool, p4_in(L, H, b), True{}, hin) : {p4_rng(p, 1n+b, L, H, Spec.fold_step(_, b, Some{v})) == Some{p4_asc(p, 1n+b, Spec.Add{v, Spec.Part{b}})} : Maybe<&2, Spec.Sum>}
      p4_mid(p, 1n+b, L, H, Spec.Add{v, Spec.Part{b}},
        ar_le_trans(L, b, 1n+b, hL, p4_le_succ(b)),
        Equal.trans(Bool, Nat.is_le(Nat.add(p, 1n+b), 1n+H), Nat.is_le(1n+Nat.add(p, b), 1n+H),
          True{},
          Equal.cong(Nat, Bool, z => Nat.is_le(z, 1n+H), Nat.add(p, 1n+b), 1n+Nat.add(p, b),
            ar_add_succ_r(p, b)),
          hH))

# From block b (L <= b, b + d = H + 1) on, m >= d more blocks sum exactly blocks b .. H.
def p4_rest(+m: Nat, +d: Nat, +b: Nat, +L: Nat, +H: Nat, +v: Spec.Sum,
    +hL: {Nat.is_le(L, b) == True{} : Bool},
    +hd: {Nat.add(d, b) == 1n+H : Nat},
    hm: {Nat.is_le(d, m) == True{} : Bool})
  -> {p4_rng(m, b, L, H, Some{v}) == Some{p4_asc(d, b, v)} : Maybe<&2, Spec.Sum>}:
  %ar_sub_add(m, d, hm) : {p4_rng(_, b, L, H, Some{v}) == Some{p4_asc(d, b, v)} : Maybe<&2, Spec.Sum>}
  Equal.trans(Maybe<&2, Spec.Sum>,
    p4_rng(Nat.add(d, Nat.sub(m, d)), b, L, H, Some{v}),
    p4_rng(Nat.sub(m, d), Nat.add(d, b), L, H, p4_rng(d, b, L, H, Some{v})),
    Some{p4_asc(d, b, v)},
    p4_split(d, Nat.sub(m, d), b, L, H, Some{v}),
    Equal.trans(Maybe<&2, Spec.Sum>,
      p4_rng(Nat.sub(m, d), Nat.add(d, b), L, H, p4_rng(d, b, L, H, Some{v})),
      p4_rng(Nat.sub(m, d), Nat.add(d, b), L, H, Some{p4_asc(d, b, v)}),
      Some{p4_asc(d, b, v)},
      Equal.cong(Maybe<&2, Spec.Sum>, Maybe<&2, Spec.Sum>,
        z => p4_rng(Nat.sub(m, d), Nat.add(d, b), L, H, z),
        p4_rng(d, b, L, H, Some{v}), Some{p4_asc(d, b, v)},
        p4_mid(d, b, L, H, v, hL,
          Equal.trans(Bool, Nat.is_le(Nat.add(d, b), 1n+H), Nat.is_le(1n+H, 1n+H), True{},
            Equal.cong(Nat, Bool, z => Nat.is_le(z, 1n+H), Nat.add(d, b), 1n+H, hd),
            ar_le_refl(1n+H)))),
      p4_post(Nat.sub(m, d), Nat.add(d, b), L, H, Some{p4_asc(d, b, v)},
        Equal.trans(Bool, Nat.is_lt(H, Nat.add(d, b)), Nat.is_lt(H, 1n+H), True{},
          Equal.cong(Nat, Bool, z => Nat.is_lt(H, z), Nat.add(d, b), 1n+H, hd),
          p4_lt_succ(H)))))

# From block L on (acc None), m >= d + 1 blocks sum exactly blocks L .. H.
def p4_start(+m: Nat, +L: Nat, +H: Nat, +d: Nat,
    +hLH: {Nat.is_le(L, H) == True{} : Bool},
    +hd: {Nat.add(d, 1n+L) == 1n+H : Nat},
    hm: {Nat.is_le(1n+d, m) == True{} : Bool})
  -> {p4_rng(m, L, L, H, None{}) == Some{p4_asc(d, 1n+L, Spec.Part{L})}
    : Maybe<&2, Spec.Sum>}:
  match m:
    case 0n:
      Empty.absurd({None{} == Some{p4_asc(d, 1n+L, Spec.Part{L})} : Maybe<&2, Spec.Sum>},
        ar_false_true(hm))
    case 1n+p:
      +hin = ar_and_intro(Nat.is_le(L, L), Nat.is_le(L, H), ar_le_refl(L), hLH)
      %Equal.sym(Bool, p4_in(L, H, L), True{}, hin) : {p4_rng(p, 1n+L, L, H, Spec.fold_step(_, L, None{})) == Some{p4_asc(d, 1n+L, Spec.Part{L})} : Maybe<&2, Spec.Sum>}
      p4_rest(p, d, 1n+L, L, H, Spec.Part{L}, p4_le_succ(L), hd, hm)

# The whole grid 0 .. G - 1 (L <= H < G) sums exactly blocks L .. H, ascending.
def p4_whole(+G: Nat, +L: Nat, +H: Nat,
    +hLH: {Nat.is_le(L, H) == True{} : Bool},
    +hHG: {Nat.is_lt(H, G) == True{} : Bool})
  -> {p4_rng(G, 0n, L, H, None{}) == Some{p4_asc(Nat.sub(H, L), 1n+L, Spec.Part{L})}
    : Maybe<&2, Spec.Sum>}:
  +hLG = ar_lt_le(L, G, ar_le_lt_trans(L, H, G, hLH, hHG))
  +m = Nat.sub(G, L)
  +d = Nat.sub(H, L)
  +hd = Equal.trans(Nat, Nat.add(d, 1n+L), 1n+Nat.add(d, L), 1n+H,
    ar_add_succ_r(d, L),
    Equal.cong(Nat, Nat, z => 1n+z, Nat.add(d, L), H,
      Equal.trans(Nat, Nat.add(d, L), Nat.add(L, d), H, ar_add_comm(d, L),
        ar_sub_add(H, L, hLH))))
  +hm = Equal.trans(Bool, Nat.is_le(1n+d, m), Nat.is_lt(d, m), True{},
    Equal.sym(Bool, Nat.is_lt(d, m), Nat.is_le(1n+d, m), ar_lt_succ_le(d, m)),
    Equal.trans(Bool, Nat.is_lt(d, m), Nat.is_lt(Nat.add(L, d), G), True{},
      ar_lt_sub_iff(G, L, d),
      Equal.trans(Bool, Nat.is_lt(Nat.add(L, d), G), Nat.is_lt(H, G), True{},
        Equal.cong(Nat, Bool, z => Nat.is_lt(z, G), Nat.add(L, d), H,
          ar_sub_add(H, L, hLH)),
        hHG)))
  +hpre = Equal.trans(Bool, Nat.is_le(Nat.add(L, 0n), L), Nat.is_le(L, L), True{},
    Equal.cong(Nat, Bool, z => Nat.is_le(z, L), Nat.add(L, 0n), L, ar_add_zero_r(L)),
    ar_le_refl(L))
  %ar_sub_add(G, L, hLG) : {p4_rng(_, 0n, L, H, None{}) == Some{p4_asc(d, 1n+L, Spec.Part{L})} : Maybe<&2, Spec.Sum>}
  Equal.trans(Maybe<&2, Spec.Sum>,
    p4_rng(Nat.add(L, m), 0n, L, H, None{}),
    p4_rng(m, Nat.add(L, 0n), L, H, p4_rng(L, 0n, L, H, None{})),
    Some{p4_asc(d, 1n+L, Spec.Part{L})},
    p4_split(L, m, 0n, L, H, None{}),
    Equal.trans(Maybe<&2, Spec.Sum>,
      p4_rng(m, Nat.add(L, 0n), L, H, p4_rng(L, 0n, L, H, None{})),
      p4_rng(m, Nat.add(L, 0n), L, H, None{}),
      Some{p4_asc(d, 1n+L, Spec.Part{L})},
      Equal.cong(Maybe<&2, Spec.Sum>, Maybe<&2, Spec.Sum>,
        z => p4_rng(m, Nat.add(L, 0n), L, H, z),
        p4_rng(L, 0n, L, H, None{}), None{}, p4_pre(L, 0n, L, H, None{}, hpre)),
      Equal.trans(Maybe<&2, Spec.Sum>,
        p4_rng(m, Nat.add(L, 0n), L, H, None{}),
        p4_rng(m, L, L, H, None{}),
        Some{p4_asc(d, 1n+L, Spec.Part{L})},
        Equal.cong(Nat, Maybe<&2, Spec.Sum>, z => p4_rng(m, z, L, H, None{}),
          Nat.add(L, 0n), L, ar_add_zero_r(L)),
        p4_start(m, L, H, d, hLH, hd, hm))))

# ---- finish: the slot-order sum is the ascending fold ----

# Blocks b0 .. b0 + cnt - 1 have all arrived.
def p4_all_arr(cnt: Nat, +b0: Nat, +order: List<&2, Nat>) -> Bool:
  match cnt:
    case 0n:
      True{}
    case 1n+c:
      Bool.and(Spec.arrived(order, b0), p4_all_arr(c, 1n+b0, order))

# Slot c holds Part{c + L} once block c + L has arrived.
def p4_slot_c(+c: Nat, +order: List<&2, Nat>, +ws: List<&2, Spec.Sum>, +L: Nat,
    +hall: {p4_all_ge(order, L) == True{} : Bool},
    harr: {Spec.arrived(order, Nat.add(c, L)) == True{} : Bool},
    hlen: {Nat.is_lt(c, List.length(&2, Spec.Sum, ws)) == True{} : Bool})
  -> {Impl.slot_at(Impl.arrive(order, ws, L), c) == Spec.Part{Nat.add(c, L)} : Spec.Sum}:
  +e = Equal.trans(Nat, Nat.sub(Nat.add(c, L), L), Nat.sub(Nat.add(L, c), L), c,
    Equal.cong(Nat, Nat, z => Nat.sub(z, L), Nat.add(c, L), Nat.add(L, c),
      ar_add_comm(c, L)),
    ar_add_sub(L, c))
  Equal.trans(Spec.Sum, Impl.slot_at(Impl.arrive(order, ws, L), c),
    Impl.slot_at(Impl.arrive(order, ws, L), Nat.sub(Nat.add(c, L), L)), Spec.Part{Nat.add(c, L)},
    Equal.cong(Nat, Spec.Sum, z => Impl.slot_at(Impl.arrive(order, ws, L), z), c,
      Nat.sub(Nat.add(c, L), L), Equal.sym(Nat, Nat.sub(Nat.add(c, L), L), c, e)),
    p4_hit(order, ws, L, Nat.add(c, L), ar_le_add_l(c, L),
      Equal.trans(Bool, Nat.is_lt(Nat.sub(Nat.add(c, L), L), List.length(&2, Spec.Sum, ws)),
        Nat.is_lt(c, List.length(&2, Spec.Sum, ws)), True{},
        Equal.cong(Nat, Bool, z => Nat.is_lt(z, List.length(&2, Spec.Sum, ws)),
          Nat.sub(Nat.add(c, L), L), c, e),
        hlen),
      hall, harr))

# finish.go over slots c .. c + a - 1 adds Part{c + L} .. Part{c + a - 1 + L} in order.
def p4_fin(+a: Nat, +c: Nat, +v: Spec.Sum, +order: List<&2, Nat>, +ws: List<&2, Spec.Sum>,
    +L: Nat, +hall: {p4_all_ge(order, L) == True{} : Bool},
    +harr: {p4_all_arr(a, Nat.add(c, L), order) == True{} : Bool},
    +hlen: {Nat.is_le(Nat.add(a, c), List.length(&2, Spec.Sum, ws)) == True{} : Bool})
  -> {Impl.finish.go(a, c, v, Impl.arrive(order, ws, L)) == p4_asc(a, Nat.add(c, L), v)
    : Spec.Sum}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      +hltc = Equal.trans(Bool, Nat.is_lt(c, List.length(&2, Spec.Sum, ws)),
        Nat.is_le(1n+c, List.length(&2, Spec.Sum, ws)), True{},
        ar_lt_succ_le(c, List.length(&2, Spec.Sum, ws)),
        ar_le_trans(1n+c, 1n+Nat.add(p, c), List.length(&2, Spec.Sum, ws),
          ar_le_add_l(p, c), hlen))
      +hs = p4_slot_c(c, order, ws, L, hall,
        ar_and_l(Spec.arrived(order, Nat.add(c, L)),
          p4_all_arr(p, 1n+Nat.add(c, L), order), harr),
        hltc)
      %Equal.sym(Spec.Sum, Impl.slot_at(Impl.arrive(order, ws, L), c), Spec.Part{Nat.add(c, L)}, hs) : {Impl.finish.go(p, 1n+c, Spec.Add{v, _}, Impl.arrive(order, ws, L)) == p4_asc(p, 1n+Nat.add(c, L), Spec.Add{v, Spec.Part{Nat.add(c, L)}}) : Spec.Sum}
      p4_fin(p, 1n+c, Spec.Add{v, Spec.Part{Nat.add(c, L)}}, order, ws, L, hall,
        ar_and_r(Spec.arrived(order, Nat.add(c, L)),
          p4_all_arr(p, 1n+Nat.add(c, L), order), harr),
        Equal.trans(Bool, Nat.is_le(Nat.add(p, 1n+c), List.length(&2, Spec.Sum, ws)),
          Nat.is_le(1n+Nat.add(p, c), List.length(&2, Spec.Sum, ws)), True{},
          Equal.cong(Nat, Bool, z => Nat.is_le(z, List.length(&2, Spec.Sum, ws)),
            Nat.add(p, 1n+c), 1n+Nat.add(p, c), ar_add_succ_r(p, c)),
          hlen))

# ---- all_arrived: every touching block below the count has arrived ----

def p4_lt_zero(b: Nat, h: {Nat.is_lt(b, 0n) == True{} : Bool}) -> Empty:
  match b:
    case 0n:
      ar_false_true(h)
    case 1n+p:
      ar_false_true(h)

def p4_not_f_t(x: Bool, h: {Bool.not(x) == False{} : Bool}) -> {x == True{} : Bool}:
  match x:
    case True{}:
      {==}
    case False{}:
      Empty.absurd({False{} == True{} : Bool}, ar_true_false(h))

def p4_or_not(x: Bool, -y: Bool, hx: {x == True{} : Bool},
    h: {Bool.or(Bool.not(x), y) == True{} : Bool}) -> {y == True{} : Bool}:
  match x:
    case True{}:
      h
    case False{}:
      Empty.absurd({y == True{} : Bool}, ar_false_true(hx))

def p4_arr_step(x: Bool, +b: Nat, +c: Nat, +order: List<&2, Nat>, +g: Nat, +KT: Nat,
    +I: Nat, +G: Nat, hx: {Nat.is_lt(b, c) == x : Bool},
    hm: {Nat.is_le(b, c) == True{} : Bool},
    ht: {Spec.touches(b, g, KT, I, G) == True{} : Bool},
    hr: {Bool.or(Bool.not(Spec.touches(c, g, KT, I, G)), Spec.arrived(order, c)) == True{}
      : Bool},
    ih: {Nat.is_lt(b, c) == True{} : Bool} -> {Spec.arrived(order, b) == True{} : Bool})
  -> {Spec.arrived(order, b) == True{} : Bool}:
  match x:
    case True{}:
      ih(hx)
    case False{}:
      +e = ar_le_antisym(c, b,
        p4_not_f_t(Nat.is_le(c, b),
          Equal.trans(Bool, Bool.not(Nat.is_le(c, b)), Nat.is_lt(b, c), False{},
            Equal.sym(Bool, Nat.is_lt(b, c), Bool.not(Nat.is_le(c, b)),
              ar_lt_not_le(b, c)),
            hx)),
        hm)
      %e : {Spec.arrived(order, _) == True{} : Bool}
      p4_or_not(Spec.touches(c, g, KT, I, G), Spec.arrived(order, c),
        Equal.trans(Bool, Spec.touches(c, g, KT, I, G), Spec.touches(b, g, KT, I, G), True{},
          Equal.cong(Nat, Bool, z => Spec.touches(z, g, KT, I, G), c, b, e), ht),
        hr)

def p4_arr(+count: Nat, +order: List<&2, Nat>, +g: Nat, +KT: Nat, +I: Nat, +G: Nat,
    +b: Nat, +h: {Spec.all_arrived(count, order, g, KT, I, G) == True{} : Bool},
    hb: {Nat.is_lt(b, count) == True{} : Bool},
    +ht: {Spec.touches(b, g, KT, I, G) == True{} : Bool})
  -> {Spec.arrived(order, b) == True{} : Bool}:
  match count:
    case 0n:
      Empty.absurd({Spec.arrived(order, b) == True{} : Bool}, p4_lt_zero(b, hb))
    case 1n+c:
      p4_arr_step(Nat.is_lt(b, c), b, c, order, g, KT, I, G, {==},
        Equal.trans(Bool, Nat.is_le(b, c), Nat.is_lt(b, 1n+c), True{},
          Equal.sym(Bool, Nat.is_lt(b, 1n+c), Nat.is_le(1n+b, 1n+c),
            ar_lt_succ_le(b, 1n+c)),
          hb),
        ht,
        ar_and_r(Spec.all_arrived(c, order, g, KT, I, G),
          Bool.or(Bool.not(Spec.touches(c, g, KT, I, G)), Spec.arrived(order, c)), h),
        hc => p4_arr(c, order, g, KT, I, G, b,
          ar_and_l(Spec.all_arrived(c, order, g, KT, I, G),
            Bool.or(Bool.not(Spec.touches(c, g, KT, I, G)), Spec.arrived(order, c)), h),
          hc, ht))

# ---- the law's context: touches(b) == b in [b_lo, b_hi] ----

def p4_fst(-A: Type, -B: Type, p: A & B) -> A:
  (a, b) = p
  a

def p4_snd(-A: Type, -B: Type, p: A & B) -> B:
  (a, b) = p
  b

def p4_ct(+mats: Nat, +n: Nat, +k: Nat, +G: Nat, +g: Nat, +b: Nat,
    +grid: {Nat.is_lt(0n, G) == True{} : Bool},
    +shaped: {Nat.is_le(128n, k) == True{} : Bool},
    +fits: {Nat.is_le(G, Impl.total(mats, n, k)) == True{} : Bool},
    +group: {Nat.is_lt(g, Impl.groups(mats, n)) == True{} : Bool})
  -> {Spec.touches(b, g, Impl.kt(k), Impl.total(mats, n, k), G) ==
    p4_in(Impl.b_lo(g, Impl.kt(k), Impl.total(mats, n, k), G),
      Impl.b_hi(g, Impl.kt(k), Impl.total(mats, n, k), G), b) : Bool}:
  p4_fst({Spec.touches(b, g, Impl.kt(k), Impl.total(mats, n, k), G) ==
      p4_in(Impl.b_lo(g, Impl.kt(k), Impl.total(mats, n, k), G),
        Impl.b_hi(g, Impl.kt(k), Impl.total(mats, n, k), G), b) : Bool},
    {Nat.is_le(Impl.b_lo(g, Impl.kt(k), Impl.total(mats, n, k), G),
      Impl.b_hi(g, Impl.kt(k), Impl.total(mats, n, k), G)) == True{} : Bool} &
    {Nat.is_lt(Impl.b_hi(g, Impl.kt(k), Impl.total(mats, n, k), G), G) == True{} : Bool},
    Laws.contributors_contiguous(mats, n, k, G, g, b, grid, shaped, fits, group))

def p4_cc23(+mats: Nat, +n: Nat, +k: Nat, +G: Nat, +g: Nat,
    +grid: {Nat.is_lt(0n, G) == True{} : Bool},
    +shaped: {Nat.is_le(128n, k) == True{} : Bool},
    +fits: {Nat.is_le(G, Impl.total(mats, n, k)) == True{} : Bool},
    +group: {Nat.is_lt(g, Impl.groups(mats, n)) == True{} : Bool})
  -> {Nat.is_le(Impl.b_lo(g, Impl.kt(k), Impl.total(mats, n, k), G),
      Impl.b_hi(g, Impl.kt(k), Impl.total(mats, n, k), G)) == True{} : Bool} &
    {Nat.is_lt(Impl.b_hi(g, Impl.kt(k), Impl.total(mats, n, k), G), G) == True{} : Bool}:
  p4_snd({Spec.touches(0n, g, Impl.kt(k), Impl.total(mats, n, k), G) ==
      p4_in(Impl.b_lo(g, Impl.kt(k), Impl.total(mats, n, k), G),
        Impl.b_hi(g, Impl.kt(k), Impl.total(mats, n, k), G), 0n) : Bool},
    {Nat.is_le(Impl.b_lo(g, Impl.kt(k), Impl.total(mats, n, k), G),
      Impl.b_hi(g, Impl.kt(k), Impl.total(mats, n, k), G)) == True{} : Bool} &
    {Nat.is_lt(Impl.b_hi(g, Impl.kt(k), Impl.total(mats, n, k), G), G) == True{} : Bool},
    Laws.contributors_contiguous(mats, n, k, G, g, 0n, grid, shaped, fits, group))

# b_lo <= b_hi
def p4_hLH(+mats: Nat, +n: Nat, +k: Nat, +G: Nat, +g: Nat,
    +grid: {Nat.is_lt(0n, G) == True{} : Bool},
    +shaped: {Nat.is_le(128n, k) == True{} : Bool},
    +fits: {Nat.is_le(G, Impl.total(mats, n, k)) == True{} : Bool},
    +group: {Nat.is_lt(g, Impl.groups(mats, n)) == True{} : Bool})
  -> {Nat.is_le(Impl.b_lo(g, Impl.kt(k), Impl.total(mats, n, k), G),
      Impl.b_hi(g, Impl.kt(k), Impl.total(mats, n, k), G)) == True{} : Bool}:
  p4_fst({Nat.is_le(Impl.b_lo(g, Impl.kt(k), Impl.total(mats, n, k), G),
      Impl.b_hi(g, Impl.kt(k), Impl.total(mats, n, k), G)) == True{} : Bool},
    {Nat.is_lt(Impl.b_hi(g, Impl.kt(k), Impl.total(mats, n, k), G), G) == True{} : Bool},
    p4_cc23(mats, n, k, G, g, grid, shaped, fits, group))

# b_hi < G
def p4_hHG(+mats: Nat, +n: Nat, +k: Nat, +G: Nat, +g: Nat,
    +grid: {Nat.is_lt(0n, G) == True{} : Bool},
    +shaped: {Nat.is_le(128n, k) == True{} : Bool},
    +fits: {Nat.is_le(G, Impl.total(mats, n, k)) == True{} : Bool},
    +group: {Nat.is_lt(g, Impl.groups(mats, n)) == True{} : Bool})
  -> {Nat.is_lt(Impl.b_hi(g, Impl.kt(k), Impl.total(mats, n, k), G), G) == True{} : Bool}:
  p4_snd({Nat.is_le(Impl.b_lo(g, Impl.kt(k), Impl.total(mats, n, k), G),
      Impl.b_hi(g, Impl.kt(k), Impl.total(mats, n, k), G)) == True{} : Bool},
    {Nat.is_lt(Impl.b_hi(g, Impl.kt(k), Impl.total(mats, n, k), G), G) == True{} : Bool},
    p4_cc23(mats, n, k, G, g, grid, shaped, fits, group))

def p4_conv_fold(+left: Nat, +b: Nat, +acc: Maybe<&2, Spec.Sum>, +mats: Nat, +n: Nat,
    +k: Nat, +G: Nat, +g: Nat,
    +grid: {Nat.is_lt(0n, G) == True{} : Bool},
    +shaped: {Nat.is_le(128n, k) == True{} : Bool},
    +fits: {Nat.is_le(G, Impl.total(mats, n, k)) == True{} : Bool},
    +group: {Nat.is_lt(g, Impl.groups(mats, n)) == True{} : Bool})
  -> {Spec.fold_touching(left, b, g, Impl.kt(k), Impl.total(mats, n, k), G, acc) ==
    p4_rng(left, b, Impl.b_lo(g, Impl.kt(k), Impl.total(mats, n, k), G),
      Impl.b_hi(g, Impl.kt(k), Impl.total(mats, n, k), G), acc) : Maybe<&2, Spec.Sum>}:
  match left:
    case 0n:
      {==}
    case 1n+p:
      Equal.trans(Maybe<&2, Spec.Sum>,
        Spec.fold_touching(p, 1n+b, g, Impl.kt(k), Impl.total(mats, n, k), G,
          Spec.fold_step(Spec.touches(b, g, Impl.kt(k), Impl.total(mats, n, k), G), b, acc)),
        p4_rng(p, 1n+b, Impl.b_lo(g, Impl.kt(k), Impl.total(mats, n, k), G),
          Impl.b_hi(g, Impl.kt(k), Impl.total(mats, n, k), G),
          Spec.fold_step(Spec.touches(b, g, Impl.kt(k), Impl.total(mats, n, k), G), b, acc)),
        p4_rng(p, 1n+b, Impl.b_lo(g, Impl.kt(k), Impl.total(mats, n, k), G),
          Impl.b_hi(g, Impl.kt(k), Impl.total(mats, n, k), G),
          Spec.fold_step(p4_in(Impl.b_lo(g, Impl.kt(k), Impl.total(mats, n, k), G),
            Impl.b_hi(g, Impl.kt(k), Impl.total(mats, n, k), G), b), b, acc)),
        p4_conv_fold(p, 1n+b,
          Spec.fold_step(Spec.touches(b, g, Impl.kt(k), Impl.total(mats, n, k), G), b, acc),
          mats, n, k, G, g, grid, shaped, fits, group),
        Equal.cong(Bool, Maybe<&2, Spec.Sum>,
          z => p4_rng(p, 1n+b, Impl.b_lo(g, Impl.kt(k), Impl.total(mats, n, k), G),
            Impl.b_hi(g, Impl.kt(k), Impl.total(mats, n, k), G), Spec.fold_step(z, b, acc)),
          Spec.touches(b, g, Impl.kt(k), Impl.total(mats, n, k), G),
          p4_in(Impl.b_lo(g, Impl.kt(k), Impl.total(mats, n, k), G),
            Impl.b_hi(g, Impl.kt(k), Impl.total(mats, n, k), G), b),
          p4_ct(mats, n, k, G, g, b, grid, shaped, fits, group)))

def p4_conv_ge(+order: List<&2, Nat>, +mats: Nat, +n: Nat, +k: Nat, +G: Nat, +g: Nat,
    +grid: {Nat.is_lt(0n, G) == True{} : Bool},
    +shaped: {Nat.is_le(128n, k) == True{} : Bool},
    +fits: {Nat.is_le(G, Impl.total(mats, n, k)) == True{} : Bool},
    +group: {Nat.is_lt(g, Impl.groups(mats, n)) == True{} : Bool},
    +h: {Spec.all_touch(order, g, Impl.kt(k), Impl.total(mats, n, k), G) == True{} : Bool})
  -> {p4_all_ge(order, Impl.b_lo(g, Impl.kt(k), Impl.total(mats, n, k), G)) == True{}
    : Bool}:
  match order:
    case Nil{}:
      {==}
    case +a <> rest:
      +hin = Equal.trans(Bool,
        p4_in(Impl.b_lo(g, Impl.kt(k), Impl.total(mats, n, k), G),
          Impl.b_hi(g, Impl.kt(k), Impl.total(mats, n, k), G), a),
        Spec.touches(a, g, Impl.kt(k), Impl.total(mats, n, k), G), True{},
        Equal.sym(Bool, Spec.touches(a, g, Impl.kt(k), Impl.total(mats, n, k), G),
          p4_in(Impl.b_lo(g, Impl.kt(k), Impl.total(mats, n, k), G),
            Impl.b_hi(g, Impl.kt(k), Impl.total(mats, n, k), G), a),
          p4_ct(mats, n, k, G, g, a, grid, shaped, fits, group)),
        ar_and_l(Spec.touches(a, g, Impl.kt(k), Impl.total(mats, n, k), G),
          Spec.all_touch(rest, g, Impl.kt(k), Impl.total(mats, n, k), G), h))
      ar_and_intro(
        Nat.is_le(Impl.b_lo(g, Impl.kt(k), Impl.total(mats, n, k), G), a),
        p4_all_ge(rest, Impl.b_lo(g, Impl.kt(k), Impl.total(mats, n, k), G)),
        ar_and_l(Nat.is_le(Impl.b_lo(g, Impl.kt(k), Impl.total(mats, n, k), G), a),
          Nat.is_le(a, Impl.b_hi(g, Impl.kt(k), Impl.total(mats, n, k), G)), hin),
        p4_conv_ge(rest, mats, n, k, G, g, grid, shaped, fits, group,
          ar_and_r(Spec.touches(a, g, Impl.kt(k), Impl.total(mats, n, k), G),
            Spec.all_touch(rest, g, Impl.kt(k), Impl.total(mats, n, k), G), h)))

# Blocks b0 .. b0 + a - 1 inside [b_lo, b_hi] have all arrived.
def p4_mk_arr(+a: Nat, +b0: Nat, +order: List<&2, Nat>, +mats: Nat, +n: Nat, +k: Nat,
    +G: Nat, +g: Nat,
    +grid: {Nat.is_lt(0n, G) == True{} : Bool},
    +shaped: {Nat.is_le(128n, k) == True{} : Bool},
    +fits: {Nat.is_le(G, Impl.total(mats, n, k)) == True{} : Bool},
    +group: {Nat.is_lt(g, Impl.groups(mats, n)) == True{} : Bool},
    +hc: {Spec.all_arrived(G, order, g, Impl.kt(k), Impl.total(mats, n, k), G) == True{}
      : Bool},
    +hL: {Nat.is_le(Impl.b_lo(g, Impl.kt(k), Impl.total(mats, n, k), G), b0) == True{} : Bool},
    +hH: {Nat.is_le(Nat.add(a, b0), 1n+Impl.b_hi(g, Impl.kt(k), Impl.total(mats, n, k), G))
      == True{} : Bool})
  -> {p4_all_arr(a, b0, order) == True{} : Bool}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      +hHG = p4_hHG(mats, n, k, G, g, grid, shaped, fits, group)
      +hbH = ar_le_trans(b0, Nat.add(p, b0),
        Impl.b_hi(g, Impl.kt(k), Impl.total(mats, n, k), G), ar_le_add_l(p, b0), hH)
      +ht = Equal.trans(Bool, Spec.touches(b0, g, Impl.kt(k), Impl.total(mats, n, k), G),
        p4_in(Impl.b_lo(g, Impl.kt(k), Impl.total(mats, n, k), G),
          Impl.b_hi(g, Impl.kt(k), Impl.total(mats, n, k), G), b0), True{},
        p4_ct(mats, n, k, G, g, b0, grid, shaped, fits, group),
        ar_and_intro(Nat.is_le(Impl.b_lo(g, Impl.kt(k), Impl.total(mats, n, k), G), b0),
          Nat.is_le(b0, Impl.b_hi(g, Impl.kt(k), Impl.total(mats, n, k), G)), hL, hbH))
      ar_and_intro(Spec.arrived(order, b0), p4_all_arr(p, 1n+b0, order),
        p4_arr(G, order, g, Impl.kt(k), Impl.total(mats, n, k), G, b0, hc,
          ar_le_lt_trans(b0, Impl.b_hi(g, Impl.kt(k), Impl.total(mats, n, k), G), G,
            hbH, hHG),
          ht),
        p4_mk_arr(p, 1n+b0, order, mats, n, k, G, g, grid, shaped, fits, group, hc,
          ar_le_trans(Impl.b_lo(g, Impl.kt(k), Impl.total(mats, n, k), G), b0, 1n+b0,
            hL, p4_le_succ(b0)),
          Equal.trans(Bool,
            Nat.is_le(Nat.add(p, 1n+b0), 1n+Impl.b_hi(g, Impl.kt(k), Impl.total(mats, n, k), G)),
            Nat.is_le(1n+Nat.add(p, b0), 1n+Impl.b_hi(g, Impl.kt(k), Impl.total(mats, n, k), G)),
            True{},
            Equal.cong(Nat, Bool,
              z => Nat.is_le(z, 1n+Impl.b_hi(g, Impl.kt(k), Impl.total(mats, n, k), G)),
              Nat.add(p, 1n+b0), 1n+Nat.add(p, b0), ar_add_succ_r(p, b0)),
            hH)))

def p4_sub1(+d: Nat) -> {Nat.sub(Nat.add(d, 1n), 1n) == d : Nat}:
  Equal.trans(Nat, Nat.sub(Nat.add(d, 1n), 1n), Nat.sub(Nat.add(1n, d), 1n), d,
    Equal.cong(Nat, Nat, z => Nat.sub(z, 1n), Nat.add(d, 1n), Nat.add(1n, d),
      ar_add_comm(d, 1n)),
    ar_sub_zero_r(d))

# ---- reduction_order ----

def Laws.reduction_order(mats, n, k, G, g, order, ws, grid, shaped, fits, group, room,
    touching, complete):
  +gr = grid
  +sh = shaped
  +fi = fits
  +gp = group
  +rm = room
  +hLH = p4_hLH(mats, n, k, G, g, gr, sh, fi, gp)
  +hHG = p4_hHG(mats, n, k, G, g, gr, sh, fi, gp)
  +L = Impl.b_lo(g, Impl.kt(k), Impl.total(mats, n, k), G)
  +H = Impl.b_hi(g, Impl.kt(k), Impl.total(mats, n, k), G)
  +d = Nat.sub(H, L)
  +W = Impl.arrive(order, ws, L)
  +hall = p4_conv_ge(order, mats, n, k, G, g, gr, sh, fi, gp, touching)
  +harr = p4_mk_arr(1n+d, L, order, mats, n, k, G, g, gr, sh, fi, gp, complete,
    ar_le_refl(L),
    Equal.trans(Bool, Nat.is_le(Nat.add(d, L), H), Nat.is_le(H, H), True{},
      Equal.cong(Nat, Bool, z => Nat.is_le(z, H), Nat.add(d, L), H,
        Equal.trans(Nat, Nat.add(d, L), Nat.add(L, d), H, ar_add_comm(d, L),
          ar_sub_add(H, L, hLH))),
      ar_le_refl(H)))
  +hs0 = p4_slot_c(0n, order, ws, L, hall,
    ar_and_l(Spec.arrived(order, L), p4_all_arr(d, 1n+L, order), harr),
    Equal.trans(Bool, Nat.is_lt(0n, List.length(&2, Spec.Sum, ws)),
      Nat.is_le(1n, List.length(&2, Spec.Sum, ws)), True{},
      ar_lt_succ_le(0n, List.length(&2, Spec.Sum, ws)),
      ar_le_trans(1n, Nat.add(d, 1n), List.length(&2, Spec.Sum, ws),
        ar_le_add_l(d, 1n), rm)))
  %Equal.sym(Nat, Nat.sub(Nat.add(d, 1n), 1n), d, p4_sub1(d)) : {Some{Impl.finish.go(_, 1n, Impl.slot_at(W, 0n), W)} == Spec.reduce(g, Impl.kt(k), Impl.total(mats, n, k), G) : Maybe<&2, Spec.Sum>}
  %Equal.sym(Spec.Sum, Impl.slot_at(W, 0n), Spec.Part{L}, hs0) : {Some{Impl.finish.go(d, 1n, _, W)} == Spec.reduce(g, Impl.kt(k), Impl.total(mats, n, k), G) : Maybe<&2, Spec.Sum>}
  Equal.trans(Maybe<&2, Spec.Sum>, Some{Impl.finish.go(d, 1n, Spec.Part{L}, W)},
    Some{p4_asc(d, 1n+L, Spec.Part{L})},
    Spec.reduce(g, Impl.kt(k), Impl.total(mats, n, k), G),
    Equal.cong(Spec.Sum, Maybe<&2, Spec.Sum>, z => Some{z},
      Impl.finish.go(d, 1n, Spec.Part{L}, W), p4_asc(d, 1n+L, Spec.Part{L}),
      p4_fin(d, 1n, Spec.Part{L}, order, ws, L, hall,
        ar_and_r(Spec.arrived(order, L), p4_all_arr(d, 1n+L, order), harr), rm)),
    Equal.sym(Maybe<&2, Spec.Sum>, Spec.reduce(g, Impl.kt(k), Impl.total(mats, n, k), G),
      Some{p4_asc(d, 1n+L, Spec.Part{L})},
      Equal.trans(Maybe<&2, Spec.Sum>,
        Spec.reduce(g, Impl.kt(k), Impl.total(mats, n, k), G),
        p4_rng(G, 0n, L, H, None{}), Some{p4_asc(d, 1n+L, Spec.Part{L})},
        p4_conv_fold(G, 0n, None{}, mats, n, k, G, g, gr, sh, fi, gp),
        p4_whole(G, L, H, hLH, hHG))))

# ==== Law mainloop_trace ====

def p5_rt(-a: Bool, -b: Bool, e: {a == b : Bool}, h: {a == True{} : Bool})
  -> {b == True{} : Bool}:
  Equal.trans(Bool, b, a, True{}, Equal.sym(Bool, a, b, e), h)

def p5_fst(-A: Type, -B: Type, p: A & B) -> A:
  match p:
    case (a, b):
      a

def p5_snd(-A: Type, -B: Type, p: A & B) -> B:
  match p:
    case (a, b):
      b

def p5_not_false(x: Bool, h: {Bool.not(x) == False{} : Bool}) -> {x == True{} : Bool}:
  match x:
    case True{}:
      {==}
    case False{}:
      Empty.absurd({False{} == True{} : Bool}, ar_true_false(h))

# b | c from a | b and b -> c.
def p5_or_mono(a: Bool, -b: Bool, -c: Bool, h: {Bool.or(a, b) == True{} : Bool},
    imp: {b == True{} : Bool} -> {c == True{} : Bool})
  -> {Bool.or(a, c) == True{} : Bool}:
  match a:
    case True{}:
      {==}
    case False{}:
      imp(h)

# a < b implies a != b.
def p5_lt_ne(a: Nat, b: Nat, lt: {Nat.is_lt(a, b) == True{} : Bool})
  -> {Nat.is_eq(a, b) == False{} : Bool}:
  match a b:
    case 0n 0n:
      Empty.absurd({Nat.is_eq(0n, 0n) == False{} : Bool}, ar_false_true(lt))
    case 0n 1n+q:
      {==}
    case 1n+p 0n:
      Empty.absurd({Nat.is_eq(1n+p, 0n) == False{} : Bool}, ar_false_true(lt))
    case 1n+p 1n+q:
      p5_lt_ne(p, q, lt)

def p5_eq_lt_absurd_go(-a: Nat, +c: Nat, eq: {a == c : Nat})
  -> {False{} == Nat.is_lt(a, c) : Bool}:
  %Equal.sym(Nat, a, c, eq) : {False{} == Nat.is_lt(_, c) : Bool}
  Equal.sym(Bool, Nat.is_lt(c, c), False{}, ar_lt_irrefl(c))

# a == c and a < c is absurd.
def p5_eq_lt_absurd(-a: Nat, +c: Nat, eq: {a == c : Nat}, lt: {Nat.is_lt(a, c) == True{} : Bool})
  -> Empty:
  ar_false_true(Equal.trans(Bool, False{}, Nat.is_lt(a, c), True{},
    p5_eq_lt_absurd_go(a, c, eq), lt))

# i < 1 + (k + i).
def p5_lt_succ_add(+k: Nat, +i: Nat) -> {Nat.is_lt(i, 1n+Nat.add(k, i)) == True{} : Bool}:
  p5_rt(Nat.is_le(i, Nat.add(k, i)), Nat.is_lt(i, 1n+Nat.add(k, i)),
    Equal.sym(Bool, Nat.is_lt(i, 1n+Nat.add(k, i)), Nat.is_le(1n+i, 1n+Nat.add(k, i)),
      ar_lt_succ_le(i, 1n+Nat.add(k, i))),
    ar_le_add_l(k, i))

# y < k + y for k >= 1.
def p5_lt_add_pos(k: Nat, +y: Nat, kpos: {Nat.is_lt(0n, k) == True{} : Bool})
  -> {Nat.is_lt(y, Nat.add(k, y)) == True{} : Bool}:
  match k:
    case 0n:
      Empty.absurd({Nat.is_lt(y, Nat.add(0n, y)) == True{} : Bool}, ar_false_true(kpos))
    case 1n+k0:
      p5_lt_succ_add(k0, y)

def p5_sub_self(n: Nat) -> {Nat.sub(n, n) == 0n : Nat}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      p5_sub_self(p)

def p5_min_le_l(+a: Nat, +b: Nat) -> {Nat.is_le(Nat.min(a, b), a) == True{} : Bool}:
  match a b:
    case 0n _:
      {==}
    case 1n+p 0n:
      {==}
    case 1n+p 1n+q:
      p5_min_le_l(p, q)

def p5_min_le_r(+a: Nat, +b: Nat) -> {Nat.is_le(Nat.min(a, b), b) == True{} : Bool}:
  match a b:
    case 0n _:
      ar_h_le_zero(b)
    case 1n+p 0n:
      {==}
    case 1n+p 1n+q:
      p5_min_le_r(p, q)

# min(a, b) is a or b.
def p5_min_split(+a: Nat, +b: Nat)
  -> {Bool.or(Nat.is_eq(Nat.min(a, b), a), Nat.is_eq(Nat.min(a, b), b)) == True{} : Bool}:
  match a b:
    case 0n _:
      {==}
    case 1n+p 0n:
      {==}
    case 1n+p 1n+q:
      p5_min_split(p, q)

# min(a, b) < a forces min(a, b) == b.
def p5_min_lt_l(+a: Nat, +b: Nat, h: {Nat.is_lt(Nat.min(a, b), a) == True{} : Bool})
  -> {Nat.min(a, b) == b : Nat}:
  match a b:
    case 0n _:
      Empty.absurd({Nat.min(0n, b) == b : Nat}, ar_false_true(h))
    case 1n+p 0n:
      {==}
    case 1n+p 1n+q:
      %p5_min_lt_l(p, q, h) : {1n+Nat.min(p, q) == 1n+_ : Nat}
      {==}

def p5_max_r(+j: Nat, +x: Nat, le: {Nat.is_le(j, x) == True{} : Bool})
  -> {Nat.max(j, x) == x : Nat}:
  match j x:
    case 0n _:
      {==}
    case 1n+p 0n:
      Empty.absurd({Nat.max(1n+p, 0n) == 0n : Nat}, ar_false_true(le))
    case 1n+p 1n+q:
      %p5_max_r(p, q, le) : {1n+Nat.max(p, q) == 1n+_ : Nat}
      {==}

def p5_sub_split_go(+n: Nat, +se: Nat, +j: Nat,
    jse: {Nat.is_le(j, se) == True{} : Bool}, sen: {Nat.is_le(se, n) == True{} : Bool})
  -> {Nat.add(j, Nat.add(Nat.sub(se, j), Nat.sub(n, se))) == n : Nat}:
  %ar_add_assoc(j, Nat.sub(se, j), Nat.sub(n, se)) : {_ == n : Nat}
  %Equal.sym(Nat, Nat.add(j, Nat.sub(se, j)), se, ar_sub_add(se, j, jse)) : {Nat.add(_, Nat.sub(n, se)) == n : Nat}
  ar_sub_add(n, se, sen)

# n - j == (se - j) + (n - se) for j <= se <= n.
def p5_sub_split(+n: Nat, +se: Nat, +j: Nat,
    jse: {Nat.is_le(j, se) == True{} : Bool}, sen: {Nat.is_le(se, n) == True{} : Bool})
  -> {Nat.sub(n, j) == Nat.add(Nat.sub(se, j), Nat.sub(n, se)) : Nat}:
  %p5_sub_split_go(n, se, j, jse, sen) : {Nat.sub(_, j) == Nat.add(Nat.sub(se, j), Nat.sub(n, se)) : Nat}
  ar_add_sub(j, Nat.add(Nat.sub(se, j), Nat.sub(n, se)))


# ---- iteration facts: group and flush condition ----

# g * KT <= c < (1 + g) * KT gives c / KT == g.
def p5_div_eq(+g: Nat, +c: Nat, +KT: Nat, +pos: {Nat.is_lt(0n, KT) == True{} : Bool},
    lo: {Nat.is_le(Nat.mul(g, KT), c) == True{} : Bool},
    hi: {Nat.is_lt(c, Nat.mul(1n+g, KT)) == True{} : Bool})
  -> {Nat.div(c, KT) == g : Nat}:
  +lt = p5_rt(Nat.is_lt(c, Nat.mul(1n+g, KT)), Nat.is_lt(Nat.div(c, KT), 1n+g),
    Equal.sym(Bool, Nat.is_lt(Nat.div(c, KT), 1n+g), Nat.is_lt(c, Nat.mul(1n+g, KT)),
      ar_div_lt(1n+g, c, KT, pos)), hi)
  ar_le_antisym(Nat.div(c, KT), g,
    p5_rt(Nat.is_lt(Nat.div(c, KT), 1n+g), Nat.is_le(Nat.div(c, KT), g),
      ar_lt_succ_le(Nat.div(c, KT), 1n+g), lt),
    p5_rt(Nat.is_le(Nat.mul(g, KT), c), Nat.is_le(g, Nat.div(c, KT)),
      ar_div_le(g, c, KT, pos), lo))

# (1 + g) * KT is a multiple of KT.
def p5_mod_zero(+g: Nat, +KT: Nat, pos: {Nat.is_lt(0n, KT) == True{} : Bool})
  -> {Nat.mod(Nat.mul(1n+g, KT), KT) == 0n : Nat}:
  %ar_add_zero_r(Nat.mul(1n+g, KT)) : {Nat.mod(_, KT) == 0n : Nat}
  p5_snd({Nat.div(Nat.add(Nat.mul(1n+g, KT), 0n), KT) == 1n+g : Nat},
    {Nat.mod(Nat.add(Nat.mul(1n+g, KT), 0n), KT) == 0n : Nat},
    ar_div_unique(1n+g, 0n, KT, pos))

def p5_mod_nz_eq(+g: Nat, +c: Nat, +KT: Nat, pos: {Nat.is_lt(0n, KT) == True{} : Bool},
    dq: {Nat.div(c, KT) == g : Nat}, h: {Nat.mod(c, KT) == 0n : Nat})
  -> {Nat.add(Nat.mul(g, KT), 0n) == c : Nat}:
  %h : {Nat.add(Nat.mul(g, KT), _) == c : Nat}
  %dq : {Nat.add(Nat.mul(_, KT), Nat.mod(c, KT)) == c : Nat}
  p5_fst({Nat.add(Nat.mul(Nat.div(c, KT), KT), Nat.mod(c, KT)) == c : Nat},
    {Nat.is_lt(Nat.mod(c, KT), KT) == True{} : Bool}, ar_divmod(c, KT, pos))

# g * KT < c < (1 + g) * KT: c is not a multiple of KT.
def p5_mod_nz(+g: Nat, +c: Nat, +KT: Nat, +pos: {Nat.is_lt(0n, KT) == True{} : Bool},
    +lo: {Nat.is_lt(Nat.mul(g, KT), c) == True{} : Bool},
    +hi: {Nat.is_lt(c, Nat.mul(1n+g, KT)) == True{} : Bool})
  -> {Nat.is_eq(Nat.mod(c, KT), 0n) == False{} : Bool}:
  ar_eq_false(Nat.mod(c, KT), 0n, h => p5_eq_lt_absurd(Nat.mul(g, KT), c,
    Equal.trans(Nat, Nat.mul(g, KT), Nat.add(Nat.mul(g, KT), 0n), c,
      Equal.sym(Nat, Nat.add(Nat.mul(g, KT), 0n), Nat.mul(g, KT),
        ar_add_zero_r(Nat.mul(g, KT))),
      p5_mod_nz_eq(g, c, KT, pos,
        p5_div_eq(g, c, KT, pos, ar_lt_le(Nat.mul(g, KT), c, lo), hi), h)),
    lo))

def p5_cond_imp(+c: Nat, +X: Nat, +KT: Nat, +g: Nat, pos: {Nat.is_lt(0n, KT) == True{} : Bool},
    hX: {X == Nat.mul(1n+g, KT) : Nat}, h: {Nat.is_eq(c, X) == True{} : Bool})
  -> {Nat.is_eq(Nat.mod(c, KT), 0n) == True{} : Bool}:
  %Equal.sym(Nat, c, X, ar_eq_true(c, X, h)) : {Nat.is_eq(Nat.mod(_, KT), 0n) == True{} : Bool}
  %Equal.sym(Nat, X, Nat.mul(1n+g, KT), hX) : {Nat.is_eq(Nat.mod(_, KT), 0n) == True{} : Bool}
  %Equal.sym(Nat, Nat.mod(Nat.mul(1n+g, KT), KT), 0n, p5_mod_zero(g, KT, pos)) : {Nat.is_eq(_, 0n) == True{} : Bool}
  {==}

# The flush fires at the segment's last iteration c + 1 == min(e, (1 + g) * KT).
def p5_cond_last(+c: Nat, +e: Nat, +KT: Nat, +g: Nat, +pos: {Nat.is_lt(0n, KT) == True{} : Bool},
    hc: {c == Nat.min(e, Nat.mul(1n+g, KT)) : Nat})
  -> {Bool.or(Nat.is_eq(c, e), Nat.is_eq(Nat.mod(c, KT), 0n)) == True{} : Bool}:
  %Equal.sym(Nat, c, Nat.min(e, Nat.mul(1n+g, KT)), hc) : {Bool.or(Nat.is_eq(_, e), Nat.is_eq(Nat.mod(_, KT), 0n)) == True{} : Bool}
  p5_or_mono(Nat.is_eq(Nat.min(e, Nat.mul(1n+g, KT)), e),
    Nat.is_eq(Nat.min(e, Nat.mul(1n+g, KT)), Nat.mul(1n+g, KT)),
    Nat.is_eq(Nat.mod(Nat.min(e, Nat.mul(1n+g, KT)), KT), 0n),
    p5_min_split(e, Nat.mul(1n+g, KT)),
    h => p5_cond_imp(Nat.min(e, Nat.mul(1n+g, KT)), Nat.mul(1n+g, KT), KT, g, pos, {==}, h))

# No flush strictly inside the segment: g * KT < c < min(e, (1 + g) * KT).
def p5_cond_mid(+c: Nat, +e: Nat, +KT: Nat, +g: Nat, +pos: {Nat.is_lt(0n, KT) == True{} : Bool},
    +lo: {Nat.is_lt(Nat.mul(g, KT), c) == True{} : Bool},
    hi: {Nat.is_lt(c, Nat.min(e, Nat.mul(1n+g, KT))) == True{} : Bool})
  -> {Bool.or(Nat.is_eq(c, e), Nat.is_eq(Nat.mod(c, KT), 0n)) == False{} : Bool}:
  +both = p5_rt(Nat.is_lt(c, Nat.min(e, Nat.mul(1n+g, KT))),
    Bool.and(Nat.is_lt(c, e), Nat.is_lt(c, Nat.mul(1n+g, KT))),
    ar_min_lt(c, e, Nat.mul(1n+g, KT)), hi)
  %Equal.sym(Bool, Nat.is_eq(c, e), False{}, p5_lt_ne(c, e, ar_and_l(Nat.is_lt(c, e), Nat.is_lt(c, Nat.mul(1n+g, KT)), both))) : {Bool.or(_, Nat.is_eq(Nat.mod(c, KT), 0n)) == False{} : Bool}
  p5_mod_nz(g, c, KT, pos, lo,
    ar_and_r(Nat.is_lt(c, e), Nat.is_lt(c, Nat.mul(1n+g, KT)), both))

# ---- one inner loop == the trace of one segment ----

def p5_lt_end(+k0: Nat, +i: Nat, -t: Nat, hend: {Nat.add(1n+k0, i) == t : Nat})
  -> {Nat.is_lt(i, t) == True{} : Bool}:
  %hend : {Nat.is_lt(i, _) == True{} : Bool}
  p5_lt_succ_add(k0, i)

def p5_lt_mid(+k1: Nat, +i: Nat, -t: Nat, hend: {Nat.add(1n+1n+k1, i) == t : Nat})
  -> {Nat.is_lt(1n+i, t) == True{} : Bool}:
  %hend : {Nat.is_lt(1n+i, _) == True{} : Bool}
  p5_lt_succ_add(k1, i)

def p5_lo_step(+s: Nat, +j: Nat, +M: Nat, lo: {Nat.is_le(M, Nat.add(s, j)) == True{} : Bool})
  -> {Nat.is_le(M, Nat.add(s, 1n+j)) == True{} : Bool}:
  %Equal.sym(Nat, Nat.add(s, 1n+j), 1n+Nat.add(s, j), ar_add_succ_r(s, j)) : {Nat.is_le(M, _) == True{} : Bool}
  ar_le_trans(M, Nat.add(s, j), 1n+Nat.add(s, j), lo, ar_h_le_succ(Nat.add(s, j)))

def p5_hend_step(+k1: Nat, +s: Nat, +j: Nat, -t: Nat,
    hend: {Nat.add(1n+1n+k1, Nat.add(s, j)) == t : Nat})
  -> {Nat.add(1n+k1, Nat.add(s, 1n+j)) == t : Nat}:
  %Equal.sym(Nat, Nat.add(s, 1n+j), 1n+Nat.add(s, j), ar_add_succ_r(s, j)) : {Nat.add(1n+k1, _) == t : Nat}
  %Equal.sym(Nat, Nat.add(k1, 1n+Nat.add(s, j)), 1n+Nat.add(k1, Nat.add(s, j)), ar_add_succ_r(k1, Nat.add(s, j))) : {1n+_ == t : Nat}
  hend

# The group of the first iteration i of a segment with k0 + 1 iterations left.
def p5_seg_dq(+k0: Nat, +s: Nat, +j: Nat, +e: Nat, +KT: Nat, +g: Nat,
    +pos: {Nat.is_lt(0n, KT) == True{} : Bool},
    lo: {Nat.is_le(Nat.mul(g, KT), Nat.add(s, j)) == True{} : Bool},
    hend: {Nat.add(1n+k0, Nat.add(s, j)) == Nat.min(e, Nat.mul(1n+g, KT)) : Nat})
  -> {Nat.div(Nat.add(s, j), KT) == g : Nat}:
  p5_div_eq(g, Nat.add(s, j), KT, pos, lo,
    ar_lt_le_trans(Nat.add(s, j), Nat.min(e, Nat.mul(1n+g, KT)), Nat.mul(1n+g, KT),
      p5_lt_end(k0, Nat.add(s, j), Nat.min(e, Nat.mul(1n+g, KT)), hend),
      p5_min_le_r(e, Nat.mul(1n+g, KT))))

# For i = s + j with g * KT <= i and i + k == t = min(e, (1 + g) * KT), k >= 1: the k inner
# iterations j .. j + k - 1 followed by Flush{g} are the trace of the iterations i .. t - 1.
def p5_seg(k: Nat, +m: Nat, +s: Nat, +e: Nat, +KT: Nat, +g: Nat, +j: Nat,
    +pos: {Nat.is_lt(0n, KT) == True{} : Bool},
    +lo: {Nat.is_le(Nat.mul(g, KT), Nat.add(s, j)) == True{} : Bool},
    kpos: {Nat.is_lt(0n, k) == True{} : Bool},
    +hend: {Nat.add(k, Nat.add(s, j)) == Nat.min(e, Nat.mul(1n+g, KT)) : Nat})
  -> {Spec.trace(Nat.add(k, m), Nat.add(s, j), e, KT) ==
    Impl.inner(k, s, g, j,
      Spec.Flush{g} <> Spec.trace(m, Nat.min(e, Nat.mul(1n+g, KT)), e, KT)) : List<&2, Spec.Ev>}:
  match k:
    case 0n:
      Empty.absurd({Spec.trace(Nat.add(0n, m), Nat.add(s, j), e, KT) ==
        Impl.inner(0n, s, g, j,
          Spec.Flush{g} <> Spec.trace(m, Nat.min(e, Nat.mul(1n+g, KT)), e, KT))
        : List<&2, Spec.Ev>}, ar_false_true(kpos))
    case 1n+0n:
      %Equal.sym(Nat, Nat.div(Nat.add(s, j), KT), g, p5_seg_dq(0n, s, j, e, KT, g, pos, lo, hend)) : {Spec.Tile{_, Nat.add(s, j)} <> Spec.trace.flush(Bool.or(Nat.is_eq(1n+Nat.add(s, j), e), Nat.is_eq(Nat.mod(1n+Nat.add(s, j), KT), 0n)), _, Spec.trace(m, 1n+Nat.add(s, j), e, KT)) == Impl.inner(1n, s, g, j, Spec.Flush{g} <> Spec.trace(m, Nat.min(e, Nat.mul(1n+g, KT)), e, KT)) : List<&2, Spec.Ev>}
      %Equal.sym(Bool, Bool.or(Nat.is_eq(1n+Nat.add(s, j), e), Nat.is_eq(Nat.mod(1n+Nat.add(s, j), KT), 0n)), True{}, p5_cond_last(1n+Nat.add(s, j), e, KT, g, pos, hend)) : {Spec.Tile{g, Nat.add(s, j)} <> Spec.trace.flush(_, g, Spec.trace(m, 1n+Nat.add(s, j), e, KT)) == Impl.inner(1n, s, g, j, Spec.Flush{g} <> Spec.trace(m, Nat.min(e, Nat.mul(1n+g, KT)), e, KT)) : List<&2, Spec.Ev>}
      %hend : {Spec.Tile{g, Nat.add(s, j)} <> Spec.Flush{g} <> Spec.trace(m, 1n+Nat.add(s, j), e, KT) == Spec.Tile{g, Nat.add(s, j)} <> Spec.Flush{g} <> Spec.trace(m, _, e, KT) : List<&2, Spec.Ev>}
      {==}
    case 1n+1n+k1:
      +r1 = k1
      %Equal.sym(Nat, Nat.div(Nat.add(s, j), KT), g, p5_seg_dq(1n+r1, s, j, e, KT, g, pos, lo, hend)) : {Spec.Tile{_, Nat.add(s, j)} <> Spec.trace.flush(Bool.or(Nat.is_eq(1n+Nat.add(s, j), e), Nat.is_eq(Nat.mod(1n+Nat.add(s, j), KT), 0n)), _, Spec.trace(Nat.add(1n+k1, m), 1n+Nat.add(s, j), e, KT)) == Impl.inner(1n+1n+k1, s, g, j, Spec.Flush{g} <> Spec.trace(m, Nat.min(e, Nat.mul(1n+g, KT)), e, KT)) : List<&2, Spec.Ev>}
      +cm = p5_cond_mid(1n+Nat.add(s, j), e, KT, g, pos,
        p5_rt(Nat.is_le(Nat.mul(g, KT), Nat.add(s, j)), Nat.is_lt(Nat.mul(g, KT), 1n+Nat.add(s, j)),
          Equal.sym(Bool, Nat.is_lt(Nat.mul(g, KT), 1n+Nat.add(s, j)),
            Nat.is_le(1n+Nat.mul(g, KT), 1n+Nat.add(s, j)),
            ar_lt_succ_le(Nat.mul(g, KT), 1n+Nat.add(s, j))), lo),
        p5_lt_mid(r1, Nat.add(s, j), Nat.min(e, Nat.mul(1n+g, KT)), hend))
      %Equal.sym(Bool, Bool.or(Nat.is_eq(1n+Nat.add(s, j), e), Nat.is_eq(Nat.mod(1n+Nat.add(s, j), KT), 0n)), False{}, cm) : {Spec.Tile{g, Nat.add(s, j)} <> Spec.trace.flush(_, g, Spec.trace(Nat.add(1n+k1, m), 1n+Nat.add(s, j), e, KT)) == Impl.inner(1n+1n+k1, s, g, j, Spec.Flush{g} <> Spec.trace(m, Nat.min(e, Nat.mul(1n+g, KT)), e, KT)) : List<&2, Spec.Ev>}
      %p5_seg(1n+r1, m, s, e, KT, g, 1n+j, pos, p5_lo_step(s, j, Nat.mul(g, KT), lo), {==}, p5_hend_step(r1, s, j, Nat.min(e, Nat.mul(1n+g, KT)), hend)) : {Spec.Tile{g, Nat.add(s, j)} <> Spec.trace(Nat.add(1n+k1, m), 1n+Nat.add(s, j), e, KT) == Spec.Tile{g, Nat.add(s, j)} <> _ : List<&2, Spec.Ev>}
      %ar_add_succ_r(s, j) : {Spec.Tile{g, Nat.add(s, j)} <> Spec.trace(Nat.add(1n+k1, m), _, e, KT) == Spec.Tile{g, Nat.add(s, j)} <> Spec.trace(Nat.add(1n+k1, m), Nat.add(s, 1n+j), e, KT) : List<&2, Spec.Ev>}
      {==}


# ---- the outer loop == the trace of the remaining range ----

def p5_lt_rw(-a: Nat, -b: Nat, +c: Nat, eq: {a == b : Nat}, h: {Nat.is_lt(a, c) == True{} : Bool})
  -> {Nat.is_lt(b, c) == True{} : Bool}:
  %eq : {Nat.is_lt(_, c) == True{} : Bool}
  h

def p5_le_div(+g: Nat, -c: Nat, -KT: Nat, dq: {Nat.div(c, KT) == g : Nat})
  -> {Nat.is_le(g, Nat.div(c, KT)) == True{} : Bool}:
  %Equal.sym(Nat, Nat.div(c, KT), g, dq) : {Nat.is_le(g, _) == True{} : Bool}
  ar_le_refl(g)

def p5_lt_div(+g: Nat, -c: Nat, -KT: Nat, dq: {Nat.div(c, KT) == g : Nat})
  -> {Nat.is_lt(Nat.div(c, KT), 1n+g) == True{} : Bool}:
  %Equal.sym(Nat, Nat.div(c, KT), g, dq) : {Nat.is_lt(_, 1n+g) == True{} : Bool}
  p5_lt_succ_add(0n, g)

def p5_div_add0(+s: Nat, +KT: Nat) -> {Nat.div(Nat.add(s, 0n), KT) == Nat.div(s, KT) : Nat}:
  %Equal.sym(Nat, Nat.add(s, 0n), s, ar_add_zero_r(s)) : {Nat.div(_, KT) == Nat.div(s, KT) : Nat}
  {==}

def p5_lt_add0(+j: Nat, +x: Nat, h: {Nat.is_lt(j, x) == True{} : Bool})
  -> {Nat.is_lt(Nat.add(j, 0n), x) == True{} : Bool}:
  %Equal.sym(Nat, Nat.add(j, 0n), j, ar_add_zero_r(j)) : {Nat.is_lt(_, x) == True{} : Bool}
  h

# t <= e == s + (e - s).
def p5_t_le(+s: Nat, +e: Nat, +X: Nat, ord: {Nat.is_le(s, e) == True{} : Bool})
  -> {Nat.is_le(Nat.min(e, X), Nat.add(s, Nat.sub(e, s))) == True{} : Bool}:
  %Equal.sym(Nat, Nat.add(s, Nat.sub(e, s)), e, ar_sub_add(e, s, ord)) : {Nat.is_le(Nat.min(e, X), _) == True{} : Bool}
  p5_min_le_l(e, X)

# (se - j) + (s + j) == s + se == t.
def p5_hend0(+s: Nat, +j: Nat, +se: Nat, -t: Nat, jse: {Nat.is_le(j, se) == True{} : Bool},
    hs: {Nat.add(s, se) == t : Nat})
  -> {Nat.add(Nat.sub(se, j), Nat.add(s, j)) == t : Nat}:
  %ar_add_comm(Nat.add(s, j), Nat.sub(se, j)) : {_ == t : Nat}
  %Equal.sym(Nat, Nat.add(Nat.add(s, j), Nat.sub(se, j)), Nat.add(s, Nat.add(j, Nat.sub(se, j))), ar_add_assoc(s, j, Nat.sub(se, j))) : {_ == t : Nat}
  %Equal.sym(Nat, Nat.add(j, Nat.sub(se, j)), se, ar_sub_add(se, j, jse)) : {Nat.add(s, _) == t : Nat}
  hs

def p5_fuel_lt(+n: Nat, +se: Nat, +j: Nat,
    nsplit: {Nat.sub(n, j) == Nat.add(Nat.sub(se, j), Nat.sub(n, se)) : Nat},
    kpos: {Nat.is_lt(0n, Nat.sub(se, j)) == True{} : Bool})
  -> {Nat.is_lt(Nat.sub(n, se), Nat.sub(n, j)) == True{} : Bool}:
  %Equal.sym(Nat, Nat.sub(n, j), Nat.add(Nat.sub(se, j), Nat.sub(n, se)), nsplit) : {Nat.is_lt(Nat.sub(n, se), _) == True{} : Bool}
  p5_lt_add_pos(Nat.sub(se, j), Nat.sub(n, se), kpos)

# The fuel shrinks: n - se < n - j <= f.
def p5_fuel_step(+n: Nat, +se: Nat, +j: Nat, +f: Nat,
    nsplit: {Nat.sub(n, j) == Nat.add(Nat.sub(se, j), Nat.sub(n, se)) : Nat},
    kpos: {Nat.is_lt(0n, Nat.sub(se, j)) == True{} : Bool},
    hfuel: {Nat.is_lt(Nat.sub(n, j), 1n+f) == True{} : Bool})
  -> {Nat.is_lt(Nat.sub(n, se), f) == True{} : Bool}:
  ar_lt_le_trans(Nat.sub(n, se), Nat.sub(n, j), f, p5_fuel_lt(n, se, j, nsplit, kpos),
    p5_rt(Nat.is_lt(Nat.sub(n, j), 1n+f), Nat.is_le(Nat.sub(n, j), f),
      ar_lt_succ_le(Nat.sub(n, j), 1n+f), hfuel))

# A segment that stops before e ends at (1 + g) * KT, so the next group is g + 1.
def p5_next_group(+s: Nat, +e: Nat, +KT: Nat, +g: Nat, pos: {Nat.is_lt(0n, KT) == True{} : Bool},
    +st: {Nat.is_le(s, Nat.min(e, Nat.mul(1n+g, KT))) == True{} : Bool},
    h: {Nat.is_lt(Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s), Nat.sub(e, s)) == True{} : Bool})
  -> {Nat.div(Nat.add(s, Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s)), KT) == 1n+g : Nat}:
  +hte = p5_lt_rw(Nat.add(s, Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s)), Nat.min(e, Nat.mul(1n+g, KT)), e, ar_sub_add(Nat.min(e, Nat.mul(1n+g, KT)), s, st),
    p5_rt(Nat.is_lt(Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s), Nat.sub(e, s)), Nat.is_lt(Nat.add(s, Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s)), e),
      ar_lt_sub_iff(e, s, Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s)), h))
  %Equal.sym(Nat, Nat.add(s, Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s)), Nat.min(e, Nat.mul(1n+g, KT)), ar_sub_add(Nat.min(e, Nat.mul(1n+g, KT)), s, st)) : {Nat.div(_, KT) == 1n+g : Nat}
  %Equal.sym(Nat, Nat.min(e, Nat.mul(1n+g, KT)), Nat.mul(1n+g, KT), p5_min_lt_l(e, Nat.mul(1n+g, KT), hte)) : {Nat.div(_, KT) == 1n+g : Nat}
  %ar_add_zero_r(Nat.mul(1n+g, KT)) : {Nat.div(_, KT) == 1n+g : Nat}
  p5_fst({Nat.div(Nat.add(Nat.mul(1n+g, KT), 0n), KT) == 1n+g : Nat},
    {Nat.mod(Nat.add(Nat.mul(1n+g, KT), 0n), KT) == 0n : Nat}, ar_div_unique(1n+g, 0n, KT, pos))

# Loop invariant of outer: at a group start j <= n with go == (j < n), group g == (s + j) / KT
# when j < n, and enough fuel, the rest of the loop is the trace of iterations s + j .. e - 1.
def p5_outer(fuel: Nat, go: Bool, +s: Nat, +e: Nat, +KT: Nat, +g: Nat, +j: Nat,
    +pos: {Nat.is_lt(0n, KT) == True{} : Bool},
    +ord: {Nat.is_le(s, e) == True{} : Bool},
    +jn: {Nat.is_le(j, Nat.sub(e, s)) == True{} : Bool},
    +hgo: {Nat.is_lt(j, Nat.sub(e, s)) == go : Bool},
    hg: {Nat.is_lt(j, Nat.sub(e, s)) == True{} : Bool} -> {Nat.div(Nat.add(s, j), KT) == g : Nat},
    +hfuel: {Nat.is_lt(Nat.sub(Nat.sub(e, s), j), fuel) == True{} : Bool})
  -> {Impl.outer(fuel, go, s, e, KT, g, j) == Spec.trace(Nat.sub(Nat.sub(e, s), j), Nat.add(s, j), e, KT) : List<&2, Spec.Ev>}:
  match fuel go:
    case 0n _:
      Empty.absurd({Nil{} == Spec.trace(Nat.sub(Nat.sub(e, s), j), Nat.add(s, j), e, KT) : List<&2, Spec.Ev>},
        ar_false_true(Equal.trans(Bool, False{}, Nat.is_lt(Nat.sub(Nat.sub(e, s), j), 0n), True{},
          Equal.sym(Bool, Nat.is_lt(Nat.sub(Nat.sub(e, s), j), 0n), False{}, ar_h_lt_zero(Nat.sub(Nat.sub(e, s), j))),
          hfuel)))
    case 1n+f False{}:
      +eqjn = ar_le_antisym(j, Nat.sub(e, s), jn,
        p5_not_false(Nat.is_le(Nat.sub(e, s), j),
          Equal.trans(Bool, Bool.not(Nat.is_le(Nat.sub(e, s), j)), Nat.is_lt(j, Nat.sub(e, s)), False{},
            Equal.sym(Bool, Nat.is_lt(j, Nat.sub(e, s)), Bool.not(Nat.is_le(Nat.sub(e, s), j)), ar_lt_not_le(j, Nat.sub(e, s))),
            hgo)))
      %Equal.sym(Nat, j, Nat.sub(e, s), eqjn) : {Nil{} == Spec.trace(Nat.sub(Nat.sub(e, s), _), Nat.add(s, j), e, KT) : List<&2, Spec.Ev>}
      %Equal.sym(Nat, Nat.sub(Nat.sub(e, s), Nat.sub(e, s)), 0n, p5_sub_self(Nat.sub(e, s))) : {Nil{} == Spec.trace(_, Nat.add(s, j), e, KT) : List<&2, Spec.Ev>}
      {==}
    case 1n+f True{}:
      +rf = f
      +dq = hg(hgo)
      +lo = p5_rt(Nat.is_le(g, Nat.div(Nat.add(s, j), KT)), Nat.is_le(Nat.mul(g, KT), Nat.add(s, j)),
        Equal.sym(Bool, Nat.is_le(Nat.mul(g, KT), Nat.add(s, j)), Nat.is_le(g, Nat.div(Nat.add(s, j), KT)),
          ar_div_le(g, Nat.add(s, j), KT, pos)),
        p5_le_div(g, Nat.add(s, j), KT, dq))
      +hiX = p5_rt(Nat.is_lt(Nat.div(Nat.add(s, j), KT), 1n+g), Nat.is_lt(Nat.add(s, j), Nat.mul(1n+g, KT)),
        ar_div_lt(1n+g, Nat.add(s, j), KT, pos), p5_lt_div(g, Nat.add(s, j), KT, dq))
      +hie = p5_rt(Nat.is_lt(j, Nat.sub(e, s)), Nat.is_lt(Nat.add(s, j), e), ar_lt_sub_iff(e, s, j), hgo)
      +hit = p5_rt(Bool.and(Nat.is_lt(Nat.add(s, j), e), Nat.is_lt(Nat.add(s, j), Nat.mul(1n+g, KT))), Nat.is_lt(Nat.add(s, j), Nat.min(e, Nat.mul(1n+g, KT))),
        Equal.sym(Bool, Nat.is_lt(Nat.add(s, j), Nat.min(e, Nat.mul(1n+g, KT))), Bool.and(Nat.is_lt(Nat.add(s, j), e), Nat.is_lt(Nat.add(s, j), Nat.mul(1n+g, KT))),
          ar_min_lt(Nat.add(s, j), e, Nat.mul(1n+g, KT))),
        ar_and_intro(Nat.is_lt(Nat.add(s, j), e), Nat.is_lt(Nat.add(s, j), Nat.mul(1n+g, KT)), hie, hiX))
      +st = ar_le_trans(s, Nat.add(s, j), Nat.min(e, Nat.mul(1n+g, KT)), ar_le_add_r(s, j), ar_lt_le(Nat.add(s, j), Nat.min(e, Nat.mul(1n+g, KT)), hit))
      +hs = ar_sub_add(Nat.min(e, Nat.mul(1n+g, KT)), s, st)
      +jse = p5_rt(Nat.is_lt(Nat.add(s, j), Nat.min(e, Nat.mul(1n+g, KT))), Nat.is_lt(j, Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s)),
        Equal.sym(Bool, Nat.is_lt(j, Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s)), Nat.is_lt(Nat.add(s, j), Nat.min(e, Nat.mul(1n+g, KT))), ar_lt_sub_iff(Nat.min(e, Nat.mul(1n+g, KT)), s, j)), hit)
      +jsele = ar_lt_le(j, Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s), jse)
      +sen = p5_rt(Nat.is_le(Nat.min(e, Nat.mul(1n+g, KT)), Nat.add(s, Nat.sub(e, s))), Nat.is_le(Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s), Nat.sub(e, s)),
        Equal.sym(Bool, Nat.is_le(Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s), Nat.sub(e, s)), Nat.is_le(Nat.min(e, Nat.mul(1n+g, KT)), Nat.add(s, Nat.sub(e, s))), ar_le_sub_iff(Nat.min(e, Nat.mul(1n+g, KT)), s, Nat.sub(e, s))),
        p5_t_le(s, e, Nat.mul(1n+g, KT), ord))
      +kpos = p5_rt(Nat.is_lt(Nat.add(j, 0n), Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s)), Nat.is_lt(0n, Nat.sub(Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s), j)),
        Equal.sym(Bool, Nat.is_lt(0n, Nat.sub(Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s), j)), Nat.is_lt(Nat.add(j, 0n), Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s)),
          ar_lt_sub_iff(Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s), j, 0n)),
        p5_lt_add0(j, Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s), jse))
      +nsplit = p5_sub_split(Nat.sub(e, s), Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s), j, jsele, sen)
      +ih = p5_outer(rf, Nat.is_lt(Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s), Nat.sub(e, s)), s, e, KT, 1n+g, Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s), pos, ord, sen, {==},
        h => p5_next_group(s, e, KT, g, pos, st, h),
        p5_fuel_step(Nat.sub(e, s), Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s), j, rf, nsplit, kpos, hfuel))
      %Equal.sym(Nat, Nat.max(j, Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s)), Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s), p5_max_r(j, Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s), jsele)) : {Impl.inner(Nat.sub(Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s), j), s, g, j, Spec.Flush{g} <> Impl.outer(rf, Nat.is_lt(_, Nat.sub(e, s)), s, e, KT, 1n+g, _)) == Spec.trace(Nat.sub(Nat.sub(e, s), j), Nat.add(s, j), e, KT) : List<&2, Spec.Ev>}
      %Equal.sym(List<&2, Spec.Ev>, Impl.outer(rf, Nat.is_lt(Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s), Nat.sub(e, s)), s, e, KT, 1n+g, Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s)), Spec.trace(Nat.sub(Nat.sub(e, s), Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s)), Nat.add(s, Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s)), e, KT), ih) : {Impl.inner(Nat.sub(Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s), j), s, g, j, Spec.Flush{g} <> _) == Spec.trace(Nat.sub(Nat.sub(e, s), j), Nat.add(s, j), e, KT) : List<&2, Spec.Ev>}
      %Equal.sym(Nat, Nat.add(s, Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s)), Nat.min(e, Nat.mul(1n+g, KT)), hs) : {Impl.inner(Nat.sub(Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s), j), s, g, j, Spec.Flush{g} <> Spec.trace(Nat.sub(Nat.sub(e, s), Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s)), _, e, KT)) == Spec.trace(Nat.sub(Nat.sub(e, s), j), Nat.add(s, j), e, KT) : List<&2, Spec.Ev>}
      %Equal.sym(Nat, Nat.sub(Nat.sub(e, s), j), Nat.add(Nat.sub(Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s), j), Nat.sub(Nat.sub(e, s), Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s))), nsplit) : {Impl.inner(Nat.sub(Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s), j), s, g, j, Spec.Flush{g} <> Spec.trace(Nat.sub(Nat.sub(e, s), Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s)), Nat.min(e, Nat.mul(1n+g, KT)), e, KT)) == Spec.trace(_, Nat.add(s, j), e, KT) : List<&2, Spec.Ev>}
      Equal.sym(List<&2, Spec.Ev>,
        Spec.trace(Nat.add(Nat.sub(Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s), j), Nat.sub(Nat.sub(e, s), Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s))), Nat.add(s, j), e, KT),
        Impl.inner(Nat.sub(Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s), j), s, g, j, Spec.Flush{g} <> Spec.trace(Nat.sub(Nat.sub(e, s), Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s)), Nat.min(e, Nat.mul(1n+g, KT)), e, KT)),
        p5_seg(Nat.sub(Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s), j), Nat.sub(Nat.sub(e, s), Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s)), s, e, KT, g, j, pos, lo, kpos,
          p5_hend0(s, j, Nat.sub(Nat.min(e, Nat.mul(1n+g, KT)), s), Nat.min(e, Nat.mul(1n+g, KT)), jsele, hs)))

# ---- mainloop_trace ----

def p5_fuel0(+n: Nat) -> {Nat.is_lt(Nat.sub(n, 0n), 1n+n) == True{} : Bool}:
  %Equal.sym(Nat, Nat.sub(n, 0n), n, ar_sub_zero_r(n)) : {Nat.is_lt(_, 1n+n) == True{} : Bool}
  p5_lt_succ_add(0n, n)

def Laws.mainloop_trace(s, e, KT, ordered, tiles):
  %ar_sub_zero_r(Nat.sub(e, s)) : {Impl.mainloop(s, e, KT) == Spec.trace(_, s, e, KT) : List<&2, Spec.Ev>}
  %ar_add_zero_r(s) : {Impl.mainloop(s, e, KT) == Spec.trace(Nat.sub(Nat.sub(e, s), 0n), _, e, KT) : List<&2, Spec.Ev>}
  p5_outer(1n+Nat.sub(e, s), Nat.is_lt(0n, Nat.sub(e, s)), s, e, KT, Nat.div(s, KT), 0n, tiles,
    ordered, ar_h_le_zero(Nat.sub(e, s)), {==}, h => p5_div_add0(s, KT),
    p5_fuel0(Nat.sub(e, s)))

# ==== Law flush_once_iff_touches ====

def p6_rt(-a: Bool, -b: Bool, e: {a == b : Bool}, h: {a == True{} : Bool})
  -> {b == True{} : Bool}:
  Equal.trans(Bool, b, a, True{}, Equal.sym(Bool, a, b, e), h)

def p6_rf(-a: Bool, -b: Bool, e: {a == b : Bool}, h: {a == False{} : Bool})
  -> {b == False{} : Bool}:
  Equal.trans(Bool, b, a, False{}, Equal.sym(Bool, a, b, e), h)

def p6_snd(-A: Type, -B: Type, p: A & B) -> B:
  match p:
    case (a, b):
      b

def p6_and_false(a: Bool) -> {Bool.and(a, False{}) == False{} : Bool}:
  match a:
    case True{}:
      {==}
    case False{}:
      {==}

def p6_and_true(a: Bool) -> {Bool.and(a, True{}) == a : Bool}:
  match a:
    case True{}:
      {==}
    case False{}:
      {==}

def p6_or_false(a: Bool) -> {Bool.or(a, False{}) == a : Bool}:
  match a:
    case True{}:
      {==}
    case False{}:
      {==}

def p6_or_true(a: Bool) -> {Bool.or(a, True{}) == True{} : Bool}:
  match a:
    case True{}:
      {==}
    case False{}:
      {==}

# and(a, b) == b when b implies a.
def p6_and_imp(+a: Bool, b: Bool, imp: {b == True{} : Bool} -> {a == True{} : Bool})
  -> {Bool.and(a, b) == b : Bool}:
  match b:
    case True{}:
      %Equal.sym(Bool, a, True{}, imp({==})) : {Bool.and(_, True{}) == True{} : Bool}
      {==}
    case False{}:
      p6_and_false(a)

def p6_lt_succ(a: Nat) -> {Nat.is_lt(a, 1n+a) == True{} : Bool}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      p6_lt_succ(p)

def p6_lt_zero(a: Nat) -> {Nat.is_lt(a, 0n) == False{} : Bool}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      {==}

def p6_lt_ne(a: Nat, b: Nat, lt: {Nat.is_lt(a, b) == True{} : Bool})
  -> {Nat.is_eq(a, b) == False{} : Bool}:
  match a b:
    case 0n 0n:
      Empty.absurd({Nat.is_eq(0n, 0n) == False{} : Bool}, ar_false_true(lt))
    case 0n 1n+q:
      {==}
    case 1n+p 0n:
      {==}
    case 1n+p 1n+q:
      p6_lt_ne(p, q, lt)

def p6_gt_ne(a: Nat, b: Nat, h: {Nat.is_le(1n+b, a) == True{} : Bool})
  -> {Nat.is_eq(a, b) == False{} : Bool}:
  match a b:
    case 0n _:
      Empty.absurd({Nat.is_eq(0n, b) == False{} : Bool}, ar_false_true(h))
    case 1n+p 0n:
      {==}
    case 1n+p 1n+q:
      p6_gt_ne(p, q, h)

def p6_nlt_le(a: Nat, b: Nat, h: {Nat.is_lt(a, b) == False{} : Bool})
  -> {Nat.is_le(b, a) == True{} : Bool}:
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      Empty.absurd({Nat.is_le(1n+q, 0n) == True{} : Bool}, ar_true_false(h))
    case 1n+p 0n:
      {==}
    case 1n+p 1n+q:
      p6_nlt_le(p, q, h)

def p6_le_nlt(a: Nat, b: Nat, h: {Nat.is_le(b, a) == True{} : Bool})
  -> {Nat.is_lt(a, b) == False{} : Bool}:
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      Empty.absurd({Nat.is_lt(0n, 1n+q) == False{} : Bool}, ar_false_true(h))
    case 1n+p 0n:
      {==}
    case 1n+p 1n+q:
      p6_le_nlt(p, q, h)

# b <= a < b + 1 forces a == b.
def p6_eq_of_le_lt(a: Nat, b: Nat, le: {Nat.is_le(b, a) == True{} : Bool},
    lt: {Nat.is_lt(a, 1n+b) == True{} : Bool})
  -> {Nat.is_eq(a, b) == True{} : Bool}:
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      Empty.absurd({Nat.is_eq(0n, 1n+q) == True{} : Bool}, ar_false_true(le))
    case 1n+p 0n:
      Empty.absurd({Nat.is_eq(1n+p, 0n) == True{} : Bool},
        ar_false_ne(Nat.is_lt(p, 0n), p6_lt_zero(p), lt))
    case 1n+p 1n+q:
      p6_eq_of_le_lt(p, q, le, lt)

# For a <= b exactly one of a == b, a < b holds.
def p6_eq_or_lt(a: Nat, b: Nat, h: {Nat.is_le(a, b) == True{} : Bool})
  -> {Nat.add(Spec.bit(Nat.is_eq(a, b)), Spec.bit(Nat.is_lt(a, b))) == 1n : Nat}:
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      {==}
    case 1n+p 0n:
      Empty.absurd({Nat.add(Spec.bit(Nat.is_eq(1n+p, 0n)), Spec.bit(Nat.is_lt(1n+p, 0n))) ==
        1n : Nat}, ar_false_true(h))
    case 1n+p 1n+q:
      p6_eq_or_lt(p, q, h)

# ---- group arithmetic (A = g * KT, B = (g + 1) * KT) ----

# KT = k / 16 >= 8 > 0 for k >= 128.
def p6_kt_pos(+k: Nat, shaped: {Nat.is_le(128n, k) == True{} : Bool})
  -> {Nat.is_lt(0n, Impl.kt(k)) == True{} : Bool}:
  ar_lt_le_trans(0n, 8n, Impl.kt(k), {==},
    p6_rt(Nat.is_le(Nat.mul(8n, 16n), k), Nat.is_le(8n, Nat.div(k, 16n)),
      ar_div_le(8n, k, 16n, {==}), shaped))

def p6_A_lt_B(+g: Nat, +KT: Nat, kpos: {Nat.is_lt(0n, KT) == True{} : Bool})
  -> {Nat.is_lt(Nat.mul(g, KT), Nat.mul(1n+g, KT)) == True{} : Bool}:
  p6_rt(Nat.is_lt(g, 1n+g), Nat.is_lt(Nat.mul(g, KT), Nat.mul(1n+g, KT)),
    Equal.sym(Bool, Nat.is_lt(Nat.mul(g, KT), Nat.mul(1n+g, KT)), Nat.is_lt(g, 1n+g),
      ar_mul_lt_cancel(g, 1n+g, KT, kpos)),
    p6_lt_succ(g))

# it < A: it is not in group g.
def p6_div_ne_lo(+it: Nat, +KT: Nat, +g: Nat, kpos: {Nat.is_lt(0n, KT) == True{} : Bool},
    hx: {Nat.is_lt(it, Nat.mul(g, KT)) == True{} : Bool})
  -> {Nat.is_eq(Nat.div(it, KT), g) == False{} : Bool}:
  p6_lt_ne(Nat.div(it, KT), g,
    p6_rt(Nat.is_lt(it, Nat.mul(g, KT)), Nat.is_lt(Nat.div(it, KT), g),
      Equal.sym(Bool, Nat.is_lt(Nat.div(it, KT), g), Nat.is_lt(it, Nat.mul(g, KT)),
        ar_div_lt(g, it, KT, kpos)),
      hx))

# B <= it: it is not in group g.
def p6_div_ne_hi(+it: Nat, +KT: Nat, +g: Nat, kpos: {Nat.is_lt(0n, KT) == True{} : Bool},
    hy: {Nat.is_lt(it, Nat.mul(1n+g, KT)) == False{} : Bool})
  -> {Nat.is_eq(Nat.div(it, KT), g) == False{} : Bool}:
  p6_gt_ne(Nat.div(it, KT), g,
    p6_nlt_le(Nat.div(it, KT), 1n+g,
      p6_rf(Nat.is_lt(it, Nat.mul(1n+g, KT)), Nat.is_lt(Nat.div(it, KT), 1n+g),
        Equal.sym(Bool, Nat.is_lt(Nat.div(it, KT), 1n+g), Nat.is_lt(it, Nat.mul(1n+g, KT)),
          ar_div_lt(1n+g, it, KT, kpos)),
        hy)))

# A <= it < B: it is in group g.
def p6_div_eq_mid(+it: Nat, +KT: Nat, +g: Nat, +kpos: {Nat.is_lt(0n, KT) == True{} : Bool},
    leA: {Nat.is_le(Nat.mul(g, KT), it) == True{} : Bool},
    hy: {Nat.is_lt(it, Nat.mul(1n+g, KT)) == True{} : Bool})
  -> {Nat.is_eq(Nat.div(it, KT), g) == True{} : Bool}:
  p6_eq_of_le_lt(Nat.div(it, KT), g,
    p6_rt(Nat.is_le(Nat.mul(g, KT), it), Nat.is_le(g, Nat.div(it, KT)),
      ar_div_le(g, it, KT, kpos), leA),
    p6_rt(Nat.is_lt(it, Nat.mul(1n+g, KT)), Nat.is_lt(Nat.div(it, KT), 1n+g),
      Equal.sym(Bool, Nat.is_lt(Nat.div(it, KT), 1n+g), Nat.is_lt(it, Nat.mul(1n+g, KT)),
        ar_div_lt(1n+g, it, KT, kpos)),
      hy))

# (A + d + 1) % KT == d + 1 != 0 when d + 1 < KT.
def p6_mod_ne_go(+d: Nat, +KT: Nat, +g: Nat,
    small: {Nat.is_lt(1n+d, KT) == True{} : Bool})
  -> {Nat.is_eq(Nat.mod(1n+Nat.add(Nat.mul(g, KT), d), KT), 0n) == False{} : Bool}:
  %ar_add_succ_r(Nat.mul(g, KT), d) : {Nat.is_eq(Nat.mod(_, KT), 0n) == False{} : Bool}
  %Equal.sym(Nat, Nat.mod(Nat.add(Nat.mul(g, KT), 1n+d), KT), 1n+d, p6_snd({Nat.div(Nat.add(Nat.mul(g, KT), 1n+d), KT) == g : Nat}, {Nat.mod(Nat.add(Nat.mul(g, KT), 1n+d), KT) == 1n+d : Nat}, ar_div_unique(g, 1n+d, KT, small))) : {Nat.is_eq(_, 0n) == False{} : Bool}
  {==}

# A <= it and it + 1 < B give (it - A) + 1 < KT.
def p6_mod_small(+it: Nat, +KT: Nat, +g: Nat,
    +leA: {Nat.is_le(Nat.mul(g, KT), it) == True{} : Bool},
    hw: {Nat.is_lt(1n+it, Nat.mul(1n+g, KT)) == True{} : Bool})
  -> {Nat.is_lt(1n+Nat.sub(it, Nat.mul(g, KT)), KT) == True{} : Bool}:
  +A = Nat.mul(g, KT)
  +d = Nat.sub(it, A)
  %ar_lt_add_cancel(A, 1n+d, KT) : {_ == True{} : Bool}
  %Equal.sym(Nat, Nat.add(A, 1n+d), 1n+Nat.add(A, d), ar_add_succ_r(A, d)) : {Nat.is_lt(_, Nat.add(A, KT)) == True{} : Bool}
  %Equal.sym(Nat, Nat.add(A, d), it, ar_sub_add(it, A, leA)) : {Nat.is_lt(1n+_, Nat.add(A, KT)) == True{} : Bool}
  %ar_add_comm(KT, A) : {Nat.is_lt(1n+it, _) == True{} : Bool}
  hw

# A <= it and it + 1 < B: iteration it + 1 does not start a group.
def p6_mod_ne(+it: Nat, +KT: Nat, +g: Nat,
    +leA: {Nat.is_le(Nat.mul(g, KT), it) == True{} : Bool},
    hw: {Nat.is_lt(1n+it, Nat.mul(1n+g, KT)) == True{} : Bool})
  -> {Nat.is_eq(Nat.mod(1n+it, KT), 0n) == False{} : Bool}:
  +A = Nat.mul(g, KT)
  +d = Nat.sub(it, A)
  +small = p6_mod_small(it, KT, g, leA, hw)
  %ar_sub_add(it, A, leA) : {Nat.is_eq(Nat.mod(1n+_, KT), 0n) == False{} : Bool}
  p6_mod_ne_go(d, KT, g, small)

# it < B and it + 1 == B: iteration it + 1 starts group g + 1.
def p6_mod_zero(+it: Nat, +KT: Nat, +g: Nat, kpos: {Nat.is_lt(0n, KT) == True{} : Bool},
    hy: {Nat.is_lt(it, Nat.mul(1n+g, KT)) == True{} : Bool},
    hw: {Nat.is_lt(1n+it, Nat.mul(1n+g, KT)) == False{} : Bool})
  -> {Nat.is_eq(Nat.mod(1n+it, KT), 0n) == True{} : Bool}:
  +B = Nat.mul(1n+g, KT)
  +eq = ar_le_antisym(1n+it, B, p6_rt(Nat.is_lt(it, B), Nat.is_le(1n+it, B), ar_lt_succ_le(it, B), hy), p6_nlt_le(1n+it, B, hw))
  %Equal.sym(Nat, 1n+it, B, eq) : {Nat.is_eq(Nat.mod(_, KT), 0n) == True{} : Bool}
  %ar_add_zero_r(B) : {Nat.is_eq(Nat.mod(_, KT), 0n) == True{} : Bool}
  %Equal.sym(Nat, Nat.mod(Nat.add(B, 0n), KT), 0n, p6_snd({Nat.div(Nat.add(B, 0n), KT) == 1n+g : Nat}, {Nat.mod(Nat.add(B, 0n), KT) == 0n : Nat}, ar_div_unique(1n+g, 0n, KT, kpos))) : {Nat.is_eq(_, 0n) == True{} : Bool}
  {==}

# ---- one trace step ----

# Flush condition after iteration it (last of the range, or next iteration starts a group).
def p6_c(+it: Nat, +e: Nat, +KT: Nat) -> Bool:
  Bool.or(Nat.is_eq(1n+it, e), Nat.is_eq(Nat.mod(1n+it, KT), 0n))

# Iteration x is in [.., e) and group g still has an iteration in [x, e).
def p6_P(+x: Nat, +e: Nat, +KT: Nat, +g: Nat) -> Bool:
  Bool.and(Nat.is_lt(x, e), Bool.and(Nat.is_lt(x, Nat.mul(1n+g, KT)), Nat.is_lt(Nat.mul(g, KT), e)))

def p6_flush_split(c: Bool, +h: Nat, +g: Nat, -rest: List<&2, Spec.Ev>)
  -> {Spec.flushes(Spec.trace.flush(c, h, rest), g) ==
    Nat.add(Spec.bit(Bool.and(c, Nat.is_eq(h, g))), Spec.flushes(rest, g)) : Nat}:
  match c:
    case True{}:
      {==}
    case False{}:
      {==}

# Region it < A.
def p6_reg_lo(+it: Nat, +e: Nat, +KT: Nat, +g: Nat,
    +kpos: {Nat.is_lt(0n, KT) == True{} : Bool},
    +hx: {Nat.is_lt(it, Nat.mul(g, KT)) == True{} : Bool},
    hy: {Nat.is_lt(it, Nat.mul(1n+g, KT)) == True{} : Bool})
  -> {Nat.add(Spec.bit(Bool.and(p6_c(it, e, KT), Nat.is_eq(Nat.div(it, KT), g))),
      Spec.bit(p6_P(1n+it, e, KT, g))) ==
    Spec.bit(Bool.and(True{}, Bool.and(Nat.is_lt(it, Nat.mul(1n+g, KT)),
      Nat.is_lt(Nat.mul(g, KT), e)))) : Nat}:
  +A = Nat.mul(g, KT)
  +B = Nat.mul(1n+g, KT)
  +c = p6_c(it, e, KT)
  %Equal.sym(Bool, Nat.is_eq(Nat.div(it, KT), g), False{}, p6_div_ne_lo(it, KT, g, kpos, hx)) : {Nat.add(Spec.bit(Bool.and(c, _)), Spec.bit(p6_P(1n+it, e, KT, g))) == Spec.bit(Bool.and(True{}, Bool.and(Nat.is_lt(it, B), Nat.is_lt(A, e)))) : Nat}
  %Equal.sym(Bool, Bool.and(c, False{}), False{}, p6_and_false(c)) : {Nat.add(Spec.bit(_), Spec.bit(p6_P(1n+it, e, KT, g))) == Spec.bit(Bool.and(True{}, Bool.and(Nat.is_lt(it, B), Nat.is_lt(A, e)))) : Nat}
  %Equal.sym(Bool, Nat.is_lt(it, B), True{}, hy) : {Nat.add(Spec.bit(False{}), Spec.bit(p6_P(1n+it, e, KT, g))) == Spec.bit(Bool.and(True{}, Bool.and(_, Nat.is_lt(A, e)))) : Nat}
  +le1 = p6_rt(Nat.is_lt(it, A), Nat.is_le(1n+it, A), ar_lt_succ_le(it, A), hx)
  %Equal.sym(Bool, Nat.is_lt(1n+it, B), True{}, ar_le_lt_trans(1n+it, A, B, le1, p6_A_lt_B(g, KT, kpos))) : {Nat.add(Spec.bit(False{}), Spec.bit(Bool.and(Nat.is_lt(1n+it, e), Bool.and(_, Nat.is_lt(A, e))))) == Spec.bit(Bool.and(True{}, Bool.and(True{}, Nat.is_lt(A, e)))) : Nat}
  Equal.cong(Bool, Nat, Spec.bit, Bool.and(Nat.is_lt(1n+it, e), Nat.is_lt(A, e)), Nat.is_lt(A, e),
    p6_and_imp(Nat.is_lt(1n+it, e), Nat.is_lt(A, e),
      h => ar_le_lt_trans(1n+it, A, e, le1, h)))

# Region A <= it < B, split on whether it + 1 < B.
def p6_mid(w: Bool, +it: Nat, +e: Nat, +KT: Nat, +g: Nat,
    +kpos: {Nat.is_lt(0n, KT) == True{} : Bool},
    +lt: {Nat.is_lt(it, e) == True{} : Bool},
    +leA: {Nat.is_le(Nat.mul(g, KT), it) == True{} : Bool},
    +hy: {Nat.is_lt(it, Nat.mul(1n+g, KT)) == True{} : Bool},
    hw: {Nat.is_lt(1n+it, Nat.mul(1n+g, KT)) == w : Bool})
  -> {Nat.add(Spec.bit(p6_c(it, e, KT)), Spec.bit(Bool.and(Nat.is_lt(1n+it, e),
      Bool.and(Nat.is_lt(1n+it, Nat.mul(1n+g, KT)), True{})))) == 1n : Nat}:
  match w:
    case True{}:
      +B = Nat.mul(1n+g, KT)
      +v = Nat.is_lt(1n+it, e)
      +u = Nat.is_eq(1n+it, e)
      +h = hw
      %Equal.sym(Bool, Nat.is_eq(Nat.mod(1n+it, KT), 0n), False{}, p6_mod_ne(it, KT, g, leA, h)) : {Nat.add(Spec.bit(Bool.or(u, _)), Spec.bit(Bool.and(v, Bool.and(Nat.is_lt(1n+it, B), True{})))) == 1n : Nat}
      %Equal.sym(Bool, Bool.or(u, False{}), u, p6_or_false(u)) : {Nat.add(Spec.bit(_), Spec.bit(Bool.and(v, Bool.and(Nat.is_lt(1n+it, B), True{})))) == 1n : Nat}
      %Equal.sym(Bool, Nat.is_lt(1n+it, B), True{}, h) : {Nat.add(Spec.bit(u), Spec.bit(Bool.and(v, Bool.and(_, True{})))) == 1n : Nat}
      %Equal.sym(Bool, Bool.and(v, True{}), v, p6_and_true(v)) : {Nat.add(Spec.bit(u), Spec.bit(_)) == 1n : Nat}
      p6_eq_or_lt(1n+it, e, p6_rt(Nat.is_lt(it, e), Nat.is_le(1n+it, e), ar_lt_succ_le(it, e), lt))
    case False{}:
      +B = Nat.mul(1n+g, KT)
      +v = Nat.is_lt(1n+it, e)
      +u = Nat.is_eq(1n+it, e)
      +h = hw
      %Equal.sym(Bool, Nat.is_eq(Nat.mod(1n+it, KT), 0n), True{}, p6_mod_zero(it, KT, g, kpos, hy, h)) : {Nat.add(Spec.bit(Bool.or(u, _)), Spec.bit(Bool.and(v, Bool.and(Nat.is_lt(1n+it, B), True{})))) == 1n : Nat}
      %Equal.sym(Bool, Bool.or(u, True{}), True{}, p6_or_true(u)) : {Nat.add(Spec.bit(_), Spec.bit(Bool.and(v, Bool.and(Nat.is_lt(1n+it, B), True{})))) == 1n : Nat}
      %Equal.sym(Bool, Nat.is_lt(1n+it, B), False{}, h) : {Nat.add(Spec.bit(True{}), Spec.bit(Bool.and(v, Bool.and(_, True{})))) == 1n : Nat}
      %Equal.sym(Bool, Bool.and(v, False{}), False{}, p6_and_false(v)) : {Nat.add(Spec.bit(True{}), Spec.bit(_)) == 1n : Nat}
      {==}

# Region A <= it < B.
def p6_reg_mid(+it: Nat, +e: Nat, +KT: Nat, +g: Nat,
    +kpos: {Nat.is_lt(0n, KT) == True{} : Bool},
    +lt: {Nat.is_lt(it, e) == True{} : Bool},
    hx: {Nat.is_lt(it, Nat.mul(g, KT)) == False{} : Bool},
    +hy: {Nat.is_lt(it, Nat.mul(1n+g, KT)) == True{} : Bool})
  -> {Nat.add(Spec.bit(Bool.and(p6_c(it, e, KT), Nat.is_eq(Nat.div(it, KT), g))),
      Spec.bit(p6_P(1n+it, e, KT, g))) ==
    Spec.bit(Bool.and(True{}, Bool.and(Nat.is_lt(it, Nat.mul(1n+g, KT)),
      Nat.is_lt(Nat.mul(g, KT), e)))) : Nat}:
  +A = Nat.mul(g, KT)
  +B = Nat.mul(1n+g, KT)
  +c = p6_c(it, e, KT)
  +leA = p6_nlt_le(it, A, hx)
  %Equal.sym(Bool, Nat.is_eq(Nat.div(it, KT), g), True{}, p6_div_eq_mid(it, KT, g, kpos, leA, hy)) : {Nat.add(Spec.bit(Bool.and(c, _)), Spec.bit(p6_P(1n+it, e, KT, g))) == Spec.bit(Bool.and(True{}, Bool.and(Nat.is_lt(it, B), Nat.is_lt(A, e)))) : Nat}
  %Equal.sym(Bool, Bool.and(c, True{}), c, p6_and_true(c)) : {Nat.add(Spec.bit(_), Spec.bit(p6_P(1n+it, e, KT, g))) == Spec.bit(Bool.and(True{}, Bool.and(Nat.is_lt(it, B), Nat.is_lt(A, e)))) : Nat}
  %Equal.sym(Bool, Nat.is_lt(it, B), True{}, hy) : {Nat.add(Spec.bit(c), Spec.bit(p6_P(1n+it, e, KT, g))) == Spec.bit(Bool.and(True{}, Bool.and(_, Nat.is_lt(A, e)))) : Nat}
  %Equal.sym(Bool, Nat.is_lt(A, e), True{}, ar_le_lt_trans(A, it, e, leA, lt)) : {Nat.add(Spec.bit(c), Spec.bit(Bool.and(Nat.is_lt(1n+it, e), Bool.and(Nat.is_lt(1n+it, B), _)))) == Spec.bit(Bool.and(True{}, Bool.and(True{}, _))) : Nat}
  p6_mid(Nat.is_lt(1n+it, B), it, e, KT, g, kpos, lt, leA, hy, {==})

# Region B <= it.
def p6_reg_hi(+it: Nat, +e: Nat, +KT: Nat, +g: Nat,
    +kpos: {Nat.is_lt(0n, KT) == True{} : Bool},
    +hy: {Nat.is_lt(it, Nat.mul(1n+g, KT)) == False{} : Bool})
  -> {Nat.add(Spec.bit(Bool.and(p6_c(it, e, KT), Nat.is_eq(Nat.div(it, KT), g))),
      Spec.bit(p6_P(1n+it, e, KT, g))) ==
    Spec.bit(Bool.and(True{}, Bool.and(Nat.is_lt(it, Nat.mul(1n+g, KT)),
      Nat.is_lt(Nat.mul(g, KT), e)))) : Nat}:
  +A = Nat.mul(g, KT)
  +B = Nat.mul(1n+g, KT)
  +c = p6_c(it, e, KT)
  %Equal.sym(Bool, Nat.is_eq(Nat.div(it, KT), g), False{}, p6_div_ne_hi(it, KT, g, kpos, hy)) : {Nat.add(Spec.bit(Bool.and(c, _)), Spec.bit(p6_P(1n+it, e, KT, g))) == Spec.bit(Bool.and(True{}, Bool.and(Nat.is_lt(it, B), Nat.is_lt(A, e)))) : Nat}
  %Equal.sym(Bool, Bool.and(c, False{}), False{}, p6_and_false(c)) : {Nat.add(Spec.bit(_), Spec.bit(p6_P(1n+it, e, KT, g))) == Spec.bit(Bool.and(True{}, Bool.and(Nat.is_lt(it, B), Nat.is_lt(A, e)))) : Nat}
  %Equal.sym(Bool, Nat.is_lt(it, B), False{}, hy) : {Nat.add(Spec.bit(False{}), Spec.bit(p6_P(1n+it, e, KT, g))) == Spec.bit(Bool.and(True{}, Bool.and(_, Nat.is_lt(A, e)))) : Nat}
  +nB = p6_le_nlt(1n+it, B, ar_le_trans(B, it, 1n+it, p6_nlt_le(it, B, hy), ar_le_add_l(1n, it)))
  %Equal.sym(Bool, Nat.is_lt(1n+it, B), False{}, nB) : {Nat.add(Spec.bit(False{}), Spec.bit(Bool.and(Nat.is_lt(1n+it, e), Bool.and(_, Nat.is_lt(A, e))))) == Spec.bit(Bool.and(True{}, Bool.and(False{}, Nat.is_lt(A, e)))) : Nat}
  %Equal.sym(Bool, Bool.and(Nat.is_lt(1n+it, e), False{}), False{}, p6_and_false(Nat.is_lt(1n+it, e))) : {Nat.add(Spec.bit(False{}), Spec.bit(_)) == Spec.bit(Bool.and(True{}, Bool.and(False{}, Nat.is_lt(A, e)))) : Nat}
  {==}

def p6_step_go(x: Bool, y: Bool, +it: Nat, +e: Nat, +KT: Nat, +g: Nat,
    +kpos: {Nat.is_lt(0n, KT) == True{} : Bool},
    +lt: {Nat.is_lt(it, e) == True{} : Bool},
    hx: {Nat.is_lt(it, Nat.mul(g, KT)) == x : Bool},
    hy: {Nat.is_lt(it, Nat.mul(1n+g, KT)) == y : Bool})
  -> {Nat.add(Spec.bit(Bool.and(p6_c(it, e, KT), Nat.is_eq(Nat.div(it, KT), g))),
      Spec.bit(p6_P(1n+it, e, KT, g))) ==
    Spec.bit(Bool.and(True{}, Bool.and(Nat.is_lt(it, Nat.mul(1n+g, KT)),
      Nat.is_lt(Nat.mul(g, KT), e)))) : Nat}:
  match x y:
    case True{} True{}:
      p6_reg_lo(it, e, KT, g, kpos, hx, hy)
    case True{} False{}:
      Empty.absurd({Nat.add(Spec.bit(Bool.and(p6_c(it, e, KT), Nat.is_eq(Nat.div(it, KT), g))),
          Spec.bit(p6_P(1n+it, e, KT, g))) ==
        Spec.bit(Bool.and(True{}, Bool.and(Nat.is_lt(it, Nat.mul(1n+g, KT)),
          Nat.is_lt(Nat.mul(g, KT), e)))) : Nat},
        ar_false_ne(Nat.is_lt(it, Nat.mul(1n+g, KT)), hy,
          ar_lt_trans(it, Nat.mul(g, KT), Nat.mul(1n+g, KT), hx, p6_A_lt_B(g, KT, kpos))))
    case False{} True{}:
      p6_reg_mid(it, e, KT, g, kpos, lt, hx, hy)
    case False{} False{}:
      p6_reg_hi(it, e, KT, g, kpos, hy)

# Flush bit of iteration it plus P(it + 1) is P(it), for it < e.
def p6_step(+it: Nat, +e: Nat, +KT: Nat, +g: Nat,
    +kpos: {Nat.is_lt(0n, KT) == True{} : Bool},
    +lt: {Nat.is_lt(it, e) == True{} : Bool})
  -> {Nat.add(Spec.bit(Bool.and(p6_c(it, e, KT), Nat.is_eq(Nat.div(it, KT), g))),
      Spec.bit(p6_P(1n+it, e, KT, g))) == Spec.bit(p6_P(it, e, KT, g)) : Nat}:
  %Equal.sym(Bool, Nat.is_lt(it, e), True{}, lt) : {Nat.add(Spec.bit(Bool.and(p6_c(it, e, KT), Nat.is_eq(Nat.div(it, KT), g))), Spec.bit(p6_P(1n+it, e, KT, g))) == Spec.bit(Bool.and(_, Bool.and(Nat.is_lt(it, Nat.mul(1n+g, KT)), Nat.is_lt(Nat.mul(g, KT), e)))) : Nat}
  p6_step_go(Nat.is_lt(it, Nat.mul(g, KT)), Nat.is_lt(it, Nat.mul(1n+g, KT)), it, e, KT, g,
    kpos, lt, {==}, {==})

# ---- the trace ----

def p6_lt_of_add(+rest: Nat, +it: Nat, +e: Nat, h: {1n+Nat.add(rest, it) == e : Nat})
  -> {Nat.is_lt(it, e) == True{} : Bool}:
  %h : {Nat.is_lt(it, _) == True{} : Bool}
  p6_rt(Nat.is_le(1n+it, 1n+Nat.add(rest, it)), Nat.is_lt(it, 1n+Nat.add(rest, it)),
    Equal.sym(Bool, Nat.is_lt(it, 1n+Nat.add(rest, it)), Nat.is_le(1n+it, 1n+Nat.add(rest, it)),
      ar_lt_succ_le(it, 1n+Nat.add(rest, it))),
    ar_le_add_l(rest, it))

# The trace over [it, e) (left = e - it iterations) flushes group g once iff some iteration of
# [it, e) lies in group g, i.e. it < e, it < (g + 1) * KT and g * KT < e.
def p6_core(left: Nat, +it: Nat, +e: Nat, +KT: Nat, +g: Nat,
    +kpos: {Nat.is_lt(0n, KT) == True{} : Bool},
    hl: {Nat.add(left, it) == e : Nat})
  -> {Spec.flushes(Spec.trace(left, it, e, KT), g) == Spec.bit(p6_P(it, e, KT, g)) : Nat}:
  match left:
    case 0n:
      %hl : {0n == Spec.bit(Bool.and(Nat.is_lt(it, _), Bool.and(Nat.is_lt(it, Nat.mul(1n+g, KT)), Nat.is_lt(Nat.mul(g, KT), _)))) : Nat}
      %Equal.sym(Bool, Nat.is_lt(it, it), False{}, ar_lt_irrefl(it)) : {0n == Spec.bit(Bool.and(_, Bool.and(Nat.is_lt(it, Nat.mul(1n+g, KT)), Nat.is_lt(Nat.mul(g, KT), it)))) : Nat}
      {==}
    case 1n+rest:
      +h = hl
      +r = rest
      +lt = p6_lt_of_add(r, it, e, h)
      +h2 = Equal.trans(Nat, Nat.add(r, 1n+it), 1n+Nat.add(r, it), e,
        ar_add_succ_r(r, it), h)
      %Equal.sym(Nat, Spec.flushes(Spec.trace.flush(p6_c(it, e, KT), Nat.div(it, KT), Spec.trace(r, 1n+it, e, KT)), g), Nat.add(Spec.bit(Bool.and(p6_c(it, e, KT), Nat.is_eq(Nat.div(it, KT), g))), Spec.flushes(Spec.trace(r, 1n+it, e, KT), g)), p6_flush_split(p6_c(it, e, KT), Nat.div(it, KT), g, Spec.trace(r, 1n+it, e, KT))) : {_ == Spec.bit(p6_P(it, e, KT, g)) : Nat}
      %Equal.sym(Nat, Spec.flushes(Spec.trace(r, 1n+it, e, KT), g), Spec.bit(p6_P(1n+it, e, KT, g)), p6_core(r, 1n+it, e, KT, g, kpos, h2)) : {Nat.add(Spec.bit(Bool.and(p6_c(it, e, KT), Nat.is_eq(Nat.div(it, KT), g))), _) == Spec.bit(p6_P(it, e, KT, g)) : Nat}
      p6_step(it, e, KT, g, kpos, lt)

# ---- flush_once_iff_touches ----

def Laws.flush_once_iff_touches(mats, n, k, G, g, b, grid, shaped):
  +gp = grid
  +I = Impl.total(mats, n, k)
  +s = Impl.start(b, I, G)
  +e = Impl.start(1n+b, I, G)
  +KT = Impl.kt(k)
  +kpos = p6_kt_pos(k, shaped)
  +ord = ar_div_mono(Nat.mul(b, I), Nat.mul(1n+b, I), G, gp,
    ar_le_add_l(I, Nat.mul(b, I)))
  %Equal.sym(List<&2, Spec.Ev>, Impl.mainloop(s, e, KT), Spec.trace(Nat.sub(e, s), s, e, KT), Laws.mainloop_trace(s, e, KT, ord, kpos)) : {Spec.flushes(_, g) == Spec.bit(Spec.touches(b, g, KT, I, G)) : Nat}
  %Equal.sym(Bool, Spec.touches(b, g, KT, I, G), p6_P(s, e, KT, g), p1_touches_range(b, g, KT, I, G, gp)) : {Spec.flushes(Spec.trace(Nat.sub(e, s), s, e, KT), g) == Spec.bit(_) : Nat}
  p6_core(Nat.sub(e, s), s, e, KT, g, kpos,
    Equal.trans(Nat, Nat.add(Nat.sub(e, s), s), Nat.add(s, Nat.sub(e, s)), e,
      ar_add_comm(Nat.sub(e, s), s), ar_sub_add(e, s, ord)))

# ==== Laws warp_columns, warp_columns_injective, warp_columns_cover, lane_rows, lane_rows_cover ====

def p7_fst(-A: Type, -B: Type, p: A & B) -> A:
  match p:
    case (a, b):
      a

def p7_snd(-A: Type, -B: Type, p: A & B) -> B:
  match p:
    case (a, b):
      b

# (q * b + r) / b == q for r < b.
def p7_div_q(+q: Nat, +r: Nat, +b: Nat, small: {Nat.is_lt(r, b) == True{} : Bool})
  -> {Nat.div(Nat.add(Nat.mul(q, b), r), b) == q : Nat}:
  p7_fst({Nat.div(Nat.add(Nat.mul(q, b), r), b) == q : Nat},
    {Nat.mod(Nat.add(Nat.mul(q, b), r), b) == r : Nat}, ar_div_unique(q, r, b, small))

# (q * b + r) % b == r for r < b.
def p7_mod_r(+q: Nat, +r: Nat, +b: Nat, small: {Nat.is_lt(r, b) == True{} : Bool})
  -> {Nat.mod(Nat.add(Nat.mul(q, b), r), b) == r : Nat}:
  p7_snd({Nat.div(Nat.add(Nat.mul(q, b), r), b) == q : Nat},
    {Nat.mod(Nat.add(Nat.mul(q, b), r), b) == r : Nat}, ar_div_unique(q, r, b, small))

# a / b * b + a % b == a (b > 0).
def p7_dm_eq(+a: Nat, +b: Nat, pos: {Nat.is_lt(0n, b) == True{} : Bool})
  -> {Nat.add(Nat.mul(Nat.div(a, b), b), Nat.mod(a, b)) == a : Nat}:
  p7_fst({Nat.add(Nat.mul(Nat.div(a, b), b), Nat.mod(a, b)) == a : Nat},
    {Nat.is_lt(Nat.mod(a, b), b) == True{} : Bool}, ar_divmod(a, b, pos))

# a % b < b (b > 0).
def p7_dm_lt(+a: Nat, +b: Nat, pos: {Nat.is_lt(0n, b) == True{} : Bool})
  -> {Nat.is_lt(Nat.mod(a, b), b) == True{} : Bool}:
  p7_snd({Nat.add(Nat.mul(Nat.div(a, b), b), Nat.mod(a, b)) == a : Nat},
    {Nat.is_lt(Nat.mod(a, b), b) == True{} : Bool}, ar_divmod(a, b, pos))

# x < m gives x * n + n <= m * n.
def p7_mr_le(+x: Nat, +m: Nat, +n: Nat, xm: {Nat.is_lt(x, m) == True{} : Bool})
  -> {Nat.is_le(Nat.add(Nat.mul(x, n), n), Nat.mul(m, n)) == True{} : Bool}:
  %ar_add_comm(n, Nat.mul(x, n)) : {Nat.is_le(_, Nat.mul(m, n)) == True{} : Bool}
  ar_le_mul_mono(1n+x, m, n, Equal.trans(Bool, Nat.is_le(1n+x, m), Nat.is_lt(x, m), True{},
    Equal.sym(Bool, Nat.is_lt(x, m), Nat.is_le(1n+x, m), ar_lt_succ_le(x, m)), xm))

# x < m and y < n give x * n + y < m * n.
def p7_mr_lt(+x: Nat, +y: Nat, +m: Nat, +n: Nat, xm: {Nat.is_lt(x, m) == True{} : Bool},
    yn: {Nat.is_lt(y, n) == True{} : Bool})
  -> {Nat.is_lt(Nat.add(Nat.mul(x, n), y), Nat.mul(m, n)) == True{} : Bool}:
  ar_lt_le_trans(Nat.add(Nat.mul(x, n), y), Nat.add(Nat.mul(x, n), n), Nat.mul(m, n),
    Equal.trans(Bool, Nat.is_lt(Nat.add(Nat.mul(x, n), y), Nat.add(Nat.mul(x, n), n)),
      Nat.is_lt(y, n), True{}, ar_lt_add_cancel(Nat.mul(x, n), y, n), yn),
    p7_mr_le(x, m, n, xm))

# With Q = x / (k n), R = x % (k n): (Q k + R / n) n + R % n == x.
def p7_mod_nest_eq(+x: Nat, +k: Nat, +n: Nat, npos: {Nat.is_lt(0n, n) == True{} : Bool},
    kpos: {Nat.is_lt(0n, Nat.mul(k, n)) == True{} : Bool})
  -> {Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(x, Nat.mul(k, n)), k), Nat.div(Nat.mod(x, Nat.mul(k, n)), n)), n), Nat.mod(Nat.mod(x, Nat.mul(k, n)), n)) == x : Nat}:
  %Equal.sym(Nat, Nat.mul(Nat.add(Nat.mul(Nat.div(x, Nat.mul(k, n)), k), Nat.div(Nat.mod(x, Nat.mul(k, n)), n)), n), Nat.add(Nat.mul(Nat.mul(Nat.div(x, Nat.mul(k, n)), k), n), Nat.mul(Nat.div(Nat.mod(x, Nat.mul(k, n)), n), n)), ar_mul_add_l(Nat.mul(Nat.div(x, Nat.mul(k, n)), k), Nat.div(Nat.mod(x, Nat.mul(k, n)), n), n)) : {Nat.add(_, Nat.mod(Nat.mod(x, Nat.mul(k, n)), n)) == x : Nat}
  %Equal.sym(Nat, Nat.add(Nat.add(Nat.mul(Nat.mul(Nat.div(x, Nat.mul(k, n)), k), n), Nat.mul(Nat.div(Nat.mod(x, Nat.mul(k, n)), n), n)), Nat.mod(Nat.mod(x, Nat.mul(k, n)), n)), Nat.add(Nat.mul(Nat.mul(Nat.div(x, Nat.mul(k, n)), k), n), Nat.add(Nat.mul(Nat.div(Nat.mod(x, Nat.mul(k, n)), n), n), Nat.mod(Nat.mod(x, Nat.mul(k, n)), n))), ar_add_assoc(Nat.mul(Nat.mul(Nat.div(x, Nat.mul(k, n)), k), n), Nat.mul(Nat.div(Nat.mod(x, Nat.mul(k, n)), n), n), Nat.mod(Nat.mod(x, Nat.mul(k, n)), n))) : {_ == x : Nat}
  %Equal.sym(Nat, Nat.mul(Nat.mul(Nat.div(x, Nat.mul(k, n)), k), n), Nat.mul(Nat.div(x, Nat.mul(k, n)), Nat.mul(k, n)), ar_mul_assoc(Nat.div(x, Nat.mul(k, n)), k, n)) : {Nat.add(_, Nat.add(Nat.mul(Nat.div(Nat.mod(x, Nat.mul(k, n)), n), n), Nat.mod(Nat.mod(x, Nat.mul(k, n)), n))) == x : Nat}
  %Equal.sym(Nat, Nat.add(Nat.mul(Nat.div(Nat.mod(x, Nat.mul(k, n)), n), n), Nat.mod(Nat.mod(x, Nat.mul(k, n)), n)), Nat.mod(x, Nat.mul(k, n)), p7_dm_eq(Nat.mod(x, Nat.mul(k, n)), n, npos)) : {Nat.add(Nat.mul(Nat.div(x, Nat.mul(k, n)), Nat.mul(k, n)), _) == x : Nat}
  p7_dm_eq(x, Nat.mul(k, n), kpos)

# (x % (k n)) % n == x % n (n > 0, k n > 0).
def p7_mod_nest(+x: Nat, +k: Nat, +n: Nat, npos: {Nat.is_lt(0n, n) == True{} : Bool},
    kpos: {Nat.is_lt(0n, Nat.mul(k, n)) == True{} : Bool})
  -> {Nat.mod(Nat.mod(x, Nat.mul(k, n)), n) == Nat.mod(x, n) : Nat}:
  +np = npos
  %p7_mod_nest_eq(x, k, n, np, kpos) : {Nat.mod(Nat.mod(x, Nat.mul(k, n)), n) == Nat.mod(_, n) : Nat}
  Equal.sym(Nat,
    Nat.mod(Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(x, Nat.mul(k, n)), k), Nat.div(Nat.mod(x, Nat.mul(k, n)), n)), n), Nat.mod(Nat.mod(x, Nat.mul(k, n)), n)), n),
    Nat.mod(Nat.mod(x, Nat.mul(k, n)), n),
    p7_mod_r(Nat.add(Nat.mul(Nat.div(x, Nat.mul(k, n)), k), Nat.div(Nat.mod(x, Nat.mul(k, n)), n)),
      Nat.mod(Nat.mod(x, Nat.mul(k, n)), n), n, p7_dm_lt(Nat.mod(x, Nat.mul(k, n)), n, np)))

# Next digit: (x % (k n)) / n * n + x % n == x % (k n).
def p7_digit(+x: Nat, +k: Nat, +n: Nat, npos: {Nat.is_lt(0n, n) == True{} : Bool},
    kpos: {Nat.is_lt(0n, Nat.mul(k, n)) == True{} : Bool})
  -> {Nat.add(Nat.mul(Nat.div(Nat.mod(x, Nat.mul(k, n)), n), n), Nat.mod(x, n)) == Nat.mod(x, Nat.mul(k, n)) : Nat}:
  +np = npos
  %p7_mod_nest(x, k, n, np, kpos) : {Nat.add(Nat.mul(Nat.div(Nat.mod(x, Nat.mul(k, n)), n), n), _) == Nat.mod(x, Nat.mul(k, n)) : Nat}
  p7_dm_eq(Nat.mod(x, Nat.mul(k, n)), n, np)

# ---- warp columns ----

# col = (w 64 + t 16) + (h 8 + q).
def p7_col_a(+w: Nat, +t: Nat, +h: Nat, +q: Nat)
  -> {Nat.add(Nat.add(Nat.mul(w, 64n), Nat.mul(t, 16n)), Nat.add(Nat.mul(h, 8n), q)) == Impl.col(w, t, h, q) : Nat}:
  Equal.sym(Nat, Impl.col(w, t, h, q), Nat.add(Nat.add(Nat.mul(w, 64n), Nat.mul(t, 16n)), Nat.add(Nat.mul(h, 8n), q)),
    ar_add_assoc(Nat.add(Nat.mul(w, 64n), Nat.mul(t, 16n)), Nat.mul(h, 8n), q))

# col = w 64 + (t 16 + (h 8 + q)).
def p7_col64(+w: Nat, +t: Nat, +h: Nat, +q: Nat)
  -> {Nat.add(Nat.mul(w, 64n), Nat.add(Nat.mul(t, 16n), Nat.add(Nat.mul(h, 8n), q))) == Impl.col(w, t, h, q) : Nat}:
  %ar_add_assoc(Nat.mul(w, 64n), Nat.mul(t, 16n), Nat.add(Nat.mul(h, 8n), q)) : {_ == Impl.col(w, t, h, q) : Nat}
  p7_col_a(w, t, h, q)

# w 64 + t 16 = wtile(w, t) 16.
def p7_wt16(+w: Nat, +t: Nat)
  -> {Nat.add(Nat.mul(w, 64n), Nat.mul(t, 16n)) == Nat.mul(Impl.wtile(w, t), 16n) : Nat}:
  %ar_mul_assoc(w, 4n, 16n) : {Nat.add(_, Nat.mul(t, 16n)) == Nat.mul(Impl.wtile(w, t), 16n) : Nat}
  Equal.sym(Nat, Nat.mul(Impl.wtile(w, t), 16n), Nat.add(Nat.mul(Nat.mul(w, 4n), 16n), Nat.mul(t, 16n)),
    ar_mul_add_l(Nat.mul(w, 4n), t, 16n))

# col = wtile(w, t) 16 + (h 8 + q).
def p7_col16(+w: Nat, +t: Nat, +h: Nat, +q: Nat)
  -> {Nat.add(Nat.mul(Impl.wtile(w, t), 16n), Nat.add(Nat.mul(h, 8n), q)) == Impl.col(w, t, h, q) : Nat}:
  %p7_wt16(w, t) : {Nat.add(_, Nat.add(Nat.mul(h, 8n), q)) == Impl.col(w, t, h, q) : Nat}
  p7_col_a(w, t, h, q)

def p7_hq16(+h: Nat, +q: Nat, h2: {Nat.is_lt(h, 2n) == True{} : Bool},
    q8: {Nat.is_lt(q, 8n) == True{} : Bool})
  -> {Nat.is_lt(Nat.add(Nat.mul(h, 8n), q), 16n) == True{} : Bool}:
  p7_mr_lt(h, q, 2n, 8n, h2, q8)

def p7_tail64(+t: Nat, +h: Nat, +q: Nat, t4: {Nat.is_lt(t, 4n) == True{} : Bool},
    h2: {Nat.is_lt(h, 2n) == True{} : Bool}, q8: {Nat.is_lt(q, 8n) == True{} : Bool})
  -> {Nat.is_lt(Nat.add(Nat.mul(t, 16n), Nat.add(Nat.mul(h, 8n), q)), 64n) == True{} : Bool}:
  p7_mr_lt(t, Nat.add(Nat.mul(h, 8n), q), 4n, 16n, t4, p7_hq16(h, q, h2, q8))

def p7_col_lt(+w: Nat, +t: Nat, +h: Nat, +q: Nat, w8: {Nat.is_lt(w, 8n) == True{} : Bool},
    tl: {Nat.is_lt(Nat.add(Nat.mul(t, 16n), Nat.add(Nat.mul(h, 8n), q)), 64n) == True{} : Bool})
  -> {Nat.is_lt(Impl.col(w, t, h, q), 512n) == True{} : Bool}:
  %p7_col64(w, t, h, q) : {Nat.is_lt(_, 512n) == True{} : Bool}
  p7_mr_lt(w, Nat.add(Nat.mul(t, 16n), Nat.add(Nat.mul(h, 8n), q)), 8n, 64n, w8, tl)

def p7_col_div64(+w: Nat, +t: Nat, +h: Nat, +q: Nat,
    tl: {Nat.is_lt(Nat.add(Nat.mul(t, 16n), Nat.add(Nat.mul(h, 8n), q)), 64n) == True{} : Bool})
  -> {Nat.div(Impl.col(w, t, h, q), 64n) == w : Nat}:
  %p7_col64(w, t, h, q) : {Nat.div(_, 64n) == w : Nat}
  p7_div_q(w, Nat.add(Nat.mul(t, 16n), Nat.add(Nat.mul(h, 8n), q)), 64n, tl)

def p7_col_div16(+w: Nat, +t: Nat, +h: Nat, +q: Nat,
    hq: {Nat.is_lt(Nat.add(Nat.mul(h, 8n), q), 16n) == True{} : Bool})
  -> {Nat.div(Impl.col(w, t, h, q), 16n) == Impl.wtile(w, t) : Nat}:
  %p7_col16(w, t, h, q) : {Nat.div(_, 16n) == Impl.wtile(w, t) : Nat}
  p7_div_q(Impl.wtile(w, t), Nat.add(Nat.mul(h, 8n), q), 16n, hq)

def Laws.warp_columns(w, t, h, q, w8, t4, h2, q8):
  +tf = t4
  +hf = h2
  +qf = q8
  (p7_col_lt(w, t, h, q, w8, p7_tail64(t, h, q, tf, hf, qf)),
    p7_col_div64(w, t, h, q, p7_tail64(t, h, q, tf, hf, qf)),
    p7_col_div16(w, t, h, q, p7_hq16(h, q, hf, qf)))

def p7_join4(-A: Type, -B: Type, -C: Type, -D: Type, p: A & B, r: C & D) -> A & B & C & D:
  match p:
    case (a, b):
      match r:
        case (c, d):
          (a, b, c, d)

def p7_col_same(+w: Nat, +t: Nat, +h: Nat, +q: Nat, +w2: Nat, +t2: Nat, +h2: Nat, +q2: Nat,
    same: {Impl.col(w, t, h, q) == Impl.col(w2, t2, h2, q2) : Nat})
  -> {Nat.add(Nat.mul(Impl.wtile(w, t), 16n), Nat.add(Nat.mul(h, 8n), q)) == Nat.add(Nat.mul(Impl.wtile(w2, t2), 16n), Nat.add(Nat.mul(h2, 8n), q2)) : Nat}:
  %Equal.sym(Nat, Nat.add(Nat.mul(Impl.wtile(w, t), 16n), Nat.add(Nat.mul(h, 8n), q)), Impl.col(w, t, h, q), p7_col16(w, t, h, q)) : {_ == Nat.add(Nat.mul(Impl.wtile(w2, t2), 16n), Nat.add(Nat.mul(h2, 8n), q2)) : Nat}
  %Equal.sym(Nat, Nat.add(Nat.mul(Impl.wtile(w2, t2), 16n), Nat.add(Nat.mul(h2, 8n), q2)), Impl.col(w2, t2, h2, q2), p7_col16(w2, t2, h2, q2)) : {Impl.col(w, t, h, q) == _ : Nat}
  same

def p7_col_inj_fin(+w: Nat, +t: Nat, +h: Nat, +q: Nat, +w2: Nat, +t2: Nat, +h2: Nat, +q2: Nat,
    tw: {Nat.is_lt(t, 4n) == True{} : Bool}, qw: {Nat.is_lt(q, 8n) == True{} : Bool},
    tw2: {Nat.is_lt(t2, 4n) == True{} : Bool}, qw2: {Nat.is_lt(q2, 8n) == True{} : Bool},
    both: {Impl.wtile(w, t) == Impl.wtile(w2, t2) : Nat} & {Nat.add(Nat.mul(h, 8n), q) == Nat.add(Nat.mul(h2, 8n), q2) : Nat})
  -> {w == w2 : Nat} & {t == t2 : Nat} & {h == h2 : Nat} & {q == q2 : Nat}:
  match both:
    case (e1, e2):
      p7_join4({w == w2 : Nat}, {t == t2 : Nat}, {h == h2 : Nat}, {q == q2 : Nat},
        ar_divmod_inj(w, t, w2, t2, 4n, tw, tw2, e1),
        ar_divmod_inj(h, q, h2, q2, 8n, qw, qw2, e2))

def Laws.warp_columns_injective(w, t, h, q, w2, t2, h2, q2, tw, hw, qw, tw2, hw2, qw2, same):
  +qa = qw
  +qb = qw2
  p7_col_inj_fin(w, t, h, q, w2, t2, h2, q2, tw, qa, tw2, qb,
    ar_divmod_inj(Impl.wtile(w, t), Nat.add(Nat.mul(h, 8n), q), Impl.wtile(w2, t2), Nat.add(Nat.mul(h2, 8n), q2), 16n,
      p7_hq16(h, q, hw, qa), p7_hq16(h2, q2, hw2, qb),
      p7_col_same(w, t, h, q, w2, t2, h2, q2, same)))

# c == col(c / 64, c % 64 / 16, c % 16 / 8, c % 8).
def p7_cover_col(+c: Nat)
  -> {Impl.col(Nat.div(c, 64n), Nat.div(Nat.mod(c, 64n), 16n), Nat.div(Nat.mod(c, 16n), 8n), Nat.mod(c, 8n)) == c : Nat}:
  %p7_col_a(Nat.div(c, 64n), Nat.div(Nat.mod(c, 64n), 16n), Nat.div(Nat.mod(c, 16n), 8n), Nat.mod(c, 8n)) : {_ == c : Nat}
  %Equal.sym(Nat, Nat.add(Nat.mul(Nat.div(Nat.mod(c, 16n), 8n), 8n), Nat.mod(c, 8n)), Nat.mod(c, 16n), p7_digit(c, 2n, 8n, {==}, {==})) : {Nat.add(Nat.add(Nat.mul(Nat.div(c, 64n), 64n), Nat.mul(Nat.div(Nat.mod(c, 64n), 16n), 16n)), _) == c : Nat}
  %Equal.sym(Nat, Nat.add(Nat.add(Nat.mul(Nat.div(c, 64n), 64n), Nat.mul(Nat.div(Nat.mod(c, 64n), 16n), 16n)), Nat.mod(c, 16n)), Nat.add(Nat.mul(Nat.div(c, 64n), 64n), Nat.add(Nat.mul(Nat.div(Nat.mod(c, 64n), 16n), 16n), Nat.mod(c, 16n))), ar_add_assoc(Nat.mul(Nat.div(c, 64n), 64n), Nat.mul(Nat.div(Nat.mod(c, 64n), 16n), 16n), Nat.mod(c, 16n))) : {_ == c : Nat}
  %Equal.sym(Nat, Nat.add(Nat.mul(Nat.div(Nat.mod(c, 64n), 16n), 16n), Nat.mod(c, 16n)), Nat.mod(c, 64n), p7_digit(c, 4n, 16n, {==}, {==})) : {Nat.add(Nat.mul(Nat.div(c, 64n), 64n), _) == c : Nat}
  p7_dm_eq(c, 64n, {==})

def Laws.warp_columns_cover(c, inside):
  (Equal.trans(Bool, Nat.is_lt(Nat.div(c, 64n), 8n), Nat.is_lt(c, Nat.mul(8n, 64n)), True{},
    ar_div_lt(8n, c, 64n, {==}), inside),
    p7_cover_col(c))

# ---- lane rows ----

# row = mt 8 + (a 2 + c1).
def p7_row8(+mt: Nat, +a: Nat, +c1: Nat)
  -> {Nat.add(Nat.mul(mt, 8n), Nat.add(Nat.mul(a, 2n), c1)) == Impl.row(mt, a, c1) : Nat}:
  %ar_mul_comm(2n, a) : {Nat.add(Nat.mul(mt, 8n), Nat.add(_, c1)) == Impl.row(mt, a, c1) : Nat}
  Equal.sym(Nat, Impl.row(mt, a, c1), Nat.add(Nat.mul(mt, 8n), Nat.add(Nat.mul(2n, a), c1)),
    ar_add_assoc(Nat.mul(mt, 8n), Nat.mul(2n, a), c1))

def p7_ac8(+a: Nat, +c1: Nat, aw: {Nat.is_lt(a, 4n) == True{} : Bool},
    cw: {Nat.is_lt(c1, 2n) == True{} : Bool})
  -> {Nat.is_lt(Nat.add(Nat.mul(a, 2n), c1), 8n) == True{} : Bool}:
  p7_mr_lt(a, c1, 4n, 2n, aw, cw)

def p7_row_same(+mt: Nat, +a: Nat, +c1: Nat, +mt2: Nat, +a2: Nat, +c2: Nat,
    same: {Impl.row(mt, a, c1) == Impl.row(mt2, a2, c2) : Nat})
  -> {Nat.add(Nat.mul(mt, 8n), Nat.add(Nat.mul(a, 2n), c1)) == Nat.add(Nat.mul(mt2, 8n), Nat.add(Nat.mul(a2, 2n), c2)) : Nat}:
  %Equal.sym(Nat, Nat.add(Nat.mul(mt, 8n), Nat.add(Nat.mul(a, 2n), c1)), Impl.row(mt, a, c1), p7_row8(mt, a, c1)) : {_ == Nat.add(Nat.mul(mt2, 8n), Nat.add(Nat.mul(a2, 2n), c2)) : Nat}
  %Equal.sym(Nat, Nat.add(Nat.mul(mt2, 8n), Nat.add(Nat.mul(a2, 2n), c2)), Impl.row(mt2, a2, c2), p7_row8(mt2, a2, c2)) : {Impl.row(mt, a, c1) == _ : Nat}
  same

def p7_join3(-A: Type, -B: Type, -C: Type, x: A, p: B & C) -> A & B & C:
  match p:
    case (b, c):
      (x, b, c)

def p7_row_inj_fin(+mt: Nat, +a: Nat, +c1: Nat, +mt2: Nat, +a2: Nat, +c2: Nat,
    cw: {Nat.is_lt(c1, 2n) == True{} : Bool}, cw2: {Nat.is_lt(c2, 2n) == True{} : Bool},
    both: {mt == mt2 : Nat} & {Nat.add(Nat.mul(a, 2n), c1) == Nat.add(Nat.mul(a2, 2n), c2) : Nat})
  -> {mt == mt2 : Nat} & {a == a2 : Nat} & {c1 == c2 : Nat}:
  match both:
    case (e1, e2):
      p7_join3({mt == mt2 : Nat}, {a == a2 : Nat}, {c1 == c2 : Nat}, e1,
        ar_divmod_inj(a, c1, a2, c2, 2n, cw, cw2, e2))

def Laws.lane_rows(mt, a, c1, mt2, a2, c2, aw, cw, aw2, cw2, same):
  +ca = cw
  +cb = cw2
  p7_row_inj_fin(mt, a, c1, mt2, a2, c2, ca, cb,
    ar_divmod_inj(mt, Nat.add(Nat.mul(a, 2n), c1), mt2, Nat.add(Nat.mul(a2, 2n), c2), 8n,
      p7_ac8(a, c1, aw, ca), p7_ac8(a2, c2, aw2, cb),
      p7_row_same(mt, a, c1, mt2, a2, c2, same)))

# row(r / 8, r % 8 / 2, r % 2) == r.
def p7_rows_cover1(+r: Nat)
  -> {Impl.row(Nat.div(r, 8n), Nat.div(Nat.mod(r, 8n), 2n), Nat.mod(r, 2n)) == r : Nat}:
  %p7_row8(Nat.div(r, 8n), Nat.div(Nat.mod(r, 8n), 2n), Nat.mod(r, 2n)) : {_ == r : Nat}
  %Equal.sym(Nat, Nat.add(Nat.mul(Nat.div(Nat.mod(r, 8n), 2n), 2n), Nat.mod(r, 2n)), Nat.mod(r, 8n), p7_digit(r, 4n, 2n, {==}, {==})) : {Nat.add(Nat.mul(Nat.div(r, 8n), 8n), _) == r : Nat}
  p7_dm_eq(r, 8n, {==})

def p7_rows_cover2(+r: Nat)
  -> {Nat.div(r, 8n) == Nat.div(Impl.row(Nat.div(r, 8n), Nat.div(Nat.mod(r, 8n), 2n), Nat.mod(r, 2n)), 8n) : Nat}:
  %Equal.sym(Nat, Impl.row(Nat.div(r, 8n), Nat.div(Nat.mod(r, 8n), 2n), Nat.mod(r, 2n)), r, p7_rows_cover1(r)) : {Nat.div(r, 8n) == Nat.div(_, 8n) : Nat}
  {==}

def Laws.lane_rows_cover(r):
  (p7_rows_cover1(r), p7_rows_cover2(r))

# ==== Law finish_jobs_once ====

def p8_fst(-A: Type, -B: Type, p: A & B) -> A:
  (a, b) = p
  a

def p8_snd(-A: Type, -B: Type, p: A & B) -> B:
  (a, b) = p
  b

# not(b) == False forces b == True.
def p8_not_false(b: Bool, e: {Bool.not(b) == False{} : Bool}) -> {b == True{} : Bool}:
  match b:
    case False{}:
      Empty.absurd({False{} == True{} : Bool}, ar_true_false(e))
    case True{}:
      {==}

def p8_bool_ext(a: Bool, b: Bool, f: {a == True{} : Bool} -> {b == True{} : Bool},
    g: {b == True{} : Bool} -> {a == True{} : Bool}) -> {a == b : Bool}:
  match a b:
    case False{} False{}:
      {==}
    case False{} True{}:
      Empty.absurd({False{} == True{} : Bool}, ar_false_true(g({==})))
    case True{} False{}:
      Empty.absurd({True{} == False{} : Bool}, ar_false_true(f({==})))
    case True{} True{}:
      {==}

def p8_add_cong(-a: Nat, -a2: Nat, -b: Nat, -b2: Nat, ea: {a == a2 : Nat}, eb: {b == b2 : Nat})
  -> {Nat.add(a, b) == Nat.add(a2, b2) : Nat}:
  %ea : {Nat.add(a, b) == Nat.add(_, b2) : Nat}
  %eb : {Nat.add(a, b) == Nat.add(a, _) : Nat}
  {==}

# Interval split in Bool form: bit(a && m) + bit(n && h) == bit(a && h) when m == !n,
# m -> h and n -> a.
def p8_split_b(a: Bool, m: Bool, n: Bool, h: Bool, nm: {m == Bool.not(n) : Bool},
    mh: {m == True{} : Bool} -> {h == True{} : Bool},
    na: {n == True{} : Bool} -> {a == True{} : Bool})
  -> {Nat.add(Spec.bit(Bool.and(a, m)), Spec.bit(Bool.and(n, h))) == Spec.bit(Bool.and(a, h)) : Nat}:
  match a m n h:
    case False{} False{} False{} False{}:
      {==}
    case False{} False{} False{} True{}:
      {==}
    case False{} False{} True{} False{}:
      {==}
    case False{} False{} True{} True{}:
      Empty.absurd({1n == 0n : Nat}, ar_false_true(na({==})))
    case False{} True{} False{} False{}:
      {==}
    case False{} True{} False{} True{}:
      {==}
    case False{} True{} True{} False{}:
      {==}
    case False{} True{} True{} True{}:
      Empty.absurd({1n == 0n : Nat}, ar_true_false(nm))
    case True{} False{} False{} False{}:
      {==}
    case True{} False{} False{} True{}:
      Empty.absurd({0n == 1n : Nat}, ar_false_true(nm))
    case True{} False{} True{} False{}:
      {==}
    case True{} False{} True{} True{}:
      {==}
    case True{} True{} False{} False{}:
      Empty.absurd({1n == 0n : Nat}, ar_false_true(mh({==})))
    case True{} True{} False{} True{}:
      {==}
    case True{} True{} True{} False{}:
      Empty.absurd({1n == 0n : Nat}, ar_true_false(nm))
    case True{} True{} True{} True{}:
      Empty.absurd({2n == 1n : Nat}, ar_true_false(nm))

# Guarding both sides of a bit sum with the same L.
def p8_band(L: Bool, -X: Bool, -Y: Bool, -Z: Bool,
    h: {Nat.add(Spec.bit(X), Spec.bit(Y)) == Spec.bit(Z) : Nat})
  -> {Nat.add(Spec.bit(Bool.and(L, X)), Spec.bit(Bool.and(L, Y))) == Spec.bit(Bool.and(L, Z)) : Nat}:
  match L:
    case False{}:
      {==}
    case True{}:
      h

def p8_zero_b(L: Bool, a: Bool, m: Bool, e: {m == Bool.not(a) : Bool})
  -> {0n == Spec.bit(Bool.and(L, Bool.and(a, m))) : Nat}:
  match L a m:
    case False{} False{} False{}:
      {==}
    case False{} False{} True{}:
      {==}
    case False{} True{} False{}:
      {==}
    case False{} True{} True{}:
      {==}
    case True{} False{} False{}:
      {==}
    case True{} False{} True{}:
      {==}
    case True{} True{} False{}:
      {==}
    case True{} True{} True{}:
      Empty.absurd({0n == 1n : Nat}, ar_true_false(e))

def p8_zero_f(L: Bool, m: Bool, e: {m == False{} : Bool})
  -> {0n == Spec.bit(Bool.and(L, m)) : Nat}:
  match L m:
    case False{} False{}:
      {==}
    case False{} True{}:
      {==}
    case True{} False{}:
      {==}
    case True{} True{}:
      Empty.absurd({0n == 1n : Nat}, ar_true_false(e))

def p8_bit_and_true(L: Bool, m: Bool, e: {m == True{} : Bool})
  -> {Spec.bit(Bool.and(L, m)) == Spec.bit(L) : Nat}:
  match L m:
    case False{} False{}:
      {==}
    case False{} True{}:
      {==}
    case True{} False{}:
      Empty.absurd({0n == 1n : Nat}, ar_false_true(e))
    case True{} True{}:
      {==}

# v == r iff v <= r < v + 1.
def p8_eq_iv(v: Nat, r: Nat)
  -> {Nat.is_eq(v, r) == Bool.and(Nat.is_le(v, r), Nat.is_lt(r, 1n+v)) : Bool}:
  match v r:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      Equal.sym(Bool, Nat.is_lt(q, 0n), False{}, ar_h_lt_zero(q))
    case 1n+p 0n:
      {==}
    case 1n+p 1n+q:
      p8_eq_iv(p, q)

def p8_and_iv(+L: Bool, +v: Nat, +r: Nat)
  -> {Bool.and(L, Nat.is_eq(v, r)) == Bool.and(L, Bool.and(Nat.is_le(v, r), Nat.is_lt(r, 1n+v))) : Bool}:
  %p8_eq_iv(v, r) : {Bool.and(L, Nat.is_eq(v, r)) == Bool.and(L, _) : Bool}
  {==}

# [lo, mid) and [mid, hi) partition [lo, hi).
def p8_split(+lo: Nat, +mid: Nat, +hi: Nat, +r: Nat, lm: {Nat.is_le(lo, mid) == True{} : Bool},
    mh: {Nat.is_le(mid, hi) == True{} : Bool})
  -> {Nat.add(Spec.bit(Bool.and(Nat.is_le(lo, r), Nat.is_lt(r, mid))),
      Spec.bit(Bool.and(Nat.is_le(mid, r), Nat.is_lt(r, hi)))) ==
      Spec.bit(Bool.and(Nat.is_le(lo, r), Nat.is_lt(r, hi))) : Nat}:
  p8_split_b(Nat.is_le(lo, r), Nat.is_lt(r, mid), Nat.is_le(mid, r), Nat.is_lt(r, hi),
    ar_lt_not_le(r, mid),
    e => ar_lt_le_trans(r, mid, hi, e, mh),
    e => ar_le_trans(lo, mid, r, lm, e))

# [0, mid) and [mid, hi) partition [0, hi).
def p8_split0(+mid: Nat, +hi: Nat, +r: Nat, mh: {Nat.is_le(mid, hi) == True{} : Bool})
  -> {Nat.add(Spec.bit(Nat.is_lt(r, mid)),
      Spec.bit(Bool.and(Nat.is_le(mid, r), Nat.is_lt(r, hi)))) ==
      Spec.bit(Nat.is_lt(r, hi)) : Nat}:
  p8_split_b(True{}, Nat.is_lt(r, mid), Nat.is_le(mid, r), Nat.is_lt(r, hi),
    ar_lt_not_le(r, mid),
    e => ar_lt_le_trans(r, mid, hi, e, mh),
    e => {==})

def p8_le_split(k: Nat, q: Nat)
  -> {Nat.is_le(k, q) == Bool.or(Nat.is_eq(k, q), Nat.is_le(1n+k, q)) : Bool}:
  match k q:
    case 0n 0n:
      {==}
    case 0n 1n+b:
      Equal.sym(Bool, Nat.is_le(0n, b), True{}, ar_h_le_zero(b))
    case 1n+a 0n:
      {==}
    case 1n+a 1n+b:
      p8_le_split(a, b)

def p8_eq_of(+a: Nat, +b: Nat, e: {a == b : Nat}) -> {Nat.is_eq(a, b) == True{} : Bool}:
  %e : {Nat.is_eq(a, _) == True{} : Bool}
  ar_eq_refl(a)

def p8_lin_eq(+k: Nat, +q: Nat, +u: Nat, +r: Nat, +s: Nat, kq: {k == q : Nat}, ur: {u == r : Nat})
  -> {Nat.add(Nat.mul(k, s), u) == Nat.add(Nat.mul(q, s), r) : Nat}:
  %kq : {Nat.add(Nat.mul(k, s), u) == Nat.add(Nat.mul(_, s), r) : Nat}
  %ur : {Nat.add(Nat.mul(k, s), u) == Nat.add(Nat.mul(k, s), _) : Nat}
  {==}

def p8_le_lin(+k: Nat, +q: Nat, +s: Nat, +u: Nat, +r: Nat,
    lk: {Nat.is_le(k, q) == True{} : Bool}, ur: {u == r : Nat})
  -> {Nat.is_le(Nat.add(Nat.mul(k, s), u), Nat.add(Nat.mul(q, s), r)) == True{} : Bool}:
  %ur : {Nat.is_le(Nat.add(Nat.mul(k, s), u), Nat.add(Nat.mul(q, s), _)) == True{} : Bool}
  %ar_add_comm(u, Nat.mul(k, s)) : {Nat.is_le(_, Nat.add(Nat.mul(q, s), u)) == True{} : Bool}
  %ar_add_comm(u, Nat.mul(q, s)) : {Nat.is_le(Nat.add(u, Nat.mul(k, s)), _) == True{} : Bool}
  Equal.trans(Bool, Nat.is_le(Nat.add(u, Nat.mul(k, s)), Nat.add(u, Nat.mul(q, s))),
    Nat.is_le(Nat.mul(k, s), Nat.mul(q, s)), True{},
    ar_le_add_cancel(u, Nat.mul(k, s), Nat.mul(q, s)), ar_le_mul_mono(k, q, s, lk))

# ---- membership of the K:358 job loop ----

# x0 = k * s + u, x = q * s + r (u, r < s): x0 == x iff k == q and u == r.
def p8_and_pair(+k: Nat, +q: Nat, +u: Nat, +r: Nat, pr: {k == q : Nat} & {u == r : Nat})
  -> {Bool.and(Nat.is_eq(k, q), Nat.is_eq(u, r)) == True{} : Bool}:
  (kq, ur) = pr
  ar_and_intro(Nat.is_eq(k, q), Nat.is_eq(u, r), p8_eq_of(k, q, kq), p8_eq_of(u, r, ur))

def p8_eqA_f(+x0: Nat, +x: Nat, +k: Nat, +q: Nat, +u: Nat, +r: Nat, +s: Nat,
    ex0: {Nat.add(Nat.mul(k, s), u) == x0 : Nat}, ex: {Nat.add(Nat.mul(q, s), r) == x : Nat},
    us: {Nat.is_lt(u, s) == True{} : Bool}, rs: {Nat.is_lt(r, s) == True{} : Bool},
    e: {Nat.is_eq(x0, x) == True{} : Bool})
  -> {Bool.and(Nat.is_eq(k, q), Nat.is_eq(u, r)) == True{} : Bool}:
  p8_and_pair(k, q, u, r, ar_divmod_inj(k, u, q, r, s, us, rs,
    Equal.trans(Nat, Nat.add(Nat.mul(k, s), u), x0, Nat.add(Nat.mul(q, s), r), ex0,
      Equal.trans(Nat, x0, x, Nat.add(Nat.mul(q, s), r), ar_eq_true(x0, x, e),
        Equal.sym(Nat, Nat.add(Nat.mul(q, s), r), x, ex)))))

def p8_eqA_g(+x0: Nat, +x: Nat, +k: Nat, +q: Nat, +u: Nat, +r: Nat, +s: Nat,
    ex0: {Nat.add(Nat.mul(k, s), u) == x0 : Nat}, ex: {Nat.add(Nat.mul(q, s), r) == x : Nat},
    +e: {Bool.and(Nat.is_eq(k, q), Nat.is_eq(u, r)) == True{} : Bool})
  -> {Nat.is_eq(x0, x) == True{} : Bool}:
  p8_eq_of(x0, x,
    Equal.trans(Nat, x0, Nat.add(Nat.mul(k, s), u), x, Equal.sym(Nat, Nat.add(Nat.mul(k, s), u), x0, ex0),
      Equal.trans(Nat, Nat.add(Nat.mul(k, s), u), Nat.add(Nat.mul(q, s), r), x,
        p8_lin_eq(k, q, u, r, s,
          ar_eq_true(k, q, ar_and_l(Nat.is_eq(k, q), Nat.is_eq(u, r), e)),
          ar_eq_true(u, r, ar_and_r(Nat.is_eq(k, q), Nat.is_eq(u, r), e))),
        ex)))

def p8_eqA(+x0: Nat, +x: Nat, +k: Nat, +q: Nat, +u: Nat, +r: Nat, +s: Nat,
    +ex0: {Nat.add(Nat.mul(k, s), u) == x0 : Nat}, +ex: {Nat.add(Nat.mul(q, s), r) == x : Nat},
    +us: {Nat.is_lt(u, s) == True{} : Bool}, +rs: {Nat.is_lt(r, s) == True{} : Bool})
  -> {Nat.is_eq(x0, x) == Bool.and(Nat.is_eq(k, q), Nat.is_eq(u, r)) : Bool}:
  p8_bool_ext(Nat.is_eq(x0, x), Bool.and(Nat.is_eq(k, q), Nat.is_eq(u, r)),
    e => p8_eqA_f(x0, x, k, q, u, r, s, ex0, ex, us, rs, e),
    e => p8_eqA_g(x0, x, k, q, u, r, s, ex0, ex, e))

# The loop step: or(x0 == x, rest) == hits from k, given x0 < J.
def p8_step_c(Ek: Bool, U: Bool, L: Bool, K1: Bool,
    h: {Bool.and(Ek, U) == True{} : Bool} -> {L == True{} : Bool})
  -> {Bool.or(Bool.and(Ek, U), Bool.and(L, Bool.and(K1, U))) ==
      Bool.and(L, Bool.and(Bool.or(Ek, K1), U)) : Bool}:
  match Ek U L K1:
    case False{} _ _ _:
      {==}
    case True{} False{} False{} _:
      {==}
    case True{} False{} True{} False{}:
      {==}
    case True{} False{} True{} True{}:
      {==}
    case True{} True{} False{} _:
      Empty.absurd({True{} == False{} : Bool}, ar_false_true(h({==})))
    case True{} True{} True{} _:
      {==}

def p8_le_x0x(+x0: Nat, +x: Nat, +k: Nat, +q: Nat, +s: Nat, +u: Nat, +r: Nat,
    ex0: {Nat.add(Nat.mul(k, s), u) == x0 : Nat}, ex: {Nat.add(Nat.mul(q, s), r) == x : Nat},
    lin: {Nat.is_le(Nat.add(Nat.mul(k, s), u), Nat.add(Nat.mul(q, s), r)) == True{} : Bool})
  -> {Nat.is_le(x0, x) == True{} : Bool}:
  %ex0 : {Nat.is_le(_, x) == True{} : Bool}
  %ex : {Nat.is_le(Nat.add(Nat.mul(k, s), u), _) == True{} : Bool}
  lin

def p8_hitL(+x0: Nat, +x: Nat, +J: Nat, +k: Nat, +q: Nat, +u: Nat, +r: Nat, +s: Nat,
    ex0: {Nat.add(Nat.mul(k, s), u) == x0 : Nat}, ex: {Nat.add(Nat.mul(q, s), r) == x : Nat},
    lt0: {Nat.is_lt(x0, J) == True{} : Bool},
    e: {Bool.and(Nat.is_eq(k, q), Nat.is_eq(u, r)) == True{} : Bool})
  -> {Nat.is_lt(x, J) == True{} : Bool}:
  %ar_eq_true(x0, x, p8_eqA_g(x0, x, k, q, u, r, s, ex0, ex, e)) : {Nat.is_lt(_, J) == True{} : Bool}
  lt0

# A hit x0 <= x < J is impossible once J <= x0.
def p8_nohit(+x0: Nat, +x: Nat, +J: Nat, +k: Nat, +q: Nat, +u: Nat, +r: Nat, +s: Nat,
    ex0: {Nat.add(Nat.mul(k, s), u) == x0 : Nat}, ex: {Nat.add(Nat.mul(q, s), r) == x : Nat},
    jx0: {Nat.is_le(J, x0) == True{} : Bool},
    +h: {Bool.and(Nat.is_lt(x, J), Bool.and(Nat.is_le(k, q), Nat.is_eq(u, r))) == True{} : Bool})
  -> Empty:
  +rest = ar_and_r(Nat.is_lt(x, J), Bool.and(Nat.is_le(k, q), Nat.is_eq(u, r)), h)
  ar_false_true(Equal.trans(Bool, False{}, Nat.is_lt(x0, x0), True{},
    Equal.sym(Bool, Nat.is_lt(x0, x0), False{}, ar_lt_irrefl(x0)),
    ar_lt_le_trans(x0, J, x0,
      ar_le_lt_trans(x0, x, J,
        p8_le_x0x(x0, x, k, q, s, u, r, ex0, ex,
          p8_le_lin(k, q, s, u, r, ar_and_l(Nat.is_le(k, q), Nat.is_eq(u, r), rest),
            ar_eq_true(u, r, ar_and_r(Nat.is_le(k, q), Nat.is_eq(u, r), rest)))),
        ar_and_l(Nat.is_lt(x, J), Bool.and(Nat.is_le(k, q), Nat.is_eq(u, r)), h)),
      jx0)))

def p8_ex0_step(+k: Nat, +s: Nat, +u: Nat, +x0: Nat, ex0: {Nat.add(Nat.mul(k, s), u) == x0 : Nat})
  -> {Nat.add(Nat.mul(1n+k, s), u) == Nat.add(x0, s) : Nat}:
  %ex0 : {Nat.add(Nat.mul(1n+k, s), u) == Nat.add(_, s) : Nat}
  %ar_add_comm(s, Nat.add(Nat.mul(k, s), u)) : {Nat.add(Nat.mul(1n+k, s), u) == _ : Nat}
  ar_add_assoc(s, Nat.mul(k, s), u)

def p8_lt_self_add(+a: Nat, +s: Nat, spos: {Nat.is_lt(0n, s) == True{} : Bool})
  -> {Nat.is_lt(a, Nat.add(a, s)) == True{} : Bool}:
  %ar_add_zero_r(a) : {Nat.is_lt(_, Nat.add(a, s)) == True{} : Bool}
  Equal.trans(Bool, Nat.is_lt(Nat.add(a, 0n), Nat.add(a, s)), Nat.is_lt(0n, s), True{},
    ar_lt_add_cancel(a, 0n, s), spos)

# Fuel stays ahead of the jobs bound: J < fuel + x0 is preserved by one loop step.
def p8_fuel_step(+f: Nat, +x0: Nat, +s: Nat, +J: Nat, spos: {Nat.is_lt(0n, s) == True{} : Bool},
    fuelok: {Nat.is_lt(J, Nat.add(1n+f, x0)) == True{} : Bool})
  -> {Nat.is_lt(J, Nat.add(f, Nat.add(x0, s))) == True{} : Bool}:
  ar_le_lt_trans(J, Nat.add(f, x0), Nat.add(f, Nat.add(x0, s)),
    Equal.trans(Bool, Nat.is_le(J, Nat.add(f, x0)), Nat.is_lt(J, 1n+Nat.add(f, x0)), True{},
      Equal.sym(Bool, Nat.is_lt(J, 1n+Nat.add(f, x0)), Nat.is_le(J, Nat.add(f, x0)),
        ar_lt_succ_le(J, 1n+Nat.add(f, x0))),
      fuelok),
    Equal.trans(Bool, Nat.is_lt(Nat.add(f, x0), Nat.add(f, Nat.add(x0, s))),
      Nat.is_lt(x0, Nat.add(x0, s)), True{},
      ar_lt_add_cancel(f, x0, Nat.add(x0, s)), p8_lt_self_add(x0, s, spos)))

# Membership in the K:358 loop from x0 = k * s + u (u < s) with enough fuel:
# x = q * s + r (r < s) is visited iff x < J, k <= q and u == r.
def p8_mem(fuel: Nat, g: Bool, +x0: Nat, +s: Nat, +J: Nat, +k: Nat, +u: Nat, +q: Nat, +r: Nat,
    +x: Nat, eg: {Nat.is_lt(x0, J) == g : Bool},
    +ex0: {Nat.add(Nat.mul(k, s), u) == x0 : Nat}, +ex: {Nat.add(Nat.mul(q, s), r) == x : Nat},
    +us: {Nat.is_lt(u, s) == True{} : Bool}, +rs: {Nat.is_lt(r, s) == True{} : Bool},
    fuelok: {Nat.is_lt(J, Nat.add(fuel, x0)) == True{} : Bool})
  -> {Spec.member(Impl.jobloop.go(fuel, g, x0, s, J), x) ==
      Bool.and(Nat.is_lt(x, J), Bool.and(Nat.is_le(k, q), Nat.is_eq(u, r))) : Bool}:
  match fuel g:
    case 0n _:
      Equal.sym(Bool, Bool.and(Nat.is_lt(x, J), Bool.and(Nat.is_le(k, q), Nat.is_eq(u, r))), False{},
        ar_not_true(Bool.and(Nat.is_lt(x, J), Bool.and(Nat.is_le(k, q), Nat.is_eq(u, r))),
          h => p8_nohit(x0, x, J, k, q, u, r, s, ex0, ex, ar_lt_le(J, x0, fuelok), h)))
    case 1n+f False{}:
      Equal.sym(Bool, Bool.and(Nat.is_lt(x, J), Bool.and(Nat.is_le(k, q), Nat.is_eq(u, r))), False{},
        ar_not_true(Bool.and(Nat.is_lt(x, J), Bool.and(Nat.is_le(k, q), Nat.is_eq(u, r))),
          h => p8_nohit(x0, x, J, k, q, u, r, s, ex0, ex,
            p8_not_false(Nat.is_le(J, x0),
              Equal.trans(Bool, Bool.not(Nat.is_le(J, x0)), Nat.is_lt(x0, J), False{},
                Equal.sym(Bool, Nat.is_lt(x0, J), Bool.not(Nat.is_le(J, x0)), ar_lt_not_le(x0, J)),
                eg)), h)))
    case 1n+f True{}:
      +fr = f
      +lt0 = eg
      %Equal.sym(Bool, Spec.member(Impl.jobloop.go(f, Nat.is_lt(Nat.add(x0, s), J), Nat.add(x0, s), s, J), x),
        Bool.and(Nat.is_lt(x, J), Bool.and(Nat.is_le(1n+k, q), Nat.is_eq(u, r))),
        p8_mem(fr, Nat.is_lt(Nat.add(x0, s), J), Nat.add(x0, s), s, J, 1n+k, u, q, r, x, {==},
          p8_ex0_step(k, s, u, x0, ex0), ex, us, rs,
          p8_fuel_step(fr, x0, s, J,
            ar_le_lt_trans(0n, u, s, ar_h_le_zero(u), us), fuelok))) : {Bool.or(Nat.is_eq(x0, x), _) == Bool.and(Nat.is_lt(x, J), Bool.and(Nat.is_le(k, q), Nat.is_eq(u, r))) : Bool}
      %Equal.sym(Bool, Nat.is_eq(x0, x), Bool.and(Nat.is_eq(k, q), Nat.is_eq(u, r)),
        p8_eqA(x0, x, k, q, u, r, s, ex0, ex, us, rs)) : {Bool.or(_, Bool.and(Nat.is_lt(x, J), Bool.and(Nat.is_le(1n+k, q), Nat.is_eq(u, r)))) == Bool.and(Nat.is_lt(x, J), Bool.and(Nat.is_le(k, q), Nat.is_eq(u, r))) : Bool}
      %Equal.sym(Bool, Nat.is_le(k, q), Bool.or(Nat.is_eq(k, q), Nat.is_le(1n+k, q)),
        p8_le_split(k, q)) : {Bool.or(Bool.and(Nat.is_eq(k, q), Nat.is_eq(u, r)), Bool.and(Nat.is_lt(x, J), Bool.and(Nat.is_le(1n+k, q), Nat.is_eq(u, r)))) == Bool.and(Nat.is_lt(x, J), Bool.and(_, Nat.is_eq(u, r))) : Bool}
      p8_step_c(Nat.is_eq(k, q), Nat.is_eq(u, r), Nat.is_lt(x, J), Nat.is_le(1n+k, q),
        e => p8_hitL(x0, x, J, k, q, u, r, s, ex0, ex, lt0, e))

# ---- one (block, warp) loop, then the double count ----

# lo <= r < hi.
def p8_I(+lo: Nat, +hi: Nat, +r: Nat) -> Bool:
  Bool.and(Nat.is_le(lo, r), Nat.is_lt(r, hi))

def p8_pos8(G: Nat, grid: {Nat.is_lt(0n, G) == True{} : Bool})
  -> {Nat.is_lt(0n, Nat.mul(8n, G)) == True{} : Bool}:
  match G:
    case 0n:
      Empty.absurd({Nat.is_lt(0n, Nat.mul(8n, 0n)) == True{} : Bool}, ar_false_true(grid))
    case 1n+g:
      {==}

def p8_fuel0(+J: Nat, +u: Nat) -> {Nat.is_lt(J, Nat.add(1n+J, u)) == True{} : Bool}:
  Equal.trans(Bool, Nat.is_lt(J, 1n+Nat.add(J, u)), Nat.is_le(J, Nat.add(J, u)), True{},
    ar_lt_succ_le(J, 1n+Nat.add(J, u)), ar_le_add_r(J, u))

def p8_drop_le0(+L: Bool, +q: Nat, +E: Bool)
  -> {Bool.and(L, Bool.and(Nat.is_le(0n, q), E)) == Bool.and(L, E) : Bool}:
  %Equal.sym(Bool, Nat.is_le(0n, q), True{}, ar_h_le_zero(q)) : {Bool.and(L, Bool.and(_, E)) == Bool.and(L, E) : Bool}
  {==}

# Loop (b, w) with u = w + 8b < 8G visits x iff x < J and u == x % 8G.
def p8_member(+b: Nat, +w: Nat, +x: Nat, +G: Nat, +J: Nat,
    ul: {Nat.is_lt(Nat.add(w, Nat.mul(8n, b)), Nat.mul(8n, G)) == True{} : Bool},
    +grid: {Nat.is_lt(0n, G) == True{} : Bool})
  -> {Spec.member(Impl.jobloop(b, w, G, J), x) ==
      Bool.and(Nat.is_lt(x, J), Nat.is_eq(Nat.add(w, Nat.mul(8n, b)), Nat.mod(x, Nat.mul(8n, G)))) : Bool}:
  Equal.trans(Bool, Spec.member(Impl.jobloop(b, w, G, J), x),
    Bool.and(Nat.is_lt(x, J), Bool.and(Nat.is_le(0n, Nat.div(x, Nat.mul(8n, G))),
      Nat.is_eq(Nat.add(w, Nat.mul(8n, b)), Nat.mod(x, Nat.mul(8n, G))))),
    Bool.and(Nat.is_lt(x, J), Nat.is_eq(Nat.add(w, Nat.mul(8n, b)), Nat.mod(x, Nat.mul(8n, G)))),
    p8_mem(1n+J, Nat.is_lt(Nat.add(w, Nat.mul(8n, b)), J), Nat.add(w, Nat.mul(8n, b)),
      Nat.mul(8n, G), J, 0n, Nat.add(w, Nat.mul(8n, b)), Nat.div(x, Nat.mul(8n, G)),
      Nat.mod(x, Nat.mul(8n, G)), x, {==}, {==},
      p8_fst({Nat.add(Nat.mul(Nat.div(x, Nat.mul(8n, G)), Nat.mul(8n, G)), Nat.mod(x, Nat.mul(8n, G))) == x : Nat},
        {Nat.is_lt(Nat.mod(x, Nat.mul(8n, G)), Nat.mul(8n, G)) == True{} : Bool},
        ar_divmod(x, Nat.mul(8n, G), p8_pos8(G, grid))),
      ul,
      p8_snd({Nat.add(Nat.mul(Nat.div(x, Nat.mul(8n, G)), Nat.mul(8n, G)), Nat.mod(x, Nat.mul(8n, G))) == x : Nat},
        {Nat.is_lt(Nat.mod(x, Nat.mul(8n, G)), Nat.mul(8n, G)) == True{} : Bool},
        ar_divmod(x, Nat.mul(8n, G), p8_pos8(G, grid))),
      p8_fuel0(J, Nat.add(w, Nat.mul(8n, b)))),
    p8_drop_le0(Nat.is_lt(x, J), Nat.div(x, Nat.mul(8n, G)),
      Nat.is_eq(Nat.add(w, Nat.mul(8n, b)), Nat.mod(x, Nat.mul(8n, G)))))

# w < 8 and t = 8b: w + t < 8 + t.
def p8_lt_shift(+w: Nat, +t: Nat, wl: {Nat.is_lt(w, 8n) == True{} : Bool})
  -> {Nat.is_lt(Nat.add(w, t), Nat.add(8n, t)) == True{} : Bool}:
  %ar_add_comm(t, w) : {Nat.is_lt(_, Nat.add(8n, t)) == True{} : Bool}
  %ar_add_comm(t, 8n) : {Nat.is_lt(Nat.add(t, w), _) == True{} : Bool}
  Equal.trans(Bool, Nat.is_lt(Nat.add(t, w), Nat.add(t, 8n)), Nat.is_lt(w, 8n), True{},
    ar_lt_add_cancel(t, w, 8n), wl)

# b < G: 8 + 8b <= 8G.
def p8_le8(+b: Nat, +G: Nat, bl: {Nat.is_lt(b, G) == True{} : Bool})
  -> {Nat.is_le(Nat.add(8n, Nat.mul(8n, b)), Nat.mul(8n, G)) == True{} : Bool}:
  %ar_mul_succ_r(8n, b) : {Nat.is_le(_, Nat.mul(8n, G)) == True{} : Bool}
  %ar_mul_comm(1n+b, 8n) : {Nat.is_le(_, Nat.mul(8n, G)) == True{} : Bool}
  %ar_mul_comm(G, 8n) : {Nat.is_le(Nat.mul(1n+b, 8n), _) == True{} : Bool}
  ar_le_mul_mono(1n+b, G, 8n,
    Equal.trans(Bool, Nat.is_le(1n+b, G), Nat.is_lt(b, G), True{},
      Equal.sym(Bool, Nat.is_lt(b, G), Nat.is_le(1n+b, G), ar_lt_succ_le(b, G)), bl))

def p8_u_lt(+w: Nat, +b: Nat, +G: Nat, wl: {Nat.is_lt(w, 8n) == True{} : Bool},
    bl: {Nat.is_lt(b, G) == True{} : Bool})
  -> {Nat.is_lt(Nat.add(w, Nat.mul(8n, b)), Nat.mul(8n, G)) == True{} : Bool}:
  ar_lt_le_trans(Nat.add(w, Nat.mul(8n, b)), Nat.add(8n, Nat.mul(8n, b)), Nat.mul(8n, G),
    p8_lt_shift(w, Nat.mul(8n, b), wl), p8_le8(b, G, bl))

# Warps w < c of block b (c <= 8, b < G) visit x once iff x < J and 8b <= x % 8G < 8b + c.
def p8_warp(c: Nat, +b: Nat, +x: Nat, +G: Nat, +J: Nat, cle: {Nat.is_le(c, 8n) == True{} : Bool},
    +bl: {Nat.is_lt(b, G) == True{} : Bool}, +grid: {Nat.is_lt(0n, G) == True{} : Bool})
  -> {Impl.warp_hits(c, b, x, G, J) == Spec.bit(Bool.and(Nat.is_lt(x, J),
      Bool.and(Nat.is_le(Nat.mul(8n, b), Nat.mod(x, Nat.mul(8n, G))),
        Nat.is_lt(Nat.mod(x, Nat.mul(8n, G)), Nat.add(c, Nat.mul(8n, b)))))) : Nat}:
  match c:
    case 0n:
      p8_zero_b(Nat.is_lt(x, J), Nat.is_le(Nat.mul(8n, b), Nat.mod(x, Nat.mul(8n, G))),
        Nat.is_lt(Nat.mod(x, Nat.mul(8n, G)), Nat.mul(8n, b)),
        ar_lt_not_le(Nat.mod(x, Nat.mul(8n, G)), Nat.mul(8n, b)))
    case 1n+p0:
      +p = p0
      +pl = Equal.trans(Bool, Nat.is_lt(p, 8n), Nat.is_le(1n+p, 8n), True{}, ar_lt_succ_le(p, 8n), cle)
      Equal.trans(Nat,
        Nat.add(Impl.warp_hits(p, b, x, G, J), Spec.bit(Spec.member(Impl.jobloop(b, p, G, J), x))),
        Nat.add(Spec.bit(Bool.and(Nat.is_lt(x, J), p8_I(Nat.mul(8n, b), Nat.add(p, Nat.mul(8n, b)), Nat.mod(x, Nat.mul(8n, G))))),
          Spec.bit(Bool.and(Nat.is_lt(x, J), p8_I(Nat.add(p, Nat.mul(8n, b)), 1n+Nat.add(p, Nat.mul(8n, b)), Nat.mod(x, Nat.mul(8n, G)))))),
        Spec.bit(Bool.and(Nat.is_lt(x, J), p8_I(Nat.mul(8n, b), 1n+Nat.add(p, Nat.mul(8n, b)), Nat.mod(x, Nat.mul(8n, G))))),
        p8_add_cong(Impl.warp_hits(p, b, x, G, J),
          Spec.bit(Bool.and(Nat.is_lt(x, J), p8_I(Nat.mul(8n, b), Nat.add(p, Nat.mul(8n, b)), Nat.mod(x, Nat.mul(8n, G))))),
          Spec.bit(Spec.member(Impl.jobloop(b, p, G, J), x)),
          Spec.bit(Bool.and(Nat.is_lt(x, J), p8_I(Nat.add(p, Nat.mul(8n, b)), 1n+Nat.add(p, Nat.mul(8n, b)), Nat.mod(x, Nat.mul(8n, G))))),
          p8_warp(p, b, x, G, J, ar_lt_le(p, 8n, pl), bl, grid),
          Equal.cong(Bool, Nat, Spec.bit, Spec.member(Impl.jobloop(b, p, G, J), x),
            Bool.and(Nat.is_lt(x, J), p8_I(Nat.add(p, Nat.mul(8n, b)), 1n+Nat.add(p, Nat.mul(8n, b)), Nat.mod(x, Nat.mul(8n, G)))),
            Equal.trans(Bool, Spec.member(Impl.jobloop(b, p, G, J), x),
              Bool.and(Nat.is_lt(x, J), Nat.is_eq(Nat.add(p, Nat.mul(8n, b)), Nat.mod(x, Nat.mul(8n, G)))),
              Bool.and(Nat.is_lt(x, J), p8_I(Nat.add(p, Nat.mul(8n, b)), 1n+Nat.add(p, Nat.mul(8n, b)), Nat.mod(x, Nat.mul(8n, G)))),
              p8_member(b, p, x, G, J, p8_u_lt(p, b, G, pl, bl), grid),
              p8_and_iv(Nat.is_lt(x, J), Nat.add(p, Nat.mul(8n, b)), Nat.mod(x, Nat.mul(8n, G)))))),
        p8_band(Nat.is_lt(x, J),
          p8_I(Nat.mul(8n, b), Nat.add(p, Nat.mul(8n, b)), Nat.mod(x, Nat.mul(8n, G))),
          p8_I(Nat.add(p, Nat.mul(8n, b)), 1n+Nat.add(p, Nat.mul(8n, b)), Nat.mod(x, Nat.mul(8n, G))),
          p8_I(Nat.mul(8n, b), 1n+Nat.add(p, Nat.mul(8n, b)), Nat.mod(x, Nat.mul(8n, G))),
          p8_split(Nat.mul(8n, b), Nat.add(p, Nat.mul(8n, b)), 1n+Nat.add(p, Nat.mul(8n, b)),
            Nat.mod(x, Nat.mul(8n, G)), ar_le_add_l(p, Nat.mul(8n, b)),
            ar_h_le_succ(Nat.add(p, Nat.mul(8n, b))))))

# Blocks b < n (n <= G) visit x once iff x < J and x % 8G < 8n.
def p8_jobs(n: Nat, +x: Nat, +G: Nat, +J: Nat, nle: {Nat.is_le(n, G) == True{} : Bool},
    +grid: {Nat.is_lt(0n, G) == True{} : Bool})
  -> {Impl.job_hits(n, x, G, J) == Spec.bit(Bool.and(Nat.is_lt(x, J),
      Nat.is_lt(Nat.mod(x, Nat.mul(8n, G)), Nat.mul(8n, n)))) : Nat}:
  match n:
    case 0n:
      p8_zero_f(Nat.is_lt(x, J), Nat.is_lt(Nat.mod(x, Nat.mul(8n, G)), 0n),
        ar_h_lt_zero(Nat.mod(x, Nat.mul(8n, G))))
    case 1n+p0:
      +p = p0
      +pl = Equal.trans(Bool, Nat.is_lt(p, G), Nat.is_le(1n+p, G), True{}, ar_lt_succ_le(p, G), nle)
      %Equal.sym(Nat, Nat.mul(8n, 1n+p), Nat.add(8n, Nat.mul(8n, p)), ar_mul_succ_r(8n, p)) : {Nat.add(Impl.job_hits(p, x, G, J), Impl.warp_hits(8n, p, x, G, J)) == Spec.bit(Bool.and(Nat.is_lt(x, J), Nat.is_lt(Nat.mod(x, Nat.mul(8n, G)), _))) : Nat}
      Equal.trans(Nat, Nat.add(Impl.job_hits(p, x, G, J), Impl.warp_hits(8n, p, x, G, J)),
        Nat.add(Spec.bit(Bool.and(Nat.is_lt(x, J), Nat.is_lt(Nat.mod(x, Nat.mul(8n, G)), Nat.mul(8n, p)))),
          Spec.bit(Bool.and(Nat.is_lt(x, J), p8_I(Nat.mul(8n, p), Nat.add(8n, Nat.mul(8n, p)), Nat.mod(x, Nat.mul(8n, G)))))),
        Spec.bit(Bool.and(Nat.is_lt(x, J), Nat.is_lt(Nat.mod(x, Nat.mul(8n, G)), Nat.add(8n, Nat.mul(8n, p))))),
        p8_add_cong(Impl.job_hits(p, x, G, J),
          Spec.bit(Bool.and(Nat.is_lt(x, J), Nat.is_lt(Nat.mod(x, Nat.mul(8n, G)), Nat.mul(8n, p)))),
          Impl.warp_hits(8n, p, x, G, J),
          Spec.bit(Bool.and(Nat.is_lt(x, J), p8_I(Nat.mul(8n, p), Nat.add(8n, Nat.mul(8n, p)), Nat.mod(x, Nat.mul(8n, G))))),
          p8_jobs(p, x, G, J, ar_lt_le(p, G, pl), grid),
          p8_warp(8n, p, x, G, J, {==}, pl, grid)),
        p8_band(Nat.is_lt(x, J), Nat.is_lt(Nat.mod(x, Nat.mul(8n, G)), Nat.mul(8n, p)),
          p8_I(Nat.mul(8n, p), Nat.add(8n, Nat.mul(8n, p)), Nat.mod(x, Nat.mul(8n, G))),
          Nat.is_lt(Nat.mod(x, Nat.mul(8n, G)), Nat.add(8n, Nat.mul(8n, p))),
          p8_split0(Nat.mul(8n, p), Nat.add(8n, Nat.mul(8n, p)), Nat.mod(x, Nat.mul(8n, G)),
            ar_le_add_l(8n, Nat.mul(8n, p)))))

def Laws.finish_jobs_once(x, G, J, grid):
  +gp = grid
  Equal.trans(Nat, Impl.job_hits(G, x, G, J),
    Spec.bit(Bool.and(Nat.is_lt(x, J), Nat.is_lt(Nat.mod(x, Nat.mul(8n, G)), Nat.mul(8n, G)))),
    Spec.bit(Nat.is_lt(x, J)),
    p8_jobs(G, x, G, J, ar_le_refl(G), gp),
    p8_bit_and_true(Nat.is_lt(x, J), Nat.is_lt(Nat.mod(x, Nat.mul(8n, G)), Nat.mul(8n, G)),
      p8_snd({Nat.add(Nat.mul(Nat.div(x, Nat.mul(8n, G)), Nat.mul(8n, G)), Nat.mod(x, Nat.mul(8n, G))) == x : Nat},
        {Nat.is_lt(Nat.mod(x, Nat.mul(8n, G)), Nat.mul(8n, G)) == True{} : Bool},
        ar_divmod(x, Nat.mul(8n, G), p8_pos8(G, gp)))))

# ==== Laws finish_job_decode, finish_job_encode ====

def p9_rt(-a: Bool, -b: Bool, e: {a == b : Bool}, h: {a == True{} : Bool})
  -> {b == True{} : Bool}:
  Equal.trans(Bool, b, a, True{}, Equal.sym(Bool, a, b, e), h)

def p9_tr(-a: Bool, -b: Bool, e: {a == b : Bool}, h: {b == True{} : Bool})
  -> {a == True{} : Bool}:
  Equal.trans(Bool, a, b, True{}, e, h)

def p9_fst(-A: Type, -B: Type, p: A & B) -> A:
  match p:
    case (a, b):
      a

def p9_snd(-A: Type, -B: Type, p: A & B) -> B:
  match p:
    case (a, b):
      b

# ---- Impl.job, staged: each stage abstracts one computed value for rewriting ----

def p9_jobc(+q: Nat, +r: Nat, +rem: Nat) -> Spec.Job:
  Spec.Job{q, r, Nat.sub(rem, Nat.mul(r, 4n))}

def p9_jobr(+q: Nat, +rem: Nat) -> Spec.Job:
  p9_jobc(q, Nat.div(rem, 4n), rem)

def p9_jobq(+q: Nat, +x: Nat, +m: Nat) -> Spec.Job:
  p9_jobr(q, Nat.sub(x, Nat.mul(Nat.mul(q, m), 4n)))

# ---- division facts ----

# a - (a / b) * b == a % b.
def p9_sub_mod(+a: Nat, +b: Nat, pos: {Nat.is_lt(0n, b) == True{} : Bool})
  -> {Nat.sub(a, Nat.mul(Nat.div(a, b), b)) == Nat.mod(a, b) : Nat}:
  %p9_fst({Nat.add(Nat.mul(Nat.div(a, b), b), Nat.mod(a, b)) == a : Nat},
    {Nat.is_lt(Nat.mod(a, b), b) == True{} : Bool}, ar_divmod(a, b, pos))
    : {Nat.sub(_, Nat.mul(Nat.div(a, b), b)) == Nat.mod(a, b) : Nat}
  ar_add_sub(Nat.mul(Nat.div(a, b), b), Nat.mod(a, b))

# a - (a / b) * b < b.
def p9_sub_mod_lt(+a: Nat, +b: Nat, +pos: {Nat.is_lt(0n, b) == True{} : Bool})
  -> {Nat.is_lt(Nat.sub(a, Nat.mul(Nat.div(a, b), b)), b) == True{} : Bool}:
  %Equal.sym(Nat, Nat.sub(a, Nat.mul(Nat.div(a, b), b)), Nat.mod(a, b), p9_sub_mod(a, b, pos))
    : {Nat.is_lt(_, b) == True{} : Bool}
  p9_snd({Nat.add(Nat.mul(Nat.div(a, b), b), Nat.mod(a, b)) == a : Nat},
    {Nat.is_lt(Nat.mod(a, b), b) == True{} : Bool}, ar_divmod(a, b, pos))

# (a / b) * b <= a.
def p9_div_mul_le(+a: Nat, +b: Nat, pos: {Nat.is_lt(0n, b) == True{} : Bool})
  -> {Nat.is_le(Nat.mul(Nat.div(a, b), b), a) == True{} : Bool}:
  p9_tr(Nat.is_le(Nat.mul(Nat.div(a, b), b), a), Nat.is_le(Nat.div(a, b), Nat.div(a, b)),
    ar_div_le(Nat.div(a, b), a, b, pos), ar_le_refl(Nat.div(a, b)))

# ---- decode ----

def p9_lt_zero_false(+groups: Nat, +x: Nat)
  -> {False{} == Nat.is_lt(x, Nat.mul(Nat.mul(groups, 0n), 4n)) : Bool}:
  %Equal.sym(Nat, Nat.mul(groups, 0n), 0n, ar_mul_zero_r(groups))
    : {False{} == Nat.is_lt(x, Nat.mul(_, 4n)) : Bool}
  Equal.sym(Bool, Nat.is_lt(x, 0n), False{}, ar_h_lt_zero(x))

# A job index exists only if m * 4 > 0.
def p9_dec_pos(+groups: Nat, m: Nat, +x: Nat,
    inside: {Nat.is_lt(x, Nat.mul(Nat.mul(groups, m), 4n)) == True{} : Bool})
  -> {Nat.is_lt(0n, Nat.mul(m, 4n)) == True{} : Bool}:
  match m:
    case 0n:
      Empty.absurd({Nat.is_lt(0n, 0n) == True{} : Bool},
        ar_false_true(Equal.trans(Bool, False{},
          Nat.is_lt(x, Nat.mul(Nat.mul(groups, 0n), 4n)), True{},
          p9_lt_zero_false(groups, x), inside)))
    case 1n+p:
      {==}

# q * (m * 4) == (q * m) * 4.
def p9_assoc(+q: Nat, +m: Nat)
  -> {Nat.mul(q, Nat.mul(m, 4n)) == Nat.mul(Nat.mul(q, m), 4n) : Nat}:
  Equal.sym(Nat, Nat.mul(Nat.mul(q, m), 4n), Nat.mul(q, Nat.mul(m, 4n)),
    ar_mul_assoc(q, m, 4n))

# (q * m) * 4 <= x for q = x / (m * 4).
def p9_dec_le(+x: Nat, +m: Nat, pos: {Nat.is_lt(0n, Nat.mul(m, 4n)) == True{} : Bool})
  -> {Nat.is_le(Nat.mul(Nat.mul(Nat.div(x, Nat.mul(m, 4n)), m), 4n), x) == True{} : Bool}:
  %p9_assoc(Nat.div(x, Nat.mul(m, 4n)), m) : {Nat.is_le(_, x) == True{} : Bool}
  p9_div_mul_le(x, Nat.mul(m, 4n), pos)

# rem = x - (q * m) * 4 < m * 4 for q = x / (m * 4).
def p9_dec_rem_lt(+x: Nat, +m: Nat, +pos: {Nat.is_lt(0n, Nat.mul(m, 4n)) == True{} : Bool})
  -> {Nat.is_lt(Nat.sub(x, Nat.mul(Nat.mul(Nat.div(x, Nat.mul(m, 4n)), m), 4n)),
    Nat.mul(m, 4n)) == True{} : Bool}:
  %p9_assoc(Nat.div(x, Nat.mul(m, 4n)), m)
    : {Nat.is_lt(Nat.sub(x, _), Nat.mul(m, 4n)) == True{} : Bool}
  p9_sub_mod_lt(x, Nat.mul(m, 4n), pos)

def p9_inside(+groups: Nat, +m: Nat, +x: Nat,
    inside: {Nat.is_lt(x, Nat.mul(Nat.mul(groups, m), 4n)) == True{} : Bool})
  -> {Nat.is_lt(x, Nat.mul(groups, Nat.mul(m, 4n))) == True{} : Bool}:
  %ar_mul_assoc(groups, m, 4n) : {Nat.is_lt(x, _) == True{} : Bool}
  inside

# q < groups for q = x / (m * 4), x < (groups * m) * 4.
def p9_dec_g_lt(+groups: Nat, +m: Nat, +x: Nat,
    pos: {Nat.is_lt(0n, Nat.mul(m, 4n)) == True{} : Bool},
    inside: {Nat.is_lt(x, Nat.mul(Nat.mul(groups, m), 4n)) == True{} : Bool})
  -> {Nat.is_lt(Nat.div(x, Nat.mul(m, 4n)), groups) == True{} : Bool}:
  p9_tr(Nat.is_lt(Nat.div(x, Nat.mul(m, 4n)), groups),
    Nat.is_lt(x, Nat.mul(groups, Nat.mul(m, 4n))),
    ar_div_lt(groups, x, Nat.mul(m, 4n), pos),
    p9_inside(groups, m, x, inside))

# job_ok of the staged decode (q, rem / 4, rem - (rem / 4) * 4) for any rem < m * 4.
def p9_ok_rem(+groups: Nat, +m: Nat, +q: Nat, +rem: Nat,
    qlt: {Nat.is_lt(q, groups) == True{} : Bool},
    rlt: {Nat.is_lt(rem, Nat.mul(m, 4n)) == True{} : Bool})
  -> {Spec.job_ok(p9_jobr(q, rem), groups, m) == True{} : Bool}:
  +r = Nat.div(rem, 4n)
  +c = Nat.sub(rem, Nat.mul(r, 4n))
  ar_and_intro(Nat.is_lt(q, groups), Bool.and(Nat.is_lt(r, m), Nat.is_lt(c, 4n)), qlt,
    ar_and_intro(Nat.is_lt(r, m), Nat.is_lt(c, 4n),
      p9_tr(Nat.is_lt(r, m), Nat.is_lt(rem, Nat.mul(m, 4n)),
        ar_div_lt(m, rem, 4n, {==}), rlt),
      p9_sub_mod_lt(rem, 4n, {==})))

# encode of the staged decode is (q * m) * 4 + rem.
def p9_enc_rem(+q: Nat, +m: Nat, +rem: Nat)
  -> {Spec.encode(p9_jobr(q, rem), m) == Nat.add(Nat.mul(Nat.mul(q, m), 4n), rem) : Nat}:
  +r = Nat.div(rem, 4n)
  +r4 = Nat.mul(r, 4n)
  +c = Nat.sub(rem, r4)
  +A = Nat.mul(Nat.mul(q, m), 4n)
  %Equal.sym(Nat, Nat.mul(Nat.add(Nat.mul(q, m), r), 4n), Nat.add(A, r4),
    ar_mul_add_l(Nat.mul(q, m), r, 4n)) : {Nat.add(_, c) == Nat.add(A, rem) : Nat}
  %Equal.sym(Nat, Nat.add(Nat.add(A, r4), c), Nat.add(A, Nat.add(r4, c)),
    ar_add_assoc(A, r4, c)) : {_ == Nat.add(A, rem) : Nat}
  %Equal.sym(Nat, Nat.add(r4, c), rem, ar_sub_add(rem, r4, p9_div_mul_le(rem, 4n, {==})))
    : {Nat.add(A, _) == Nat.add(A, rem) : Nat}
  {==}

def Laws.finish_job_decode(groups, m, x, inside):
  +ins = inside
  +pos = p9_dec_pos(groups, m, x, ins)
  +q = Nat.div(x, Nat.mul(m, 4n))
  +A = Nat.mul(Nat.mul(q, m), 4n)
  +rem = Nat.sub(x, A)
  (p9_ok_rem(groups, m, q, rem, p9_dec_g_lt(groups, m, x, pos, ins), p9_dec_rem_lt(x, m, pos)),
    Equal.trans(Nat, Spec.encode(p9_jobr(q, rem), m), Nat.add(A, rem), x,
      p9_enc_rem(q, m, rem), ar_sub_add(x, A, p9_dec_le(x, m, pos))))

# ---- encode ----

def p9_m4_pos(+m: Nat, pos: {Nat.is_lt(0n, m) == True{} : Bool})
  -> {Nat.is_lt(0n, Nat.mul(m, 4n)) == True{} : Bool}:
  p9_tr(Nat.is_lt(Nat.mul(0n, 4n), Nat.mul(m, 4n)), Nat.is_lt(0n, m),
    ar_mul_lt_cancel(0n, m, 4n, {==}), pos)

# r * 4 + 4 <= m * 4 for r < m.
def p9_succ_le(+r: Nat, +m: Nat, rlt: {Nat.is_lt(r, m) == True{} : Bool})
  -> {Nat.is_le(Nat.add(Nat.mul(r, 4n), 4n), Nat.mul(m, 4n)) == True{} : Bool}:
  %ar_add_comm(4n, Nat.mul(r, 4n)) : {Nat.is_le(_, Nat.mul(m, 4n)) == True{} : Bool}
  ar_le_mul_mono(1n+r, m, 4n,
    p9_rt(Nat.is_lt(r, m), Nat.is_le(1n+r, m), ar_lt_succ_le(r, m), rlt))

# r * 4 + ch < m * 4 for r < m, ch < 4.
def p9_slt(+r: Nat, +m: Nat, +ch: Nat,
    rlt: {Nat.is_lt(r, m) == True{} : Bool},
    chlt: {Nat.is_lt(ch, 4n) == True{} : Bool})
  -> {Nat.is_lt(Nat.add(Nat.mul(r, 4n), ch), Nat.mul(m, 4n)) == True{} : Bool}:
  ar_lt_le_trans(Nat.add(Nat.mul(r, 4n), ch), Nat.add(Nat.mul(r, 4n), 4n), Nat.mul(m, 4n),
    p9_tr(Nat.is_lt(Nat.add(Nat.mul(r, 4n), ch), Nat.add(Nat.mul(r, 4n), 4n)), Nat.is_lt(ch, 4n),
      ar_lt_add_cancel(Nat.mul(r, 4n), ch, 4n), chlt),
    p9_succ_le(r, m, rlt))

# encode(g, r, ch) == g * (m * 4) + (r * 4 + ch).
def p9_enc_split(+g: Nat, +m: Nat, +r: Nat, +ch: Nat)
  -> {Spec.encode(Spec.Job{g, r, ch}, m) ==
    Nat.add(Nat.mul(g, Nat.mul(m, 4n)), Nat.add(Nat.mul(r, 4n), ch)) : Nat}:
  +A = Nat.mul(Nat.mul(g, m), 4n)
  +r4 = Nat.mul(r, 4n)
  +R = Nat.add(Nat.mul(g, Nat.mul(m, 4n)), Nat.add(r4, ch))
  %Equal.sym(Nat, Nat.mul(Nat.add(Nat.mul(g, m), r), 4n), Nat.add(A, r4),
    ar_mul_add_l(Nat.mul(g, m), r, 4n)) : {Nat.add(_, ch) == R : Nat}
  %Equal.sym(Nat, Nat.add(Nat.add(A, r4), ch), Nat.add(A, Nat.add(r4, ch)),
    ar_add_assoc(A, r4, ch)) : {_ == R : Nat}
  %p9_assoc(g, m) : {Nat.add(_, Nat.add(r4, ch)) == R : Nat}
  {==}

# (g * (m * 4) + s) - (g * m) * 4 == s.
def p9_sub_E(+g: Nat, +m: Nat, +s: Nat)
  -> {Nat.sub(Nat.add(Nat.mul(g, Nat.mul(m, 4n)), s), Nat.mul(Nat.mul(g, m), 4n)) == s : Nat}:
  %p9_assoc(g, m) : {Nat.sub(Nat.add(Nat.mul(g, Nat.mul(m, 4n)), s), _) == s : Nat}
  ar_add_sub(Nat.mul(g, Nat.mul(m, 4n)), s)

def p9_enc_job(+g: Nat, +m: Nat, +r: Nat, +ch: Nat,
    slt: {Nat.is_lt(Nat.add(Nat.mul(r, 4n), ch), Nat.mul(m, 4n)) == True{} : Bool},
    chlt: {Nat.is_lt(ch, 4n) == True{} : Bool})
  -> {Impl.job(Spec.encode(Spec.Job{g, r, ch}, m), m) == Spec.Job{g, r, ch} : Spec.Job}:
  +M = Nat.mul(m, 4n)
  +r4 = Nat.mul(r, 4n)
  +s = Nat.add(r4, ch)
  +E = Nat.add(Nat.mul(g, M), s)
  %Equal.sym(Nat, Spec.encode(Spec.Job{g, r, ch}, m), E, p9_enc_split(g, m, r, ch))
    : {Impl.job(_, m) == Spec.Job{g, r, ch} : Spec.Job}
  %Equal.sym(Nat, Nat.div(E, M), g,
    p9_fst({Nat.div(E, M) == g : Nat}, {Nat.mod(E, M) == s : Nat},
      ar_div_unique(g, s, M, slt)))
    : {p9_jobq(_, E, m) == Spec.Job{g, r, ch} : Spec.Job}
  %Equal.sym(Nat, Nat.sub(E, Nat.mul(Nat.mul(g, m), 4n)), s, p9_sub_E(g, m, s))
    : {p9_jobr(g, _) == Spec.Job{g, r, ch} : Spec.Job}
  %Equal.sym(Nat, Nat.div(s, 4n), r,
    p9_fst({Nat.div(s, 4n) == r : Nat}, {Nat.mod(s, 4n) == ch : Nat},
      ar_div_unique(r, ch, 4n, chlt)))
    : {p9_jobc(g, _, s) == Spec.Job{g, r, ch} : Spec.Job}
  %Equal.sym(Nat, Nat.sub(s, r4), ch, ar_add_sub(r4, ch))
    : {Spec.Job{g, r, _} == Spec.Job{g, r, ch} : Spec.Job}
  {==}

def p9_div_lt_g(+groups: Nat, +E: Nat, +M: Nat, +g: Nat, dq: {Nat.div(E, M) == g : Nat},
    glt: {Nat.is_lt(g, groups) == True{} : Bool})
  -> {Nat.is_lt(Nat.div(E, M), groups) == True{} : Bool}:
  %Equal.sym(Nat, Nat.div(E, M), g, dq) : {Nat.is_lt(_, groups) == True{} : Bool}
  glt

def p9_enc_lt(+groups: Nat, +m: Nat, +g: Nat, +r: Nat, +ch: Nat,
    glt: {Nat.is_lt(g, groups) == True{} : Bool},
    pos: {Nat.is_lt(0n, Nat.mul(m, 4n)) == True{} : Bool},
    slt: {Nat.is_lt(Nat.add(Nat.mul(r, 4n), ch), Nat.mul(m, 4n)) == True{} : Bool})
  -> {Nat.is_lt(Spec.encode(Spec.Job{g, r, ch}, m), Impl.jobs(groups, m)) == True{} : Bool}:
  +M = Nat.mul(m, 4n)
  +s = Nat.add(Nat.mul(r, 4n), ch)
  +E = Nat.add(Nat.mul(g, M), s)
  %Equal.sym(Nat, Spec.encode(Spec.Job{g, r, ch}, m), E, p9_enc_split(g, m, r, ch))
    : {Nat.is_lt(_, Nat.mul(Nat.mul(groups, m), 4n)) == True{} : Bool}
  %p9_assoc(groups, m) : {Nat.is_lt(E, _) == True{} : Bool}
  p9_rt(Nat.is_lt(Nat.div(E, M), groups), Nat.is_lt(E, Nat.mul(groups, M)),
    ar_div_lt(groups, E, M, pos),
    p9_div_lt_g(groups, E, M, g,
      p9_fst({Nat.div(E, M) == g : Nat}, {Nat.mod(E, M) == s : Nat},
        ar_div_unique(g, s, M, slt)),
      glt))

def p9_encode_parts(+groups: Nat, +m: Nat, +g: Nat, +r: Nat, +ch: Nat,
    +ok: {Spec.job_ok(Spec.Job{g, r, ch}, groups, m) == True{} : Bool})
  -> {Impl.job(Spec.encode(Spec.Job{g, r, ch}, m), m) == Spec.Job{g, r, ch} : Spec.Job} &
    {Nat.is_lt(Spec.encode(Spec.Job{g, r, ch}, m), Impl.jobs(groups, m)) == True{} : Bool}:
  +rest = ar_and_r(Nat.is_lt(g, groups), Bool.and(Nat.is_lt(r, m), Nat.is_lt(ch, 4n)), ok)
  +rlt = ar_and_l(Nat.is_lt(r, m), Nat.is_lt(ch, 4n), rest)
  +chlt = ar_and_r(Nat.is_lt(r, m), Nat.is_lt(ch, 4n), rest)
  +slt = p9_slt(r, m, ch, rlt, chlt)
  (p9_enc_job(g, m, r, ch, slt, chlt),
    p9_enc_lt(groups, m, g, r, ch,
      ar_and_l(Nat.is_lt(g, groups), Bool.and(Nat.is_lt(r, m), Nat.is_lt(ch, 4n)), ok),
      p9_m4_pos(m, ar_le_lt_trans(0n, r, m, ar_h_le_zero(r), rlt)),
      slt))

def p9_encode(+groups: Nat, +m: Nat, j: Spec.Job,
    ok: {Spec.job_ok(j, groups, m) == True{} : Bool})
  -> {Impl.job(Spec.encode(j, m), m) == j : Spec.Job} &
    {Nat.is_lt(Spec.encode(j, m), Impl.jobs(groups, m)) == True{} : Bool}:
  match j:
    case Spec.Job{g, r, ch}:
      p9_encode_parts(groups, m, g, r, ch, ok)

def Laws.finish_job_encode(groups, m, j, ok):
  p9_encode(groups, m, j, ok)
