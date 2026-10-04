# Proofs of the ext 5109 laws (gdn_tree_laws.bend): finite facts over all 5040 shapes, decided by the checker's
# evaluation. No unsafe definitions, axioms or holes. all_ok over the 5040 shapes is split into the 24 groups of
# 210 shapes sharing the prefix [0, 0, q2, q3, q4] (one evaluation per def stays within the kernel's fuel), glued
# by levels / ext_all distributing over append.
import Base
import ./gdn_tree.bend as Impl
import ./gdn_tree_spec.bend as Spec
import ./gdn_tree_laws.bend as Laws

def gt_app_assoc(xs: List<&2, List<&2, Nat>>, +ys: List<&2, List<&2, Nat>>, +zs: List<&2, List<&2, Nat>>)
  -> {List.append(&2, List<&2, Nat>, List.append(&2, List<&2, Nat>, xs, ys), zs) == List.append(&2, List<&2, Nat>, xs, List.append(&2, List<&2, Nat>, ys, zs)) : List<&2, List<&2, Nat>>}:
  match xs:
    case Nil{}:
      {==}
    case +h <> t:
      Equal.cong(List<&2, List<&2, Nat>>, List<&2, List<&2, Nat>>, x => h <> x,
        List.append(&2, List<&2, Nat>, List.append(&2, List<&2, Nat>, t, ys), zs), List.append(&2, List<&2, Nat>, t, List.append(&2, List<&2, Nat>, ys, zs)),
        gt_app_assoc(t, ys, zs))

def gt_and_assoc(a: Bool, -b: Bool, -c: Bool) -> {Bool.and(a, Bool.and(b, c)) == Bool.and(Bool.and(a, b), c) : Bool}:
  match a:
    case True{}:
      {==}
    case False{}:
      {==}

def gt_all_app(xs: List<&2, List<&2, Nat>>, +ys: List<&2, List<&2, Nat>>)
  -> {Laws.all_ok(List.append(&2, List<&2, Nat>, xs, ys)) == Bool.and(Laws.all_ok(xs), Laws.all_ok(ys)) : Bool}:
  match xs:
    case Nil{}:
      {==}
    case +p <> +rest:
      %Equal.sym(Bool, Laws.all_ok(List.append(&2, List<&2, Nat>, rest, ys)), Bool.and(Laws.all_ok(rest), Laws.all_ok(ys)), gt_all_app(rest, ys)) :
        {Bool.and(Laws.shape_ok(p), _) == Bool.and(Bool.and(Laws.shape_ok(p), Laws.all_ok(rest)), Laws.all_ok(ys)) : Bool}
      gt_and_assoc(Laws.shape_ok(p), Laws.all_ok(rest), Laws.all_ok(ys))

def gt_ext_all_app(xs: List<&2, List<&2, Nat>>, +ys: List<&2, List<&2, Nat>>, +r: Nat)
  -> {Spec.ext_all(List.append(&2, List<&2, Nat>, xs, ys), r) == List.append(&2, List<&2, Nat>, Spec.ext_all(xs, r), Spec.ext_all(ys, r)) : List<&2, List<&2, Nat>>}:
  match xs:
    case Nil{}:
      {==}
    case +p <> +rest:
      %Equal.sym(List<&2, List<&2, Nat>>, Spec.ext_all(List.append(&2, List<&2, Nat>, rest, ys), r),
          List.append(&2, List<&2, Nat>, Spec.ext_all(rest, r), Spec.ext_all(ys, r)), gt_ext_all_app(rest, ys, r)) :
        {List.append(&2, List<&2, Nat>, Spec.ext_one(r, p), _) ==
          List.append(&2, List<&2, Nat>, List.append(&2, List<&2, Nat>, Spec.ext_one(r, p), Spec.ext_all(rest, r)), Spec.ext_all(ys, r)) : List<&2, List<&2, Nat>>}
      Equal.sym(List<&2, List<&2, Nat>>,
        List.append(&2, List<&2, Nat>, List.append(&2, List<&2, Nat>, Spec.ext_one(r, p), Spec.ext_all(rest, r)), Spec.ext_all(ys, r)),
        List.append(&2, List<&2, Nat>, Spec.ext_one(r, p), List.append(&2, List<&2, Nat>, Spec.ext_all(rest, r), Spec.ext_all(ys, r))),
        gt_app_assoc(Spec.ext_one(r, p), Spec.ext_all(rest, r), Spec.ext_all(ys, r)))

def gt_levels_app(n: Nat, +r: Nat, +xs: List<&2, List<&2, Nat>>, +ys: List<&2, List<&2, Nat>>)
  -> {Spec.levels(n, r, List.append(&2, List<&2, Nat>, xs, ys)) == List.append(&2, List<&2, Nat>, Spec.levels(n, r, xs), Spec.levels(n, r, ys)) : List<&2, List<&2, Nat>>}:
  match n:
    case 0n:
      {==}
    case 1n+k:
      %Equal.sym(List<&2, List<&2, Nat>>, Spec.ext_all(List.append(&2, List<&2, Nat>, xs, ys), r),
          List.append(&2, List<&2, Nat>, Spec.ext_all(xs, r), Spec.ext_all(ys, r)), gt_ext_all_app(xs, ys, r)) :
        {Spec.levels(k, 1n+r, _) == List.append(&2, List<&2, Nat>, Spec.levels(k, 1n+r, Spec.ext_all(xs, r)), Spec.levels(k, 1n+r, Spec.ext_all(ys, r))) : List<&2, List<&2, Nat>>}
      gt_levels_app(k, 1n+r, Spec.ext_all(xs, r), Spec.ext_all(ys, r))

def gt_and_tt(a: Bool, -b: Bool, ha: {a == True{} : Bool}, hb: {b == True{} : Bool}) -> {Bool.and(a, b) == True{} : Bool}:
  %Equal.sym(Bool, a, True{}, ha) : {Bool.and(_, b) == True{} : Bool}
  hb

# all_ok over the shapes grown from x <> xs is the conjunction of the two parts
def gt_cons(+n: Nat, +r: Nat, +x: List<&2, Nat>, +xs: List<&2, List<&2, Nat>>,
    h1: {Laws.all_ok(Spec.levels(n, r, [x])) == True{} : Bool}, h2: {Laws.all_ok(Spec.levels(n, r, xs)) == True{} : Bool})
  -> {Laws.all_ok(Spec.levels(n, r, x <> xs)) == True{} : Bool}:
  %Equal.sym(List<&2, List<&2, Nat>>, Spec.levels(n, r, List.append(&2, List<&2, Nat>, [x], xs)), List.append(&2, List<&2, Nat>, Spec.levels(n, r, [x]), Spec.levels(n, r, xs)),
      gt_levels_app(n, r, [x], xs)) : {Laws.all_ok(_) == True{} : Bool}
  %Equal.sym(Bool, Laws.all_ok(List.append(&2, List<&2, Nat>, Spec.levels(n, r, [x]), Spec.levels(n, r, xs))),
      Bool.and(Laws.all_ok(Spec.levels(n, r, [x])), Laws.all_ok(Spec.levels(n, r, xs))),
      gt_all_app(Spec.levels(n, r, [x]), Spec.levels(n, r, xs))) : {_ == True{} : Bool}
  gt_and_tt(Laws.all_ok(Spec.levels(n, r, [x])), Laws.all_ok(Spec.levels(n, r, xs)), h1, h2)

def gt_nil() -> {Laws.all_ok(Spec.levels(3n, 5n, Nil{})) == True{} : Bool}:
  {==}

# the 24 prefixes [0, 0, q2, q3, q4] (q2 < 2, q3 < 3, q4 < 4) in the order levels builds them
def gt_shapes_eq() -> {Spec.levels(3n, 5n, [[0n, 0n, 0n, 0n, 0n], [0n, 0n, 0n, 0n, 1n], [0n, 0n, 0n, 0n, 2n], [0n, 0n, 0n, 0n, 3n], [0n, 0n, 0n, 1n, 0n], [0n, 0n, 0n, 1n, 1n], [0n, 0n, 0n, 1n, 2n], [0n, 0n, 0n, 1n, 3n], [0n, 0n, 0n, 2n, 0n], [0n, 0n, 0n, 2n, 1n], [0n, 0n, 0n, 2n, 2n], [0n, 0n, 0n, 2n, 3n], [0n, 0n, 1n, 0n, 0n], [0n, 0n, 1n, 0n, 1n], [0n, 0n, 1n, 0n, 2n], [0n, 0n, 1n, 0n, 3n], [0n, 0n, 1n, 1n, 0n], [0n, 0n, 1n, 1n, 1n], [0n, 0n, 1n, 1n, 2n], [0n, 0n, 1n, 1n, 3n], [0n, 0n, 1n, 2n, 0n], [0n, 0n, 1n, 2n, 1n], [0n, 0n, 1n, 2n, 2n], [0n, 0n, 1n, 2n, 3n]]) == Spec.shapes() : List<&2, List<&2, Nat>>}:
  {==}

# each group of 210 shapes, by evaluation
def gt_c00() -> {Laws.all_ok(Spec.levels(3n, 5n, [[0n, 0n, 0n, 0n, 0n]])) == True{} : Bool}:
  {==}

def gt_c01() -> {Laws.all_ok(Spec.levels(3n, 5n, [[0n, 0n, 0n, 0n, 1n]])) == True{} : Bool}:
  {==}

def gt_c02() -> {Laws.all_ok(Spec.levels(3n, 5n, [[0n, 0n, 0n, 0n, 2n]])) == True{} : Bool}:
  {==}

def gt_c03() -> {Laws.all_ok(Spec.levels(3n, 5n, [[0n, 0n, 0n, 0n, 3n]])) == True{} : Bool}:
  {==}

def gt_c04() -> {Laws.all_ok(Spec.levels(3n, 5n, [[0n, 0n, 0n, 1n, 0n]])) == True{} : Bool}:
  {==}

def gt_c05() -> {Laws.all_ok(Spec.levels(3n, 5n, [[0n, 0n, 0n, 1n, 1n]])) == True{} : Bool}:
  {==}

def gt_c06() -> {Laws.all_ok(Spec.levels(3n, 5n, [[0n, 0n, 0n, 1n, 2n]])) == True{} : Bool}:
  {==}

def gt_c07() -> {Laws.all_ok(Spec.levels(3n, 5n, [[0n, 0n, 0n, 1n, 3n]])) == True{} : Bool}:
  {==}

def gt_c08() -> {Laws.all_ok(Spec.levels(3n, 5n, [[0n, 0n, 0n, 2n, 0n]])) == True{} : Bool}:
  {==}

def gt_c09() -> {Laws.all_ok(Spec.levels(3n, 5n, [[0n, 0n, 0n, 2n, 1n]])) == True{} : Bool}:
  {==}

def gt_c10() -> {Laws.all_ok(Spec.levels(3n, 5n, [[0n, 0n, 0n, 2n, 2n]])) == True{} : Bool}:
  {==}

def gt_c11() -> {Laws.all_ok(Spec.levels(3n, 5n, [[0n, 0n, 0n, 2n, 3n]])) == True{} : Bool}:
  {==}

def gt_c12() -> {Laws.all_ok(Spec.levels(3n, 5n, [[0n, 0n, 1n, 0n, 0n]])) == True{} : Bool}:
  {==}

def gt_c13() -> {Laws.all_ok(Spec.levels(3n, 5n, [[0n, 0n, 1n, 0n, 1n]])) == True{} : Bool}:
  {==}

def gt_c14() -> {Laws.all_ok(Spec.levels(3n, 5n, [[0n, 0n, 1n, 0n, 2n]])) == True{} : Bool}:
  {==}

def gt_c15() -> {Laws.all_ok(Spec.levels(3n, 5n, [[0n, 0n, 1n, 0n, 3n]])) == True{} : Bool}:
  {==}

def gt_c16() -> {Laws.all_ok(Spec.levels(3n, 5n, [[0n, 0n, 1n, 1n, 0n]])) == True{} : Bool}:
  {==}

def gt_c17() -> {Laws.all_ok(Spec.levels(3n, 5n, [[0n, 0n, 1n, 1n, 1n]])) == True{} : Bool}:
  {==}

def gt_c18() -> {Laws.all_ok(Spec.levels(3n, 5n, [[0n, 0n, 1n, 1n, 2n]])) == True{} : Bool}:
  {==}

def gt_c19() -> {Laws.all_ok(Spec.levels(3n, 5n, [[0n, 0n, 1n, 1n, 3n]])) == True{} : Bool}:
  {==}

def gt_c20() -> {Laws.all_ok(Spec.levels(3n, 5n, [[0n, 0n, 1n, 2n, 0n]])) == True{} : Bool}:
  {==}

def gt_c21() -> {Laws.all_ok(Spec.levels(3n, 5n, [[0n, 0n, 1n, 2n, 1n]])) == True{} : Bool}:
  {==}

def gt_c22() -> {Laws.all_ok(Spec.levels(3n, 5n, [[0n, 0n, 1n, 2n, 2n]])) == True{} : Bool}:
  {==}

def gt_c23() -> {Laws.all_ok(Spec.levels(3n, 5n, [[0n, 0n, 1n, 2n, 3n]])) == True{} : Bool}:
  {==}

def gt_all_shapes() -> {Laws.all_ok(Spec.shapes()) == True{} : Bool}:
  %gt_shapes_eq() : {Laws.all_ok(_) == True{} : Bool}
  gt_cons(3n, 5n, [0n, 0n, 0n, 0n, 0n], [[0n, 0n, 0n, 0n, 1n], [0n, 0n, 0n, 0n, 2n], [0n, 0n, 0n, 0n, 3n], [0n, 0n, 0n, 1n, 0n], [0n, 0n, 0n, 1n, 1n], [0n, 0n, 0n, 1n, 2n], [0n, 0n, 0n, 1n, 3n], [0n, 0n, 0n, 2n, 0n], [0n, 0n, 0n, 2n, 1n], [0n, 0n, 0n, 2n, 2n], [0n, 0n, 0n, 2n, 3n], [0n, 0n, 1n, 0n, 0n], [0n, 0n, 1n, 0n, 1n], [0n, 0n, 1n, 0n, 2n], [0n, 0n, 1n, 0n, 3n], [0n, 0n, 1n, 1n, 0n], [0n, 0n, 1n, 1n, 1n], [0n, 0n, 1n, 1n, 2n], [0n, 0n, 1n, 1n, 3n], [0n, 0n, 1n, 2n, 0n], [0n, 0n, 1n, 2n, 1n], [0n, 0n, 1n, 2n, 2n], [0n, 0n, 1n, 2n, 3n]], gt_c00(),
    gt_cons(3n, 5n, [0n, 0n, 0n, 0n, 1n], [[0n, 0n, 0n, 0n, 2n], [0n, 0n, 0n, 0n, 3n], [0n, 0n, 0n, 1n, 0n], [0n, 0n, 0n, 1n, 1n], [0n, 0n, 0n, 1n, 2n], [0n, 0n, 0n, 1n, 3n], [0n, 0n, 0n, 2n, 0n], [0n, 0n, 0n, 2n, 1n], [0n, 0n, 0n, 2n, 2n], [0n, 0n, 0n, 2n, 3n], [0n, 0n, 1n, 0n, 0n], [0n, 0n, 1n, 0n, 1n], [0n, 0n, 1n, 0n, 2n], [0n, 0n, 1n, 0n, 3n], [0n, 0n, 1n, 1n, 0n], [0n, 0n, 1n, 1n, 1n], [0n, 0n, 1n, 1n, 2n], [0n, 0n, 1n, 1n, 3n], [0n, 0n, 1n, 2n, 0n], [0n, 0n, 1n, 2n, 1n], [0n, 0n, 1n, 2n, 2n], [0n, 0n, 1n, 2n, 3n]], gt_c01(),
    gt_cons(3n, 5n, [0n, 0n, 0n, 0n, 2n], [[0n, 0n, 0n, 0n, 3n], [0n, 0n, 0n, 1n, 0n], [0n, 0n, 0n, 1n, 1n], [0n, 0n, 0n, 1n, 2n], [0n, 0n, 0n, 1n, 3n], [0n, 0n, 0n, 2n, 0n], [0n, 0n, 0n, 2n, 1n], [0n, 0n, 0n, 2n, 2n], [0n, 0n, 0n, 2n, 3n], [0n, 0n, 1n, 0n, 0n], [0n, 0n, 1n, 0n, 1n], [0n, 0n, 1n, 0n, 2n], [0n, 0n, 1n, 0n, 3n], [0n, 0n, 1n, 1n, 0n], [0n, 0n, 1n, 1n, 1n], [0n, 0n, 1n, 1n, 2n], [0n, 0n, 1n, 1n, 3n], [0n, 0n, 1n, 2n, 0n], [0n, 0n, 1n, 2n, 1n], [0n, 0n, 1n, 2n, 2n], [0n, 0n, 1n, 2n, 3n]], gt_c02(),
    gt_cons(3n, 5n, [0n, 0n, 0n, 0n, 3n], [[0n, 0n, 0n, 1n, 0n], [0n, 0n, 0n, 1n, 1n], [0n, 0n, 0n, 1n, 2n], [0n, 0n, 0n, 1n, 3n], [0n, 0n, 0n, 2n, 0n], [0n, 0n, 0n, 2n, 1n], [0n, 0n, 0n, 2n, 2n], [0n, 0n, 0n, 2n, 3n], [0n, 0n, 1n, 0n, 0n], [0n, 0n, 1n, 0n, 1n], [0n, 0n, 1n, 0n, 2n], [0n, 0n, 1n, 0n, 3n], [0n, 0n, 1n, 1n, 0n], [0n, 0n, 1n, 1n, 1n], [0n, 0n, 1n, 1n, 2n], [0n, 0n, 1n, 1n, 3n], [0n, 0n, 1n, 2n, 0n], [0n, 0n, 1n, 2n, 1n], [0n, 0n, 1n, 2n, 2n], [0n, 0n, 1n, 2n, 3n]], gt_c03(),
    gt_cons(3n, 5n, [0n, 0n, 0n, 1n, 0n], [[0n, 0n, 0n, 1n, 1n], [0n, 0n, 0n, 1n, 2n], [0n, 0n, 0n, 1n, 3n], [0n, 0n, 0n, 2n, 0n], [0n, 0n, 0n, 2n, 1n], [0n, 0n, 0n, 2n, 2n], [0n, 0n, 0n, 2n, 3n], [0n, 0n, 1n, 0n, 0n], [0n, 0n, 1n, 0n, 1n], [0n, 0n, 1n, 0n, 2n], [0n, 0n, 1n, 0n, 3n], [0n, 0n, 1n, 1n, 0n], [0n, 0n, 1n, 1n, 1n], [0n, 0n, 1n, 1n, 2n], [0n, 0n, 1n, 1n, 3n], [0n, 0n, 1n, 2n, 0n], [0n, 0n, 1n, 2n, 1n], [0n, 0n, 1n, 2n, 2n], [0n, 0n, 1n, 2n, 3n]], gt_c04(),
    gt_cons(3n, 5n, [0n, 0n, 0n, 1n, 1n], [[0n, 0n, 0n, 1n, 2n], [0n, 0n, 0n, 1n, 3n], [0n, 0n, 0n, 2n, 0n], [0n, 0n, 0n, 2n, 1n], [0n, 0n, 0n, 2n, 2n], [0n, 0n, 0n, 2n, 3n], [0n, 0n, 1n, 0n, 0n], [0n, 0n, 1n, 0n, 1n], [0n, 0n, 1n, 0n, 2n], [0n, 0n, 1n, 0n, 3n], [0n, 0n, 1n, 1n, 0n], [0n, 0n, 1n, 1n, 1n], [0n, 0n, 1n, 1n, 2n], [0n, 0n, 1n, 1n, 3n], [0n, 0n, 1n, 2n, 0n], [0n, 0n, 1n, 2n, 1n], [0n, 0n, 1n, 2n, 2n], [0n, 0n, 1n, 2n, 3n]], gt_c05(),
    gt_cons(3n, 5n, [0n, 0n, 0n, 1n, 2n], [[0n, 0n, 0n, 1n, 3n], [0n, 0n, 0n, 2n, 0n], [0n, 0n, 0n, 2n, 1n], [0n, 0n, 0n, 2n, 2n], [0n, 0n, 0n, 2n, 3n], [0n, 0n, 1n, 0n, 0n], [0n, 0n, 1n, 0n, 1n], [0n, 0n, 1n, 0n, 2n], [0n, 0n, 1n, 0n, 3n], [0n, 0n, 1n, 1n, 0n], [0n, 0n, 1n, 1n, 1n], [0n, 0n, 1n, 1n, 2n], [0n, 0n, 1n, 1n, 3n], [0n, 0n, 1n, 2n, 0n], [0n, 0n, 1n, 2n, 1n], [0n, 0n, 1n, 2n, 2n], [0n, 0n, 1n, 2n, 3n]], gt_c06(),
    gt_cons(3n, 5n, [0n, 0n, 0n, 1n, 3n], [[0n, 0n, 0n, 2n, 0n], [0n, 0n, 0n, 2n, 1n], [0n, 0n, 0n, 2n, 2n], [0n, 0n, 0n, 2n, 3n], [0n, 0n, 1n, 0n, 0n], [0n, 0n, 1n, 0n, 1n], [0n, 0n, 1n, 0n, 2n], [0n, 0n, 1n, 0n, 3n], [0n, 0n, 1n, 1n, 0n], [0n, 0n, 1n, 1n, 1n], [0n, 0n, 1n, 1n, 2n], [0n, 0n, 1n, 1n, 3n], [0n, 0n, 1n, 2n, 0n], [0n, 0n, 1n, 2n, 1n], [0n, 0n, 1n, 2n, 2n], [0n, 0n, 1n, 2n, 3n]], gt_c07(),
    gt_cons(3n, 5n, [0n, 0n, 0n, 2n, 0n], [[0n, 0n, 0n, 2n, 1n], [0n, 0n, 0n, 2n, 2n], [0n, 0n, 0n, 2n, 3n], [0n, 0n, 1n, 0n, 0n], [0n, 0n, 1n, 0n, 1n], [0n, 0n, 1n, 0n, 2n], [0n, 0n, 1n, 0n, 3n], [0n, 0n, 1n, 1n, 0n], [0n, 0n, 1n, 1n, 1n], [0n, 0n, 1n, 1n, 2n], [0n, 0n, 1n, 1n, 3n], [0n, 0n, 1n, 2n, 0n], [0n, 0n, 1n, 2n, 1n], [0n, 0n, 1n, 2n, 2n], [0n, 0n, 1n, 2n, 3n]], gt_c08(),
    gt_cons(3n, 5n, [0n, 0n, 0n, 2n, 1n], [[0n, 0n, 0n, 2n, 2n], [0n, 0n, 0n, 2n, 3n], [0n, 0n, 1n, 0n, 0n], [0n, 0n, 1n, 0n, 1n], [0n, 0n, 1n, 0n, 2n], [0n, 0n, 1n, 0n, 3n], [0n, 0n, 1n, 1n, 0n], [0n, 0n, 1n, 1n, 1n], [0n, 0n, 1n, 1n, 2n], [0n, 0n, 1n, 1n, 3n], [0n, 0n, 1n, 2n, 0n], [0n, 0n, 1n, 2n, 1n], [0n, 0n, 1n, 2n, 2n], [0n, 0n, 1n, 2n, 3n]], gt_c09(),
    gt_cons(3n, 5n, [0n, 0n, 0n, 2n, 2n], [[0n, 0n, 0n, 2n, 3n], [0n, 0n, 1n, 0n, 0n], [0n, 0n, 1n, 0n, 1n], [0n, 0n, 1n, 0n, 2n], [0n, 0n, 1n, 0n, 3n], [0n, 0n, 1n, 1n, 0n], [0n, 0n, 1n, 1n, 1n], [0n, 0n, 1n, 1n, 2n], [0n, 0n, 1n, 1n, 3n], [0n, 0n, 1n, 2n, 0n], [0n, 0n, 1n, 2n, 1n], [0n, 0n, 1n, 2n, 2n], [0n, 0n, 1n, 2n, 3n]], gt_c10(),
    gt_cons(3n, 5n, [0n, 0n, 0n, 2n, 3n], [[0n, 0n, 1n, 0n, 0n], [0n, 0n, 1n, 0n, 1n], [0n, 0n, 1n, 0n, 2n], [0n, 0n, 1n, 0n, 3n], [0n, 0n, 1n, 1n, 0n], [0n, 0n, 1n, 1n, 1n], [0n, 0n, 1n, 1n, 2n], [0n, 0n, 1n, 1n, 3n], [0n, 0n, 1n, 2n, 0n], [0n, 0n, 1n, 2n, 1n], [0n, 0n, 1n, 2n, 2n], [0n, 0n, 1n, 2n, 3n]], gt_c11(),
    gt_cons(3n, 5n, [0n, 0n, 1n, 0n, 0n], [[0n, 0n, 1n, 0n, 1n], [0n, 0n, 1n, 0n, 2n], [0n, 0n, 1n, 0n, 3n], [0n, 0n, 1n, 1n, 0n], [0n, 0n, 1n, 1n, 1n], [0n, 0n, 1n, 1n, 2n], [0n, 0n, 1n, 1n, 3n], [0n, 0n, 1n, 2n, 0n], [0n, 0n, 1n, 2n, 1n], [0n, 0n, 1n, 2n, 2n], [0n, 0n, 1n, 2n, 3n]], gt_c12(),
    gt_cons(3n, 5n, [0n, 0n, 1n, 0n, 1n], [[0n, 0n, 1n, 0n, 2n], [0n, 0n, 1n, 0n, 3n], [0n, 0n, 1n, 1n, 0n], [0n, 0n, 1n, 1n, 1n], [0n, 0n, 1n, 1n, 2n], [0n, 0n, 1n, 1n, 3n], [0n, 0n, 1n, 2n, 0n], [0n, 0n, 1n, 2n, 1n], [0n, 0n, 1n, 2n, 2n], [0n, 0n, 1n, 2n, 3n]], gt_c13(),
    gt_cons(3n, 5n, [0n, 0n, 1n, 0n, 2n], [[0n, 0n, 1n, 0n, 3n], [0n, 0n, 1n, 1n, 0n], [0n, 0n, 1n, 1n, 1n], [0n, 0n, 1n, 1n, 2n], [0n, 0n, 1n, 1n, 3n], [0n, 0n, 1n, 2n, 0n], [0n, 0n, 1n, 2n, 1n], [0n, 0n, 1n, 2n, 2n], [0n, 0n, 1n, 2n, 3n]], gt_c14(),
    gt_cons(3n, 5n, [0n, 0n, 1n, 0n, 3n], [[0n, 0n, 1n, 1n, 0n], [0n, 0n, 1n, 1n, 1n], [0n, 0n, 1n, 1n, 2n], [0n, 0n, 1n, 1n, 3n], [0n, 0n, 1n, 2n, 0n], [0n, 0n, 1n, 2n, 1n], [0n, 0n, 1n, 2n, 2n], [0n, 0n, 1n, 2n, 3n]], gt_c15(),
    gt_cons(3n, 5n, [0n, 0n, 1n, 1n, 0n], [[0n, 0n, 1n, 1n, 1n], [0n, 0n, 1n, 1n, 2n], [0n, 0n, 1n, 1n, 3n], [0n, 0n, 1n, 2n, 0n], [0n, 0n, 1n, 2n, 1n], [0n, 0n, 1n, 2n, 2n], [0n, 0n, 1n, 2n, 3n]], gt_c16(),
    gt_cons(3n, 5n, [0n, 0n, 1n, 1n, 1n], [[0n, 0n, 1n, 1n, 2n], [0n, 0n, 1n, 1n, 3n], [0n, 0n, 1n, 2n, 0n], [0n, 0n, 1n, 2n, 1n], [0n, 0n, 1n, 2n, 2n], [0n, 0n, 1n, 2n, 3n]], gt_c17(),
    gt_cons(3n, 5n, [0n, 0n, 1n, 1n, 2n], [[0n, 0n, 1n, 1n, 3n], [0n, 0n, 1n, 2n, 0n], [0n, 0n, 1n, 2n, 1n], [0n, 0n, 1n, 2n, 2n], [0n, 0n, 1n, 2n, 3n]], gt_c18(),
    gt_cons(3n, 5n, [0n, 0n, 1n, 1n, 3n], [[0n, 0n, 1n, 2n, 0n], [0n, 0n, 1n, 2n, 1n], [0n, 0n, 1n, 2n, 2n], [0n, 0n, 1n, 2n, 3n]], gt_c19(),
    gt_cons(3n, 5n, [0n, 0n, 1n, 2n, 0n], [[0n, 0n, 1n, 2n, 1n], [0n, 0n, 1n, 2n, 2n], [0n, 0n, 1n, 2n, 3n]], gt_c20(),
    gt_cons(3n, 5n, [0n, 0n, 1n, 2n, 1n], [[0n, 0n, 1n, 2n, 2n], [0n, 0n, 1n, 2n, 3n]], gt_c21(),
    gt_cons(3n, 5n, [0n, 0n, 1n, 2n, 2n], [[0n, 0n, 1n, 2n, 3n]], gt_c22(),
    gt_cons(3n, 5n, [0n, 0n, 1n, 2n, 3n], Nil{}, gt_c23(),
    gt_nil()))))))))))))))))))))))))

def Laws.tree_all_shapes():
  ({==}, gt_all_shapes())

def Laws.tree_chain():
  ({==}, {==}, {==})
