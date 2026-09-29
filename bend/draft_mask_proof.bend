# Proofs of the DFlash2 draft attention mask laws (draft_mask_laws.bend) about the Bend model
# draft_mask.bend. Structural evidence only: no unsafe definitions, axioms, holes, foreign proof
# code or answer tables. The Nat lemmas below are local copies (renamed dm_*) of the ones in
# gemm_m16_partition_proof.bend, so this module stands alone.
import Base
import ./draft_mask.bend as Impl
import ./draft_mask_spec.bend as Spec
import ./draft_mask_laws.bend as Laws

# ==== Nat / Bool lemmas ====

def dm_Truth(value: Bool) -> Type:
  match value:
    case False{}:
      Empty
    case True{}:
      Unit

def dm_false_true(equal: {False{} == True{} : Bool}) -> Empty:
  %Equal.sym(Bool, False{}, True{}, equal) : dm_Truth(_)
  Unit{}

def dm_add_zero_r(+a: Nat) -> {Nat.add(a, 0n) == a : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      %dm_add_zero_r(p) : {1n+Nat.add(p, 0n) == 1n+_ : Nat}
      {==}

def dm_add_succ_r(+a: Nat, +b: Nat) -> {Nat.add(a, 1n+b) == 1n+Nat.add(a, b) : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      %dm_add_succ_r(p, b) : {1n+Nat.add(p, 1n+b) == 1n+_ : Nat}
      {==}

def dm_add_comm(+a: Nat, +b: Nat) -> {Nat.add(a, b) == Nat.add(b, a) : Nat}:
  match a:
    case 0n:
      Equal.sym(Nat, Nat.add(b, 0n), b, dm_add_zero_r(b))
    case 1n+p:
      %Equal.sym(Nat, Nat.add(b, 1n+p), 1n+Nat.add(b, p), dm_add_succ_r(b, p)) : {1n+Nat.add(p, b) == _ : Nat}
      %dm_add_comm(p, b) : {1n+Nat.add(p, b) == 1n+_ : Nat}
      {==}

def dm_add_assoc(+a: Nat, +b: Nat, +c: Nat)
    -> {Nat.add(Nat.add(a, b), c) == Nat.add(a, Nat.add(b, c)) : Nat}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      %dm_add_assoc(p, b, c) : {1n+Nat.add(Nat.add(p, b), c) == 1n+_ : Nat}
      {==}

def dm_h_le_zero(+n: Nat) -> {Nat.is_le(0n, n) == True{} : Bool}:
  match n:
    case 0n:
      {==}
    case 1n+p:
      {==}

def dm_le_add_r(+a: Nat, +b: Nat) -> {Nat.is_le(a, Nat.add(a, b)) == True{} : Bool}:
  match a:
    case 0n:
      dm_h_le_zero(b)
    case 1n+p:
      dm_le_add_r(p, b)

def dm_le_trans(+a: Nat, +b: Nat, +c: Nat, ab: {Nat.is_le(a, b) == True{} : Bool},
    bc: {Nat.is_le(b, c) == True{} : Bool}) -> {Nat.is_le(a, c) == True{} : Bool}:
  match a b c:
    case 0n _ _:
      dm_h_le_zero(c)
    case 1n+p 0n _:
      Empty.absurd({Nat.is_le(1n+p, c) == True{} : Bool}, dm_false_true(ab))
    case 1n+p 1n+q 0n:
      Empty.absurd({Nat.is_le(1n+p, 0n) == True{} : Bool}, dm_false_true(bc))
    case 1n+p 1n+q 1n+r:
      dm_le_trans(p, q, r, ab, bc)

# p + 1 < 1 iff p + 1 <= 0 (both false).
def dm_h_lt_one(+p: Nat) -> {Nat.is_lt(1n+p, 1n) == Nat.is_le(1n+p, 0n) : Bool}:
  match p:
    case 0n:
      {==}
    case 1n+pp:
      {==}

# a < b + 1 iff a <= b.
def dm_lt_succ(+a: Nat, +b: Nat) -> {Nat.is_lt(a, 1n+b) == Nat.is_le(a, b) : Bool}:
  match a b:
    case 0n 0n:
      {==}
    case 0n 1n+q:
      {==}
    case 1n+p 0n:
      dm_h_lt_one(p)
    case 1n+p 1n+q:
      dm_lt_succ(p, q)

# a < b + (1 + w) iff a <= b + w.
def dm_lt_add_succ(+a: Nat, +b: Nat, +w: Nat)
    -> {Nat.is_le(a, Nat.add(b, w)) == Nat.is_lt(a, Nat.add(b, 1n+w)) : Bool}:
  %Equal.sym(Nat, Nat.add(b, 1n+w), 1n+Nat.add(b, w), dm_add_succ_r(b, w)) : {Nat.is_le(a, Nat.add(b, w)) == Nat.is_lt(a, _) : Bool}
  Equal.sym(Bool, Nat.is_lt(a, 1n+Nat.add(b, w)), Nat.is_le(a, Nat.add(b, w)), dm_lt_succ(a, Nat.add(b, w)))

# n <= q + 0 iff n <= q.
def dm_le_add_zero(+n: Nat, +q: Nat) -> {Nat.is_le(n, Nat.add(q, 0n)) == Nat.is_le(n, q) : Bool}:
  %Equal.sym(Nat, Nat.add(q, 0n), q, dm_add_zero_r(q)) : {Nat.is_le(n, _) == Nat.is_le(n, q) : Bool}
  {==}

# s <= n + lp iff s + 1 <= n + (lp + 1).
def dm_h_sub_step(+s: Nat, +n: Nat, +lp: Nat)
    -> {Nat.is_le(s, Nat.add(n, lp)) == Nat.is_le(1n+s, Nat.add(n, 1n+lp)) : Bool}:
  %Equal.sym(Nat, Nat.add(n, 1n+lp), 1n+Nat.add(n, lp), dm_add_succ_r(n, lp)) : {Nat.is_le(s, Nat.add(n, lp)) == Nat.is_le(1n+s, _) : Bool}
  {==}

# S <= n + l gives S - l <= n (truncated subtraction).
def dm_sub_le(+S: Nat, +l: Nat, +n: Nat, h: {Nat.is_le(S, Nat.add(n, l)) == True{} : Bool})
    -> {Nat.is_le(Nat.sub(S, l), n) == True{} : Bool}:
  match S l:
    case 0n 0n:
      dm_h_le_zero(n)
    case 0n 1n+lp:
      dm_h_le_zero(n)
    case 1n+s 0n:
      %dm_add_zero_r(n) : {Nat.is_le(1n+s, _) == True{} : Bool}
      h
    case 1n+s 1n+lp:
      +eq = dm_h_sub_step(s, n, lp)
      dm_sub_le(s, lp, n, Equal.trans(Bool, Nat.is_le(s, Nat.add(n, lp)),
        Nat.is_le(1n+s, Nat.add(n, 1n+lp)), True{}, eq, h))

# ==== Floor division: (a / b) * b <= a (via the Nat.divmod.go invariant) ====

# 1 + ((y + x) + n) == x + (z + (1 + n)) when z == y.
def dm_shuffle(+x: Nat, -y: Nat, +z: Nat, +n: Nat, e: {z == y : Nat})
    -> {1n+Nat.add(Nat.add(y, x), n) == Nat.add(x, Nat.add(z, 1n+n)) : Nat}:
  %e : {1n+Nat.add(Nat.add(_, x), n) == Nat.add(x, Nat.add(z, 1n+n)) : Nat}
  %Equal.sym(Nat, Nat.add(z, 1n+n), 1n+Nat.add(z, n), dm_add_succ_r(z, n)) : {1n+Nat.add(Nat.add(z, x), n) == Nat.add(x, _) : Nat}
  %Equal.sym(Nat, Nat.add(x, 1n+Nat.add(z, n)), 1n+Nat.add(x, Nat.add(z, n)), dm_add_succ_r(x, Nat.add(z, n))) : {1n+Nat.add(Nat.add(z, x), n) == _ : Nat}
  %dm_add_comm(x, z) : {1n+Nat.add(_, n) == 1n+Nat.add(x, Nat.add(z, n)) : Nat}
  %Equal.sym(Nat, Nat.add(Nat.add(x, z), n), Nat.add(x, Nat.add(z, n)), dm_add_assoc(x, z, n)) : {1n+_ == 1n+Nat.add(x, Nat.add(z, n)) : Nat}
  {==}

# Invariant of Nat.divmod.go with divisor b = 1 + bp: when r + m == bp,
# q * b + r2 == d * b + (r + n).
def dm_go_eq(+n: Nat, +m: Nat, +d: Nat, +r: Nat, +bp: Nat, inv: {Nat.add(r, m) == bp : Nat})
    -> {Nat.add(Nat.mul(Pair.fst(Nat, Nat, Nat.divmod.go(n, m, d, r)), 1n+bp), Pair.snd(Nat, Nat, Nat.divmod.go(n, m, d, r))) == Nat.add(Nat.mul(d, 1n+bp), Nat.add(r, n)) : Nat}:
  match n:
    case 0n:
      %dm_add_zero_r(r) : {Nat.add(Nat.mul(d, 1n+bp), _) == Nat.add(Nat.mul(d, 1n+bp), Nat.add(r, 0n)) : Nat}
      {==}
    case 1n+np:
      match m:
        case 0n:
          +inv2 = Equal.trans(Nat, r, Nat.add(r, 0n), bp,
            Equal.sym(Nat, Nat.add(r, 0n), r, dm_add_zero_r(r)), inv)
          %dm_shuffle(Nat.mul(d, 1n+bp), bp, r, np, inv2) : {Nat.add(Nat.mul(Pair.fst(Nat, Nat, Nat.divmod.go(np, r, 1n+d, 0n)), 1n+bp), Pair.snd(Nat, Nat, Nat.divmod.go(np, r, 1n+d, 0n))) == _ : Nat}
          dm_go_eq(np, r, 1n+d, 0n, bp, inv2)
        case 1n+mp:
          %Equal.sym(Nat, Nat.add(r, 1n+np), 1n+Nat.add(r, np), dm_add_succ_r(r, np)) : {Nat.add(Nat.mul(Pair.fst(Nat, Nat, Nat.divmod.go(np, mp, d, 1n+r)), 1n+bp), Pair.snd(Nat, Nat, Nat.divmod.go(np, mp, d, 1n+r))) == Nat.add(Nat.mul(d, 1n+bp), _) : Nat}
          dm_go_eq(np, mp, d, 1n+r, bp,
            Equal.trans(Nat, 1n+Nat.add(r, mp), Nat.add(r, 1n+mp), bp,
              Equal.sym(Nat, Nat.add(r, 1n+mp), 1n+Nat.add(r, mp), dm_add_succ_r(r, mp)), inv))

# (a / (1 + bp)) * (1 + bp) <= a.
def dm_le_mul_div(+a: Nat, +bp: Nat)
    -> {Nat.is_le(Nat.mul(Nat.div(a, 1n+bp), 1n+bp), a) == True{} : Bool}:
  %dm_go_eq(a, bp, 0n, 0n, bp, {==}) : {Nat.is_le(Nat.mul(Nat.div(a, 1n+bp), 1n+bp), _) == True{} : Bool}
  dm_le_add_r(Nat.mul(Nat.div(a, 1n+bp), 1n+bp), Nat.mod(a, 1n+bp))

# ==== start_keeps_left ====

def Laws.start_keeps_left(l, S, r, n, BN, left):
  match l:
    case Impl.NoWin{}:
      dm_h_le_zero(n)
    case Impl.At{+v}:
      match BN:
        case 0n:
          dm_h_le_zero(n)
        case 1n+bp:
          dm_le_trans(Nat.mul(Nat.div(Nat.sub(S, v), 1n+bp), 1n+bp), Nat.sub(S, v), n,
            dm_le_mul_div(Nat.sub(S, v), bp),
            dm_sub_le(S, v, n, dm_le_trans(S, Nat.add(S, r), Nat.add(n, v),
              dm_le_add_r(S, r), left)))

# ==== mask_refines ====

# Non-causal Bool core: p is implied by x.
def dm_absorb_nc(+x: Bool, +p: Bool, +a: Bool, +y: Bool,
    h: {x == True{} : Bool} -> {p == True{} : Bool})
    -> {Bool.and(p, Bool.and(a, Bool.and(True{}, Bool.and(x, y)))) ==
      Bool.and(a, Bool.and(True{}, Bool.and(x, y))) : Bool}:
  match x:
    case False{}:
      match p:
        case True{}:
          {==}
        case False{}:
          match a:
            case False{}:
              {==}
            case True{}:
              {==}
    case True{}:
      match p:
        case True{}:
          {==}
        case False{}:
          Empty.absurd({Bool.and(False{}, Bool.and(a, Bool.and(True{}, Bool.and(True{}, y)))) ==
            Bool.and(a, Bool.and(True{}, Bool.and(True{}, y))) : Bool}, dm_false_true(h({==})))

# Causal Bool core: p is implied by x, and the right test z moves in front of x.
def dm_absorb_c(+x: Bool, +p: Bool, +a: Bool, +z: Bool,
    h: {x == True{} : Bool} -> {p == True{} : Bool})
    -> {Bool.and(p, Bool.and(a, Bool.and(True{}, Bool.and(x, z)))) ==
      Bool.and(a, Bool.and(z, Bool.and(x, True{}))) : Bool}:
  match x:
    case False{}:
      match p:
        case True{}:
          match a:
            case False{}:
              {==}
            case True{}:
              match z:
                case False{}:
                  {==}
                case True{}:
                  {==}
        case False{}:
          match a:
            case False{}:
              {==}
            case True{}:
              match z:
                case False{}:
                  {==}
                case True{}:
                  {==}
    case True{}:
      match p:
        case True{}:
          match a:
            case False{}:
              {==}
            case True{}:
              match z:
                case False{}:
                  {==}
                case True{}:
                  {==}
        case False{}:
          Empty.absurd({Bool.and(False{}, Bool.and(a, Bool.and(True{}, Bool.and(True{}, z)))) ==
            Bool.and(a, Bool.and(z, Bool.and(True{}, True{}))) : Bool}, dm_false_true(h({==})))

# Non-causal shape: the window-start conjunct p is implied by the left test x, and the kernel's
# x, y equal the reference's x2, y2.
def dm_nc(+p: Bool, +a: Bool, +x: Bool, +y: Bool, -x2: Bool, -y2: Bool,
    h: {x == True{} : Bool} -> {p == True{} : Bool},
    ex: {x == x2 : Bool}, ey: {y == y2 : Bool})
    -> {Bool.and(p, Bool.and(a, Bool.and(True{}, Bool.and(x, y)))) ==
      Bool.and(a, Bool.and(True{}, Bool.and(x2, y2))) : Bool}:
  %ex : {Bool.and(p, Bool.and(a, Bool.and(True{}, Bool.and(x, y)))) == Bool.and(a, Bool.and(True{}, Bool.and(_, y2))) : Bool}
  %ey : {Bool.and(p, Bool.and(a, Bool.and(True{}, Bool.and(x, y)))) == Bool.and(a, Bool.and(True{}, Bool.and(x, _))) : Bool}
  dm_absorb_nc(x, p, a, y, h)

# Causal shape: additionally the kernel's right test z (n <= q + 0) is the reference's c
# (n <= q), which the reference applies before its window test.
def dm_c(+p: Bool, +a: Bool, +x: Bool, +z: Bool, -c: Bool, -x2: Bool,
    h: {x == True{} : Bool} -> {p == True{} : Bool},
    ex: {x == x2 : Bool}, ez: {z == c : Bool})
    -> {Bool.and(p, Bool.and(a, Bool.and(True{}, Bool.and(x, z)))) ==
      Bool.and(a, Bool.and(c, Bool.and(x2, True{}))) : Bool}:
  %ex : {Bool.and(p, Bool.and(a, Bool.and(True{}, Bool.and(x, z)))) == Bool.and(a, Bool.and(c, Bool.and(_, True{}))) : Bool}
  %ez : {Bool.and(p, Bool.and(a, Bool.and(True{}, Bool.and(x, z)))) == Bool.and(a, Bool.and(_, Bool.and(x, True{}))) : Bool}
  dm_absorb_c(x, p, a, z, h)

# Causal windows (is_causal absent or true): kernel window (w, 0).
def dm_causal(+w: Nat, +S: Nat, +L: Nat, +r: Nat, +n: Nat, +BN: Nat)
    -> {Bool.and(Nat.is_le(Impl.start(Impl.At{w}, S, BN), n), Bool.and(Nat.is_lt(n, Nat.add(S, L)),
      Bool.and(True{}, Bool.and(Nat.is_le(Nat.add(S, r), Nat.add(n, w)),
        Nat.is_le(n, Nat.add(Nat.add(S, r), 0n)))))) ==
      Spec.visible(True{}, 1n+w, Nat.add(S, L), Nat.add(S, r), n) : Bool}:
  +q = Nat.add(S, r)
  dm_c(Nat.is_le(Impl.start(Impl.At{w}, S, BN), n), Nat.is_lt(n, Nat.add(S, L)),
    Nat.is_le(q, Nat.add(n, w)), Nat.is_le(n, Nat.add(q, 0n)), Nat.is_le(n, q),
    Nat.is_lt(q, Nat.add(n, 1n+w)),
    e => Laws.start_keeps_left(Impl.At{w}, S, r, n, BN, e),
    dm_lt_add_succ(q, n, w), dm_le_add_zero(n, q))

# Non-causal windows (is_causal false): kernel window (w, w).
def dm_noncausal(+w: Nat, +S: Nat, +L: Nat, +r: Nat, +n: Nat, +BN: Nat)
    -> {Bool.and(Nat.is_le(Impl.start(Impl.At{w}, S, BN), n), Bool.and(Nat.is_lt(n, Nat.add(S, L)),
      Bool.and(True{}, Bool.and(Nat.is_le(Nat.add(S, r), Nat.add(n, w)),
        Nat.is_le(n, Nat.add(Nat.add(S, r), w)))))) ==
      Spec.visible(False{}, 1n+w, Nat.add(S, L), Nat.add(S, r), n) : Bool}:
  +q = Nat.add(S, r)
  dm_nc(Nat.is_le(Impl.start(Impl.At{w}, S, BN), n), Nat.is_lt(n, Nat.add(S, L)),
    Nat.is_le(q, Nat.add(n, w)), Nat.is_le(n, Nat.add(q, w)), Nat.is_lt(q, Nat.add(n, 1n+w)),
    Nat.is_lt(n, Nat.add(q, 1n+w)),
    e => Laws.start_keeps_left(Impl.At{w}, S, r, n, BN, e),
    dm_lt_add_succ(q, n, w), dm_lt_add_succ(n, q, w))

def Laws.mask_refines(cfg, w, S, L, r, n, BN):
  match cfg:
    case None{}:
      dm_causal(w, S, L, r, n, BN)
    case Some{c}:
      match c:
        case True{}:
          dm_causal(w, S, L, r, n, BN)
        case False{}:
          dm_noncausal(w, S, L, r, n, BN)

# ==== window_admitted, served_window, g7n_differs ====

def Laws.window_admitted(cfg, w):
  ({==}, {==})

def Laws.served_window():
  {==}

def Laws.g7n_differs():
  ({==}, {==})
