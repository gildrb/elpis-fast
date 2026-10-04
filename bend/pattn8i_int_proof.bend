# Proofs of pattn8i_int_laws.bend (ext 3021c int8 QK^T integer core). Nat lemmas from gemm_m16_partition_proof.bend (Ar)
# and attn_stride_proof.bend (SP). Big constants (2^31, 2^32 - 1, 0x4B400000) only ever appear as the second operand of
# Nat.add / Nat.mul / Nat.is_lt next to a variable, so no step unfolds them (Bend Nat literals are unary past 2^32 - 1).
import Base
import ./pattn8i_int.bend as Impl
import ./pattn8i_int_spec.bend as Spec
import ./pattn8i_int_laws.bend as Laws
import ./gemm_m16_partition_proof.bend as Ar
import ./attn_stride_proof.bend as SP

# ---------------------------------------------------------------------------------------------
# A. Nat helpers

# a == b gives a <= b
def pi_le_eq(+a: Nat, -b: Nat, e: {a == b : Nat}) -> {Nat.is_le(a, b) == True{} : Bool}:
  %e : {Nat.is_le(a, _) == True{} : Bool}
  Ar.ar_le_refl(a)

# a < b, a == a2, b == b2 give a2 < b2
def pi_lt_rw(-a: Nat, -a2: Nat, -b: Nat, -b2: Nat, ea: {a == a2 : Nat}, eb: {b == b2 : Nat},
    h: {Nat.is_lt(a, b) == True{} : Bool}) -> {Nat.is_lt(a2, b2) == True{} : Bool}:
  %ea : {Nat.is_lt(_, b2) == True{} : Bool}
  %eb : {Nat.is_lt(a, _) == True{} : Bool}
  h

def pi_le_rw(-a: Nat, -a2: Nat, -b: Nat, -b2: Nat, ea: {a == a2 : Nat}, eb: {b == b2 : Nat},
    h: {Nat.is_le(a, b) == True{} : Bool}) -> {Nat.is_le(a2, b2) == True{} : Bool}:
  %ea : {Nat.is_le(_, b2) == True{} : Bool}
  %eb : {Nat.is_le(a, _) == True{} : Bool}
  h

# c + a == c + b gives a == b
def pi_add_cancel(+c: Nat, +a: Nat, +b: Nat, +e: {Nat.add(c, a) == Nat.add(c, b) : Nat}) -> {a == b : Nat}:
  +l1 = pi_le_eq(Nat.add(c, a), Nat.add(c, b), e)
  +l2 = pi_le_eq(Nat.add(c, b), Nat.add(c, a), Equal.sym(Nat, Nat.add(c, a), Nat.add(c, b), e))
  Ar.ar_le_antisym(a, b,
    Ar.ard_rt(Nat.is_le(Nat.add(c, a), Nat.add(c, b)), Nat.is_le(a, b), Ar.ar_le_add_cancel(c, a, b), l1),
    Ar.ard_rt(Nat.is_le(Nat.add(c, b), Nat.add(c, a)), Nat.is_le(b, a), Ar.ar_le_add_cancel(c, b, a), l2))

# (k + a) - (k + b) == a - b
def pi_sub_cancel_l(k: Nat, +a: Nat, +b: Nat) -> {Nat.sub(Nat.add(k, a), Nat.add(k, b)) == Nat.sub(a, b) : Nat}:
  match k:
    case 0n:
      {==}
    case 1n+j:
      pi_sub_cancel_l(j, a, b)

# a + d == c + b gives a - b == c - d
def pi_sub_eq(+a: Nat, +b: Nat, +c: Nat, +d: Nat, h: {Nat.add(a, d) == Nat.add(c, b) : Nat})
  -> {Nat.sub(a, b) == Nat.sub(c, d) : Nat}:
  %pi_sub_cancel_l(d, a, b) : {_ == Nat.sub(c, d) : Nat}
  %Ar.ar_add_comm(a, d) : {Nat.sub(_, Nat.add(d, b)) == Nat.sub(c, d) : Nat}
  %Equal.sym(Nat, Nat.add(a, d), Nat.add(c, b), h) : {Nat.sub(_, Nat.add(d, b)) == Nat.sub(c, d) : Nat}
  %Ar.ar_add_comm(b, c) : {Nat.sub(_, Nat.add(d, b)) == Nat.sub(c, d) : Nat}
  %Ar.ar_add_comm(b, d) : {Nat.sub(Nat.add(b, c), _) == Nat.sub(c, d) : Nat}
  pi_sub_cancel_l(b, c, d)

# a - b <= a
def pi_sub_le(+a: Nat, +b: Nat) -> {Nat.is_le(Nat.sub(a, b), a) == True{} : Bool}:
  match a b:
    case 0n _:
      {==}
    case 1n+x 0n:
      Ar.ar_le_refl(1n+x)
    case 1n+x 1n+y:
      Ar.ar_le_trans(Nat.sub(x, y), x, 1n+x, pi_sub_le(x, y), Ar.ar_h_le_succ(x))

# a <= b is false gives b < a
def pi_lt_of_nle(+a: Nat, +b: Nat, e: {Nat.is_le(a, b) == False{} : Bool}) -> {Nat.is_lt(b, a) == True{} : Bool}:
  Equal.trans(Bool, Nat.is_lt(b, a), Bool.not(Nat.is_le(a, b)), True{}, Ar.ar_lt_not_le(b, a),
    (%Equal.sym(Bool, Nat.is_le(a, b), False{}, e) : {Bool.not(_) == True{} : Bool}
     {==}))

# a <= b, c <= d give a + c <= b + d
def pi_le_add2(+a: Nat, +b: Nat, +c: Nat, +d: Nat, ab: {Nat.is_le(a, b) == True{} : Bool},
    cd: {Nat.is_le(c, d) == True{} : Bool}) -> {Nat.is_le(Nat.add(a, c), Nat.add(b, d)) == True{} : Bool}:
  +s1 = Ar.ard_tr(Nat.is_le(Nat.add(a, c), Nat.add(a, d)), Nat.is_le(c, d), Ar.ar_le_add_cancel(a, c, d), cd)
  +s2 = pi_le_rw(Nat.add(d, a), Nat.add(a, d), Nat.add(d, b), Nat.add(b, d), Ar.ar_add_comm(d, a), Ar.ar_add_comm(d, b),
    Ar.ard_tr(Nat.is_le(Nat.add(d, a), Nat.add(d, b)), Nat.is_le(a, b), Ar.ar_le_add_cancel(d, a, b), ab))
  Ar.ar_le_trans(Nat.add(a, c), Nat.add(a, d), Nat.add(b, d), s1, s2)

# a <= b gives a < b + 1 style: a <= c gives a < 1 + c
def pi_lt_succ_of_le(+a: Nat, +c: Nat, h: {Nat.is_le(a, c) == True{} : Bool}) -> {Nat.is_lt(a, 1n+c) == True{} : Bool}:
  Ar.ard_rt(Nat.is_le(a, c), Nat.is_lt(a, 1n+c), Ar.p1_le_lt_succ(a, c), h)

# a < 1 + c gives a <= c
def pi_le_of_lt_succ(+a: Nat, +c: Nat, h: {Nat.is_lt(a, 1n+c) == True{} : Bool}) -> {Nat.is_le(a, c) == True{} : Bool}:
  Ar.ard_tr(Nat.is_le(a, c), Nat.is_lt(a, 1n+c), Ar.p1_le_lt_succ(a, c), h)

# ---------------------------------------------------------------------------------------------
# B. |z|, canonical pairs, value equality

# equal integers have equal |.| and equal canonical pairs
def pi_eq_flip(+xp: Nat, +xn: Nat, +yp: Nat, +yn: Nat, e: {Nat.add(xp, yn) == Nat.add(yp, xn) : Nat})
  -> {Nat.add(xn, yp) == Nat.add(yn, xp) : Nat}:
  Equal.trans(Nat, Nat.add(xn, yp), Nat.add(yp, xn), Nat.add(yn, xp), Ar.ar_add_comm(xn, yp),
    Equal.trans(Nat, Nat.add(yp, xn), Nat.add(xp, yn), Nat.add(yn, xp), Equal.sym(Nat, Nat.add(xp, yn), Nat.add(yp, xn), e),
      Ar.ar_add_comm(xp, yn)))

def pi_abs_inv(+xp: Nat, +xn: Nat, +yp: Nat, +yn: Nat, +e: {Nat.add(xp, yn) == Nat.add(yp, xn) : Nat})
  -> {Spec.abs(Spec.Z{xp, xn}) == Spec.abs(Spec.Z{yp, yn}) : Nat}:
  +s1 = pi_sub_eq(xp, xn, yp, yn, e)
  +s2 = pi_sub_eq(xn, xp, yn, yp, pi_eq_flip(xp, xn, yp, yn, e))
  %s1 : {Nat.add(Nat.sub(xp, xn), Nat.sub(xn, xp)) == Nat.add(_, Nat.sub(yn, yp)) : Nat}
  %s2 : {Nat.add(Nat.sub(xp, xn), Nat.sub(xn, xp)) == Nat.add(Nat.sub(xp, xn), _) : Nat}
  {==}

def pi_norm_inv(+xp: Nat, +xn: Nat, +yp: Nat, +yn: Nat, +e: {Nat.add(xp, yn) == Nat.add(yp, xn) : Nat})
  -> {Spec.norm(Spec.Z{xp, xn}) == Spec.norm(Spec.Z{yp, yn}) : Spec.Z}:
  +s1 = pi_sub_eq(xp, xn, yp, yn, e)
  +s2 = pi_sub_eq(xn, xp, yn, yp, pi_eq_flip(xp, xn, yp, yn, e))
  %s1 : {Spec.Z{Nat.sub(xp, xn), Nat.sub(xn, xp)} == Spec.Z{_, Nat.sub(yn, yp)} : Spec.Z}
  %s2 : {Spec.Z{Nat.sub(xp, xn), Nat.sub(xn, xp)} == Spec.Z{Nat.sub(xp, xn), _} : Spec.Z}
  {==}

# |z| <= B from p <= n + B and n <= p + B
def pi_abs_le.go(v: Bool, +p: Nat, +n: Nat, +B: Nat, e: {Nat.is_le(p, n) == v : Bool},
    h1: {Nat.is_le(p, Nat.add(n, B)) == True{} : Bool}, h2: {Nat.is_le(n, Nat.add(p, B)) == True{} : Bool})
  -> {Nat.is_le(Spec.abs(Spec.Z{p, n}), B) == True{} : Bool}:
  match v:
    case True{}:
      %Equal.sym(Nat, Nat.sub(p, n), 0n, SP.sp_sub_le0(p, n, e)) : {Nat.is_le(Nat.add(_, Nat.sub(n, p)), B) == True{} : Bool}
      Ar.ard_tr(Nat.is_le(Nat.sub(n, p), B), Nat.is_le(n, Nat.add(p, B)), Ar.ar_le_sub_iff(n, p, B), h2)
    case False{}:
      +np = Ar.ar_lt_le(n, p, pi_lt_of_nle(p, n, e))
      %Equal.sym(Nat, Nat.sub(n, p), 0n, SP.sp_sub_le0(n, p, np)) : {Nat.is_le(Nat.add(Nat.sub(p, n), _), B) == True{} : Bool}
      %Equal.sym(Nat, Nat.add(Nat.sub(p, n), 0n), Nat.sub(p, n), Ar.ar_add_zero_r(Nat.sub(p, n))) : {Nat.is_le(_, B) == True{} : Bool}
      Ar.ard_tr(Nat.is_le(Nat.sub(p, n), B), Nat.is_le(p, Nat.add(n, B)), Ar.ar_le_sub_iff(p, n, B), h1)

def pi_abs_le(+p: Nat, +n: Nat, +B: Nat, h1: {Nat.is_le(p, Nat.add(n, B)) == True{} : Bool},
    h2: {Nat.is_le(n, Nat.add(p, B)) == True{} : Bool}) -> {Nat.is_le(Spec.abs(Spec.Z{p, n}), B) == True{} : Bool}:
  pi_abs_le.go(Nat.is_le(p, n), p, n, B, {==}, h1, h2)

# |z| <= p + n
def pi_abs_le_mass(+p: Nat, +n: Nat) -> {Nat.is_le(Spec.abs(Spec.Z{p, n}), Nat.add(p, n)) == True{} : Bool}:
  pi_abs_le(p, n, Nat.add(p, n),
    Ar.ar_le_trans(p, Nat.add(p, n), Nat.add(n, Nat.add(p, n)), Ar.ar_le_add_r(p, n), Ar.ar_le_add_l(n, Nat.add(p, n))),
    Ar.ar_le_trans(n, Nat.add(p, n), Nat.add(p, Nat.add(p, n)), Ar.ar_le_add_l(p, n), Ar.ar_le_add_l(p, Nat.add(p, n))))

# |z| <= B gives p <= n + B and n <= p + B
def pi_abs_split1(+p: Nat, +n: Nat, +B: Nat, h: {Nat.is_le(Spec.abs(Spec.Z{p, n}), B) == True{} : Bool})
  -> {Nat.is_le(p, Nat.add(n, B)) == True{} : Bool}:
  Ar.ard_rt(Nat.is_le(Nat.sub(p, n), B), Nat.is_le(p, Nat.add(n, B)), Ar.ar_le_sub_iff(p, n, B),
    Ar.ar_le_trans(Nat.sub(p, n), Nat.add(Nat.sub(p, n), Nat.sub(n, p)), B, Ar.ar_le_add_r(Nat.sub(p, n), Nat.sub(n, p)), h))

def pi_abs_split2(+p: Nat, +n: Nat, +B: Nat, h: {Nat.is_le(Spec.abs(Spec.Z{p, n}), B) == True{} : Bool})
  -> {Nat.is_le(n, Nat.add(p, B)) == True{} : Bool}:
  Ar.ard_rt(Nat.is_le(Nat.sub(n, p), B), Nat.is_le(n, Nat.add(p, B)), Ar.ar_le_sub_iff(n, p, B),
    Ar.ar_le_trans(Nat.sub(n, p), Nat.add(Nat.sub(p, n), Nat.sub(n, p)), B, Ar.ar_le_add_l(Nat.sub(p, n), Nat.sub(n, p)), h))

# the canonical pair names the same integer
def pi_norm_eq.go(v: Bool, +p: Nat, +n: Nat, +e: {Nat.is_le(n, p) == v : Bool})
  -> Spec.zeq(Spec.norm(Spec.Z{p, n}), Spec.Z{p, n}):
  match v:
    case True{}:
      %Equal.sym(Nat, Nat.sub(n, p), 0n, SP.sp_sub_le0(n, p, e)) : {Nat.add(Nat.sub(p, n), n) == Nat.add(p, _) : Nat}
      %Ar.ar_add_comm(n, Nat.sub(p, n)) : {_ == Nat.add(p, 0n) : Nat}
      %Equal.sym(Nat, Nat.add(n, Nat.sub(p, n)), p, Ar.ar_sub_add(p, n, e)) : {_ == Nat.add(p, 0n) : Nat}
      Equal.sym(Nat, Nat.add(p, 0n), p, Ar.ar_add_zero_r(p))
    case False{}:
      +pn = Ar.ar_lt_le(p, n, pi_lt_of_nle(n, p, e))
      %Equal.sym(Nat, Nat.sub(p, n), 0n, SP.sp_sub_le0(p, n, pn)) : {Nat.add(_, n) == Nat.add(p, Nat.sub(n, p)) : Nat}
      Equal.sym(Nat, Nat.add(p, Nat.sub(n, p)), n, Ar.ar_sub_add(n, p, pn))

def pi_norm_eq(+p: Nat, +n: Nat) -> Spec.zeq(Spec.norm(Spec.Z{p, n}), Spec.Z{p, n}):
  pi_norm_eq.go(Nat.is_le(n, p), p, n, {==})

# ---------------------------------------------------------------------------------------------
# C. Q clamp (laws clamp_bound, clamp_keeps)

def pi_clamp_hi(v: Bool, +xp: Nat, +xn: Nat,
    e: {Nat.is_lt(xn, Nat.add(xp, 127n)) == v : Bool},
    lt: {Nat.is_lt(Nat.add(xp, 0n), Nat.add(127n, xn)) == True{} : Bool})
  -> {Nat.is_le(Spec.mass(Spec.norm(Impl.pickz(v, Spec.Z{xp, xn}, Spec.Z{0n, 127n}))), 127n) == True{} : Bool}:
  match v:
    case True{}:
      +h1 = Ar.ar_lt_le(xp, Nat.add(xn, 127n),
        pi_lt_rw(Nat.add(xp, 0n), xp, Nat.add(127n, xn), Nat.add(xn, 127n), Ar.ar_add_zero_r(xp), Ar.ar_add_comm(127n, xn), lt))
      pi_abs_le(xp, xn, 127n, h1, Ar.ar_lt_le(xn, Nat.add(xp, 127n), e))
    case False{}:
      {==}

def pi_clamp_lo(v: Bool, +xp: Nat, +xn: Nat, e: {Nat.is_lt(Nat.add(xp, 0n), Nat.add(127n, xn)) == v : Bool})
  -> {Nat.is_le(Spec.mass(Spec.norm(Impl.imax(Spec.Z{0n, 127n}, Impl.pickz(v, Spec.Z{xp, xn}, Spec.Z{127n, 0n})))), 127n)
    == True{} : Bool}:
  match v:
    case True{}:
      pi_clamp_hi(Nat.is_lt(xn, Nat.add(xp, 127n)), xp, xn, {==}, e)
    case False{}:
      {==}

def Laws.clamp_bound(xp, xn):
  pi_clamp_lo(Nat.is_lt(Nat.add(xp, 0n), Nat.add(127n, xn)), xp, xn, {==})

def pi_keep_hi(v: Bool, +xp: Nat, +xn: Nat, e: {Nat.is_lt(xn, Nat.add(xp, 127n)) == v : Bool},
    small: {Nat.is_le(Spec.abs(Spec.Z{xp, xn}), 127n) == True{} : Bool})
  -> Spec.zeq(Spec.norm(Impl.pickz(v, Spec.Z{xp, xn}, Spec.Z{0n, 127n})), Spec.Z{xp, xn}):
  match v:
    case True{}:
      pi_norm_eq(xp, xn)
    case False{}:
      Ar.ar_le_antisym(xn, Nat.add(xp, 127n), pi_abs_split2(xp, xn, 127n, small), Ar.p6_nlt_le(xn, Nat.add(xp, 127n), e))

def pi_keep_lo(v: Bool, +xp: Nat, +xn: Nat, e: {Nat.is_lt(Nat.add(xp, 0n), Nat.add(127n, xn)) == v : Bool},
    small: {Nat.is_le(Spec.abs(Spec.Z{xp, xn}), 127n) == True{} : Bool})
  -> Spec.zeq(Spec.norm(Impl.imax(Spec.Z{0n, 127n}, Impl.pickz(v, Spec.Z{xp, xn}, Spec.Z{127n, 0n}))), Spec.Z{xp, xn}):
  match v:
    case True{}:
      pi_keep_hi(Nat.is_lt(xn, Nat.add(xp, 127n)), xp, xn, {==}, small)
    case False{}:
      +le2 = pi_le_rw(xp, Nat.add(xp, 0n), Nat.add(xn, 127n), Nat.add(127n, xn),
        Equal.sym(Nat, Nat.add(xp, 0n), xp, Ar.ar_add_zero_r(xp)), Ar.ar_add_comm(xn, 127n), pi_abs_split1(xp, xn, 127n, small))
      Ar.ar_le_antisym(Nat.add(127n, xn), Nat.add(xp, 0n), Ar.p6_nlt_le(Nat.add(xp, 0n), Nat.add(127n, xn), e), le2)

def Laws.clamp_keeps(xp, xn, small):
  pi_keep_lo(Nat.is_lt(Nat.add(xp, 0n), Nat.add(127n, xn)), xp, xn, {==}, small)

# ---------------------------------------------------------------------------------------------
# D. K code (laws code_field, code_bijection)

def pi_field_lt8(+hi: Nat, +lo: Nat, +i: Nat) -> {Nat.is_lt(Impl.field(hi, lo, i), 8n) == True{} : Bool}:
  +m4 = Nat.mod(Nat.div(hi, Nat.pow(2n, Nat.mul(2n, i))), 4n)
  +m2 = Nat.mod(Nat.div(lo, Nat.pow(2n, i)), 2n)
  +l4 = pi_le_of_lt_succ(m4, 3n, Ar.ard_dm_lt(Nat.div(hi, Nat.pow(2n, Nat.mul(2n, i))), 4n, {==}))
  +l2 = pi_le_of_lt_succ(m2, 1n, Ar.ard_dm_lt(Nat.div(lo, Nat.pow(2n, i)), 2n, {==}))
  pi_lt_succ_of_le(Nat.add(Nat.mul(m4, 2n), m2), 7n,
    pi_le_add2(Nat.mul(m4, 2n), 6n, m2, 1n, Ar.ar_le_mul_mono(m4, 3n, 2n, l4), l2))

def pi_lt0_absurd(-T: Type, +r: Nat, h: {Nat.is_lt(r, 0n) == True{} : Bool}) -> T:
  Empty.absurd(T, Ar.ar_true_false(Equal.trans(Bool, True{}, Nat.is_lt(r, 0n), False{},
    Equal.sym(Bool, Nat.is_lt(r, 0n), True{}, h), Ar.p3_lt_zero(r))))

def pi_code_mem(c: Nat, h: {Nat.is_lt(c, 8n) == True{} : Bool})
  -> {Spec.member(Impl.code(c), Spec.odd_codes()) == True{} : Bool}:
  match c:
    case 0n:
      {==}
    case 1n+c1:
      match c1:
        case 0n:
          {==}
        case 1n+c2:
          match c2:
            case 0n:
              {==}
            case 1n+c3:
              match c3:
                case 0n:
                  {==}
                case 1n+c4:
                  match c4:
                    case 0n:
                      {==}
                    case 1n+c5:
                      match c5:
                        case 0n:
                          {==}
                        case 1n+c6:
                          match c6:
                            case 0n:
                              {==}
                            case 1n+c7:
                              match c7:
                                case 0n:
                                  {==}
                                case 1n+r:
                                  pi_lt0_absurd({Spec.member(Impl.code(8n+r), Spec.odd_codes()) == True{} : Bool}, r, h)

def Laws.code_field(hi, lo, i):
  (pi_field_lt8(hi, lo, i), pi_code_mem(Impl.field(hi, lo, i), pi_field_lt8(hi, lo, i)))

def Laws.code_bijection():
  ({==}, {==})

# ---------------------------------------------------------------------------------------------
# E. K requantisation (laws kq_bound, kq_max_group_exact, kq_bound_reached)

def pi_add_zero_in_r(+x: Nat, +y: Nat) -> {Nat.add(x, Nat.add(y, 0n)) == Nat.add(x, y) : Nat}:
  Equal.cong(Nat, Nat, z => Nat.add(x, z), Nat.add(y, 0n), y, Ar.ar_add_zero_r(y))

# add(add(x, b), y) == add(add(y, b), x)
def pi_swap_outer(+x: Nat, +b: Nat, +y: Nat) -> {Nat.add(Nat.add(x, b), y) == Nat.add(Nat.add(y, b), x) : Nat}:
  Equal.trans(Nat, Nat.add(Nat.add(x, b), y), Nat.add(y, Nat.add(x, b)), Nat.add(Nat.add(y, b), x),
    Ar.ar_add_comm(Nat.add(x, b), y),
    Equal.trans(Nat, Nat.add(y, Nat.add(x, b)), Nat.add(y, Nat.add(b, x)), Nat.add(Nat.add(y, b), x),
      Equal.cong(Nat, Nat, z => Nat.add(y, z), Nat.add(x, b), Nat.add(b, x), Ar.ar_add_comm(x, b)),
      Equal.sym(Nat, Nat.add(Nat.add(y, b), x), Nat.add(y, Nat.add(b, x)), Ar.ar_add_assoc(y, b, x))))

# kp b + cn a < (cp a + b) + kn b with cp <= cn + 7 and a <= 18 b gives kp <= kn + 126
def pi_kq_core(+kp: Nat, +kn: Nat, +cp: Nat, +cn: Nat, +a: Nat, +b: Nat,
    bpos: {Nat.is_lt(0n, b) == True{} : Bool},
    c7: {Nat.is_le(cp, Nat.add(cn, 7n)) == True{} : Bool},
    f18: {Nat.is_le(a, Nat.mul(b, 18n)) == True{} : Bool},
    h: {Nat.is_lt(Nat.add(Nat.mul(kp, b), Nat.mul(cn, a)), Nat.add(Nat.add(Nat.mul(cp, a), b), Nat.mul(kn, b))) == True{} : Bool})
  -> {Nat.is_le(kp, Nat.add(kn, 126n)) == True{} : Bool}:
  +X = Nat.mul(kp, b)
  +Y = Nat.mul(cn, a)
  +P = Nat.mul(cp, a)
  +K = Nat.mul(kn, b)
  +M = Nat.mul(b, 126n)
  +m1 = pi_le_rw(P, P, Nat.mul(Nat.add(cn, 7n), a), Nat.add(Y, Nat.mul(7n, a)), {==}, Ar.ar_mul_add_l(cn, 7n, a),
    Ar.ar_le_mul_mono(cp, Nat.add(cn, 7n), a, c7))
  +m2 = pi_le_rw(Nat.mul(a, 7n), Nat.mul(7n, a), Nat.mul(Nat.mul(b, 18n), 7n), M, Ar.ar_mul_comm(a, 7n), Ar.ar_mul_assoc(b, 18n, 7n),
    Ar.ar_le_mul_mono(a, Nat.mul(b, 18n), 7n, f18))
  +m3 = Ar.ard_tr(Nat.is_le(Nat.add(Y, Nat.mul(7n, a)), Nat.add(Y, M)), Nat.is_le(Nat.mul(7n, a), M),
    Ar.ar_le_add_cancel(Y, Nat.mul(7n, a), M), m2)
  +s1 = Ar.ar_le_trans(P, Nat.add(Y, Nat.mul(7n, a)), Nat.add(Y, M), m1, m3)
  +s2 = pi_le_add2(Nat.add(P, b), Nat.add(Nat.add(Y, M), b), K, K, pi_le_add2(P, Nat.add(Y, M), b, b, s1, Ar.ar_le_refl(b)),
    Ar.ar_le_refl(K))
  +s3 = Ar.ar_lt_le_trans(Nat.add(X, Y), Nat.add(Nat.add(P, b), K), Nat.add(Nat.add(Nat.add(Y, M), b), K), h, s2)
  +T = Nat.add(Nat.add(M, b), K)
  +eR = Equal.trans(Nat, Nat.add(Nat.add(Nat.add(Y, M), b), K), Nat.add(Nat.add(Y, Nat.add(M, b)), K), Nat.add(Y, T),
    Equal.cong(Nat, Nat, z => Nat.add(z, K), Nat.add(Nat.add(Y, M), b), Nat.add(Y, Nat.add(M, b)), Ar.ar_add_assoc(Y, M, b)),
    Ar.ar_add_assoc(Y, Nat.add(M, b), K))
  +s4 = pi_lt_rw(Nat.add(X, Y), Nat.add(Y, X), Nat.add(Nat.add(Nat.add(Y, M), b), K), Nat.add(Y, T), Ar.ar_add_comm(X, Y), eR, s3)
  +s5 = Ar.ard_rt(Nat.is_lt(Nat.add(Y, X), Nat.add(Y, T)), Nat.is_lt(X, T), Ar.ar_lt_add_cancel(Y, X, T), s4)
  +eT = Equal.trans(Nat, T, Nat.add(K, Nat.add(M, b)), Nat.mul(Nat.add(kn, 127n), b),
    Ar.ar_add_comm(Nat.add(M, b), K),
    Equal.trans(Nat, Nat.add(K, Nat.add(M, b)), Nat.add(K, Nat.add(b, M)), Nat.mul(Nat.add(kn, 127n), b),
      Equal.cong(Nat, Nat, z => Nat.add(K, z), Nat.add(M, b), Nat.add(b, M), Ar.ar_add_comm(M, b)),
      Equal.sym(Nat, Nat.mul(Nat.add(kn, 127n), b), Nat.add(K, Nat.add(b, M)),
        Equal.trans(Nat, Nat.mul(Nat.add(kn, 127n), b), Nat.add(K, Nat.mul(127n, b)), Nat.add(K, Nat.add(b, M)),
          Ar.ar_mul_add_l(kn, 127n, b),
          Equal.cong(Nat, Nat, z => Nat.add(K, z), Nat.mul(127n, b), Nat.add(b, M),
            Equal.trans(Nat, Nat.mul(127n, b), Nat.mul(b, 127n), Nat.add(b, M), Ar.ar_mul_comm(127n, b), Ar.ar_mul_succ_r(b, 126n)))))))
  +s6 = pi_lt_rw(X, X, T, Nat.mul(Nat.add(kn, 127n), b), {==}, eT, s5)
  +s7 = Ar.ard_rt(Nat.is_lt(X, Nat.mul(Nat.add(kn, 127n), b)), Nat.is_lt(kp, Nat.add(kn, 127n)),
    Ar.ar_mul_lt_cancel(kp, Nat.add(kn, 127n), b, bpos), s6)
  pi_le_of_lt_succ(kp, Nat.add(kn, 126n),
    pi_lt_rw(kp, kp, Nat.add(kn, 127n), 1n+Nat.add(kn, 126n), {==}, Ar.ar_add_succ_r(kn, 126n), s7))

# both adjacency inequalities and |code| <= 7, f <= 18 give |kq| <= 126
def pi_kq_bound(+kp: Nat, +kn: Nat, +cp: Nat, +cn: Nat, +a: Nat, +b: Nat,
    +bpos: {Nat.is_lt(0n, b) == True{} : Bool},
    c7a: {Nat.is_le(cp, Nat.add(cn, 7n)) == True{} : Bool},
    c7b: {Nat.is_le(cn, Nat.add(cp, 7n)) == True{} : Bool},
    +f18: {Nat.is_le(a, Nat.mul(b, 18n)) == True{} : Bool},
    +rn: {Spec.adjacent(Spec.Z{kp, kn}, Spec.Z{cp, cn}, a, b) == True{} : Bool})
  -> {Nat.is_le(Spec.abs(Spec.Z{kp, kn}), 126n) == True{} : Bool}:
  +X = Nat.mul(kp, b)
  +Y = Nat.mul(cn, a)
  +P = Nat.mul(cp, a)
  +K = Nat.mul(kn, b)
  +i1 = Nat.is_lt(Nat.add(X, Nat.add(Y, 0n)), Nat.add(Nat.add(P, b), K))
  +i2 = Nat.is_lt(Nat.add(P, Nat.add(K, 0n)), Nat.add(Nat.add(X, b), Y))
  +h1 = pi_lt_rw(Nat.add(X, Nat.add(Y, 0n)), Nat.add(X, Y), Nat.add(Nat.add(P, b), K), Nat.add(Nat.add(P, b), K),
    pi_add_zero_in_r(X, Y), {==}, Ar.ar_and_l(i1, i2, rn))
  +h2 = pi_lt_rw(Nat.add(P, Nat.add(K, 0n)), Nat.add(K, P), Nat.add(Nat.add(X, b), Y), Nat.add(Nat.add(Y, b), X),
    Equal.trans(Nat, Nat.add(P, Nat.add(K, 0n)), Nat.add(P, K), Nat.add(K, P), pi_add_zero_in_r(P, K), Ar.ar_add_comm(P, K)),
    pi_swap_outer(X, b, Y), Ar.ar_and_r(i1, i2, rn))
  pi_abs_le(kp, kn, 126n, pi_kq_core(kp, kn, cp, cn, a, b, bpos, c7a, f18, h1), pi_kq_core(kn, kp, cn, cp, a, b, bpos, c7b, f18, h2))

def pi_field_le7(+hi: Nat, +lo: Nat, +i: Nat) -> {Nat.is_le(Impl.field(hi, lo, i), 7n) == True{} : Bool}:
  pi_le_of_lt_succ(Impl.field(hi, lo, i), 7n, pi_field_lt8(hi, lo, i))

def Laws.kq_bound(hi, lo, i, kp, kn, a, b, bpos, f18, rn):
  +F = Impl.field(hi, lo, i)
  pi_kq_bound(kp, kn, Nat.mul(F, 2n), 7n, a, b, bpos, Ar.ar_le_mul_mono(F, 7n, 2n, pi_field_le7(hi, lo, i)),
    Ar.ar_le_add_l(Nat.mul(F, 2n), 7n), f18, rn)

# f = 18: kp b + cn (18 b) < (cp (18 b) + b) + kn b gives kp + 18 cn <= 18 cp + kn
def pi_kq18_core(+kp: Nat, +kn: Nat, +cp: Nat, +cn: Nat, +b: Nat,
    bpos: {Nat.is_lt(0n, b) == True{} : Bool},
    h: {Nat.is_lt(Nat.add(Nat.mul(kp, b), Nat.mul(cn, Nat.mul(b, 18n))),
      Nat.add(Nat.add(Nat.mul(cp, Nat.mul(b, 18n)), b), Nat.mul(kn, b))) == True{} : Bool})
  -> {Nat.is_le(Nat.add(kp, Nat.mul(cn, 18n)), Nat.add(Nat.mul(cp, 18n), kn)) == True{} : Bool}:
  +B18 = Nat.mul(b, 18n)
  +L = Nat.add(kp, Nat.mul(cn, 18n))
  +Q = Nat.add(Nat.mul(cp, 18n), kn)
  +A = Nat.mul(cp, B18)
  +K = Nat.mul(kn, b)
  +eL = Equal.sym(Nat, Nat.mul(L, b), Nat.add(Nat.mul(kp, b), Nat.mul(cn, B18)),
    Equal.trans(Nat, Nat.mul(L, b), Nat.add(Nat.mul(kp, b), Nat.mul(Nat.mul(cn, 18n), b)), Nat.add(Nat.mul(kp, b), Nat.mul(cn, B18)),
      Ar.ar_mul_add_l(kp, Nat.mul(cn, 18n), b),
      Equal.cong(Nat, Nat, z => Nat.add(Nat.mul(kp, b), z), Nat.mul(Nat.mul(cn, 18n), b), Nat.mul(cn, B18),
        Equal.trans(Nat, Nat.mul(Nat.mul(cn, 18n), b), Nat.mul(cn, Nat.mul(18n, b)), Nat.mul(cn, B18),
          Ar.ar_mul_assoc(cn, 18n, b),
          Equal.cong(Nat, Nat, z => Nat.mul(cn, z), Nat.mul(18n, b), B18, Ar.ar_mul_comm(18n, b))))))
  +eQ = Equal.trans(Nat, Nat.mul(Q, b), Nat.add(Nat.mul(Nat.mul(cp, 18n), b), K), Nat.add(A, K),
    Ar.ar_mul_add_l(Nat.mul(cp, 18n), kn, b),
    Equal.cong(Nat, Nat, z => Nat.add(z, K), Nat.mul(Nat.mul(cp, 18n), b), A,
      Equal.trans(Nat, Nat.mul(Nat.mul(cp, 18n), b), Nat.mul(cp, Nat.mul(18n, b)), A,
        Ar.ar_mul_assoc(cp, 18n, b),
        Equal.cong(Nat, Nat, z => Nat.mul(cp, z), Nat.mul(18n, b), B18, Ar.ar_mul_comm(18n, b)))))
  +eR = Equal.trans(Nat, Nat.add(Nat.add(A, b), K), Nat.add(b, Nat.add(A, K)), Nat.add(b, Nat.mul(Q, b)),
    Equal.trans(Nat, Nat.add(Nat.add(A, b), K), Nat.add(Nat.add(b, A), K), Nat.add(b, Nat.add(A, K)),
      Equal.cong(Nat, Nat, z => Nat.add(z, K), Nat.add(A, b), Nat.add(b, A), Ar.ar_add_comm(A, b)),
      Ar.ar_add_assoc(b, A, K)),
    Equal.cong(Nat, Nat, z => Nat.add(b, z), Nat.add(A, K), Nat.mul(Q, b), Equal.sym(Nat, Nat.mul(Q, b), Nat.add(A, K), eQ)))
  +s = pi_lt_rw(Nat.add(Nat.mul(kp, b), Nat.mul(cn, B18)), Nat.mul(L, b), Nat.add(Nat.add(A, b), K), Nat.mul(1n+Q, b), eL, eR, h)
  pi_le_of_lt_succ(L, Q, Ar.ard_rt(Nat.is_lt(Nat.mul(L, b), Nat.mul(1n+Q, b)), Nat.is_lt(L, 1n+Q),
    Ar.ar_mul_lt_cancel(L, 1n+Q, b, bpos), s))

def pi_kq18(+kp: Nat, +kn: Nat, +cp: Nat, +cn: Nat, +b: Nat, +bpos: {Nat.is_lt(0n, b) == True{} : Bool},
    +rn: {Spec.adjacent(Spec.Z{kp, kn}, Spec.Z{cp, cn}, Nat.mul(b, 18n), b) == True{} : Bool})
  -> Spec.zeq(Spec.Z{kp, kn}, Spec.zscale(Spec.Z{cp, cn}, 18n)):
  +B18 = Nat.mul(b, 18n)
  +X = Nat.mul(kp, b)
  +Y = Nat.mul(cn, B18)
  +P = Nat.mul(cp, B18)
  +K = Nat.mul(kn, b)
  +i1 = Nat.is_lt(Nat.add(X, Nat.add(Y, 0n)), Nat.add(Nat.add(P, b), K))
  +i2 = Nat.is_lt(Nat.add(P, Nat.add(K, 0n)), Nat.add(Nat.add(X, b), Y))
  +h1 = pi_lt_rw(Nat.add(X, Nat.add(Y, 0n)), Nat.add(X, Y), Nat.add(Nat.add(P, b), K), Nat.add(Nat.add(P, b), K),
    pi_add_zero_in_r(X, Y), {==}, Ar.ar_and_l(i1, i2, rn))
  +h2 = pi_lt_rw(Nat.add(P, Nat.add(K, 0n)), Nat.add(K, P), Nat.add(Nat.add(X, b), Y), Nat.add(Nat.add(Y, b), X),
    Equal.trans(Nat, Nat.add(P, Nat.add(K, 0n)), Nat.add(P, K), Nat.add(K, P), pi_add_zero_in_r(P, K), Ar.ar_add_comm(P, K)),
    pi_swap_outer(X, b, Y), Ar.ar_and_r(i1, i2, rn))
  +le1 = pi_kq18_core(kp, kn, cp, cn, b, bpos, h1)
  +le2 = pi_le_rw(Nat.add(kn, Nat.mul(cp, 18n)), Nat.add(Nat.mul(cp, 18n), kn), Nat.add(Nat.mul(cn, 18n), kp),
    Nat.add(kp, Nat.mul(cn, 18n)), Ar.ar_add_comm(kn, Nat.mul(cp, 18n)), Ar.ar_add_comm(Nat.mul(cn, 18n), kp),
    pi_kq18_core(kn, kp, cn, cp, b, bpos, h2))
  Ar.ar_le_antisym(Nat.add(kp, Nat.mul(cn, 18n)), Nat.add(Nat.mul(cp, 18n), kn), le1, le2)

def Laws.kq_max_group_exact(hi, lo, i, kp, kn, b, bpos, rn):
  pi_kq18(kp, kn, Nat.mul(Impl.field(hi, lo, i), 2n), 7n, b, bpos, rn)

def Laws.kq_bound_reached():
  ({==}, {==})

# ---------------------------------------------------------------------------------------------
# F. Finding (laws finding_k_rounded, finding_no_exact_kq, finding_partial_bound), issue pattern, rounding points

def Laws.finding_k_rounded():
  ({==}, {==}, {==})

# k + x == k + y gives x == y
def pi_add_inj_l(k: Nat, -x: Nat, -y: Nat, e: {Nat.add(k, x) == Nat.add(k, y) : Nat}) -> {x == y : Nat}:
  match k:
    case 0n:
      e
    case 1n+j:
      pi_add_inj_l(j, x, y, Ar.ar_succ_inj(Nat.add(j, x), Nat.add(j, y), e))

# 5 u is never 5 s + 3 + 5 t
def pi_not_mul5(u: Nat, +s: Nat, +t: Nat, e: {Nat.mul(u, 5n) == Nat.add(Nat.add(Nat.mul(s, 5n), 3n), Nat.mul(t, 5n)) : Nat})
  -> Empty:
  match u:
    case 0n:
      match s:
        case 0n:
          Ar.ar_zero_succ(Nat.add(2n, Nat.mul(t, 5n)), e)
        case 1n+s1:
          Ar.ar_zero_succ(Nat.add(Nat.add(Nat.add(4n, Nat.mul(s1, 5n)), 3n), Nat.mul(t, 5n)), e)
    case 1n+u1:
      match s t:
        case 1n+s1 _:
          pi_not_mul5(u1, s1, t, pi_add_inj_l(5n, Nat.mul(u1, 5n), Nat.add(Nat.add(Nat.mul(s1, 5n), 3n), Nat.mul(t, 5n)), e))
        case 0n 0n:
          Ar.ar_zero_succ(Nat.add(1n, Nat.mul(u1, 5n)),
            Equal.sym(Nat, Nat.add(2n, Nat.mul(u1, 5n)), 0n, pi_add_inj_l(3n, Nat.add(2n, Nat.mul(u1, 5n)), 0n, e)))
        case 0n 1n+t1:
          +e2 = pi_add_inj_l(3n, Nat.add(2n, Nat.mul(u1, 5n)), Nat.mul(1n+t1, 5n), e)
          pi_not_mul5(u1, 0n, t1, Equal.sym(Nat, Nat.add(3n, Nat.mul(t1, 5n)), Nat.mul(u1, 5n),
            pi_add_inj_l(2n, Nat.add(3n, Nat.mul(t1, 5n)), Nat.mul(u1, 5n), Equal.sym(Nat, Nat.add(2n, Nat.mul(u1, 5n)), Nat.mul(1n+t1, 5n), e2))))

def Laws.finding_no_exact_kq(kp, kn, same):
  +e1 = Equal.trans(Nat, Nat.add(378n, Nat.mul(kp, 5n)), Nat.add(Nat.mul(kp, 5n), 378n), Nat.add(756n, Nat.mul(kn, 5n)),
    Ar.ar_add_comm(378n, Nat.mul(kp, 5n)), same)
  pi_not_mul5(kp, 75n, kn, pi_add_cancel(378n, Nat.mul(kp, 5n), Nat.add(378n, Nat.mul(kn, 5n)), e1))

def Laws.imma_issue():
  ({==}, {==}, {==}, {==}, {==}, {==})

def Laws.rounding_points():
  ({==}, {==})

# ---------------------------------------------------------------------------------------------
# G. int32 wrap (laws wrap_identity, wrap_at_2_31)

# r < B gives (r + q B) mod B == r
def pi_mod_r(+q: Nat, +r: Nat, +B: Nat, small: {Nat.is_lt(r, B) == True{} : Bool})
  -> {Nat.mod(Nat.add(r, Nat.mul(q, B)), B) == r : Nat}:
  %Ar.ar_add_comm(Nat.mul(q, B), r) : {Nat.mod(_, B) == r : Nat}
  Ar.p7_mod_r(q, r, B, small)

def pi_div_r(+q: Nat, +r: Nat, +B: Nat, small: {Nat.is_lt(r, B) == True{} : Bool})
  -> {Nat.div(Nat.add(r, Nat.mul(q, B)), B) == q : Nat}:
  %Ar.ar_add_comm(Nat.mul(q, B), r) : {Nat.div(_, B) == q : Nat}
  Ar.p7_div_q(q, r, B, small)

# a < a + b for b > 0
def pi_lt_add_pos(+a: Nat, +b: Nat, bpos: {Nat.is_lt(0n, b) == True{} : Bool}) -> {Nat.is_lt(a, Nat.add(a, b)) == True{} : Bool}:
  match b:
    case 0n:
      Empty.absurd({Nat.is_lt(a, Nat.add(a, 0n)) == True{} : Bool}, Ar.ar_false_true(bpos))
    case 1n+c:
      pi_lt_rw(a, a, 1n+Nat.add(a, c), Nat.add(a, 1n+c), {==}, Equal.sym(Nat, Nat.add(a, 1n+c), 1n+Nat.add(a, c), Ar.ar_add_succ_r(a, c)),
        pi_lt_succ_of_le(a, Nat.add(a, c), Ar.ar_le_add_r(a, c)))

# b <= a gives (a + L) - b == (a - b) + L
def pi_sub_add_l(+a: Nat, +L: Nat, +b: Nat, +h: {Nat.is_le(b, a) == True{} : Bool})
  -> {Nat.sub(Nat.add(a, L), b) == Nat.add(Nat.sub(a, b), L) : Nat}:
  %Ar.ar_sub_add(a, b, h) : {Nat.sub(Nat.add(_, L), b) == Nat.add(Nat.sub(a, b), L) : Nat}
  %Equal.sym(Nat, Nat.add(Nat.add(b, Nat.sub(a, b)), L), Nat.add(b, Nat.add(Nat.sub(a, b), L)), Ar.ar_add_assoc(b, Nat.sub(a, b), L))
    : {Nat.sub(_, b) == Nat.add(Nat.sub(a, b), L) : Nat}
  Ar.ar_add_sub(b, Nat.add(Nat.sub(a, b), L))

# b <= a gives a - (a - b) == b
def pi_sub_sub(+a: Nat, +b: Nat, +h: {Nat.is_le(b, a) == True{} : Bool}) -> {Nat.sub(a, Nat.sub(a, b)) == b : Nat}:
  %Ar.ar_sub_add(a, b, h) : {Nat.sub(_, Nat.sub(a, b)) == b : Nat}
  %Ar.ar_add_comm(Nat.sub(a, b), b) : {Nat.sub(_, Nat.sub(a, b)) == b : Nat}
  Ar.ar_add_sub(Nat.sub(a, b), b)

# Below, m = INT32_MAX stays a variable: 2^31 = 1 + m, 2^32 - 1 = W = m + 2^31, 2^32 = N = 2^31 + 2^31 = 1 + W.

# n <= p, p - n < 2^31: the pattern is d = p - n and decodes to d
def pi_wrap_pos(+p: Nat, +n: Nat, +m: Nat, +np: {Nat.is_le(n, p) == True{} : Bool},
    +small: {Nat.is_lt(Nat.sub(p, n), 1n+m) == True{} : Bool})
  -> Spec.zeq(Impl.wrap(Spec.Z{p, n}, m), Spec.Z{p, n}):
  +W = Nat.add(m, 1n+m)
  +N = Nat.add(1n+m, 1n+m)
  +d = Nat.sub(p, n)
  +eq = Equal.trans(Nat, Nat.add(d, Nat.mul(n, N)), Nat.add(d, Nat.add(n, Nat.mul(n, W))), Nat.add(p, Nat.mul(n, W)),
    Equal.cong(Nat, Nat, z => Nat.add(d, z), Nat.mul(n, N), Nat.add(n, Nat.mul(n, W)), Ar.ar_mul_succ_r(n, W)),
    Equal.trans(Nat, Nat.add(d, Nat.add(n, Nat.mul(n, W))), Nat.add(Nat.add(d, n), Nat.mul(n, W)), Nat.add(p, Nat.mul(n, W)),
      Equal.sym(Nat, Nat.add(Nat.add(d, n), Nat.mul(n, W)), Nat.add(d, Nat.add(n, Nat.mul(n, W))), Ar.ar_add_assoc(d, n, Nat.mul(n, W))),
      Equal.cong(Nat, Nat, z => Nat.add(z, Nat.mul(n, W)), Nat.add(d, n), p,
        Equal.trans(Nat, Nat.add(d, n), Nat.add(n, d), p, Ar.ar_add_comm(d, n), Ar.ar_sub_add(p, n, np)))))
  +em = Equal.trans(Nat, Nat.mod(Nat.add(p, Nat.mul(n, W)), N), Nat.mod(Nat.add(d, Nat.mul(n, N)), N), d,
    Equal.cong(Nat, Nat, z => Nat.mod(z, N), Nat.add(p, Nat.mul(n, W)), Nat.add(d, Nat.mul(n, N)),
      Equal.sym(Nat, Nat.add(d, Nat.mul(n, N)), Nat.add(p, Nat.mul(n, W)), eq)),
    pi_mod_r(n, d, N, Ar.ar_lt_le_trans(d, 1n+m, N, small, Ar.ar_le_add_r(1n+m, 1n+m))))
  %Equal.sym(Nat, Nat.mod(Nat.add(p, Nat.mul(n, W)), N), d, em) : Spec.zeq(Impl.dec(_, m), Spec.Z{p, n})
  %Equal.sym(Bool, Nat.is_lt(d, 1n+m), True{}, small) : Spec.zeq(Impl.pickz(_, Spec.Z{d, 0n}, Spec.Z{0n, Nat.sub(N, d)}), Spec.Z{p, n})
  Equal.trans(Nat, Nat.add(d, n), p, Nat.add(p, 0n),
    Equal.trans(Nat, Nat.add(d, n), Nat.add(n, d), p, Ar.ar_add_comm(d, n), Ar.ar_sub_add(p, n, np)),
    Equal.sym(Nat, Nat.add(p, 0n), p, Ar.ar_add_zero_r(p)))

# n = p + 1 + e2 with e2 < m (so 0 < n - p <= 2^31 - 1): the pattern is W - e2 = 2^32 - (n - p), which decodes to p - n
def pi_wrap_neg(+p: Nat, +n: Nat, +m: Nat, +e2: Nat, +ep: {Nat.add(p, 1n+e2) == n : Nat},
    +small: {Nat.is_lt(e2, m) == True{} : Bool})
  -> Spec.zeq(Impl.wrap(Spec.Z{p, n}, m), Spec.Z{p, n}):
  +W = Nat.add(m, 1n+m)
  +N = Nat.add(1n+m, 1n+m)
  +S = Nat.sub(W, e2)
  +pW = Nat.mul(p, W)
  +eW = Nat.mul(e2, W)
  +Q = Nat.add(p, e2)
  +Y = Nat.add(pW, eW)
  +Z2 = Nat.add(e2, S)
  +le_m = Ar.ar_lt_le(e2, m, small)
  +le_W = Ar.ar_le_trans(e2, m, W, le_m, Ar.ar_le_add_r(m, 1n+m))
  +C = Nat.add(p, Nat.add(pW, Nat.add(eW, Z2)))
  +eL = Equal.trans(Nat, Nat.add(p, Nat.mul(n, W)), Nat.add(p, Nat.mul(Nat.add(p, 1n+e2), W)), C,
    Equal.cong(Nat, Nat, z => Nat.add(p, Nat.mul(z, W)), n, Nat.add(p, 1n+e2), Equal.sym(Nat, Nat.add(p, 1n+e2), n, ep)),
    Equal.cong(Nat, Nat, z => Nat.add(p, z), Nat.mul(Nat.add(p, 1n+e2), W), Nat.add(pW, Nat.add(eW, Z2)),
      Equal.trans(Nat, Nat.mul(Nat.add(p, 1n+e2), W), Nat.add(pW, Nat.add(W, eW)), Nat.add(pW, Nat.add(eW, Z2)),
        Ar.ar_mul_add_l(p, 1n+e2, W),
        Equal.cong(Nat, Nat, z => Nat.add(pW, z), Nat.add(W, eW), Nat.add(eW, Z2),
          Equal.trans(Nat, Nat.add(W, eW), Nat.add(eW, W), Nat.add(eW, Z2), Ar.ar_add_comm(W, eW),
            Equal.cong(Nat, Nat, z => Nat.add(eW, z), W, Z2, Equal.sym(Nat, Z2, W, Ar.ar_sub_add(W, e2, le_W))))))))
  +r4 = Equal.trans(Nat, Nat.add(e2, Nat.add(Y, S)), Nat.add(Nat.add(e2, Y), S), Nat.add(Y, Z2),
    Equal.sym(Nat, Nat.add(Nat.add(e2, Y), S), Nat.add(e2, Nat.add(Y, S)), Ar.ar_add_assoc(e2, Y, S)),
    Equal.trans(Nat, Nat.add(Nat.add(e2, Y), S), Nat.add(Nat.add(Y, e2), S), Nat.add(Y, Z2),
      Equal.cong(Nat, Nat, z => Nat.add(z, S), Nat.add(e2, Y), Nat.add(Y, e2), Ar.ar_add_comm(e2, Y)),
      Ar.ar_add_assoc(Y, e2, S)))
  +eR1 = Equal.trans(Nat, Nat.add(S, Nat.add(Q, Y)), Nat.add(Nat.add(Q, Y), S), Nat.add(p, Nat.add(e2, Nat.add(Y, S))),
    Ar.ar_add_comm(S, Nat.add(Q, Y)),
    Equal.trans(Nat, Nat.add(Nat.add(Q, Y), S), Nat.add(Q, Nat.add(Y, S)), Nat.add(p, Nat.add(e2, Nat.add(Y, S))),
      Ar.ar_add_assoc(Q, Y, S), Ar.ar_add_assoc(p, e2, Nat.add(Y, S))))
  +eR2 = Equal.trans(Nat, Nat.add(p, Nat.add(e2, Nat.add(Y, S))), Nat.add(p, Nat.add(Y, Z2)), C,
    Equal.cong(Nat, Nat, z => Nat.add(p, z), Nat.add(e2, Nat.add(Y, S)), Nat.add(Y, Z2), r4),
    Equal.cong(Nat, Nat, z => Nat.add(p, z), Nat.add(Y, Z2), Nat.add(pW, Nat.add(eW, Z2)), Ar.ar_add_assoc(pW, eW, Z2)))
  +eR0 = Equal.trans(Nat, Nat.add(S, Nat.mul(Q, N)), Nat.add(S, Nat.add(Q, Nat.mul(Q, W))), Nat.add(S, Nat.add(Q, Y)),
    Equal.cong(Nat, Nat, z => Nat.add(S, z), Nat.mul(Q, N), Nat.add(Q, Nat.mul(Q, W)), Ar.ar_mul_succ_r(Q, W)),
    Equal.cong(Nat, Nat, z => Nat.add(S, Nat.add(Q, z)), Nat.mul(Q, W), Y, Ar.ar_mul_add_l(p, e2, W)))
  +eR = Equal.trans(Nat, Nat.add(S, Nat.mul(Q, N)), Nat.add(S, Nat.add(Q, Y)), C, eR0, Equal.trans(Nat, Nat.add(S, Nat.add(Q, Y)),
    Nat.add(p, Nat.add(e2, Nat.add(Y, S))), C, eR1, eR2))
  +eLR = Equal.trans(Nat, Nat.add(p, Nat.mul(n, W)), C, Nat.add(S, Nat.mul(Q, N)), eL, Equal.sym(Nat, Nat.add(S, Nat.mul(Q, N)), C, eR))
  +S_le_W = pi_sub_le(W, e2)
  +em = Equal.trans(Nat, Nat.mod(Nat.add(p, Nat.mul(n, W)), N), Nat.mod(Nat.add(S, Nat.mul(Q, N)), N), S,
    Equal.cong(Nat, Nat, z => Nat.mod(z, N), Nat.add(p, Nat.mul(n, W)), Nat.add(S, Nat.mul(Q, N)), eLR),
    pi_mod_r(Q, S, N, pi_lt_succ_of_le(S, W, S_le_W)))
  +eS = pi_sub_add_l(m, 1n+m, e2, le_m)
  +H_le_S = pi_le_rw(1n+m, 1n+m, Nat.add(Nat.sub(m, e2), 1n+m), S, {==}, Equal.sym(Nat, S, Nat.add(Nat.sub(m, e2), 1n+m), eS),
    Ar.ar_le_add_l(Nat.sub(m, e2), 1n+m))
  +lt_false = Ar.p6_le_nlt(S, 1n+m, H_le_S)
  +eN = Equal.trans(Nat, Nat.sub(N, S), 1n+Nat.sub(W, S), 1n+e2, SP.sp_sub_succ(W, S, S_le_W),
    Equal.cong(Nat, Nat, z => 1n+z, Nat.sub(W, S), e2, pi_sub_sub(W, e2, le_W)))
  %Equal.sym(Nat, Nat.mod(Nat.add(p, Nat.mul(n, W)), N), S, em) : Spec.zeq(Impl.dec(_, m), Spec.Z{p, n})
  %Equal.sym(Bool, Nat.is_lt(S, 1n+m), False{}, lt_false) : Spec.zeq(Impl.pickz(_, Spec.Z{S, 0n}, Spec.Z{0n, Nat.sub(N, S)}), Spec.Z{p, n})
  %Equal.sym(Nat, Nat.sub(N, S), 1n+e2, eN) : {n == Nat.add(p, _) : Nat}
  Equal.sym(Nat, Nat.add(p, 1n+e2), n, ep)

def pi_wrap_neg.go(k: Nat, +p: Nat, +n: Nat, +m: Nat, +ep: {Nat.add(p, k) == n : Nat},
    e: {Nat.is_le(n, p) == False{} : Bool}, small: {Nat.is_lt(k, 1n+m) == True{} : Bool})
  -> Spec.zeq(Impl.wrap(Spec.Z{p, n}, m), Spec.Z{p, n}):
  match k:
    case 0n:
      +np = pi_le_rw(Nat.add(p, 0n), n, p, p, ep, {==}, pi_le_eq(Nat.add(p, 0n), p, Ar.ar_add_zero_r(p)))
      Empty.absurd(Spec.zeq(Impl.wrap(Spec.Z{p, n}, m), Spec.Z{p, n}),
        Ar.ar_true_false(Equal.trans(Bool, True{}, Nat.is_le(n, p), False{}, Equal.sym(Bool, Nat.is_le(n, p), True{}, np), e)))
    case 1n+e2:
      pi_wrap_neg(p, n, m, e2, ep, small)

def pi_wrap.go(v: Bool, +p: Nat, +n: Nat, +m: Nat, +e: {Nat.is_le(n, p) == v : Bool},
    +small: {Nat.is_lt(Spec.abs(Spec.Z{p, n}), 1n+m) == True{} : Bool})
  -> Spec.zeq(Impl.wrap(Spec.Z{p, n}, m), Spec.Z{p, n}):
  match v:
    case True{}:
      pi_wrap_pos(p, n, m, e, Ar.ar_le_lt_trans(Nat.sub(p, n), Spec.abs(Spec.Z{p, n}), 1n+m,
        Ar.ar_le_add_r(Nat.sub(p, n), Nat.sub(n, p)), small))
    case False{}:
      +pn = Ar.ar_lt_le(p, n, pi_lt_of_nle(n, p, e))
      pi_wrap_neg.go(Nat.sub(n, p), p, n, m, Ar.ar_sub_add(n, p, pn), e,
        Ar.ar_le_lt_trans(Nat.sub(n, p), Spec.abs(Spec.Z{p, n}), 1n+m, Ar.ar_le_add_l(Nat.sub(p, n), Nat.sub(n, p)), small))

# ---------------------------------------------------------------------------------------------
# Big constants (laws 5-8): 2^22 = 2048 * 2048, 2^31 = 32768 * 65536, INT32_MAX = 2^31 - 1. Every fact about them is
# derived from small evaluations (at most 524288 unary steps, lazily) and monotonicity, never by evaluating a product.

# 1 <= 2^31 (lazy: one step of the product)
def pi_t31_pos() -> {Nat.is_le(1n, Nat.mul(32768n, 65536n)) == True{} : Bool}:
  {==}

# 1 + y == 1n+y, for any y
def pi_add1(-y: Nat) -> {Nat.add(1n, y) == 1n+y : Nat}:
  {==}

# 1 + (2^31 - 1) == 2^31
def pi_t31_succ() -> {1n+Nat.sub(Nat.mul(32768n, 65536n), 1n) == Nat.mul(32768n, 65536n) : Nat}:
  Equal.trans(Nat, 1n+Nat.sub(Nat.mul(32768n, 65536n), 1n), Nat.add(1n, Nat.sub(Nat.mul(32768n, 65536n), 1n)), Nat.mul(32768n, 65536n),
    Equal.sym(Nat, Nat.add(1n, Nat.sub(Nat.mul(32768n, 65536n), 1n)), 1n+Nat.sub(Nat.mul(32768n, 65536n), 1n),
      pi_add1(Nat.sub(Nat.mul(32768n, 65536n), 1n))),
    Ar.ar_sub_add(Nat.mul(32768n, 65536n), 1n, pi_t31_pos()))

# x == x, for any x (instances are not evaluated)
def pi_refl(-x: Nat) -> {x == x : Nat}:
  {==}

# P = 32 * (127 * 126) (512064, the per-IMMA partial bound) stays a product: every fact about it below evaluates at
# most 127 * 126 = 16002 unary steps.

# 8 P < 2048 * 2048: 127 * 126 < 8 * 2048 (16002 < 16384, evaluated), times 256 = 8 * 32 on both sides
def pi_chain_lt22() -> {Nat.is_lt(Nat.mul(8n, Nat.mul(32n, Nat.mul(127n, 126n))), Nat.mul(2048n, 2048n)) == True{} : Bool}:
  +X = Nat.mul(127n, 126n)
  +Y = Nat.mul(8n, 2048n)
  +s = Ar.ard_tr(Nat.is_lt(Nat.mul(X, 256n), Nat.mul(Y, 256n)), Nat.is_lt(X, Y), Ar.ar_mul_lt_cancel(X, Y, 256n, {==}), {==})
  +l2 = Equal.cong(Nat, Nat, z => Nat.mul(z, X), 256n, Nat.mul(8n, 32n), {==})
  +r2 = Equal.sym(Nat, Nat.mul(Nat.mul(256n, 8n), 2048n), Nat.mul(256n, Y), Ar.ar_mul_assoc(256n, 8n, 2048n))
  +r3 = Equal.cong(Nat, Nat, z => Nat.mul(z, 2048n), Nat.mul(256n, 8n), 2048n, {==})
  pi_lt_rw(Nat.mul(X, 256n), Nat.mul(8n, Nat.mul(32n, X)), Nat.mul(Y, 256n), Nat.mul(2048n, 2048n),
    Equal.trans(Nat, Nat.mul(X, 256n), Nat.mul(256n, X), Nat.mul(8n, Nat.mul(32n, X)), Ar.ar_mul_comm(X, 256n),
      Equal.trans(Nat, Nat.mul(256n, X), Nat.mul(Nat.mul(8n, 32n), X), Nat.mul(8n, Nat.mul(32n, X)), l2, Ar.ar_mul_assoc(8n, 32n, X))),
    Equal.trans(Nat, Nat.mul(Y, 256n), Nat.mul(256n, Y), Nat.mul(2048n, 2048n), Ar.ar_mul_comm(Y, 256n),
      Equal.trans(Nat, Nat.mul(256n, Y), Nat.mul(Nat.mul(256n, 8n), 2048n), Nat.mul(2048n, 2048n), r2, r3)),
    s)

# P < 4096 * 4096: 127 * 126 < 128 * 4096 (evaluated), times 32 = 4096 / 128 on both sides
def pi_p_lt24() -> {Nat.is_lt(Nat.mul(32n, Nat.mul(127n, 126n)), Nat.mul(4096n, 4096n)) == True{} : Bool}:
  +X = Nat.mul(127n, 126n)
  +Y = Nat.mul(128n, 4096n)
  +s = Ar.ard_tr(Nat.is_lt(Nat.mul(X, 32n), Nat.mul(Y, 32n)), Nat.is_lt(X, Y), Ar.ar_mul_lt_cancel(X, Y, 32n, {==}), {==})
  +r2 = Equal.sym(Nat, Nat.mul(Nat.mul(32n, 128n), 4096n), Nat.mul(32n, Y), Ar.ar_mul_assoc(32n, 128n, 4096n))
  +r3 = Equal.cong(Nat, Nat, z => Nat.mul(z, 4096n), Nat.mul(32n, 128n), 4096n, {==})
  pi_lt_rw(Nat.mul(X, 32n), Nat.mul(32n, X), Nat.mul(Y, 32n), Nat.mul(4096n, 4096n), Ar.ar_mul_comm(X, 32n),
    Equal.trans(Nat, Nat.mul(Y, 32n), Nat.mul(32n, Y), Nat.mul(4096n, 4096n), Ar.ar_mul_comm(Y, 32n),
      Equal.trans(Nat, Nat.mul(32n, Y), Nat.mul(Nat.mul(32n, 128n), 4096n), Nat.mul(4096n, 4096n), r2, r3)),
    s)

# 2048 * 2048 < 32768 * 2048 <= 32768 * 65536
def pi_22_lt_31() -> {Nat.is_lt(Nat.mul(2048n, 2048n), Nat.mul(32768n, 65536n)) == True{} : Bool}:
  +a1 = Ar.ard_tr(Nat.is_lt(Nat.mul(2048n, 2048n), Nat.mul(32768n, 2048n)), Nat.is_lt(2048n, 32768n),
    Ar.ar_mul_lt_cancel(2048n, 32768n, 2048n, {==}), {==})
  +a2 = pi_le_rw(Nat.mul(2048n, 32768n), Nat.mul(32768n, 2048n), Nat.mul(65536n, 32768n), Nat.mul(32768n, 65536n),
    Ar.ar_mul_comm(2048n, 32768n), Ar.ar_mul_comm(65536n, 32768n), Ar.ar_le_mul_mono(2048n, 65536n, 32768n, {==}))
  Ar.ar_lt_le_trans(Nat.mul(2048n, 2048n), Nat.mul(32768n, 2048n), Nat.mul(32768n, 65536n), a1, a2)

# x < 2^31 gives x <= INT32_MAX
def pi_le_max(+x: Nat, h: {Nat.is_lt(x, Nat.mul(32768n, 65536n)) == True{} : Bool})
  -> {Nat.is_le(x, Nat.sub(Nat.mul(32768n, 65536n), 1n)) == True{} : Bool}:
  Ar.ard_tr(Nat.is_le(x, Nat.sub(Nat.mul(32768n, 65536n), 1n)), Nat.is_lt(x, Nat.mul(32768n, 65536n)),
    Ar.p1_le_pred(x, Nat.mul(32768n, 65536n), pi_t31_pos()), h)

# 8 P < 2^31 and P < 2^31
def pi_8p_lt31() -> {Nat.is_lt(Nat.mul(8n, Nat.mul(32n, Nat.mul(127n, 126n))), Nat.mul(32768n, 65536n)) == True{} : Bool}:
  Ar.ar_lt_trans(Nat.mul(8n, Nat.mul(32n, Nat.mul(127n, 126n))), Nat.mul(2048n, 2048n), Nat.mul(32768n, 65536n), pi_chain_lt22(), pi_22_lt_31())

def pi_p_lt31() -> {Nat.is_lt(Nat.mul(32n, Nat.mul(127n, 126n)), Nat.mul(32768n, 65536n)) == True{} : Bool}:
  +P = Nat.mul(32n, Nat.mul(127n, 126n))
  +le = pi_le_rw(Nat.mul(1n, P), P, Nat.mul(8n, P), Nat.mul(8n, P), Ar.ar_mul_one_l(P), pi_refl(Nat.mul(8n, P)),
    Ar.ar_le_mul_mono(1n, 8n, P, {==}))
  Ar.ar_le_lt_trans(P, Nat.mul(8n, P), Nat.mul(32768n, 65536n), le, pi_8p_lt31())

def Laws.wrap_identity(p, n, small):
  +sm = pi_lt_rw(Spec.abs(Spec.Z{p, n}), Spec.abs(Spec.Z{p, n}), Nat.mul(32768n, 65536n), 1n+Nat.sub(Nat.mul(32768n, 65536n), 1n),
    {==}, Equal.sym(Nat, 1n+Nat.sub(Nat.mul(32768n, 65536n), 1n), Nat.mul(32768n, 65536n), pi_t31_succ()), small)
  pi_wrap.go(Nat.is_le(n, p), p, n, Nat.sub(Nat.mul(32768n, 65536n), 1n), {==}, sm)

# 2^31 wraps to -2^31
def pi_wrap_top(+n: Nat, +m: Nat) -> {Impl.wrap(Spec.Z{Nat.add(n, 1n+m), n}, m) == Spec.Z{0n, 1n+m} : Spec.Z}:
  +H : Nat = 1n+m
  +W = Nat.add(m, 1n+m)
  +N = Nat.add(1n+m, 1n+m)
  +nW = Nat.mul(n, W)
  +eq = Equal.trans(Nat, Nat.add(Nat.add(n, H), nW), Nat.add(nW, Nat.add(n, H)), Nat.add(H, Nat.mul(n, N)),
    Ar.ar_add_comm(Nat.add(n, H), nW),
    Equal.trans(Nat, Nat.add(nW, Nat.add(n, H)), Nat.add(Nat.add(nW, n), H), Nat.add(H, Nat.mul(n, N)),
      Equal.sym(Nat, Nat.add(Nat.add(nW, n), H), Nat.add(nW, Nat.add(n, H)), Ar.ar_add_assoc(nW, n, H)),
      Equal.trans(Nat, Nat.add(Nat.add(nW, n), H), Nat.add(Nat.mul(n, N), H), Nat.add(H, Nat.mul(n, N)),
        Equal.cong(Nat, Nat, z => Nat.add(z, H), Nat.add(nW, n), Nat.mul(n, N),
          Equal.trans(Nat, Nat.add(nW, n), Nat.add(n, nW), Nat.mul(n, N), Ar.ar_add_comm(nW, n),
            Equal.sym(Nat, Nat.mul(n, N), Nat.add(n, nW), Ar.ar_mul_succ_r(n, W)))),
        Ar.ar_add_comm(Nat.mul(n, N), H))))
  +em = Equal.trans(Nat, Nat.mod(Nat.add(Nat.add(n, H), nW), N), Nat.mod(Nat.add(H, Nat.mul(n, N)), N), H,
    Equal.cong(Nat, Nat, z => Nat.mod(z, N), Nat.add(Nat.add(n, H), nW), Nat.add(H, Nat.mul(n, N)), eq),
    pi_mod_r(n, H, N, pi_lt_add_pos(H, H, {==})))
  %Equal.sym(Nat, Nat.mod(Nat.add(Nat.add(n, H), nW), N), H, em) : {Impl.dec(_, m) == Spec.Z{0n, H} : Spec.Z}
  %Equal.sym(Bool, Nat.is_lt(H, H), False{}, Ar.ar_lt_irrefl(H)) : {Impl.pickz(_, Spec.Z{H, 0n}, Spec.Z{0n, Nat.sub(N, H)}) == Spec.Z{0n, H} : Spec.Z}
  %Equal.sym(Nat, Nat.sub(N, H), H, Ar.ar_add_sub(H, H)) : {Spec.Z{0n, _} == Spec.Z{0n, H} : Spec.Z}
  {==}

def Laws.wrap_at_2_31(n):
  %pi_t31_succ() : {Impl.wrap(Spec.Z{Nat.add(n, _), n}, Nat.sub(Nat.mul(32768n, 65536n), 1n)) == Spec.Z{0n, _} : Spec.Z}
  pi_wrap_top(n, Nat.sub(Nat.mul(32768n, 65536n), 1n))

# ---------------------------------------------------------------------------------------------
# H. Dot products and the IMMA chain (laws imma_partial, imma_chain_exact), generic in m = INT32_MAX

# (a + c) + (b + d) == (a + b) + (c + d)
def pi_swap4(+a: Nat, +c: Nat, +b: Nat, +d: Nat)
  -> {Nat.add(Nat.add(a, c), Nat.add(b, d)) == Nat.add(Nat.add(a, b), Nat.add(c, d)) : Nat}:
  Equal.trans(Nat, Nat.add(Nat.add(a, c), Nat.add(b, d)), Nat.add(a, Nat.add(c, Nat.add(b, d))), Nat.add(Nat.add(a, b), Nat.add(c, d)),
    Ar.ar_add_assoc(a, c, Nat.add(b, d)),
    Equal.trans(Nat, Nat.add(a, Nat.add(c, Nat.add(b, d))), Nat.add(a, Nat.add(b, Nat.add(c, d))), Nat.add(Nat.add(a, b), Nat.add(c, d)),
      Equal.cong(Nat, Nat, z => Nat.add(a, z), Nat.add(c, Nat.add(b, d)), Nat.add(b, Nat.add(c, d)),
        Equal.trans(Nat, Nat.add(c, Nat.add(b, d)), Nat.add(Nat.add(c, b), d), Nat.add(b, Nat.add(c, d)),
          Equal.sym(Nat, Nat.add(Nat.add(c, b), d), Nat.add(c, Nat.add(b, d)), Ar.ar_add_assoc(c, b, d)),
          Equal.trans(Nat, Nat.add(Nat.add(c, b), d), Nat.add(Nat.add(b, c), d), Nat.add(b, Nat.add(c, d)),
            Equal.cong(Nat, Nat, z => Nat.add(z, d), Nat.add(c, b), Nat.add(b, c), Ar.ar_add_comm(c, b)),
            Ar.ar_add_assoc(b, c, d)))),
      Equal.sym(Nat, Nat.add(Nat.add(a, b), Nat.add(c, d)), Nat.add(a, Nat.add(b, Nat.add(c, d))), Ar.ar_add_assoc(a, b, Nat.add(c, d)))))

def pi_mass_add(x: Spec.Z, y: Spec.Z) -> {Spec.mass(Spec.zadd(x, y)) == Nat.add(Spec.mass(x), Spec.mass(y)) : Nat}:
  match x y:
    case Spec.Z{a, b} Spec.Z{c, d}:
      pi_swap4(a, c, b, d)

# a (c + d) == a c + a d
def pi_mul_add_r(+a: Nat, +c: Nat, +d: Nat) -> {Nat.mul(a, Nat.add(c, d)) == Nat.add(Nat.mul(a, c), Nat.mul(a, d)) : Nat}:
  Equal.trans(Nat, Nat.mul(a, Nat.add(c, d)), Nat.mul(Nat.add(c, d), a), Nat.add(Nat.mul(a, c), Nat.mul(a, d)),
    Ar.ar_mul_comm(a, Nat.add(c, d)),
    Equal.trans(Nat, Nat.mul(Nat.add(c, d), a), Nat.add(Nat.mul(c, a), Nat.mul(d, a)), Nat.add(Nat.mul(a, c), Nat.mul(a, d)),
      Ar.ar_mul_add_l(c, d, a),
      Equal.trans(Nat, Nat.add(Nat.mul(c, a), Nat.mul(d, a)), Nat.add(Nat.mul(a, c), Nat.mul(d, a)), Nat.add(Nat.mul(a, c), Nat.mul(a, d)),
        Equal.cong(Nat, Nat, z => Nat.add(z, Nat.mul(d, a)), Nat.mul(c, a), Nat.mul(a, c), Ar.ar_mul_comm(c, a)),
        Equal.cong(Nat, Nat, z => Nat.add(Nat.mul(a, c), z), Nat.mul(d, a), Nat.mul(a, d), Ar.ar_mul_comm(d, a)))))

# mass(x y) == mass(x) mass(y)
def pi_mass_mul(x: Spec.Z, y: Spec.Z) -> {Spec.mass(Spec.zmul(x, y)) == Nat.mul(Spec.mass(x), Spec.mass(y)) : Nat}:
  match x y:
    case Spec.Z{+a, +b} Spec.Z{+c, +d}:
      +ac = Nat.mul(a, c)
      +ad = Nat.mul(a, d)
      +bc = Nat.mul(b, c)
      +bd = Nat.mul(b, d)
      +M = Nat.add(Nat.add(ac, ad), Nat.add(bc, bd))
      +eL = Equal.trans(Nat, Nat.add(Nat.add(ac, bd), Nat.add(ad, bc)), Nat.add(Nat.add(ac, ad), Nat.add(bd, bc)), M,
        pi_swap4(ac, bd, ad, bc),
        Equal.cong(Nat, Nat, z => Nat.add(Nat.add(ac, ad), z), Nat.add(bd, bc), Nat.add(bc, bd), Ar.ar_add_comm(bd, bc)))
      +eR = Equal.trans(Nat, Nat.mul(Nat.add(a, b), Nat.add(c, d)), Nat.add(Nat.mul(a, Nat.add(c, d)), Nat.mul(b, Nat.add(c, d))), M,
        Ar.ar_mul_add_l(a, b, Nat.add(c, d)),
        Equal.trans(Nat, Nat.add(Nat.mul(a, Nat.add(c, d)), Nat.mul(b, Nat.add(c, d))), Nat.add(Nat.add(ac, ad), Nat.mul(b, Nat.add(c, d))), M,
          Equal.cong(Nat, Nat, z => Nat.add(z, Nat.mul(b, Nat.add(c, d))), Nat.mul(a, Nat.add(c, d)), Nat.add(ac, ad), pi_mul_add_r(a, c, d)),
          Equal.cong(Nat, Nat, z => Nat.add(Nat.add(ac, ad), z), Nat.mul(b, Nat.add(c, d)), Nat.add(bc, bd), pi_mul_add_r(b, c, d))))
      Equal.trans(Nat, Nat.add(Nat.add(ac, bd), Nat.add(ad, bc)), M, Nat.mul(Nat.add(a, b), Nat.add(c, d)), eL,
        Equal.sym(Nat, Nat.mul(Nat.add(a, b), Nat.add(c, d)), M, eR))

def pi_mass_mul_le(+x: Spec.Z, +y: Spec.Z, +A: Nat, +B: Nat, ha: {Nat.is_le(Spec.mass(x), A) == True{} : Bool},
    hb: {Nat.is_le(Spec.mass(y), B) == True{} : Bool}) -> {Nat.is_le(Spec.mass(Spec.zmul(x, y)), Nat.mul(A, B)) == True{} : Bool}:
  +mx = Spec.mass(x)
  +my = Spec.mass(y)
  +m1 = Ar.ar_le_mul_mono(mx, A, my, ha)
  +m2 = pi_le_rw(Nat.mul(my, A), Nat.mul(A, my), Nat.mul(B, A), Nat.mul(A, B), Ar.ar_mul_comm(my, A), Ar.ar_mul_comm(B, A),
    Ar.ar_le_mul_mono(my, B, A, hb))
  pi_le_rw(Nat.mul(mx, my), Spec.mass(Spec.zmul(x, y)), Nat.mul(A, B), Nat.mul(A, B),
    Equal.sym(Nat, Spec.mass(Spec.zmul(x, y)), Nat.mul(mx, my), pi_mass_mul(x, y)), {==},
    Ar.ar_le_trans(Nat.mul(mx, my), Nat.mul(A, my), Nat.mul(A, B), m1, m2))

# mass of a dot product of bounded operands
def pi_dot_mass(q: List<&2, Spec.Z>, k: List<&2, Spec.Z>, +A: Nat, +B: Nat, +ok: {Spec.zip_ok(q, k, A, B) == True{} : Bool})
  -> {Nat.is_le(Spec.mass(Spec.dot(q, k)), Nat.mul(Spec.len(k), Nat.mul(A, B))) == True{} : Bool}:
  match q k:
    case Nil{} Nil{}:
      {==}
    case Nil{} +h <> +t:
      Empty.absurd({Nat.is_le(Spec.mass(Spec.dot(Nil{}, h <> t)), Nat.mul(Spec.len(h <> t), Nat.mul(A, B))) == True{} : Bool},
        Ar.ar_false_true(ok))
    case +h <> +t Nil{}:
      Empty.absurd({Nat.is_le(Spec.mass(Spec.dot(h <> t, Nil{})), Nat.mul(Spec.len(Nil{}), Nat.mul(A, B))) == True{} : Bool},
        Ar.ar_false_true(ok))
    case +a <> +qs +b <> +ks:
      +hb_all = Ar.ar_and_r(Nat.is_le(Spec.mass(a), A), Bool.and(Nat.is_le(Spec.mass(b), B), Spec.zip_ok(qs, ks, A, B)), ok)
      +ha = Ar.ar_and_l(Nat.is_le(Spec.mass(a), A), Bool.and(Nat.is_le(Spec.mass(b), B), Spec.zip_ok(qs, ks, A, B)), ok)
      +hb = Ar.ar_and_l(Nat.is_le(Spec.mass(b), B), Spec.zip_ok(qs, ks, A, B), hb_all)
      +hr = Ar.ar_and_r(Nat.is_le(Spec.mass(b), B), Spec.zip_ok(qs, ks, A, B), hb_all)
      +AB = Nat.mul(A, B)
      pi_le_rw(Nat.add(Spec.mass(Spec.zmul(a, b)), Spec.mass(Spec.dot(qs, ks))), Spec.mass(Spec.zadd(Spec.zmul(a, b), Spec.dot(qs, ks))),
        Nat.add(AB, Nat.mul(Spec.len(ks), AB)), Nat.add(AB, Nat.mul(Spec.len(ks), AB)),
        Equal.sym(Nat, Spec.mass(Spec.zadd(Spec.zmul(a, b), Spec.dot(qs, ks))), Nat.add(Spec.mass(Spec.zmul(a, b)), Spec.mass(Spec.dot(qs, ks))),
          pi_mass_add(Spec.zmul(a, b), Spec.dot(qs, ks))), {==},
        pi_le_add2(Spec.mass(Spec.zmul(a, b)), AB, Spec.mass(Spec.dot(qs, ks)), Nat.mul(Spec.len(ks), AB),
          pi_mass_mul_le(a, b, A, B, ha, hb), pi_dot_mass(qs, ks, A, B, hr)))

def pi_zadd_zero_l(x: Spec.Z) -> {Spec.zadd(Spec.Z{0n, 0n}, x) == x : Spec.Z}:
  match x:
    case Spec.Z{p, n}:
      {==}

def pi_zadd_zero_r(x: Spec.Z) -> {Spec.zadd(x, Spec.Z{0n, 0n}) == x : Spec.Z}:
  match x:
    case Spec.Z{+p, +n}:
      %Equal.sym(Nat, Nat.add(p, 0n), p, Ar.ar_add_zero_r(p)) : {Spec.Z{_, Nat.add(n, 0n)} == Spec.Z{p, n} : Spec.Z}
      %Equal.sym(Nat, Nat.add(n, 0n), n, Ar.ar_add_zero_r(n)) : {Spec.Z{p, _} == Spec.Z{p, n} : Spec.Z}
      {==}

def pi_zadd_assoc(x: Spec.Z, y: Spec.Z, z: Spec.Z) -> {Spec.zadd(Spec.zadd(x, y), z) == Spec.zadd(x, Spec.zadd(y, z)) : Spec.Z}:
  match x y z:
    case Spec.Z{+a, +b} Spec.Z{+c, +d} Spec.Z{+e, +f}:
      %Equal.sym(Nat, Nat.add(Nat.add(a, c), e), Nat.add(a, Nat.add(c, e)), Ar.ar_add_assoc(a, c, e))
        : {Spec.Z{_, Nat.add(Nat.add(b, d), f)} == Spec.Z{Nat.add(a, Nat.add(c, e)), Nat.add(b, Nat.add(d, f))} : Spec.Z}
      %Equal.sym(Nat, Nat.add(Nat.add(b, d), f), Nat.add(b, Nat.add(d, f)), Ar.ar_add_assoc(b, d, f))
        : {Spec.Z{Nat.add(a, Nat.add(c, e)), _} == Spec.Z{Nat.add(a, Nat.add(c, e)), Nat.add(b, Nat.add(d, f))} : Spec.Z}
      {==}

# dot over concatenated equal-length pairs
def pi_dot_app(q: List<&2, Spec.Z>, k: List<&2, Spec.Z>, +Q: List<&2, Spec.Z>, +K: List<&2, Spec.Z>, +A: Nat, +B: Nat,
    ok: {Spec.zip_ok(q, k, A, B) == True{} : Bool})
  -> {Spec.dot(List.append(&2, Spec.Z, q, Q), List.append(&2, Spec.Z, k, K)) == Spec.zadd(Spec.dot(q, k), Spec.dot(Q, K)) : Spec.Z}:
  match q k:
    case Nil{} Nil{}:
      Equal.sym(Spec.Z, Spec.zadd(Spec.Z{0n, 0n}, Spec.dot(Q, K)), Spec.dot(Q, K), pi_zadd_zero_l(Spec.dot(Q, K)))
    case Nil{} +h <> +t:
      Empty.absurd({Spec.dot(List.append(&2, Spec.Z, Nil{}, Q), List.append(&2, Spec.Z, h <> t, K)) ==
        Spec.zadd(Spec.dot(Nil{}, h <> t), Spec.dot(Q, K)) : Spec.Z}, Ar.ar_false_true(ok))
    case +h <> +t Nil{}:
      Empty.absurd({Spec.dot(List.append(&2, Spec.Z, h <> t, Q), List.append(&2, Spec.Z, Nil{}, K)) ==
        Spec.zadd(Spec.dot(h <> t, Nil{}), Spec.dot(Q, K)) : Spec.Z}, Ar.ar_false_true(ok))
    case +a <> +qs +b <> +ks:
      +hr = Ar.ar_and_r(Nat.is_le(Spec.mass(b), B), Spec.zip_ok(qs, ks, A, B),
        Ar.ar_and_r(Nat.is_le(Spec.mass(a), A), Bool.and(Nat.is_le(Spec.mass(b), B), Spec.zip_ok(qs, ks, A, B)), ok))
      +R = Spec.dot(List.append(&2, Spec.Z, qs, Q), List.append(&2, Spec.Z, ks, K))
      %Equal.sym(Spec.Z, R, Spec.zadd(Spec.dot(qs, ks), Spec.dot(Q, K)), pi_dot_app(qs, ks, Q, K, A, B, hr))
        : {Spec.zadd(Spec.zmul(a, b), _) == Spec.zadd(Spec.zadd(Spec.zmul(a, b), Spec.dot(qs, ks)), Spec.dot(Q, K)) : Spec.Z}
      Equal.sym(Spec.Z, Spec.zadd(Spec.zadd(Spec.zmul(a, b), Spec.dot(qs, ks)), Spec.dot(Q, K)),
        Spec.zadd(Spec.zmul(a, b), Spec.zadd(Spec.dot(qs, ks), Spec.dot(Q, K))), pi_zadd_assoc(Spec.zmul(a, b), Spec.dot(qs, ks), Spec.dot(Q, K)))

# 32 bounded pairs: mass(dot) <= B for B == 32 * (127 * 126), generic in B so the bound is never evaluated
def pi_chunk_mass(+q: List<&2, Spec.Z>, +k: List<&2, Spec.Z>, +B: Nat, hB: {Nat.mul(32n, Nat.mul(127n, 126n)) == B : Nat},
    ok: {Spec.zip_ok(q, k, 127n, 126n) == True{} : Bool},
    n32: {Nat.is_eq(Spec.len(k), 32n) == True{} : Bool}) -> {Nat.is_le(Spec.mass(Spec.dot(q, k)), B) == True{} : Bool}:
  pi_le_rw(Spec.mass(Spec.dot(q, k)), Spec.mass(Spec.dot(q, k)), Nat.mul(Spec.len(k), Nat.mul(127n, 126n)), B, {==},
    Equal.trans(Nat, Nat.mul(Spec.len(k), Nat.mul(127n, 126n)), Nat.mul(32n, Nat.mul(127n, 126n)), B,
      Equal.cong(Nat, Nat, z => Nat.mul(z, Nat.mul(127n, 126n)), Spec.len(k), 32n, Ar.ar_eq_true(Spec.len(k), 32n, n32)), hB),
    pi_dot_mass(q, k, 127n, 126n, ok))

def pi_flat_mass(ch: List<&2, Spec.Chunk>, +B: Nat, +hB: {Nat.mul(32n, Nat.mul(127n, 126n)) == B : Nat},
    +ok: {Spec.chunks_ok(ch, 127n, 126n) == True{} : Bool})
  -> {Nat.is_le(Spec.mass(Spec.dot(Spec.flat_q(ch), Spec.flat_k(ch))), Nat.mul(Spec.chunks_len(ch), B)) == True{} : Bool}:
  match ch:
    case Nil{}:
      {==}
    case Spec.Chunk{+q, +k} <> +t:
      +b1 = Nat.is_eq(Spec.len(k), 32n)
      +b2 = Spec.zip_ok(q, k, 127n, 126n)
      +b3 = Spec.chunks_ok(t, 127n, 126n)
      +n32 = Ar.ar_and_l(b1, Bool.and(b2, b3), ok)
      +zok = Ar.ar_and_l(b2, b3, Ar.ar_and_r(b1, Bool.and(b2, b3), ok))
      +tok = Ar.ar_and_r(b2, b3, Ar.ar_and_r(b1, Bool.and(b2, b3), ok))
      +d = Spec.dot(q, k)
      +R = Spec.dot(Spec.flat_q(t), Spec.flat_k(t))
      %Equal.sym(Spec.Z, Spec.dot(List.append(&2, Spec.Z, q, Spec.flat_q(t)), List.append(&2, Spec.Z, k, Spec.flat_k(t))), Spec.zadd(d, R),
        pi_dot_app(q, k, Spec.flat_q(t), Spec.flat_k(t), 127n, 126n, zok))
        : {Nat.is_le(Spec.mass(_), Nat.mul(1n+Spec.chunks_len(t), B)) == True{} : Bool}
      %Equal.sym(Nat, Spec.mass(Spec.zadd(d, R)), Nat.add(Spec.mass(d), Spec.mass(R)), pi_mass_add(d, R))
        : {Nat.is_le(_, Nat.mul(1n+Spec.chunks_len(t), B)) == True{} : Bool}
      pi_le_add2(Spec.mass(d), B, Spec.mass(R), Nat.mul(Spec.chunks_len(t), B), pi_chunk_mass(q, k, B, hB, zok, n32),
        pi_flat_mass(t, B, hB, tok))

def pi_mma_sum(a: List<&2, Spec.Z>, b: List<&2, Spec.Z>) -> {Impl.mma_sum(a, b) == Spec.dot(a, b) : Spec.Z}:
  match a b:
    case Nil{} Nil{}:
      {==}
    case Nil{} h <> t:
      {==}
    case h <> t Nil{}:
      {==}
    case +x <> +xs +y <> +ys:
      %pi_mma_sum(xs, ys) : {Spec.zadd(Spec.zmul(x, y), Impl.mma_sum(xs, ys)) == Spec.zadd(Spec.zmul(x, y), _) : Spec.Z}
      {==}

# value equality: transitivity, from equality, adding the same term
def pi_trans_nat(+xp: Nat, +xn: Nat, +yp: Nat, +yn: Nat, +zp: Nat, +zn: Nat,
    h1: {Nat.add(xp, yn) == Nat.add(yp, xn) : Nat}, h2: {Nat.add(yp, zn) == Nat.add(zp, yn) : Nat})
  -> {Nat.add(xp, zn) == Nat.add(zp, xn) : Nat}:
  +s = Equal.trans(Nat, Nat.add(Nat.add(xp, yn), Nat.add(yp, zn)), Nat.add(Nat.add(yp, xn), Nat.add(yp, zn)),
    Nat.add(Nat.add(yp, xn), Nat.add(zp, yn)),
    Equal.cong(Nat, Nat, w => Nat.add(w, Nat.add(yp, zn)), Nat.add(xp, yn), Nat.add(yp, xn), h1),
    Equal.cong(Nat, Nat, w => Nat.add(Nat.add(yp, xn), w), Nat.add(yp, zn), Nat.add(zp, yn), h2))
  +eL = Equal.trans(Nat, Nat.add(Nat.add(xp, yn), Nat.add(yp, zn)), Nat.add(Nat.add(yn, xp), Nat.add(yp, zn)),
    Nat.add(Nat.add(yn, yp), Nat.add(xp, zn)),
    Equal.cong(Nat, Nat, w => Nat.add(w, Nat.add(yp, zn)), Nat.add(xp, yn), Nat.add(yn, xp), Ar.ar_add_comm(xp, yn)),
    pi_swap4(yn, xp, yp, zn))
  +eR = Equal.trans(Nat, Nat.add(Nat.add(yp, xn), Nat.add(zp, yn)), Nat.add(Nat.add(zp, yn), Nat.add(yp, xn)),
    Nat.add(Nat.add(yn, yp), Nat.add(zp, xn)),
    Ar.ar_add_comm(Nat.add(yp, xn), Nat.add(zp, yn)),
    Equal.trans(Nat, Nat.add(Nat.add(zp, yn), Nat.add(yp, xn)), Nat.add(Nat.add(yn, zp), Nat.add(yp, xn)),
      Nat.add(Nat.add(yn, yp), Nat.add(zp, xn)),
      Equal.cong(Nat, Nat, w => Nat.add(w, Nat.add(yp, xn)), Nat.add(zp, yn), Nat.add(yn, zp), Ar.ar_add_comm(zp, yn)),
      pi_swap4(yn, zp, yp, xn)))
  pi_add_cancel(Nat.add(yn, yp), Nat.add(xp, zn), Nat.add(zp, xn),
    Equal.trans(Nat, Nat.add(Nat.add(yn, yp), Nat.add(xp, zn)), Nat.add(Nat.add(xp, yn), Nat.add(yp, zn)),
      Nat.add(Nat.add(yn, yp), Nat.add(zp, xn)),
      Equal.sym(Nat, Nat.add(Nat.add(xp, yn), Nat.add(yp, zn)), Nat.add(Nat.add(yn, yp), Nat.add(xp, zn)), eL),
      Equal.trans(Nat, Nat.add(Nat.add(xp, yn), Nat.add(yp, zn)), Nat.add(Nat.add(yp, xn), Nat.add(zp, yn)),
        Nat.add(Nat.add(yn, yp), Nat.add(zp, xn)), s, eR)))

def pi_zeq_trans(+x: Spec.Z, +y: Spec.Z, +z: Spec.Z, h1: Spec.zeq(x, y), h2: Spec.zeq(y, z)) -> Spec.zeq(x, z):
  pi_trans_nat(Spec.zp(x), Spec.zn(x), Spec.zp(y), Spec.zn(y), Spec.zp(z), Spec.zn(z), h1, h2)

def pi_zeq_of_eq(+x: Spec.Z, -y: Spec.Z, e: {x == y : Spec.Z}) -> Spec.zeq(x, y):
  %e : Spec.zeq(x, _)
  {==}

def pi_zeq_add_l(acc: Spec.Z, D: Spec.Z, d: Spec.Z, h: Spec.zeq(acc, D)) -> Spec.zeq(Spec.zadd(acc, d), Spec.zadd(D, d)):
  match acc D d:
    case Spec.Z{+a, +b} Spec.Z{+c, +e} Spec.Z{+f, +g}:
      Equal.trans(Nat, Nat.add(Nat.add(a, f), Nat.add(e, g)), Nat.add(Nat.add(a, e), Nat.add(f, g)), Nat.add(Nat.add(c, f), Nat.add(b, g)),
        pi_swap4(a, f, e, g),
        Equal.trans(Nat, Nat.add(Nat.add(a, e), Nat.add(f, g)), Nat.add(Nat.add(c, b), Nat.add(f, g)), Nat.add(Nat.add(c, f), Nat.add(b, g)),
          Equal.cong(Nat, Nat, w => Nat.add(w, Nat.add(f, g)), Nat.add(a, e), Nat.add(c, b), h),
          Equal.sym(Nat, Nat.add(Nat.add(c, f), Nat.add(b, g)), Nat.add(Nat.add(c, b), Nat.add(f, g)), pi_swap4(c, f, b, g))))

def pi_abs_inv_z(x: Spec.Z, y: Spec.Z, h: Spec.zeq(x, y)) -> {Spec.abs(x) == Spec.abs(y) : Nat}:
  match x y:
    case Spec.Z{+a, +b} Spec.Z{+c, +d}:
      pi_abs_inv(a, b, c, d, h)

def pi_abs_le_mass_z(x: Spec.Z) -> {Nat.is_le(Spec.abs(x), Spec.mass(x)) == True{} : Bool}:
  match x:
    case Spec.Z{+p, +n}:
      pi_abs_le_mass(p, n)

def pi_wrap_z(z: Spec.Z, +m: Nat, small: {Nat.is_lt(Spec.abs(z), 1n+m) == True{} : Bool}) -> Spec.zeq(Impl.wrap(z, m), z):
  match z:
    case Spec.Z{+p, +n}:
      pi_wrap.go(Nat.is_le(n, p), p, n, m, {==}, small)

# one IMMA from accumulator acc is acc + the exact partial when that sum is below 2^31 in magnitude
def pi_mma_wrap(+acc: Spec.Z, +q: List<&2, Spec.Z>, +k: List<&2, Spec.Z>, +m: Nat,
    small: {Nat.is_lt(Spec.abs(Spec.zadd(acc, Spec.dot(q, k))), 1n+m) == True{} : Bool})
  -> Spec.zeq(Impl.mma(acc, q, k, m), Spec.zadd(acc, Spec.dot(q, k))):
  %Equal.sym(Spec.Z, Impl.mma_sum(q, k), Spec.dot(q, k), pi_mma_sum(q, k))
    : Spec.zeq(Impl.wrap(Spec.zadd(acc, _), m), Spec.zadd(acc, Spec.dot(q, k)))
  pi_wrap_z(Spec.zadd(acc, Spec.dot(q, k)), m, small)

# zeq(x, y) and y == y2 give zeq(x, y2)
def pi_zeq_eq_r(+x: Spec.Z, -y: Spec.Z, -y2: Spec.Z, e: {y == y2 : Spec.Z}, h: Spec.zeq(x, y)) -> Spec.zeq(x, y2):
  %e : Spec.zeq(x, _)
  h

# The chain from accumulator acc ~ D: as long as every prefix sum stays within m, the int32 chain is D plus the exact
# dot product over the covered dims.
def pi_chain(ch: List<&2, Spec.Chunk>, +acc: Spec.Z, +D: Spec.Z, +m: Nat, +B: Nat, +hB: {Nat.mul(32n, Nat.mul(127n, 126n)) == B : Nat},
    +ok: {Spec.chunks_ok(ch, 127n, 126n) == True{} : Bool},
    inv: Spec.zeq(acc, D),
    +room: {Nat.is_le(Nat.add(Spec.mass(D), Nat.mul(Spec.chunks_len(ch), B)), m) == True{} : Bool})
  -> Spec.zeq(Impl.chain(ch, acc, m), Spec.zadd(D, Spec.dot(Spec.flat_q(ch), Spec.flat_k(ch)))):
  match ch:
    case Nil{}:
      %Equal.sym(Spec.Z, Spec.zadd(D, Spec.Z{0n, 0n}), D, pi_zadd_zero_r(D)) : Spec.zeq(acc, _)
      inv
    case Spec.Chunk{+q, +k} <> +t:
      +b1 = Nat.is_eq(Spec.len(k), 32n)
      +b2 = Spec.zip_ok(q, k, 127n, 126n)
      +b3 = Spec.chunks_ok(t, 127n, 126n)
      +n32 = Ar.ar_and_l(b1, Bool.and(b2, b3), ok)
      +zok = Ar.ar_and_l(b2, b3, Ar.ar_and_r(b1, Bool.and(b2, b3), ok))
      +tok = Ar.ar_and_r(b2, b3, Ar.ar_and_r(b1, Bool.and(b2, b3), ok))
      +d = Spec.dot(q, k)
      +M : Nat = B
      +L = Spec.chunks_len(t)
      +LM = Nat.mul(L, M)
      +mD = Spec.mass(D)
      +md = Spec.mass(d)
      +Dp = Spec.zadd(D, d)
      +eA : {Nat.add(Spec.zp(Spec.zadd(acc, d)), Spec.zn(Dp)) == Nat.add(Spec.zp(Dp), Spec.zn(Spec.zadd(acc, d))) : Nat} =
        pi_zeq_add_l(acc, D, d, inv)
      +le1 = Ar.ard_tr(Nat.is_le(Nat.add(mD, md), Nat.add(mD, M)), Nat.is_le(md, M), Ar.ar_le_add_cancel(mD, md, M),
        pi_chunk_mass(q, k, B, hB, zok, n32))
      +le2 = Ar.ard_tr(Nat.is_le(Nat.add(mD, M), Nat.add(mD, Nat.add(M, LM))), Nat.is_le(M, Nat.add(M, LM)),
        Ar.ar_le_add_cancel(mD, M, Nat.add(M, LM)), Ar.ar_le_add_r(M, LM))
      +bm = pi_le_rw(Nat.add(mD, md), Spec.mass(Dp), m, m, Equal.sym(Nat, Spec.mass(Dp), Nat.add(mD, md), pi_mass_add(D, d)), {==},
        Ar.ar_le_trans(Nat.add(mD, md), Nat.add(mD, M), m, le1, Ar.ar_le_trans(Nat.add(mD, M), Nat.add(mD, Nat.add(M, LM)), m, le2, room)))
      +ba = pi_le_rw(Spec.abs(Dp), Spec.abs(Spec.zadd(acc, d)), m, m,
        Equal.sym(Nat, Spec.abs(Spec.zadd(acc, d)), Spec.abs(Dp), pi_abs_inv_z(Spec.zadd(acc, d), Dp, eA)), {==},
        Ar.ar_le_trans(Spec.abs(Dp), Spec.mass(Dp), m, pi_abs_le_mass_z(Dp), bm))
      +w : {Nat.add(Spec.zp(Impl.mma(acc, q, k, m)), Spec.zn(Spec.zadd(acc, d))) == Nat.add(Spec.zp(Spec.zadd(acc, d)), Spec.zn(Impl.mma(acc, q, k, m))) : Nat} =
        pi_mma_wrap(acc, q, k, m, pi_lt_succ_of_le(Spec.abs(Spec.zadd(acc, d)), m, ba))
      +e1 : {Nat.add(Spec.zp(Impl.mma(acc, q, k, m)), Spec.zn(Dp)) == Nat.add(Spec.zp(Dp), Spec.zn(Impl.mma(acc, q, k, m))) : Nat} =
        pi_zeq_trans(Impl.mma(acc, q, k, m), Spec.zadd(acc, d), Dp, w, eA)
      +room2 = pi_le_rw(Nat.add(Nat.add(mD, md), LM), Nat.add(Spec.mass(Dp), LM), m, m,
        Equal.cong(Nat, Nat, z => Nat.add(z, LM), Nat.add(mD, md), Spec.mass(Dp),
          Equal.sym(Nat, Spec.mass(Dp), Nat.add(mD, md), pi_mass_add(D, d))), {==},
        Ar.ar_le_trans(Nat.add(Nat.add(mD, md), LM), Nat.add(Nat.add(mD, M), LM), m,
          pi_le_add2(Nat.add(mD, md), Nat.add(mD, M), LM, LM, le1, Ar.ar_le_refl(LM)),
          pi_le_rw(Nat.add(mD, Nat.add(M, LM)), Nat.add(Nat.add(mD, M), LM), m, m,
            Equal.sym(Nat, Nat.add(Nat.add(mD, M), LM), Nat.add(mD, Nat.add(M, LM)), Ar.ar_add_assoc(mD, M, LM)), {==}, room)))
      +R = Spec.dot(Spec.flat_q(t), Spec.flat_k(t))
      %Equal.sym(Spec.Z, Spec.dot(List.append(&2, Spec.Z, q, Spec.flat_q(t)), List.append(&2, Spec.Z, k, Spec.flat_k(t))), Spec.zadd(d, R),
        pi_dot_app(q, k, Spec.flat_q(t), Spec.flat_k(t), 127n, 126n, zok))
        : Spec.zeq(Impl.chain(t, Impl.mma(acc, q, k, m), m), Spec.zadd(D, _))
      %pi_zadd_assoc(D, d, R) : Spec.zeq(Impl.chain(t, Impl.mma(acc, q, k, m), m), _)
      pi_chain(t, Impl.mma(acc, q, k, m), Dp, m, B, hB, tok, e1, room2)

def pi_chain_p1(+ch: List<&2, Spec.Chunk>, +m: Nat, +B: Nat, +hB: {Nat.mul(32n, Nat.mul(127n, 126n)) == B : Nat},
    +ok: {Spec.chunks_ok(ch, 127n, 126n) == True{} : Bool},
    room: {Nat.is_le(Nat.mul(Spec.chunks_len(ch), B), m) == True{} : Bool})
  -> Spec.zeq(Impl.chain(ch, Spec.Z{0n, 0n}, m), Spec.dot(Spec.flat_q(ch), Spec.flat_k(ch))):
  +X = Spec.dot(Spec.flat_q(ch), Spec.flat_k(ch))
  pi_zeq_eq_r(Impl.chain(ch, Spec.Z{0n, 0n}, m), Spec.zadd(Spec.Z{0n, 0n}, X), X, pi_zadd_zero_l(X),
    pi_chain(ch, Spec.Z{0n, 0n}, Spec.Z{0n, 0n}, m, B, hB, ok, {==}, room))

def pi_chain_p2(+ch: List<&2, Spec.Chunk>, +m: Nat, +B: Nat, +hB: {Nat.mul(32n, Nat.mul(127n, 126n)) == B : Nat},
    +ok: {Spec.chunks_ok(ch, 127n, 126n) == True{} : Bool},
    room: {Nat.is_le(Nat.mul(Spec.chunks_len(ch), B), m) == True{} : Bool})
  -> {Nat.is_le(Spec.abs(Impl.chain(ch, Spec.Z{0n, 0n}, m)), Nat.mul(Spec.chunks_len(ch), B)) == True{} : Bool}:
  +X = Spec.dot(Spec.flat_q(ch), Spec.flat_k(ch))
  +C = Impl.chain(ch, Spec.Z{0n, 0n}, m)
  +LM = Nat.mul(Spec.chunks_len(ch), B)
  pi_le_rw(Spec.abs(X), Spec.abs(C), LM, LM, Equal.sym(Nat, Spec.abs(C), Spec.abs(X), pi_abs_inv_z(C, X, pi_chain_p1(ch, m, B, hB, ok, room))), {==},
    Ar.ar_le_trans(Spec.abs(X), Spec.mass(X), LM, pi_abs_le_mass_z(X), pi_flat_mass(ch, B, hB, ok)))

# chunks_len <= 8 gives len * P <= INT32_MAX
def pi_room0(+ch: List<&2, Spec.Chunk>, le8: {Nat.is_le(Spec.chunks_len(ch), 8n) == True{} : Bool})
  -> {Nat.is_le(Nat.mul(Spec.chunks_len(ch), Nat.mul(32n, Nat.mul(127n, 126n))), Nat.sub(Nat.mul(32768n, 65536n), 1n)) == True{} : Bool}:
  +P = Nat.mul(32n, Nat.mul(127n, 126n))
  Ar.ar_le_trans(Nat.mul(Spec.chunks_len(ch), P), Nat.mul(8n, P), Nat.sub(Nat.mul(32768n, 65536n), 1n),
    Ar.ar_le_mul_mono(Spec.chunks_len(ch), 8n, P, le8), pi_le_max(Nat.mul(8n, P), pi_8p_lt31()))

def pi_abs_dot_le(+q: List<&2, Spec.Z>, +k: List<&2, Spec.Z>, +B: Nat, hB: {Nat.mul(32n, Nat.mul(127n, 126n)) == B : Nat},
    ok: {Spec.zip_ok(q, k, 127n, 126n) == True{} : Bool},
    n32: {Nat.is_eq(Spec.len(k), 32n) == True{} : Bool}) -> {Nat.is_le(Spec.abs(Spec.dot(q, k)), B) == True{} : Bool}:
  Ar.ar_le_trans(Spec.abs(Spec.dot(q, k)), Spec.mass(Spec.dot(q, k)), B, pi_abs_le_mass_z(Spec.dot(q, k)), pi_chunk_mass(q, k, B, hB, ok, n32))

def pi_partial(+q: List<&2, Spec.Z>, +k: List<&2, Spec.Z>, +m: Nat, +B: Nat, hB: {Nat.mul(32n, Nat.mul(127n, 126n)) == B : Nat},
    ok: {Spec.zip_ok(q, k, 127n, 126n) == True{} : Bool},
    n32: {Nat.is_eq(Spec.len(k), 32n) == True{} : Bool}, big: {Nat.is_le(B, m) == True{} : Bool})
  -> Spec.zeq(Impl.mma(Spec.Z{0n, 0n}, q, k, m), Spec.dot(q, k)):
  +d = Spec.dot(q, k)
  +ba = pi_le_rw(Spec.abs(d), Spec.abs(Spec.zadd(Spec.Z{0n, 0n}, d)), m, m,
    Equal.cong(Spec.Z, Nat, z => Spec.abs(z), d, Spec.zadd(Spec.Z{0n, 0n}, d), Equal.sym(Spec.Z, Spec.zadd(Spec.Z{0n, 0n}, d), d, pi_zadd_zero_l(d))),
    {==}, Ar.ar_le_trans(Spec.abs(d), B, m, pi_abs_dot_le(q, k, B, hB, ok, n32), big))
  pi_zeq_eq_r(Impl.mma(Spec.Z{0n, 0n}, q, k, m), Spec.zadd(Spec.Z{0n, 0n}, d), d, pi_zadd_zero_l(d),
    pi_mma_wrap(Spec.Z{0n, 0n}, q, k, m, pi_lt_succ_of_le(Spec.abs(Spec.zadd(Spec.Z{0n, 0n}, d)), m, ba)))

def Laws.imma_partial(q, k, ok, n32):
  +P = Nat.mul(32n, Nat.mul(127n, 126n))
  +M = Nat.sub(Nat.mul(32768n, 65536n), 1n)
  (pi_abs_dot_le(q, k, P, pi_refl(P), ok, n32), {==}, pi_p_lt24(),
   pi_partial(q, k, M, P, pi_refl(P), ok, n32, pi_le_max(P, pi_p_lt31())))

def Laws.imma_chain_exact(ch, ok, le8):
  +P = Nat.mul(32n, Nat.mul(127n, 126n))
  +M = Nat.sub(Nat.mul(32768n, 65536n), 1n)
  (pi_chain_p1(ch, M, P, pi_refl(P), ok, pi_room0(ch, le8)),
   pi_chain_p2(ch, M, P, pi_refl(P), ok, pi_room0(ch, le8)))

def Laws.chain_bound_2_22():
  pi_chain_lt22()

def Laws.bound_2_22_lt_2_31():
  pi_22_lt_31()

# ---------------------------------------------------------------------------------------------
# I. i2f_exact (laws i2f_exact_range, i2f_at_2_22), generic in h = 2^22 (T = h + h = 2^23)

# the pattern of x in [-h, h) is r + 150 T with r = x + h < T
def pi_bits_eq(+xp: Nat, +xn: Nat, +h: Nat, lo: {Nat.is_le(xn, Nat.add(xp, h)) == True{} : Bool})
  -> {Impl.bits(Spec.Z{xp, xn}, h) == Nat.add(Nat.sub(Nat.add(xp, h), xn), Nat.mul(150n, Nat.add(h, h))) : Nat}:
  +M = Nat.mul(150n, Nat.add(h, h))
  Equal.trans(Nat, Nat.sub(Nat.add(xp, Nat.add(h, M)), xn), Nat.sub(Nat.add(Nat.add(xp, h), M), xn),
    Nat.add(Nat.sub(Nat.add(xp, h), xn), M),
    Equal.cong(Nat, Nat, z => Nat.sub(z, xn), Nat.add(xp, Nat.add(h, M)), Nat.add(Nat.add(xp, h), M),
      Equal.sym(Nat, Nat.add(Nat.add(xp, h), M), Nat.add(xp, Nat.add(h, M)), Ar.ar_add_assoc(xp, h, M))),
    pi_sub_add_l(Nat.add(xp, h), M, xn, lo))

def pi_r_lt(+xp: Nat, +xn: Nat, +h: Nat, +hpos: {Nat.is_lt(0n, h) == True{} : Bool},
    hi: {Nat.is_lt(xp, Nat.add(xn, h)) == True{} : Bool})
  -> {Nat.is_lt(Nat.sub(Nat.add(xp, h), xn), Nat.add(h, h)) == True{} : Bool}:
  +T = Nat.add(h, h)
  +a1 = pi_lt_rw(Nat.add(h, xp), Nat.add(xp, h), Nat.add(h, Nat.add(xn, h)), Nat.add(xn, T),
    Ar.ar_add_comm(h, xp),
    Equal.trans(Nat, Nat.add(h, Nat.add(xn, h)), Nat.add(Nat.add(xn, h), h), Nat.add(xn, T), Ar.ar_add_comm(h, Nat.add(xn, h)),
      Ar.ar_add_assoc(xn, h, h)),
    Ar.ard_tr(Nat.is_lt(Nat.add(h, xp), Nat.add(h, Nat.add(xn, h))), Nat.is_lt(xp, Nat.add(xn, h)), Ar.ar_lt_add_cancel(h, xp, Nat.add(xn, h)), hi))
  Ar.ard_tr(Nat.is_lt(Nat.sub(Nat.add(xp, h), xn), T), Nat.is_lt(Nat.add(xp, h), Nat.add(xn, T)),
    Ar.ar_sub_lt_iff(Nat.add(xp, h), xn, T, Ar.ar_lt_le_trans(0n, h, T, hpos, Ar.ar_le_add_r(h, h))), a1)

def pi_div_bits(+xp: Nat, +xn: Nat, +h: Nat, +hpos: {Nat.is_lt(0n, h) == True{} : Bool},
    +lo: {Nat.is_le(xn, Nat.add(xp, h)) == True{} : Bool}, +hi: {Nat.is_lt(xp, Nat.add(xn, h)) == True{} : Bool})
  -> {Nat.div(Impl.bits(Spec.Z{xp, xn}, h), Nat.add(h, h)) == 150n : Nat}:
  %Equal.sym(Nat, Impl.bits(Spec.Z{xp, xn}, h), Nat.add(Nat.sub(Nat.add(xp, h), xn), Nat.mul(150n, Nat.add(h, h))), pi_bits_eq(xp, xn, h, lo))
    : {Nat.div(_, Nat.add(h, h)) == 150n : Nat}
  pi_div_r(150n, Nat.sub(Nat.add(xp, h), xn), Nat.add(h, h), pi_r_lt(xp, xn, h, hpos, hi))

def pi_mod_bits(+xp: Nat, +xn: Nat, +h: Nat, +hpos: {Nat.is_lt(0n, h) == True{} : Bool},
    +lo: {Nat.is_le(xn, Nat.add(xp, h)) == True{} : Bool}, +hi: {Nat.is_lt(xp, Nat.add(xn, h)) == True{} : Bool})
  -> {Nat.mod(Impl.bits(Spec.Z{xp, xn}, h), Nat.add(h, h)) == Nat.sub(Nat.add(xp, h), xn) : Nat}:
  %Equal.sym(Nat, Impl.bits(Spec.Z{xp, xn}, h), Nat.add(Nat.sub(Nat.add(xp, h), xn), Nat.mul(150n, Nat.add(h, h))), pi_bits_eq(xp, xn, h, lo))
    : {Nat.mod(_, Nat.add(h, h)) == Nat.sub(Nat.add(xp, h), xn) : Nat}
  pi_mod_r(150n, Nat.sub(Nat.add(xp, h), xn), Nat.add(h, h), pi_r_lt(xp, xn, h, hpos, hi))

def pi_exp150(+xp: Nat, +xn: Nat, +h: Nat, +hpos: {Nat.is_lt(0n, h) == True{} : Bool},
    +lo: {Nat.is_le(xn, Nat.add(xp, h)) == True{} : Bool}, +hi: {Nat.is_lt(xp, Nat.add(xn, h)) == True{} : Bool})
  -> {Impl.exp_field(Impl.bits(Spec.Z{xp, xn}, h), h) == 150n : Nat}:
  %Equal.sym(Nat, Nat.div(Impl.bits(Spec.Z{xp, xn}, h), Nat.add(h, h)), 150n, pi_div_bits(xp, xn, h, hpos, lo, hi))
    : {Nat.mod(_, 256n) == 150n : Nat}
  {==}

# RN of the representable difference v - c: when v - c names x and |x| <= 2^24 the result is x
def pi_fsub(+v: Nat, +xp: Nat, +xn: Nat, +h: Nat,
    +e: {Nat.add(v, xn) == Nat.add(xp, Nat.add(h, Nat.add(h, h))) : Nat},
    bound: {Nat.is_le(Spec.abs(Spec.Z{xp, xn}), Nat.add(Nat.add(h, h), Nat.add(h, h))) == True{} : Bool})
  -> {Impl.fsub_rn_int(v, Nat.add(h, Nat.add(h, h)), h) == Some{Spec.norm(Spec.Z{xp, xn})} : Maybe<&2, Spec.Z>}:
  +C = Nat.add(h, Nat.add(h, h))
  +F = Nat.add(Nat.add(h, h), Nat.add(h, h))
  %pi_abs_inv(xp, xn, v, C, Equal.sym(Nat, Nat.add(v, xn), Nat.add(xp, C), e))
    : {Impl.pickr(Nat.is_le(_, F), Spec.norm(Spec.Z{v, C})) == Some{Spec.norm(Spec.Z{xp, xn})} : Maybe<&2, Spec.Z>}
  %Equal.sym(Bool, Nat.is_le(Spec.abs(Spec.Z{xp, xn}), F), True{}, bound)
    : {Impl.pickr(_, Spec.norm(Spec.Z{v, C})) == Some{Spec.norm(Spec.Z{xp, xn})} : Maybe<&2, Spec.Z>}
  Equal.cong(Spec.Z, Maybe<&2, Spec.Z>, z => Some{z}, Spec.norm(Spec.Z{v, C}), Spec.norm(Spec.Z{xp, xn}), pi_norm_inv(v, C, xp, xn, e))

def pi_i2f_val(+xp: Nat, +xn: Nat, +h: Nat, +lo: {Nat.is_le(xn, Nat.add(xp, h)) == True{} : Bool})
  -> {Nat.add(Nat.mul(1n, Nat.add(Nat.sub(Nat.add(xp, h), xn), Nat.add(h, h))), xn) == Nat.add(xp, Nat.add(h, Nat.add(h, h))) : Nat}:
  +r = Nat.sub(Nat.add(xp, h), xn)
  +T = Nat.add(h, h)
  Equal.trans(Nat, Nat.add(Nat.mul(1n, Nat.add(r, T)), xn), Nat.add(Nat.add(r, T), xn), Nat.add(xp, Nat.add(h, T)),
    Equal.cong(Nat, Nat, z => Nat.add(z, xn), Nat.mul(1n, Nat.add(r, T)), Nat.add(r, T), Ar.ar_mul_one_l(Nat.add(r, T))),
    Equal.trans(Nat, Nat.add(Nat.add(r, T), xn), Nat.add(Nat.add(r, xn), T), Nat.add(xp, Nat.add(h, T)),
      Equal.trans(Nat, Nat.add(Nat.add(r, T), xn), Nat.add(r, Nat.add(T, xn)), Nat.add(Nat.add(r, xn), T), Ar.ar_add_assoc(r, T, xn),
        Equal.trans(Nat, Nat.add(r, Nat.add(T, xn)), Nat.add(r, Nat.add(xn, T)), Nat.add(Nat.add(r, xn), T),
          Equal.cong(Nat, Nat, z => Nat.add(r, z), Nat.add(T, xn), Nat.add(xn, T), Ar.ar_add_comm(T, xn)),
          Equal.sym(Nat, Nat.add(Nat.add(r, xn), T), Nat.add(r, Nat.add(xn, T)), Ar.ar_add_assoc(r, xn, T)))),
      Equal.trans(Nat, Nat.add(Nat.add(r, xn), T), Nat.add(Nat.add(xp, h), T), Nat.add(xp, Nat.add(h, T)),
        Equal.cong(Nat, Nat, z => Nat.add(z, T), Nat.add(r, xn), Nat.add(xp, h),
          Equal.trans(Nat, Nat.add(r, xn), Nat.add(xn, r), Nat.add(xp, h), Ar.ar_add_comm(r, xn), Ar.ar_sub_add(Nat.add(xp, h), xn, lo))),
        Ar.ar_add_assoc(xp, h, T))))

def pi_i2f_bound(+xp: Nat, +xn: Nat, +h: Nat, lo: {Nat.is_le(xn, Nat.add(xp, h)) == True{} : Bool},
    hi: {Nat.is_lt(xp, Nat.add(xn, h)) == True{} : Bool})
  -> {Nat.is_le(Spec.abs(Spec.Z{xp, xn}), Nat.add(Nat.add(h, h), Nat.add(h, h))) == True{} : Bool}:
  +T = Nat.add(h, h)
  +F = Nat.add(T, T)
  +hF = Ar.ar_le_trans(h, T, F, Ar.ar_le_add_r(h, h), Ar.ar_le_add_r(T, T))
  pi_abs_le(xp, xn, F,
    Ar.ar_le_trans(xp, Nat.add(xn, h), Nat.add(xn, F), Ar.ar_lt_le(xp, Nat.add(xn, h), hi),
      Ar.ard_tr(Nat.is_le(Nat.add(xn, h), Nat.add(xn, F)), Nat.is_le(h, F), Ar.ar_le_add_cancel(xn, h, F), hF)),
    Ar.ar_le_trans(xn, Nat.add(xp, h), Nat.add(xp, F), lo,
      Ar.ard_tr(Nat.is_le(Nat.add(xp, h), Nat.add(xp, F)), Nat.is_le(h, F), Ar.ar_le_add_cancel(xp, h, F), hF)))

def pi_i2f(+xp: Nat, +xn: Nat, +h: Nat, +hpos: {Nat.is_lt(0n, h) == True{} : Bool},
    +lo: {Nat.is_le(xn, Nat.add(xp, h)) == True{} : Bool}, +hi: {Nat.is_lt(xp, Nat.add(xn, h)) == True{} : Bool})
  -> {Impl.i2f_exact(Spec.Z{xp, xn}, h) == Some{Spec.norm(Spec.Z{xp, xn})} : Maybe<&2, Spec.Z>}:
  +u = Impl.bits(Spec.Z{xp, xn}, h)
  +T = Nat.add(h, h)
  +r = Nat.sub(Nat.add(xp, h), xn)
  +G : Maybe<&2, Spec.Z> = Some{Spec.norm(Spec.Z{xp, xn})}
  %Equal.sym(Nat, Nat.div(u, T), 150n, pi_div_bits(xp, xn, h, hpos, lo, hi))
    : {Impl.i2f_exact.go(Impl.pickm(Bool.and(Nat.is_eq(Nat.div(_, 256n), 0n), Nat.is_le(150n, Nat.mod(_, 256n))),
        Some{Nat.mul(Nat.pow(2n, Nat.sub(Nat.mod(_, 256n), 150n)), Nat.add(Nat.mod(u, T), T))}), h) == G : Maybe<&2, Spec.Z>}
  %Equal.sym(Nat, Nat.mod(u, T), r, pi_mod_bits(xp, xn, h, hpos, lo, hi))
    : {Impl.i2f_exact.go(Impl.pickm(Bool.and(Nat.is_eq(Nat.div(150n, 256n), 0n), Nat.is_le(150n, Nat.mod(150n, 256n))),
        Some{Nat.mul(Nat.pow(2n, Nat.sub(Nat.mod(150n, 256n), 150n)), Nat.add(_, T))}), h) == G : Maybe<&2, Spec.Z>}
  pi_fsub(Nat.mul(1n, Nat.add(r, T)), xp, xn, h, pi_i2f_val(xp, xn, h, lo), pi_i2f_bound(xp, xn, h, lo, hi))

def pi_frac_bits(+xp: Nat, +xn: Nat, +h: Nat, +hpos: {Nat.is_lt(0n, h) == True{} : Bool},
    +lo: {Nat.is_le(xn, Nat.add(xp, h)) == True{} : Bool}, +hi: {Nat.is_lt(xp, Nat.add(xn, h)) == True{} : Bool})
  -> {Impl.frac(Impl.bits(Spec.Z{xp, xn}, h), h) == Nat.sub(Nat.add(xp, h), xn) : Nat}:
  pi_mod_bits(xp, xn, h, hpos, lo, hi)

def Laws.i2f_exact_range(xp, xn, lo, hi):
  (pi_exp150(xp, xn, Nat.mul(2048n, 2048n), {==}, lo, hi), pi_frac_bits(xp, xn, Nat.mul(2048n, 2048n), {==}, lo, hi),
   pi_i2f(xp, xn, Nat.mul(2048n, 2048n), {==}, lo, hi))

# x = h: the pattern is 151 T
def pi_at_h(+xn: Nat, +h: Nat, +hpos: {Nat.is_lt(0n, h) == True{} : Bool})
  -> {Impl.exp_field(Impl.bits(Spec.Z{Nat.add(xn, h), xn}, h), h) == 151n : Nat}:
  +T = Nat.add(h, h)
  +M = Nat.mul(150n, T)
  +eb = Equal.trans(Nat, Nat.sub(Nat.add(Nat.add(xn, h), Nat.add(h, M)), xn), Nat.sub(Nat.add(xn, Nat.add(h, Nat.add(h, M))), xn),
    Nat.add(0n, Nat.mul(151n, T)),
    Equal.cong(Nat, Nat, z => Nat.sub(z, xn), Nat.add(Nat.add(xn, h), Nat.add(h, M)), Nat.add(xn, Nat.add(h, Nat.add(h, M))),
      Ar.ar_add_assoc(xn, h, Nat.add(h, M))),
    Equal.trans(Nat, Nat.sub(Nat.add(xn, Nat.add(h, Nat.add(h, M))), xn), Nat.add(h, Nat.add(h, M)), Nat.add(0n, Nat.mul(151n, T)),
      Ar.ar_add_sub(xn, Nat.add(h, Nat.add(h, M))),
      Equal.sym(Nat, Nat.add(T, M), Nat.add(h, Nat.add(h, M)), Ar.ar_add_assoc(h, h, M))))
  %Equal.sym(Nat, Impl.bits(Spec.Z{Nat.add(xn, h), xn}, h), Nat.add(0n, Nat.mul(151n, T)), eb) : {Nat.mod(Nat.div(_, T), 256n) == 151n : Nat}
  %Equal.sym(Nat, Nat.div(Nat.add(0n, Nat.mul(151n, T)), T), 151n, pi_div_r(151n, 0n, T, Ar.ar_lt_le_trans(0n, h, T, hpos, Ar.ar_le_add_r(h, h))))
    : {Nat.mod(_, 256n) == 151n : Nat}
  {==}

def Laws.i2f_at_2_22(xn):
  pi_at_h(xn, Nat.mul(2048n, 2048n), {==})

# ---------------------------------------------------------------------------------------------
# J. Law 9's partial bound (finding_partial_bound), without building the unary value of P

# the dot product of n copies of (a, 0) and (b, 0) is n times the product term
def pi_rep_dot(n: Nat, +a: Nat, +b: Nat)
  -> {Spec.dot(Spec.rep(n, Spec.Z{a, 0n}), Spec.rep(n, Spec.Z{b, 0n})) ==
      Spec.Z{Nat.mul(n, Nat.add(Nat.mul(a, b), 0n)), Nat.mul(n, Nat.add(Nat.mul(a, 0n), 0n))} : Spec.Z}:
  match n:
    case 0n:
      {==}
    case 1n+j:
      %Equal.sym(Spec.Z, Spec.dot(Spec.rep(j, Spec.Z{a, 0n}), Spec.rep(j, Spec.Z{b, 0n})),
          Spec.Z{Nat.mul(j, Nat.add(Nat.mul(a, b), 0n)), Nat.mul(j, Nat.add(Nat.mul(a, 0n), 0n))}, pi_rep_dot(j, a, b))
        : {Spec.zadd(Spec.zmul(Spec.Z{a, 0n}, Spec.Z{b, 0n}), _) ==
           Spec.Z{Nat.mul(1n+j, Nat.add(Nat.mul(a, b), 0n)), Nat.mul(1n+j, Nat.add(Nat.mul(a, 0n), 0n))} : Spec.Z}
      {==}

# (u, v) names p when u + 0 == p and v == 0
def pi_zeq_pos(+u: Nat, +v: Nat, +p: Nat, eu: {Nat.add(u, 0n) == p : Nat}, ev: {v == 0n : Nat})
  -> Spec.zeq(Spec.Z{u, v}, Spec.Z{p, 0n}):
  %Equal.sym(Nat, v, 0n, ev) : {Nat.add(u, 0n) == Nat.add(p, _) : Nat}
  Equal.trans(Nat, Nat.add(u, 0n), p, Nat.add(p, 0n), eu, Equal.sym(Nat, Nat.add(p, 0n), p, Ar.ar_add_zero_r(p)))

def Laws.finding_partial_bound():
  +X = Nat.mul(127n, 126n)
  +U = Nat.add(X, 0n)
  +V = Nat.add(Nat.mul(127n, 0n), 0n)
  +D = Spec.dot(Spec.rep(32n, Spec.Z{127n, 0n}), Spec.rep(32n, Spec.Z{126n, 0n}))
  +eu = Equal.trans(Nat, Nat.add(Nat.mul(32n, U), 0n), Nat.mul(32n, U), Nat.mul(32n, X), Ar.ar_add_zero_r(Nat.mul(32n, U)),
    Equal.cong(Nat, Nat, z => Nat.mul(32n, z), U, X, Ar.ar_add_zero_r(X)))
  +A = Nat.mul(127n, 7n)
  +s = Ar.ard_tr(Nat.is_lt(Nat.mul(A, 32n), Nat.mul(X, 32n)), Nat.is_lt(A, X), Ar.ar_mul_lt_cancel(A, X, 32n, {==}), {==})
  ({==},
   pi_zeq_trans(D, Spec.Z{Nat.mul(32n, U), Nat.mul(32n, V)}, Spec.Z{Nat.mul(32n, X), 0n},
     pi_zeq_of_eq(D, Spec.Z{Nat.mul(32n, U), Nat.mul(32n, V)}, pi_rep_dot(32n, 127n, 126n)),
     pi_zeq_pos(Nat.mul(32n, U), Nat.mul(32n, V), Nat.mul(32n, X), eu, {==})),
   pi_lt_rw(Nat.mul(A, 32n), Nat.mul(32n, A), Nat.mul(X, 32n), Nat.mul(32n, X), Ar.ar_mul_comm(A, 32n), Ar.ar_mul_comm(X, 32n), s))
