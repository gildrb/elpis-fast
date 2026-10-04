# Reference for the integer core of ext 3021c's int8 QK^T (pattn8i_int.bend), independent of the kernel text:
# integers as offset pairs over Nat, the exact integer dot product over the 256 head dims, the odd 3-bit code set, the
# adjacent-integer rounding relation, and the steps of a score. Laws: pattn8i_int_laws.bend.
import Base

# An integer as an offset pair: Z{p, n} is the integer p - n (many pairs name one integer; Z{p - n, n - p} is the
# canonical one, with one side 0).
type Z is Data:
  Z{p: Nat, n: Nat}

def zp(z: Z) -> Nat:
  match z:
    case Z{p, n}:
      p

def zn(z: Z) -> Nat:
  match z:
    case Z{p, n}:
      n

# x and y name the same integer: x.p - x.n == y.p - y.n
def zeq(+x: Z, +y: Z) -> Type:
  {Nat.add(zp(x), zn(y)) == Nat.add(zp(y), zn(x)) : Nat}

# |z| (truncated subtraction: one of the two terms is 0)
def abs(z: Z) -> Nat:
  match z:
    case Z{+p, +n}:
      Nat.add(Nat.sub(p, n), Nat.sub(n, p))

# the canonical pair of z (an int8 / int32 register value: sign and magnitude)
def norm(z: Z) -> Z:
  match z:
    case Z{+p, +n}:
      Z{Nat.sub(p, n), Nat.sub(n, p)}

# p + n: equals |z| on a canonical pair and bounds |z| on every pair
def mass(z: Z) -> Nat:
  match z:
    case Z{p, n}:
      Nat.add(p, n)

def zadd(a: Z, b: Z) -> Z:
  match a b:
    case Z{ap, an} Z{bp, bn}:
      Z{Nat.add(ap, bp), Nat.add(an, bn)}

def zmul(a: Z, b: Z) -> Z:
  match a b:
    case Z{+ap, +an} Z{+bp, +bn}:
      Z{Nat.add(Nat.mul(ap, bp), Nat.mul(an, bn)), Nat.add(Nat.mul(ap, bn), Nat.mul(an, bp))}

# z * c for a natural c
def zscale(z: Z, +c: Nat) -> Z:
  match z:
    case Z{p, n}:
      Z{Nat.mul(p, c), Nat.mul(n, c)}

# x < y as integers
def zlt(x: Z, y: Z) -> Bool:
  match x y:
    case Z{xp, xn} Z{yp, yn}:
      Nat.is_lt(Nat.add(xp, yn), Nat.add(yp, xn))

# Exact integer dot product sum_i q_i k_i (lists of equal length; the reference for one score's 256-dim QK^T)
def dot(q: List<&2, Z>, k: List<&2, Z>) -> Z:
  match q k:
    case Nil{} _:
      Z{0n, 0n}
    case _ Nil{}:
      Z{0n, 0n}
    case a <> qs b <> ks:
      zadd(zmul(a, b), dot(qs, ks))

# Every operand pair is bounded: |q_i| <= A, |k_i| <= B by mass (= |.| on the canonical int8 values), lengths equal.
def zip_ok(q: List<&2, Z>, k: List<&2, Z>, +A: Nat, +B: Nat) -> Bool:
  match q k:
    case Nil{} Nil{}:
      True{}
    case Nil{} _ <> _:
      False{}
    case _ <> _ Nil{}:
      False{}
    case a <> qs b <> ks:
      Bool.and(Nat.is_le(mass(a), A), Bool.and(Nat.is_le(mass(b), B), zip_ok(qs, ks, A, B)))

def len(xs: List<&2, Z>) -> Nat:
  match xs:
    case Nil{}:
      0n
    case _ <> t:
      1n+len(t)

# One k32 slice of the contraction: the Q row's and the key's int8 values over 32 consecutive head dims.
type Chunk is Data:
  Chunk{q: List<&2, Z>, k: List<&2, Z>}

# Every slice has 32 bounded operand pairs.
def chunks_ok(ch: List<&2, Chunk>, +A: Nat, +B: Nat) -> Bool:
  match ch:
    case Nil{}:
      True{}
    case Chunk{q, +k} <> t:
      Bool.and(Nat.is_eq(len(k), 32n), Bool.and(zip_ok(q, k, A, B), chunks_ok(t, A, B)))

def chunks_len(ch: List<&2, Chunk>) -> Nat:
  match ch:
    case Nil{}:
      0n
    case _ <> t:
      1n+chunks_len(t)

# The head-dim vectors the slices cover, in dim order.
def flat_q(ch: List<&2, Chunk>) -> List<&2, Z>:
  match ch:
    case Nil{}:
      Nil{}
    case Chunk{q, k} <> t:
      List.append(&2, Z, q, flat_q(t))

def flat_k(ch: List<&2, Chunk>) -> List<&2, Z>:
  match ch:
    case Nil{}:
      Nil{}
    case Chunk{q, k} <> t:
      List.append(&2, Z, k, flat_k(t))

# The odd integers -7, -5, ..., 7 (ascending; canonical pairs).
def odd_codes() -> List<&2, Z>:
  [Z{0n, 7n}, Z{0n, 5n}, Z{0n, 3n}, Z{0n, 1n}, Z{1n, 0n}, Z{3n, 0n}, Z{5n, 0n}, Z{7n, 0n}]

def zeq_b(x: Z, y: Z) -> Bool:
  match x y:
    case Z{xp, xn} Z{yp, yn}:
      Nat.is_eq(Nat.add(xp, yn), Nat.add(yp, xn))

def member(+x: Z, ys: List<&2, Z>) -> Bool:
  match ys:
    case Nil{}:
      False{}
    case y <> t:
      Bool.or(zeq_b(x, y), member(x, t))

# n copies of z
def rep(n: Nat, +z: Z) -> List<&2, Z>:
  match n:
    case 0n:
      []
    case 1n+m:
      z <> rep(m, z)

def ascending.go(t: List<&2, Z>, +x: Z) -> Bool:
  match t:
    case Nil{}:
      True{}
    case +y <> u:
      Bool.and(zlt(x, y), ascending.go(u, y))

# strictly ascending integers
def ascending(xs: List<&2, Z>) -> Bool:
  match xs:
    case Nil{}:
      True{}
    case +x <> t:
      ascending.go(t, x)

def bit(b: Bool) -> Nat:
  match b:
    case True{}:
      1n
    case False{}:
      0n

# how many times x occurs in xs
def count(+x: Nat, xs: List<&2, Nat>) -> Nat:
  match xs:
    case Nil{}:
      0n
    case +y <> t:
      Nat.add(bit(Nat.is_eq(x, y)), count(x, t))

# Rounding t = z * a / b (a rational times an integer, b > 0) to an adjacent integer r, floor(t) <= r <= ceil(t):
# (r - 1) b < z a < (r + 1) b, i.e. r b < z a + b and z a < r b + b. Round-to-nearest (either tie rule) picks one.
def adjacent(+r: Z, +z: Z, +a: Nat, +b: Nat) -> Bool:
  Bool.and(zlt(zscale(r, b), zadd(zscale(z, a), Z{b, 0n})), zlt(zscale(z, a), zadd(zscale(r, b), Z{b, 0n})))

# The steps from the fp16 Q row and the CQ K cache to one log2-domain score, in kernel order.
type Step is Data:
  QRotate{}
  QQuant{}
  KRequant{}
  RowFactor{}
  KeyScale{}
  ScaleProd{}
  ImmaChain{}
  ToFloat{}
  ScoreMul{}

def score_steps() -> List<&2, Step>:
  [QRotate{}, QQuant{}, KRequant{}, RowFactor{}, KeyScale{}, ScaleProd{}, ImmaChain{}, ToFloat{}, ScoreMul{}]
