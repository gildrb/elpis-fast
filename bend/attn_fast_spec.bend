# Reference for the int8 QK^T operands of ext 3032 (attn_fast.bend), independent of the kernel text: which head dim each
# k-slot of the m16n8k32 s8 B operand carries, the K value of a 3-bit code in units of s / 8, the s8 reading of a byte,
# and four distinct lanes of one fragment row. Integers are the offset pairs of pattn8i_int_spec.bend. Laws:
# attn_fast_laws.bend.
import Base
import ./pattn8i_int_spec.bend as Z

def pickz(c: Bool, x: Z.Z, y: Z.Z) -> Z.Z:
  match c:
    case True{}:
      x
    case False{}:
      y

# The B operand of mma.m16n8k32.row.col.s32.s8.s8.s32 (PTX fragment layout): lane 4 gid + t holds b0 = k-slots
# 4 t .. 4 t + 3 and b1 = k-slots 16 + 4 t .. 16 + 4 t + 3 of column gid (byte i of b_h = k-slot 16 h + 4 t + i). The K
# operand puts head dim 16 h + t + 4 i of the 32-dim group in k-slot 16 h + 4 t + i; kdim(s) is the dim of k-slot s.
def kdim(+s: Nat) -> Nat:
  Nat.add(Nat.mul(16n, Nat.div(s, 16n)), Nat.add(Nat.div(Nat.mod(s, 16n), 4n), Nat.mul(4n, Nat.mod(s, 4n))))

# The K value of 3-bit code c in units of s / 8 (in the H32 basis K = (2 c - 7) s / 8), as a canonical integer (an s8
# register value).
def kval(+c: Nat) -> Z.Z:
  Z.norm(Z.Z{Nat.mul(c, 2n), 7n})

# A byte u < 256 read as a signed s8 (two's complement).
def s8(+u: Nat) -> Z.Z:
  pickz(Nat.is_lt(u, 128n), Z.Z{u, 0n}, Z.Z{0n, Nat.sub(256n, u)})

def lt4(+x: Nat) -> Bool:
  Nat.is_lt(x, 4n)

def ne(+x: Nat, +y: Nat) -> Bool:
  Bool.not(Nat.is_eq(x, y))

# a, b, c, d are four distinct lanes t < 4 of one fragment row (so they are all four of them).
def distinct4(+a: Nat, +b: Nat, +c: Nat, +d: Nat) -> Bool:
  Bool.and(Bool.and(Bool.and(lt4(a), lt4(b)), Bool.and(lt4(c), lt4(d))),
    Bool.and(Bool.and(Bool.and(ne(a, b), ne(a, c)), ne(a, d)), Bool.and(Bool.and(ne(b, c), ne(b, d)), ne(c, d))))
