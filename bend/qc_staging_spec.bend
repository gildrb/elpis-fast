# Independent reference for ext 9503f's staging-scratch size (qc_staging.bend): for a block-table span of n pages the
# scratch is the smaller of the smallest power of two >= n and the smallest multiple of 64 >= n, each found by search
# over the candidates in increasing order (not by bit tricks or division); the kill switch EXL3_QC_STAGING_ROUND64:
# unset or "1" = that, "0" = the power of two alone, anything else = an error (9503f's comment, triton_paged.py:713).
import Base

# EXL3_QC_STAGING_ROUND64: unset, "1", "0", anything else
type QcEnv is Data:
  QcUnset{}
  QcOne{}
  QcZero{}
  QcOther{}

# the call's result: a page count, or the ValueError
type QcOut is Data:
  QcPages{n: Nat}
  QcError{}

# smallest 2^k >= n: try k = 0, 1, 2, ... in order; c = (n <= 2^k) for the current candidate, f = candidates left to
# test (fuel n: 2^n >= n, so the last candidate 2^n always qualifies)
def spow2.go(f: Nat, c: Bool, +k: Nat, +n: Nat) -> Nat:
  match f:
    case 0n:
      Nat.pow(2n, k)
    case 1n+g:
      match c:
        case True{}:
          Nat.pow(2n, k)
        case False{}:
          spow2.go(g, Nat.is_le(n, Nat.pow(2n, 1n+k)), 1n+k, n)

def spow2(+n: Nat) -> Nat:
  spow2.go(n, Nat.is_le(n, 1n), 0n, n)

# smallest 64 q >= n: try q = 0, 1, 2, ... in order (fuel n: 64 n >= n)
def smul64.go(f: Nat, c: Bool, +q: Nat, +n: Nat) -> Nat:
  match f:
    case 0n:
      Nat.mul(q, 64n)
    case 1n+g:
      match c:
        case True{}:
          Nat.mul(q, 64n)
        case False{}:
          smul64.go(g, Nat.is_le(n, Nat.mul(1n+q, 64n)), 1n+q, n)

def smul64(+n: Nat) -> Nat:
  smul64.go(n, Nat.is_le(n, 0n), 0n, n)

def sround64(+n: Nat) -> Nat:
  Nat.min(spow2(n), smul64(n))

def sstage(env: QcEnv, +n: Nat) -> QcOut:
  match env:
    case QcUnset{}:
      QcPages{sround64(n)}
    case QcOne{}:
      QcPages{sround64(n)}
    case QcZero{}:
      QcPages{spow2(n)}
    case QcOther{}:
      QcError{}

# the switch values that select the multiple-of-64 rounding (unset or "1")
def rounds64(env: QcEnv) -> Bool:
  match env:
    case QcUnset{}:
      True{}
    case QcOne{}:
      True{}
    case QcZero{}:
      False{}
    case QcOther{}:
      False{}
