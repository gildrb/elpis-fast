# Ext 5112: the epilogue of gemm_wide_silu_kernel (exllamav3_ext/hgemm_act.cu), transcribed over the term language of
# act_fuse_spec.bend. E = patches/exl3-ext/5112-mlp-act-fuse.patch (hgemm_act.cu line = E line - 101). The kernel's K
# loop and its (row, col) store map are gemm_wide_kernel<false>'s text (checked by act_fuse_diff.py), so the index maps
# are hgemm_wide.bend's (row_of / col_of) and the accumulator pair of (row, col) is the same fp32 value AccPair{row, col}
# as in the stock wide kernel (hgemm_wide k_order). Reference: act_fuse_spec.bend; laws: act_fuse_laws.bend.
import Base
import ./act_fuse_spec.bend as Spec

# E:131-140 silu_h2(half2 x): one = __float2half2_rn(1.0f); neg_x = __hneg2(x); e = h2exp(neg_x);
# sum = __hadd2(one, e); r = h2rcp(sum); result = __hmul2(x, r)
def silu_h2(+x: Spec.H2) -> Spec.H2:
  Spec.Mul2{x, Spec.Rcp2{Spec.Add2{Spec.One2{}, Spec.Exp2{Spec.Neg2{x}}}}}

# E:260 limit = act_limit != 0.0f; E:275-280: y2 = __hmax2(y2, -lim); y2 = __hmin2(y2, lim); x2 = __hmin2(x2, lim);
# E:281 store __hmul2(x2, y2)
def epi.pick(limit: Bool, +x2: Spec.H2, +y2: Spec.H2) -> Spec.H2:
  match limit:
    case True{}:
      Spec.Mul2{Spec.Min2{x2, Spec.LimPos{}}, Spec.Min2{Spec.Max2{y2, Spec.LimNeg{}}, Spec.LimPos{}}}
    case False{}:
      Spec.Mul2{x2, y2}

# E:273 y2 = __floats2half2_rn(v0, v1) of the (row, col) accumulator pair; E:274 x2 = silu_h2 of the gate half2 at
# G + row * ldc + col
def epi(limit: Bool, +ldc: Nat, +row: Nat, +col: Nat) -> Spec.H2:
  epi.pick(limit, silu_h2(Spec.GateAt{Nat.add(Nat.mul(row, ldc), col)}), Spec.Rn2{Spec.AccPair{row, col}})

# E:281 the store's element offset C + row * ldc + col
def epi_off(+ldc: Nat, +row: Nat, +col: Nat) -> Nat:
  Nat.add(Nat.mul(row, ldc), col)

# E:303, 306, 308, 310: the host route (fused_route): both switches on (unset or "1"), sm86, 2-D operands (batch 1),
# K == 5120, N == 17408, M >= 1024, M <= 65535 * BM (BM = 128), and the f16acc status is 1. The tensor dtype /
# contiguity / alignment / aliasing checks (E:304, 307-308, 312-315) are not modelled.
type Sw is Data:
  SwUnset{}
  SwOne{}
  SwZero{}
  SwOther{}

type Route is Data:
  Fused{}
  Unfused{}
  Refused{}

def sw_on(s: Sw) -> Route:
  match s:
    case SwUnset{}:
      Fused{}
    case SwOne{}:
      Fused{}
    case SwZero{}:
      Unfused{}
    case SwOther{}:
      Refused{}

def route.bool(b: Bool) -> Route:
  match b:
    case True{}:
      Fused{}
    case False{}:
      Unfused{}

# dev: E:304-308, 312-315 (CUDA, one device, sm86, fp16, 2-D, contiguous, aligned, gate != c) all pass (not modelled
# further); mfit: E:311 M <= 65535 * WideAct::BM (int64, not modelled); status: E:316 hgemm_f16acc_status == 1.
# E:310: K == 5120, N == 17408, M >= 1024 (b / c shape consistency folded into dev).
def route.shape(dev: Bool, +M: Nat, +K: Nat, +N: Nat, mfit: Bool, status: Bool) -> Route:
  match dev:
    case False{}:
      Unfused{}
    case True{}:
      match mfit:
        case False{}:
          Unfused{}
        case True{}:
          match status:
            case False{}:
              Unfused{}
            case True{}:
              route.bool(Bool.and(Nat.is_eq(K, 5120n), Bool.and(Nat.is_eq(N, 17408n), Bool.not(Nat.is_lt(M, 1024n)))))

# E:303 !env_on(MLP) || !env_on(WIDE) -> false (env_on raises for anything but unset / "1" / "0", MLP first)
def route.env(a: Route, b: Route, dev: Bool, +M: Nat, +K: Nat, +N: Nat, mfit: Bool, status: Bool) -> Route:
  match a:
    case Refused{}:
      Refused{}
    case Unfused{}:
      Unfused{}
    case Fused{}:
      match b:
        case Refused{}:
          Refused{}
        case Unfused{}:
          Unfused{}
        case Fused{}:
          route.shape(dev, M, K, N, mfit, status)

def route(mlp: Sw, wide: Sw, dev: Bool, +M: Nat, +K: Nat, +N: Nat, mfit: Bool, status: Bool) -> Route:
  route.env(sw_on(mlp), sw_on(wide), dev, M, K, N, mfit, status)

def is_fused(r: Route) -> Bool:
  match r:
    case Fused{}:
      True{}
    case Unfused{}:
      False{}
    case Refused{}:
      False{}

# E:266 col = bn + wn * WN + j * 8 + t * 2 (WN = 64, bn = tile_n * BN, BN = 128) is even: col = 2 pair_col, and the pair's
# second element col + 1 is in the same row; pair_col is the half2 column the store writes.
def pair_col(+tn: Nat, +tid: Nat, +j: Nat) -> Nat:
  Nat.add(Nat.add(Nat.add(Nat.mul(tn, 64n), Nat.mul(Nat.mod(Nat.div(tid, 32n), 2n), 32n)), Nat.mul(j, 4n)), Nat.mod(Nat.mod(tid, 32n), 4n))
