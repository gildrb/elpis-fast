# Independent reference for ext 5112 (the SiLU-gate product fused into the prefill up-projection GEMM epilogue,
# bend/act_fuse.bend): the stock two-step path it replaces, hgemm_recon(x, W_up) -> u (the 3011 wide kernel, its store)
# followed by silu_mul(g, u, u, act_limit) (act_mul_kernel_h<ACT_SILU>). Sources: S = stock ExLlamaV3 355c6ee
# exllamav3_ext/activation_kernels.cuh (unchanged by the ext series), A = exllamav3_ext/activation.cu (unchanged),
# W = patches/exl3-ext/3011-hgemm-f16acc-wide.patch (the wide kernel), F = exllamav3_ext/hgemm_f16acc.cu of the served
# tree (route predicates). Values are opaque terms: a term names which operands meet which half2 operation (each
# operation node is one IEEE fp16 / fp32->fp16 rounding point); two equal terms are the same bits on the same inputs.
import Base

# half2 value terms
type H2 is Data:
  AccPair{row: Nat, col: Nat}   # the fp32 accumulator pair of output elements (row, col), (row, col + 1) after the K loop
  GateAt{e: Nat}                # a half2 load of the gate tensor at element offset e (elements e, e + 1)
  One2{}                        # __float2half2_rn(1.0f)
  LimPos{}                      # __float2half2_rn(act_limit)
  LimNeg{}                      # __float2half2_rn(-act_limit)
  Rn2{x: H2}                    # __floats2half2_rn(v0, v1) of an fp32 pair
  Neg2{x: H2}                   # __hneg2
  Exp2{x: H2}                   # h2exp
  Rcp2{x: H2}                   # h2rcp
  Add2{a: H2, b: H2}            # __hadd2
  Mul2{a: H2, b: H2}            # __hmul2
  Max2{a: H2, b: H2}            # __hmax2
  Min2{a: H2, b: H2}            # __hmin2

# S:22-31 _silu(half2 x): one = __float2half2_rn(1.0f); neg_x = __hneg2(x); e = h2exp(neg_x); sum = __hadd2(one, e);
# r = h2rcp(sum); result = __hmul2(x, r)
def silu(+x: H2) -> H2:
  Mul2{x, Rcp2{Add2{One2{}, Exp2{Neg2{x}}}}}

# W:295-310 (gemm_wide_kernel<false>): the thread owning output pair (row, col) (col even) stores
# __floats2half2_rn(acc[..][2 h], acc[..][2 h + 1]) at C + row * ldc + col. With C contiguous (ldc = N) the u buffer read
# as a half2 at element offset e holds the pair stored for output (e / N, e % N) (hgemm_wide global_inj / global_cover:
# every element stored exactly once).
def u_at(+N: Nat, +e: Nat) -> H2:
  Rn2{AccPair{Nat.div(e, N), Nat.mod(e, N)}}

# S:144-186 act_mul_kernel_h<ACT_SILU>(x = g, y = u, z = u): x2 = ((const half2*) x)[idx]; y2 = ((const half2*) y)[idx];
# x2 = _silu(x2); if (act_limit != 0) { y2 = __hmax2(y2, -lim); y2 = __hmin2(y2, lim); x2 = __hmin2(x2, lim); }
# z[idx] = __hmul2(x2, y2). The half2 index idx covers elements 2 idx, 2 idx + 1.
def act_mul.pick(limit: Bool, +x2: H2, +y2: H2) -> H2:
  match limit:
    case True{}:
      Mul2{Min2{x2, LimPos{}}, Min2{Max2{y2, LimNeg{}}, LimPos{}}}
    case False{}:
      Mul2{x2, y2}

def act_mul(limit: Bool, +N: Nat, +idx: Nat) -> H2:
  act_mul.pick(limit, silu(GateAt{Nat.mul(idx, 2n)}), u_at(N, Nat.mul(idx, 2n)))

# A:49-50, 71-78 and S:153-154: blocks = CEIL_DIVIDE(numel, 2 * NUM_THREADS), NUM_THREADS = 256;
# idx = blockIdx.x * NUM_THREADS + threadIdx.x, and the thread returns when idx >= numel / 2
def stock_idx(+blk: Nat, +tid: Nat) -> Nat:
  Nat.add(Nat.mul(blk, 256n), tid)

def stock_blocks(+numel: Nat) -> Nat:
  Nat.div(Nat.sub(Nat.add(numel, 512n), 1n), 512n)

# F:447-464 worthwhile() on a non-tuned device (F:429-434: sm86 major 8 != 12), batch 1: M >= MIN_ROWS (384) and
# ceil(M / BM) * (N / BN) >= multiProcessorCount (BM = BN = 128, F:29)
def worthwhile86(+M: Nat, +N: Nat, +nsm: Nat) -> Bool:
  Bool.and(Bool.not(Nat.is_lt(M, 384n)), Nat.is_le(nsm, Nat.mul(Nat.div(Nat.sub(Nat.add(M, 128n), 1n), 128n), Nat.div(N, 128n))))
