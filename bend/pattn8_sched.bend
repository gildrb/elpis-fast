# Warp map and tile schedule of the ext 3020 8-warp prefill attention kernel (exllamav3_ext/pattn8_kernel.cuh,
# pattn8_kernel<F16>), transcribed from its integer expressions over Nat. K8 = patches/exl3-ext/3020-prefill-pattn8.patch
# (line numbers of the patch file; pattn8_kernel.cuh line = K8 line - 59). C `int` arithmetic on the non-negative
# values used here is Nat arithmetic while nothing overflows (keys <= 2^18 + 2^11, head indices < 2^8).
# Reference: the 3010 6-warp kernel as modelled by pattn_sched.bend, re-exported per warp by pattn8_sched_spec.bend;
# laws: pattn8_sched_laws.bend; finite link to the CUDA text: pattn8_sched_diff.py.
# Names: L = total = cache_seqlens[b] + kv_append_len (keys the chunk sees), q = q_len, qc = the CTA's query chunk of
# 64 positions (K8:108 qc = gridDim.y - 1 - blockIdx.y), pair = the CTA's q-head pair (K8:107), w = warp (K8:110),
# wb = w % 4 its query block in the chunk, wh = w / 4 its head in the pair (K8:111), r = a thread row of the warp's
# 16 positions (gid or gid + 8), it = the kv tile iteration, p = a kv position (key).
import Base
import ./attn_stride.bend as Stride

# K8:111 const int wb = warp % P8_NB, wh = warp / P8_NB;   (P8_NB = 4, K8:77)
def wb(+w: Nat) -> Nat:
  Nat.mod(w, 4n)

def wh(+w: Nat) -> Nat:
  Nat.div(w, 4n)

# K8:120 const int hq = kvh * PA_G + pair * P8_NH + wh;   (PA_G = 6, P8_NH = 2, K8:78)
def q_head(+kvh: Nat, +pair: Nat, +w: Nat) -> Nat:
  Nat.add(Nat.add(Nat.mul(kvh, 6n), Nat.mul(pair, 2n)), wh(w))

# K8:113 const int c0 = qc * P8_NB * PA_BQ;   (PA_BQ = 16)
def c0(+qc: Nat) -> Nat:
  Nat.mul(Nat.mul(qc, 4n), 16n)

# K8:119 const int p0 = c0 + wb * PA_BQ;  (the warp's first query position in the chunk)
def p0(+qc: Nat, +b: Nat) -> Nat:
  Nat.add(c0(qc), Nat.mul(b, 16n))

# K8:121 const int q_abs0 = total - q_len + p0;
def q_abs0(+L: Nat, +q: Nat, +qc: Nat, +b: Nat) -> Nat:
  Nat.add(Nat.sub(L, q), p0(qc, b))

# K8:176 q_abs_r0 = q_abs0 + gid, q_abs_r1 = q_abs0 + gid + 8: row r's causal bound
def row_abs(+L: Nat, +q: Nat, +qc: Nat, +b: Nat, +r: Nat) -> Nat:
  Nat.add(q_abs0(L, q, qc, b), r)

# K8:122 const int n_hi = min(q_abs0 + PA_BQ, total);
def n_hi(+L: Nat, +q: Nat, +qc: Nat, +b: Nat) -> Nat:
  Nat.min(Nat.add(q_abs0(L, q, qc, b), 16n), L)

# K8:123 const int ntiles = p0 < q_len ? (n_hi + PA_BN - 1) / PA_BN : 0;   (PA_BN = 32)
def ntiles.pick(live: Bool, +L: Nat, +q: Nat, +qc: Nat, +b: Nat) -> Nat:
  match live:
    case True{}:
      Stride.cdiv(n_hi(L, q, qc, b), 32n)
    case False{}:
      0n

def ntiles(+L: Nat, +q: Nat, +qc: Nat, +b: Nat) -> Nat:
  ntiles.pick(Nat.is_lt(p0(qc, b), q), L, q, qc, b)

# K8:114 const int n_hi_cta = min(total - q_len + c0 + P8_NB * PA_BQ, total);
def n_hi_cta(+L: Nat, +q: Nat, +qc: Nat) -> Nat:
  Nat.min(Nat.add(Nat.add(Nat.sub(L, q), c0(qc)), 64n), L)

# K8:115 const int ntiles_cta = (n_hi_cta + PA_BN - 1) / PA_BN;   (the CTA loop K8:180 runs it < ntiles_cta)
def ntiles_cta(+L: Nat, +q: Nat, +qc: Nat) -> Nat:
  Stride.cdiv(n_hi_cta(L, q, qc), 32n)

# K8:128 / K8:184 const int n0 = it * PA_BN;
def n0(+it: Nat) -> Nat:
  Nat.mul(it, 32n)

# K8:180 for (it < ntiles_cta) and K8:185 const bool act = it < ntiles: the warp computes in iteration it
# (K8:187 QK / softmax, K8:288 PV) iff both hold
def active(+L: Nat, +q: Nat, +qc: Nat, +b: Nat, +it: Nat) -> Bool:
  Bool.and(Nat.is_lt(it, ntiles_cta(L, q, qc)), Nat.is_lt(it, ntiles(L, q, qc, b)))

# K8:225 the causal / length mask runs only when n0 + PA_BN - 1 > q_abs0
def masked(+L: Nat, +q: Nat, +qc: Nat, +b: Nat, +it: Nat) -> Bool:
  Nat.is_lt(q_abs0(L, q, qc, b), Nat.add(n0(it), 31n))

# K8:232-234 under the mask, key is dropped iff key > qa || key >= total
def mask_ok(m: Bool, +L: Nat, +q: Nat, +qc: Nat, +b: Nat, +r: Nat, +p: Nat) -> Bool:
  match m:
    case False{}:
      True{}
    case True{}:
      Bool.and(Nat.is_le(p, row_abs(L, q, qc, b, r)), Nat.is_lt(p, L))

# Tile it stages keys n0 + c, c < PA_BN (K8:126-142, one (tok) per key); key p enters row r's softmax in iteration it
# iff it is one of them, the warp computes in that iteration, and the mask keeps it
def visits(+L: Nat, +q: Nat, +qc: Nat, +b: Nat, +r: Nat, +it: Nat, +p: Nat) -> Bool:
  Bool.and(Nat.is_le(n0(it), p),
    Bool.and(Nat.is_lt(p, Nat.add(n0(it), 32n)),
      Bool.and(active(L, q, qc, b, it), mask_ok(masked(L, q, qc, b, it), L, q, qc, b, r, p))))

# The warp's work trace: the iterations it = s .. s + k - 1 of the CTA loop in which it computes, in loop order
def steps.keep(c: Bool, +it: Nat, rest: List<&2, Nat>) -> List<&2, Nat>:
  match c:
    case True{}:
      it <> rest
    case False{}:
      rest

def steps(k: Nat, +s: Nat, +n: Nat) -> List<&2, Nat>:
  match k:
    case 0n:
      []
    case 1n+j:
      steps.keep(Nat.is_lt(s, n), s, steps(j, 1n+s, n))

# K8:180 + K8:185: the whole CTA loop 0 .. ntiles_cta - 1, keeping the warp's active iterations
def trace(+L: Nat, +q: Nat, +qc: Nat, +b: Nat) -> List<&2, Nat>:
  steps(ntiles_cta(L, q, qc), 0n, ntiles(L, q, qc, b))

# K8:152-153 the q prologue: chunk c < P8_NW * PA_BQ * 32 fills smem row = c >> 5; w = row / PA_BQ, pos = row - w * PA_BQ
def prow_w(+row: Nat) -> Nat:
  Nat.div(row, 16n)

def prow_pos(+row: Nat) -> Nat:
  Nat.sub(row, Nat.mul(prow_w(row), 16n))

# K8:153 p = c0 + (w % P8_NB) * PA_BQ + pos, h = kvh * PA_G + pair * P8_NH + w / P8_NB
def prow_p(+qc: Nat, +row: Nat) -> Nat:
  Nat.add(Nat.add(c0(qc), Nat.mul(Nat.mod(prow_w(row), 4n), 16n)), prow_pos(row))

def prow_h(+kvh: Nat, +pair: Nat, +row: Nat) -> Nat:
  Nat.add(Nat.add(Nat.mul(kvh, 6n), Nat.mul(pair, 2n)), Nat.div(prow_w(row), 4n))

# K8:175 const int qrow = warp * PA_BQ; K8:201 the warp's QK MMA reads smem rows qrow + (lane & 15), i.e. qrow + r
def qrow(+w: Nat, +r: Nat) -> Nat:
  Nat.add(Nat.mul(w, 16n), r)

# K8:131-134 / 3010 K:223-226: the tile issue loop, chunk c = tid + i * THREADS (THREADS = 256 here, 192 in 3010)
def issue_c(+T: Nat, +tid: Nat, +i: Nat) -> Nat:
  Nat.add(tid, Nat.mul(i, T))

# K8:38 host: const int nqc = (q_len + P8_NB * PA_BQ - 1) / (P8_NB * PA_BQ);  (grid.y)
def nqc(+q: Nat) -> Nat:
  Stride.cdiv(q, 64n)
