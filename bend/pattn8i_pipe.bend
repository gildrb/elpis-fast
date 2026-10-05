# Load / barrier schedule of the ext 3022 ping-pong K / V pipeline in pattn8i_kernel (exllamav3_ext/pattn8i_kernel.cuh,
# patches/exl3-ext/3022-prefill-pattn8i-pingpong.patch on top of 3021c), transcribed over Nat. K8I = that file in the
# patched tree. Names as in pattn8_sched.bend: nt = ntiles_cta (the CTA loop bound), nw = ntiles (the warp's own tiles),
# it = the loop iteration, a stage = one [32][256] int8 K + [32] fp32 scales + [32][256] fp16 V slot of shared memory
# (P8I_STAGE bytes at P8I_ST_OFF + st * P8I_STAGE, P8I_NST = 2 of them).
# Epoch semantics (the link pattn8i_pipe_diff.py checks the loop text): iteration it starts with `cp_wait0();
# __syncthreads();`, so every cp.async group issued before it has landed and is visible to every warp, and every warp
# has finished every read of earlier iterations; between that barrier and the next one the threads issue tile it + 1
# and the warps read tile it. An epoch's writes are therefore readable from the next epoch on, and a write may race only
# with reads of the same epoch. Checker of that discipline: pattn8i_pipe_spec.bend; laws: pattn8i_pipe_laws.bend.
import Base
import ./pattn8_sched.bend as P8

# K8I `const int st = it & 1;` (it >= 0, so it & 1 == it % 2)
def stage(+it: Nat) -> Nat:
  Nat.mod(it, 2n)

# K8I `st ^ 1` for st < 2: the other stage
def other(+s: Nat) -> Nat:
  match s:
    case 0n:
      1n
    case 1n+p:
      0n

# One epoch of the CTA: the warps read tile rt from stage rs (K8I qk(st) / pv(st)); when wr, the threads issue tile wt
# into stage ws (K8I issue_k / issue_v (it + 1, st ^ 1, ...)).
type Ep is Data:
  Ep{rt: Nat, rs: Nat, wr: Bool, wt: Nat, ws: Nat}

# K8I loop iterations it .. it + k - 1 of `for (int it = 0; it < ntiles_cta; ++it)`, one epoch each: read tile it from
# stage st, and `if (it + 1 < ntiles_cta)` issue tile it + 1 into stage st ^ 1. The read is the union over warps (a
# warp with act = false reads nothing), so the epoch's reads are at most this one stage.
def epochs(k: Nat, +it: Nat, +nt: Nat) -> List<&2, Ep>:
  match k:
    case 0n:
      []
    case 1n+j:
      Ep{it, stage(it), Nat.is_lt(1n+it, nt), 1n+it, other(stage(it))} <> epochs(j, 1n+it, nt)

# the whole CTA loop
def loop(+nt: Nat) -> List<&2, Ep>:
  epochs(nt, 0n, nt)

# K8I prologue `if (ntiles_cta > 0) issue_k(0, 0, tid, P8_THREADS);` (and issue_v(0, 0, ...)), waited by the first
# iteration's cp_wait0: stage 0 holds tile 0 when the loop runs at all. Slot values are tile + 1 (0 = never written).
def pro.pick(c: Bool) -> Nat:
  match c:
    case True{}:
      1n
    case False{}:
      0n

def pro_c0(+nt: Nat) -> Nat:
  pro.pick(Nat.is_lt(0n, nt))

# K8I `const bool act = it < ntiles; if (act) { qk(st); softmax(it); pv(st); }`: the iterations in which the warp
# computes, in loop order
def acts(es: List<&2, Ep>, +nw: Nat) -> List<&2, Nat>:
  match es:
    case Nil{}:
      []
    case Con{e, rest}:
      match e:
        case Ep{+rt, rs, wr, wt, ws}:
          P8.steps.keep(Nat.is_lt(rt, nw), rt, acts(rest, nw))

# the warp's work trace in the ping-pong loop
def trace(+L: Nat, +q: Nat, +qc: Nat, +b: Nat) -> List<&2, Nat>:
  acts(loop(P8.ntiles_cta(L, q, qc)), P8.ntiles(L, q, qc, b))
