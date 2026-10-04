# Independent reference for the ext 3020 8-warp prefill attention kernel (pattn8_sched.bend): the kernel it replaces,
# ext 3010's 6-warp pattn_kernel, as modelled by pattn_sched.bend (checked against 3010's source by
# pattn_sched_diff.py). K = patches/exl3-ext/3010-prefill-pattn.patch. In 3010 the CTA of query block qb and kv head kvh
# (K:207-208) runs its tile loop over every iteration it = 0 .. ntiles - 1 in order (K:270), and each of its 6 warps
# computes, in every iteration, the MMA / mask / online-softmax / PV work of q head kvh * 6 + warp (K:266, K:419) for the
# 16 positions 16 qb + r (K:210, K:423). The per-iteration work of a warp is a function of (its q rows, the tile it,
# the mask state and bounds of K:312-322): this file names the sequence of iterations; pattn_sched.bend names the rest.
import Base
import ./pattn_sched.bend as P6

# [s, s + 1, .., s + k - 1]
def upto(k: Nat, +s: Nat) -> List<&2, Nat>:
  match k:
    case 0n:
      []
    case 1n+j:
      s <> upto(j, 1n+s)

# K:270 for (int it = 0; it < ntiles; ++it): the iterations every warp of 3010's CTA qb computes, in order
def trace(+L: Nat, +q: Nat, +qb: Nat) -> List<&2, Nat>:
  upto(P6.ntiles(L, q, qb), 0n)
