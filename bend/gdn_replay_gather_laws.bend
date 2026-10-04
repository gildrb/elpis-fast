# Obligations for the input loader of ext 5108 (batched_state_replay_kernel<GATHER = true>) over the Nat model
# gdn_replay_gather.bend against gdn_replay_gather_spec.bend. With them the commit replay's shared memory holds, in
# every cell RULE_COMMIT reads, the value the pre-5108 loader stored there, so the replayed state is bit-identical
# (the token routine gdn_rule_tokens is unchanged; its k normalization now skips the unread q rows).
# Domains: every S in 1..GR_MAX_S = 16, every block v chunk c < 4, every thread / slot; symbolic F, Nk, kh, h and
# chunk element q for the addresses; symbolic bf16 bit patterns for the widening.
# Out of scope: the host's 16-byte alignment check (a TORCH_CHECK; misalignment faults, it cannot corrupt), the
# CUDA definition of __bfloat162float (bits << 16) and the little-endian element order inside a word (premises of
# the W model), memory ordering beyond the block barrier after the stores (the kernel's single __syncthreads).
import Base
import ./gdn_replay_gather.bend as Impl
import ./gdn_replay_gather_spec.bend as Spec

# (1) Exactly the read chunks: for every S <= 16 and block c < 4, each active element e < 20 S stores one chunk
# RULE_COMMIT reads and is its owner (so no two elements store the same chunk), and every read chunk (16 k chunks
# and the block's 4 v chunks per token) is stored by its owner, which is active. No q chunk and no other v columns.
law gather_cover:
  {Spec.cover() == True{} : Bool}

# cover() evaluates the checks at the largest token count only (Spec.all_S); they are equivalent to the checks at
# every token count S .. S + n - 1 (the reference conjunction Spec.all_S_ref), for every n and S. With gather_cover
# this gives Spec.all_S_ref(GR_MAX_S, 1) == True: the checks hold at every S in 1..16.
law gather_all_S:
  for n: Nat
  for +S: Nat
  {Spec.all_S(n, S) == Spec.all_S_ref(n, S) : Bool}

# (1) Threads: REPLAY_GI = 3, every element e < 16 * 20 is reached by thread e % 128 at slot e / 128 < 3, and
# distinct (thread < 128, slot < 3) pairs reach distinct elements.
law gather_threads:
  {Spec.threads() == True{} : Bool}

# (1) Inside a chunk: a column x < 8 n (k: n = 16, x = col; v: n = 4, x = col - 32 c) is element x % 8 < 8 of
# chunk x / 8 < n, so the chunks' 8 elements cover exactly the read cells.
law chunk_lanes:
  for +x: Nat
  for +n: Nat
  for xr: {Nat.is_lt(x, Nat.mul(n, 8n)) == True{} : Bool}
  {Nat.is_lt(Nat.div(x, 8n), n) == True{} : Bool} &
  {Nat.is_lt(Nat.mod(x, 8n), 8n) == True{} : Bool} &
  {Nat.add(Nat.mul(Nat.div(x, 8n), 8n), Nat.mod(x, 8n)) == x : Nat}

# (2) Same source: element q of the chunk element e loads is the conv_out element the old loader read for the
# cell it lands in (any F, Nk, kh, h, c, e, q).
law gather_src:
  for +F: Nat
  for +Nk: Nat
  for +kh: Nat
  for +h: Nat
  for +c: Nat
  for +e: Nat
  for +q: Nat
  {Impl.src(F, Nk, kh, h, c, e, q) == Impl.old_src(F, Nk, kh, h, Impl.at(Impl.dst(c, e), q)) : Nat}

# (2) Same value: the widened elements 2m (`w << 16`) and 2m + 1 (`w & 0xffff0000u`) of a word holding bf16 bit
# patterns b0 (lower address) and b1 are the old path's __bfloat162float(b0) and __bfloat162float(b1).
law gather_widen:
  for +b0: Nat
  for +b1: Nat
  {Impl.widen_new(Impl.W{b0, b1}, 0n) == Spec.old_widen(b0) : Impl.W} &
  {Impl.widen_new(Impl.W{b0, b1}, 1n) == Spec.old_widen(b1) : Impl.W}

# (3) EXL3_GDN_REPLAY_GATHER: unset = 1, "0" = 0, "1" = 1; empty, other characters and longer strings refused.
law gather_knob:
  {Impl.knob(Impl.Unset{}) == Impl.Ok{1n} : Impl.Knob} &
  {Impl.knob(Impl.Str{[48n]}) == Impl.Ok{0n} : Impl.Knob} &
  {Impl.knob(Impl.Str{[49n]}) == Impl.Ok{1n} : Impl.Knob} &
  {Impl.knob(Impl.Str{Nil{}}) == Impl.Refused{} : Impl.Knob} &
  {Impl.knob(Impl.Str{[50n]}) == Impl.Refused{} : Impl.Knob} &
  {Impl.knob(Impl.Str{[48n, 49n]}) == Impl.Refused{} : Impl.Knob}
