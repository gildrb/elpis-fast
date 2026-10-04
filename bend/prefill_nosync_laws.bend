# Obligations for ext 9502 (EXL3_PREFILL_NOSYNC; impl model prefill_nosync.bend, reference prefill_nosync_spec.bend).
#
# Covered: the host/stream event order of the RecurrentCache deferred-stash state machine (begin_round, settle,
# put(defer), get_stashed / prune_stranded / persist) driven by Generator.iterate (begin_round, iterate_start_jobs,
# job.prefill with its last-page put, recurrent_checkpoint, iterate_gen with the requeue put, a reader after the
# iterate), for EVERY finite sequence of iterates of the served single-job recurrent (GDN, non-TP) model: every
# combination of a start reader, a prefill forward, the three put sites (no call / cached key / new stash), the switch,
# a decode forward and an end reader, starting from a fresh RecurrentCache. Laws 1-5 read the trace through the
# happens-before observer of the spec (host program order + one CUDA stream in FIFO order + event record -> wait).
#
# Trust base (assumed, not proven): one CUDA stream per device carries the prefill/decode forwards, the uploads and
# the checkpoint copies in enqueue order (FIFO); event.record after the copies completes only after them and
# event.synchronize returns only after the event completes; torch non_blocking copy_ into pinned host memory is an
# ordinary stream operation; the caching host allocator hands a pinned block out again only after the event recorded
# for its last upload has completed (so a pin_memory() upload block of chunk c is not overwritten while in flight);
# .cpu() of a device tensor synchronizes. Python exceptions, other generator threads, several concurrent jobs, TP and
# eviction (popitem) of a still-pending stash dict are outside the model; the kernels themselves are not modelled.
import Base
import ./prefill_nosync.bend as Impl
import ./prefill_nosync_spec.bend as Spec

# 1. Every upload of a chunk's inputs (pinned block table, cache_seqlens, embedded rows) is enqueued before that
# chunk's prefill forward on the stream (and each forward consumes its own upload).
law uploads_before_forwards:
  for +its: List<&2, Spec.It>
  {Spec.ok1(Spec.run(Impl.trace(its, Impl.init()), Spec.init())) == True{} : Bool}

# 2. Every checkpoint copy reads the state the host stashed: at its stream position exactly as many forwards were
# enqueued as the job's state had gone through when put() was called, so the copy of chunk c's state is enqueued
# before the next forward (prefill or decode) that overwrites the recurrent state.
law copy_reads_its_chunk:
  for +its: List<&2, Spec.It>
  {Spec.ok2(Spec.run(Impl.trace(its, Impl.init()), Spec.init())) == True{} : Bool}

# 3. The wait window of a deferred copy enqueued in round r: it is waited before any copy of a later round is
# enqueued and before the stash loop of the next round's recurrent_checkpoint (must happen); outside a reader it is
# waited only in round r + 1 (never in round r: the GPU is not idled by the round that queued it), and only after
# round r + 1's prefill forward was enqueued or inside its recurrent_checkpoint, which follows every prefill forward
# of the round (the wait does not idle the GPU when that round has a forward).
law deferred_wait_window:
  for +its: List<&2, Spec.It>
  {Spec.ok3(Spec.run(Impl.trace(its, Impl.init()), Spec.init())) == True{} : Bool}

# 3'. Finding (not a failure of law 3, the reader exception made concrete): a stash read in iterate_start_jobs
# (pagetable.py:304 get_stashed) settles round 1's deferred copy before round 2's forward is enqueued (Wt{0} before
# Fw{2}); without the reader the same wait comes after Fw{2} (here inside recurrent_checkpoint).
law start_reader_waits_early:
  {Impl.trace([Spec.Iter{False{}, True{}, Spec.New{}, Spec.No{}, True{}, False{}, Spec.No{}, False{}},
               Spec.Iter{True{}, True{}, Spec.No{}, Spec.No{}, True{}, False{}, Spec.No{}, False{}}], Impl.init()) ==
   [Spec.Bg{}, Spec.Up{1n}, Spec.Fw{1n}, Spec.Cp{0n, 1n, True{}}, Spec.Ck0{}, Spec.Ck{},
    Spec.Bg{}, Spec.Rd0{}, Spec.Wt{0n}, Spec.Rd{}, Spec.Up{2n}, Spec.Fw{2n}, Spec.Ck0{}, Spec.Ck{}] : List<&2, Spec.Ev>} &
  {Impl.trace([Spec.Iter{False{}, True{}, Spec.New{}, Spec.No{}, True{}, False{}, Spec.No{}, False{}},
               Spec.Iter{False{}, True{}, Spec.No{}, Spec.No{}, True{}, False{}, Spec.No{}, False{}}], Impl.init()) ==
   [Spec.Bg{}, Spec.Up{1n}, Spec.Fw{1n}, Spec.Cp{0n, 1n, True{}}, Spec.Ck0{}, Spec.Ck{},
    Spec.Bg{}, Spec.Up{2n}, Spec.Fw{2n}, Spec.Ck0{}, Spec.Wt{0n}, Spec.Ck{}] : List<&2, Spec.Ev>}

# 4. No reader of stash contents (get_stashed, prune_stranded, persist) reads while a copy is outstanding: every copy
# before the read has its wait before the read. And a reader's settle() leaves nothing pending, for every pending
# list, so persist's _require(not self.rc.pending, ...) never fails.
law readers_see_no_pending:
  for +its: List<&2, Spec.It>
  {Spec.ok4(Spec.run(Impl.trace(its, Impl.init()), Spec.init())) == True{} : Bool}

law persist_require:
  for pend: List<&2, Impl.Pe>
  for +b: Nat
  {Impl.settle_keep(pend, False{}, b) == [] : List<&2, Impl.Pe>}

# 5. Conservation: copies get fresh stash ids, every wait matches exactly one copy not yet waited (no double settle),
# a blocking copy is waited right after it, and the copies not yet waited at the end are exactly the RecurrentCache's
# pending stashes (no lost settle: the next put / recurrent_checkpoint / reader waits them, laws 3 and 4).
law conservation:
  for +its: List<&2, Spec.It>
  {Spec.ok5(Spec.run(Impl.trace(its, Impl.init()), Spec.init())) == True{} : Bool} &
  {Spec.out_ids(Spec.run(Impl.trace(its, Impl.init()), Spec.init())) == Impl.pend_ids(Impl.final(its, Impl.init())) : List<&2, Nat>} &
  {Spec.blk(Spec.run(Impl.trace(its, Impl.init()), Spec.init())) == 0n : Nat}

# 5'. The off switch is the old behaviour: with EXL3_PREFILL_NOSYNC=0 in every iterate the trace is exactly the
# pre-9502 blocking trace (each stash copies and waits at once, nothing is ever pending, settle() waits nothing).
law defer_off_is_blocking:
  for +its: List<&2, Spec.It>
  for off: {Spec.all_off(its) == True{} : Bool}
  {Impl.trace(its, Impl.init()) == Spec.old_trace(its, 0n, 0n) : List<&2, Spec.Ev>}

# 6. State invariant "every pending entry has round >= current round - 1 (and <= current round)": it holds initially
# and is preserved by settle (any before_round), put (any outcome, defer on or off), the readers' settle, forwards and
# recurrent_checkpoint; begin_round preserves it from a state whose pending entries are all of the current round,
# which every iterate re-establishes (recurrent_checkpoint's settle), so it holds at every step of every run.
law inv_init:
  {Spec.inv(Impl.round(Impl.init()), Impl.rounds(Impl.init())) == True{} : Bool}

law inv_settle:
  for +rnd: Nat
  for +pend: List<&2, Impl.Pe>
  for +hasb: Bool
  for +b: Nat
  for h: {Spec.inv(rnd, Impl.rounds.go(pend)) == True{} : Bool}
  {Spec.inv(rnd, Impl.rounds.go(Impl.settle_keep(pend, hasb, b))) == True{} : Bool}

law inv_ops:
  for +st: Impl.St
  for +u: Spec.Pu
  for +d: Bool
  for +r: Bool
  for +h: {Spec.inv(Impl.round(st), Impl.rounds(st)) == True{} : Bool}
  {Spec.inv(Impl.round(Impl.put_st(u, d, st)), Impl.rounds(Impl.put_st(u, d, st))) == True{} : Bool} &
  {Spec.inv(Impl.round(Impl.rd_st(r, st)), Impl.rounds(Impl.rd_st(r, st))) == True{} : Bool} &
  {Spec.inv(Impl.round(Impl.ck_st(st)), Impl.rounds(Impl.ck_st(st))) == True{} : Bool} &
  {Spec.inv(Impl.round(Impl.fwd(st)), Impl.rounds(Impl.fwd(st))) == True{} : Bool}

law inv_begin:
  for +st: Impl.St
  for h0: {Spec.inv(Impl.round(st), Impl.rounds(st)) == True{} : Bool}
  for h: {Spec.all_at_least(Impl.round(st), Impl.rounds(st)) == True{} : Bool}
  {Spec.inv(Impl.round(Impl.begin(st)), Impl.rounds(Impl.begin(st))) == True{} : Bool}

# begin_round alone does not preserve the invariant (a pending entry of the current round becomes two rounds old):
# the invariant rests on recurrent_checkpoint settling every round.
law inv_begin_needs_settle:
  {Spec.inv(1n, Impl.rounds(Impl.St{1n, [Impl.Pe{0n, 0n}], 1n, 1n})) == True{} : Bool} &
  {Spec.inv(Impl.round(Impl.begin(Impl.St{1n, [Impl.Pe{0n, 0n}], 1n, 1n})), Impl.rounds(Impl.begin(Impl.St{1n, [Impl.Pe{0n, 0n}], 1n, 1n}))) == False{} : Bool}

law inv_iterates:
  for +its: List<&2, Spec.It>
  {Spec.all_at_least(Impl.round(Impl.final(its, Impl.init())), Impl.rounds(Impl.final(its, Impl.init()))) == True{} : Bool} &
  {Spec.inv(Impl.round(Impl.final(its, Impl.init())), Impl.rounds(Impl.final(its, Impl.init()))) == True{} : Bool}
