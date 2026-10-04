# Ext 9502 (EXL3_PREFILL_NOSYNC): the RecurrentCache deferred-stash state machine and the Generator.iterate sequence
# of the served single-job recurrent (GDN) model, transcribed over Nat/Bool/List from the patched engine tree
# (OUT/patched = main's full series). Files: R = cache/recurrent.py, G = generator/generator.py, J = generator/job.py,
# D = modules/gated_delta_net.py, E = modules/embedding.py, P = generator/persist.py. The model emits the trace of
# host actions in host program order, in the alphabet of prefill_nosync_spec.bend (Spec.Ev): stream enqueues Up / Fw /
# Dc / Cp, host waits Wt, markers Bg (iterate start), Rd0 / Rd (a stash reader starts / reads), Ck0 / Ck
# (recurrent_checkpoint starts / starts its stash loop). Stash ids s count the stashes taken (0, 1, 2, ...); ver
# counts the forwards that have written the job's recurrent state. Reference: prefill_nosync_spec.bend; laws:
# prefill_nosync_laws.bend.
import Base
import ./prefill_nosync_spec.bend as Spec

# R:50-51 self.pending = [] holds (round, stash dict); s names the stash dict
type Pe is Data:
  Pe{r: Nat, s: Nat}

# rnd = R self.round, pend = R self.pending, nxt = id of the next stash, ver = forwards through the job's state
type St is Data:
  St{rnd: Nat, pend: List<&2, Pe>, nxt: Nat, ver: Nat}

# R:51-52 self.pending = []; self.round = 0
def init() -> St:
  St{0n, [], 0n, 0n}

# R:55-57 def begin_round(self): self.round += 1
def begin(st: St) -> St:
  match st:
    case St{rnd, pend, nxt, ver}:
      St{1n+rnd, pend, nxt, ver}

# R:69 if before_round is not None and rnd >= before_round: (hasb = before_round is not None, b = before_round)
def keep_t(+hasb: Bool, +b: Nat, +r: Nat) -> Bool:
  Bool.and(hasb, Nat.is_le(b, r))

# R:68-80 for rnd, stashed in self.pending: kept entries `continue` (R:70-71); the others are waited
# (R:72-74 events, pinned = stashed.pop(STASH_PENDING); for event in events: event.synchronize()) and installed
# (R:75-80), one Wt per settled stash, in pending order
def settle_ev.pick(k: Bool, +s: Nat, rest: List<&2, Spec.Ev>) -> List<&2, Spec.Ev>:
  match k:
    case True{}:
      rest
    case False{}:
      Spec.Wt{s} <> rest

def settle_ev(pend: List<&2, Pe>, +hasb: Bool, +b: Nat) -> List<&2, Spec.Ev>:
  match pend:
    case Nil{}:
      []
    case Pe{+r, +s} <> t:
      settle_ev.pick(keep_t(hasb, b, r), s, settle_ev(t, hasb, b))

# R:67 keep = []; R:70 keep.append((rnd, stashed)); R:81 self.pending = keep
def settle_keep.pick(k: Bool, +p: Pe, rest: List<&2, Pe>) -> List<&2, Pe>:
  match k:
    case True{}:
      p <> rest
    case False{}:
      rest

def settle_keep(pend: List<&2, Pe>, +hasb: Bool, +b: Nat) -> List<&2, Pe>:
  match pend:
    case Nil{}:
      []
    case Pe{+r, +s} <> t:
      settle_keep.pick(keep_t(hasb, b, r), Pe{r, s}, settle_keep(t, hasb, b))

# D:153-178 GDNState.stash(defer): defer (D:158-170): per layer D:400-412 stash(slot, pinned = True) enqueues
# rec_h.copy_(rec, non_blocking = True); conv_h.copy_(conv, non_blocking = True) into pinned host buffers, then
# D:166-169 event.record(torch.cuda.current_stream(dev)) after the copies; no wait. Blocking (D:171-173, D:413
# return rec.cpu(), conv.cpu()): the same copies, waited before stash() returns.
def stash_ev(d: Bool, +s: Nat, +c: Nat) -> List<&2, Spec.Ev>:
  match d:
    case True{}:
      [Spec.Cp{s, c, True{}}]
    case False{}:
      [Spec.Cp{s, c, False{}}, Spec.Wt{s}]

# R:105-106 if STASH_PENDING in stashed_state: self.pending.append((self.round, stashed_state))
def pend_add(d: Bool, +r: Nat, +s: Nat) -> List<&2, Pe>:
  match d:
    case True{}:
      [Pe{r, s}]
    case False{}:
      []

# R:95-106 put(key, state, defer): R:100 self.settle(self.round) first; R:101-102 key cached: no stash (Hit);
# R:104 stashed_state = state.stash(defer = True) if defer else state.stash() (New). No: the caller made no put.
def put_ev(u: Spec.Pu, d: Bool, st: St) -> List<&2, Spec.Ev>:
  match u:
    case Spec.No{}:
      []
    case Spec.Hit{}:
      match st:
        case St{+rnd, pend, nxt, ver}:
          settle_ev(pend, True{}, rnd)
    case Spec.New{}:
      match st:
        case St{+rnd, pend, +nxt, +ver}:
          List.append(&2, Spec.Ev, settle_ev(pend, True{}, rnd), stash_ev(d, nxt, ver))

def put_st(u: Spec.Pu, d: Bool, st: St) -> St:
  match u:
    case Spec.No{}:
      st
    case Spec.Hit{}:
      match st:
        case St{+rnd, pend, nxt, ver}:
          St{rnd, settle_keep(pend, True{}, rnd), nxt, ver}
    case Spec.New{}:
      match st:
        case St{+rnd, pend, +nxt, ver}:
          St{rnd, List.append(&2, Pe, settle_keep(pend, True{}, rnd), pend_add(d, rnd, nxt)), 1n+nxt, ver}

# A reader of stash contents settles everything first: R:88 get_stashed: self.settle(); R:149 prune_stranded:
# self.settle(); P:677-678 persist: self.rc.settle(); _require(not self.rc.pending, ...). Rd0 = the call, Rd = the
# read after the settle.
def rd_ev(r: Bool, st: St) -> List<&2, Spec.Ev>:
  match r:
    case True{}:
      match st:
        case St{rnd, pend, nxt, ver}:
          Spec.Rd0{} <> List.append(&2, Spec.Ev, settle_ev(pend, False{}, 0n), [Spec.Rd{}])
    case False{}:
      []

def rd_st(r: Bool, st: St) -> St:
  match r:
    case True{}:
      match st:
        case St{rnd, pend, nxt, ver}:
          St{rnd, settle_keep(pend, False{}, 0n), nxt, ver}
    case False{}:
      st

# J:1472-1477 if nosync: block_table = seq.block_index_tensor.pin_memory(); cache_seqlens = torch.tensor(...).pin_memory()
# and E:170-171 x = x.pin_memory() (pinned_upload): the chunk's inputs are uploaded (Up) ahead of
# J:1501 self.generator.model.prefill(input_ids = prefill_ids, params = params) (Fw); then J:1573-1578
# self.stash_defer = (nosync and can_defer_stash and not loaded_tp) (= en) and the last-page put
# J:1578 self.maybe_stash_recurrent(..., PAGE_SIZE) -> J:1730 cache.put(page.phash, self.recurrent_state, defer = self.stash_defer)
def fwd(st: St) -> St:
  match st:
    case St{rnd, pend, nxt, ver}:
      St{rnd, pend, nxt, 1n+ver}

def pf_ev(fw: Bool, lp: Spec.Pu, en: Bool, st: St) -> List<&2, Spec.Ev>:
  match fw:
    case True{}:
      match st:
        case St{rnd, pend, nxt, +ver}:
          Spec.Up{1n+ver} <> Spec.Fw{1n+ver} <> put_ev(lp, en, St{rnd, pend, nxt, 1n+ver})
    case False{}:
      []

def pf_st(fw: Bool, lp: Spec.Pu, en: Bool, st: St) -> St:
  match fw:
    case True{}:
      put_st(lp, en, fwd(st))
    case False{}:
      st

# J:1269 self.stash_defer = False at the start of every prefill() call, J:1573 set after a forward: the defer flag
# of the later puts of the iterate (recurrent_checkpoint's and the requeue put) is en only if a forward ran
def sd(fw: Bool, en: Bool) -> Bool:
  match fw:
    case True{}:
      en
    case False{}:
      False{}

# G:692-697 recurrent_checkpoint(): self.recurrent_cache.settle(self.recurrent_cache.round) before
# for job in self.active_jobs: job.maybe_stash_recurrent(self.recurrent_cache)
def ck_ev(st: St) -> List<&2, Spec.Ev>:
  match st:
    case St{+rnd, pend, nxt, ver}:
      Spec.Ck0{} <> List.append(&2, Spec.Ev, settle_ev(pend, True{}, rnd), [Spec.Ck{}])

def ck_st(st: St) -> St:
  match st:
    case St{+rnd, pend, nxt, ver}:
      St{rnd, settle_keep(pend, True{}, rnd), nxt, ver}

# G:645-663 iterate_gen: a decode forward through the job's recurrent state
def dc_ev(gf: Bool, st: St) -> List<&2, Spec.Ev>:
  match gf:
    case True{}:
      match st:
        case St{rnd, pend, nxt, ver}:
          [Spec.Dc{1n+ver}]
    case False{}:
      []

def dc_st(gf: Bool, st: St) -> St:
  match gf:
    case True{}:
      fwd(st)
    case False{}:
      st

# G:628-642 iterate(): G:630 self.recurrent_cache.begin_round(); G:631 self.iterate_start_jobs(results) (may read a
# stash: pagetable.py:304 recurrent_cache.get_stashed); G:636 job.prefill(results); G:642 self.recurrent_checkpoint();
# then iterate_gen (G:645-663), whose requeue path G:1711-1712 job.maybe_stash_recurrent(self.recurrent_cache,
# PAGE_SIZE) puts with the job's stash_defer; after the iterate a reader may run (G:682 on_queue_drained ->
# prune_stranded, or persist).
def it_ev(it: Spec.It, +st: St) -> List<&2, Spec.Ev>:
  match it:
    case Spec.Iter{+rs, +fw, +lp, +cp, +en, +gf, +rq, +re}:
      Spec.Bg{} <> List.append(&2, Spec.Ev, rd_ev(rs, begin(st)),
        List.append(&2, Spec.Ev, pf_ev(fw, lp, en, rd_st(rs, begin(st))),
          List.append(&2, Spec.Ev, ck_ev(pf_st(fw, lp, en, rd_st(rs, begin(st)))),
            List.append(&2, Spec.Ev, put_ev(cp, sd(fw, en), ck_st(pf_st(fw, lp, en, rd_st(rs, begin(st))))),
              List.append(&2, Spec.Ev, dc_ev(gf, put_st(cp, sd(fw, en), ck_st(pf_st(fw, lp, en, rd_st(rs, begin(st)))))),
                List.append(&2, Spec.Ev,
                  put_ev(rq, sd(fw, en), dc_st(gf, put_st(cp, sd(fw, en), ck_st(pf_st(fw, lp, en, rd_st(rs, begin(st))))))),
                  rd_ev(re, put_st(rq, sd(fw, en), dc_st(gf, put_st(cp, sd(fw, en), ck_st(pf_st(fw, lp, en, rd_st(rs, begin(st))))))))))))))

def it_st(it: Spec.It, st: St) -> St:
  match it:
    case Spec.Iter{+rs, +fw, +lp, +cp, +en, +gf, +rq, +re}:
      rd_st(re, put_st(rq, sd(fw, en), dc_st(gf, put_st(cp, sd(fw, en), ck_st(pf_st(fw, lp, en, rd_st(rs, begin(st))))))))

# A run of iterates from state st: the concatenated iterate traces, and the final state
def trace(its: List<&2, Spec.It>, +st: St) -> List<&2, Spec.Ev>:
  match its:
    case Nil{}:
      []
    case +it <> rest:
      List.append(&2, Spec.Ev, it_ev(it, st), trace(rest, it_st(it, st)))

def final(its: List<&2, Spec.It>, +st: St) -> St:
  match its:
    case Nil{}:
      st
    case +it <> rest:
      final(rest, it_st(it, st))

def rounds.go(pend: List<&2, Pe>) -> List<&2, Nat>:
  match pend:
    case Nil{}:
      []
    case Pe{r, s} <> t:
      r <> rounds.go(t)

def ids.go(pend: List<&2, Pe>) -> List<&2, Nat>:
  match pend:
    case Nil{}:
      []
    case Pe{r, s} <> t:
      s <> ids.go(t)

# the rounds / stash ids of the pending (deferred, unsettled) stashes
def rounds(st: St) -> List<&2, Nat>:
  match st:
    case St{rnd, pend, nxt, ver}:
      rounds.go(pend)

def pend_ids(st: St) -> List<&2, Nat>:
  match st:
    case St{rnd, pend, nxt, ver}:
      ids.go(pend)

def round(st: St) -> Nat:
  match st:
    case St{rnd, pend, nxt, ver}:
      rnd

def pending(st: St) -> List<&2, Pe>:
  match st:
    case St{rnd, pend, nxt, ver}:
      pend
