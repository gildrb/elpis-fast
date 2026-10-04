# Independent reference for ext 9502 (EXL3_PREFILL_NOSYNC): what the host/stream event trace of a run of generator
# iterates must satisfy, and the pre-9502 (blocking) trace. Nothing here reads the impl model (prefill_nosync.bend);
# the impl imports this file only for the trace alphabet (Ev) and the iterate inputs (Pu, It).
#
# The trace is the sequence of host actions in host program order. Stream events (Up, Fw, Dc, Cp) are the host
# enqueueing work on the one CUDA stream the generator uses, so their trace order is their stream (FIFO) order; host
# events (Bg, Rd0, Rd, Ck0, Ck, Wt) run on the host at their trace position. Happens-before is therefore:
#   - host program order: an event happens before every later event of the trace;
#   - stream FIFO: a stream event completes before any later stream event starts;
#   - record -> wait: Wt{s} returns only after the event recorded right after Cp{s} has completed, i.e. after Cp{s}
#     and every stream event before it (a blocking stash's .cpu() synchronizes the same way).
# The observer below reads the trace left to right; every field is a function of the prefix before the current
# position (the round = number of Bg so far, the forwards so far, the copies enqueued and not yet waited, ...), and
# each ok bit is the conjunction, over all positions, of a check of the event at that position against its prefix.
import Base

# What a put call meets (Generator / Job.maybe_stash_recurrent): no put call, a put whose key is already cached
# (settle only), or a put that stashes a new checkpoint.
type Pu is Data:
  No{}
  Hit{}
  New{}

# One Generator.iterate of the served single-job recurrent model:
#   rs: iterate_start_jobs reads a stash (get_stashed);       fw: job.prefill runs one chunk forward;
#   lp: the last-page put after that forward;                cp: recurrent_checkpoint's put;
#   en: EXL3_PREFILL_NOSYNC on (and the state can defer, no TP); gf: iterate_gen runs a decode forward;
#   rq: the requeue put after the decode;                     re: a reader after the iterate (prune_stranded / persist).
type It is Data:
  Iter{rs: Bool, fw: Bool, lp: Pu, cp: Pu, en: Bool, gf: Bool, rq: Pu, re: Bool}

type Ev is Data:
  Bg{}                          # host: Generator.iterate starts a round
  Rd0{}                         # host: a reader of stash contents starts
  Rd{}                          # host: the reader reads stash contents
  Up{c: Nat}                    # stream: upload of forward c's inputs (block table, cache_seqlens, embedded rows)
  Fw{c: Nat}                    # stream: prefill forward c (writes the recurrent state)
  Dc{c: Nat}                    # stream: decode forward c (writes the recurrent state)
  Ck0{}                         # host: recurrent_checkpoint() starts
  Ck{}                          # host: recurrent_checkpoint() starts its stash loop
  Cp{s: Nat, c: Nat, d: Bool}   # stream: device-to-host copies of stash s (+ event record when d = deferred);
                                #   c = the number of forwards the host intends the checkpoint to contain
  Wt{s: Nat}                    # host: wait for stash s's copies and install its layers

# A copy enqueued and not yet waited: stash s, enqueued in round r.
type Out is Data:
  Oc{s: Nat, r: Nat}

type Obs is Data:
  Ob{rnd: Nat, nf: Nat, ups: List<&2, Nat>, out: List<&2, Out>, ns: Nat, bk: Nat, fq: Bool, ks: Bool, rdq: Bool,
     o1: Bool, o2: Bool, o3: Bool, o4: Bool, o5: Bool}
# rnd: rounds begun; nf: forwards enqueued (Fw, Dc); ups: uploads enqueued and not yet consumed by their forward;
# out: deferred copies enqueued, not yet waited, in enqueue order; ns: copies enqueued; bk: 1 + s while the previous
# event was the blocking copy Cp{s, _, False} (0 otherwise); fq: a prefill forward was enqueued in this round; ks:
# recurrent_checkpoint has started in this round; rdq: a reader is settling (between Rd0 and Rd);
# o1..o5: the checks of laws 1..5 at every position so far.

def init() -> Obs:
  Ob{0n, 0n, [], [], 0n, 0n, False{}, False{}, False{}, True{}, True{}, True{}, True{}, True{}}

def mem(xs: List<&2, Nat>, +x: Nat) -> Bool:
  match xs:
    case Nil{}:
      False{}
    case y <> rest:
      Bool.or(Nat.is_eq(y, x), mem(rest, x))

def del.pick(hit: Bool, +y: Nat, t: List<&2, Nat>, rest: List<&2, Nat>) -> List<&2, Nat>:
  match hit:
    case True{}:
      t
    case False{}:
      y <> rest

# xs without its first x
def del(xs: List<&2, Nat>, +x: Nat) -> List<&2, Nat>:
  match xs:
    case Nil{}:
      []
    case +y <> +t:
      del.pick(Nat.is_eq(y, x), y, t, del(t, x))

def is_nil(xs: List<&2, Out>) -> Bool:
  match xs:
    case Nil{}:
      True{}
    case y <> rest:
      False{}

# every outstanding copy was enqueued in round >= r (none left from an earlier round)
def all_ge(xs: List<&2, Out>, +r: Nat) -> Bool:
  match xs:
    case Nil{}:
      True{}
    case Oc{s, +q} <> rest:
      Bool.and(Nat.is_le(r, q), all_ge(rest, r))

# a deferred copy of stash s is outstanding
def has(xs: List<&2, Out>, +s: Nat) -> Bool:
  match xs:
    case Nil{}:
      False{}
    case Oc{+s2, q} <> rest:
      Bool.or(Nat.is_eq(s2, s), has(rest, s))

# Law 3 at a wait for the deferred copy enqueued in round r: inside a reader's settle; or in the next round (1 + r ==
# rnd), after that round's prefill forward was enqueued (fq) or inside its recurrent_checkpoint (ks).
def w_ok.pick(hit: Bool, +r: Nat, +rnd: Nat, +fq: Bool, +ks: Bool, +rdq: Bool, rest: Bool) -> Bool:
  match hit:
    case True{}:
      Bool.or(rdq, Bool.and(Bool.or(ks, fq), Nat.is_eq(1n+r, rnd)))
    case False{}:
      rest

def w_ok(xs: List<&2, Out>, +s: Nat, +rnd: Nat, +fq: Bool, +ks: Bool, +rdq: Bool) -> Bool:
  match xs:
    case Nil{}:
      True{}
    case Oc{+s2, +q} <> rest:
      w_ok.pick(Nat.is_eq(s2, s), q, rnd, fq, ks, rdq, w_ok(rest, s, rnd, fq, ks, rdq))

def rm.pick(hit: Bool, +y: Out, t: List<&2, Out>, rest: List<&2, Out>) -> List<&2, Out>:
  match hit:
    case True{}:
      t
    case False{}:
      y <> rest

# xs without the first copy of stash s
def rm(xs: List<&2, Out>, +s: Nat) -> List<&2, Out>:
  match xs:
    case Nil{}:
      []
    case Oc{+s2, +q} <> +t:
      rm.pick(Nat.is_eq(s2, s), Oc{s2, q}, t, rm(t, s))

# 1 + b: the blocking copy of stash b was the previous event, so this event must be its wait
def nb(o5: Bool, +bk: Nat) -> Bool:
  Bool.and(o5, Nat.is_eq(bk, 0n))

# Wt{s}: the blocking copy's wait (bk = 1 + s), or the wait of the outstanding deferred copy of s
def wt(bk: Nat, +s: Nat, +rnd: Nat, +nf: Nat, ups: List<&2, Nat>, +out: List<&2, Out>, +ns: Nat, +fq: Bool, +ks: Bool,
    +rdq: Bool, o1: Bool, o2: Bool, o3: Bool, o4: Bool, o5: Bool) -> Obs:
  match bk:
    case 0n:
      Ob{rnd, nf, ups, rm(out, s), ns, 0n, fq, ks, rdq, o1, o2, Bool.and(o3, w_ok(out, s, rnd, fq, ks, rdq)), o4,
        Bool.and(o5, has(out, s))}
    case 1n+b:
      Ob{rnd, nf, ups, out, ns, 0n, fq, ks, rdq, o1, o2, o3, o4, Bool.and(o5, Nat.is_eq(b, s))}

# Cp{s, c, d}: law 2 (c == forwards enqueued before it), law 3 (no copy of an earlier round outstanding), law 5 (ids
# fresh: s == copies so far); a deferred copy becomes outstanding, a blocking one must be waited next.
def cp(d: Bool, +s: Nat, +c: Nat, +rnd: Nat, +nf: Nat, ups: List<&2, Nat>, +out: List<&2, Out>, +ns: Nat, +bk: Nat,
    fq: Bool, ks: Bool, rdq: Bool, o1: Bool, o2: Bool, o3: Bool, o4: Bool, o5: Bool) -> Obs:
  match d:
    case True{}:
      Ob{rnd, nf, ups, List.append(&2, Out, out, [Oc{s, rnd}]), 1n+ns, bk, fq, ks, rdq, o1, Bool.and(o2, Nat.is_eq(c, nf)),
        Bool.and(o3, all_ge(out, rnd)), o4, Bool.and(nb(o5, bk), Nat.is_eq(s, ns))}
    case False{}:
      Ob{rnd, nf, ups, out, 1n+ns, 1n+s, fq, ks, rdq, o1, Bool.and(o2, Nat.is_eq(c, nf)),
        Bool.and(o3, all_ge(out, rnd)), o4, Bool.and(nb(o5, bk), Nat.is_eq(s, ns))}

def step(o: Obs, e: Ev) -> Obs:
  match o:
    case Ob{+rnd, +nf, +ups, +out, +ns, +bk, +fq, +ks, +rdq, o1, o2, o3, o4, o5}:
      match e:
        case Bg{}:
          Ob{1n+rnd, nf, ups, out, ns, bk, False{}, False{}, rdq, o1, o2, o3, o4, nb(o5, bk)}
        case Rd0{}:
          Ob{rnd, nf, ups, out, ns, bk, fq, ks, True{}, o1, o2, o3, o4, nb(o5, bk)}
        case Rd{}:
          # law 4: no copy outstanding when stash contents are read
          Ob{rnd, nf, ups, out, ns, bk, fq, ks, False{}, o1, o2, o3, Bool.and(o4, is_nil(out)), nb(o5, bk)}
        case Up{+c}:
          Ob{rnd, nf, c <> ups, out, ns, bk, fq, ks, rdq, o1, o2, o3, o4, nb(o5, bk)}
        case Fw{+c}:
          # law 1: its upload was enqueued before it; law 3: prefill forwards precede the round's recurrent_checkpoint
          Ob{rnd, 1n+nf, del(ups, c), out, ns, bk, True{}, ks, rdq, Bool.and(o1, mem(ups, c)), o2,
            Bool.and(o3, Bool.not(ks)), o4, nb(o5, bk)}
        case Dc{c}:
          Ob{rnd, 1n+nf, ups, out, ns, bk, fq, ks, rdq, o1, o2, o3, o4, nb(o5, bk)}
        case Ck0{}:
          Ob{rnd, nf, ups, out, ns, bk, fq, True{}, rdq, o1, o2, o3, o4, nb(o5, bk)}
        case Ck{}:
          # law 3: every copy of an earlier round was waited before the checkpoint's stash loop
          Ob{rnd, nf, ups, out, ns, bk, fq, ks, rdq, o1, o2, Bool.and(o3, all_ge(out, rnd)), o4, nb(o5, bk)}
        case Cp{+s, +c, d}:
          cp(d, s, c, rnd, nf, ups, out, ns, bk, fq, ks, rdq, o1, o2, o3, o4, o5)
        case Wt{+s}:
          wt(bk, s, rnd, nf, ups, out, ns, fq, ks, rdq, o1, o2, o3, o4, o5)

def run(t: List<&2, Ev>, o: Obs) -> Obs:
  match t:
    case Nil{}:
      o
    case e <> rest:
      run(rest, step(o, e))

def ok1(o: Obs) -> Bool:
  match o:
    case Ob{rnd, nf, ups, out, ns, bk, fq, ks, rdq, o1, o2, o3, o4, o5}:
      o1

def ok2(o: Obs) -> Bool:
  match o:
    case Ob{rnd, nf, ups, out, ns, bk, fq, ks, rdq, o1, o2, o3, o4, o5}:
      o2

def ok3(o: Obs) -> Bool:
  match o:
    case Ob{rnd, nf, ups, out, ns, bk, fq, ks, rdq, o1, o2, o3, o4, o5}:
      o3

def ok4(o: Obs) -> Bool:
  match o:
    case Ob{rnd, nf, ups, out, ns, bk, fq, ks, rdq, o1, o2, o3, o4, o5}:
      o4

def ok5(o: Obs) -> Bool:
  match o:
    case Ob{rnd, nf, ups, out, ns, bk, fq, ks, rdq, o1, o2, o3, o4, o5}:
      o5

def out_ids.go(xs: List<&2, Out>) -> List<&2, Nat>:
  match xs:
    case Nil{}:
      []
    case Oc{s, q} <> rest:
      s <> out_ids.go(rest)

# the stashes whose copies are enqueued and not yet waited at the end of the trace
def out_ids(o: Obs) -> List<&2, Nat>:
  match o:
    case Ob{rnd, nf, ups, out, ns, bk, fq, ks, rdq, o1, o2, o3, o4, o5}:
      out_ids.go(out)

def blk(o: Obs) -> Nat:
  match o:
    case Ob{rnd, nf, ups, out, ns, bk, fq, ks, rdq, o1, o2, o3, o4, o5}:
      bk

# ---------------------------------------------------------------------------------------------
# The pre-9502 behaviour (blocking stash: .cpu() copies then waits; no settle anywhere). s = next stash id,
# c = forwards so far.

def old_put(u: Pu, +s: Nat, +c: Nat) -> List<&2, Ev>:
  match u:
    case No{}:
      []
    case Hit{}:
      []
    case New{}:
      [Cp{s, c, False{}}, Wt{s}]

def old_ns(u: Pu, +s: Nat) -> Nat:
  match u:
    case No{}:
      s
    case Hit{}:
      s
    case New{}:
      1n+s

def old_rd(r: Bool) -> List<&2, Ev>:
  match r:
    case True{}:
      [Rd0{}, Rd{}]
    case False{}:
      []

# prefill: the chunk's upload and forward, then the last-page put
def old_pf(fw: Bool, lp: Pu, +s: Nat, +c: Nat) -> List<&2, Ev>:
  match fw:
    case True{}:
      List.append(&2, Ev, [Up{1n+c}, Fw{1n+c}], old_put(lp, s, 1n+c))
    case False{}:
      []

def old_pf_s(fw: Bool, lp: Pu, +s: Nat) -> Nat:
  match fw:
    case True{}:
      old_ns(lp, s)
    case False{}:
      s

def old_fwd(fw: Bool, +c: Nat) -> Nat:
  match fw:
    case True{}:
      1n+c
    case False{}:
      c

def old_dc(gf: Bool, +c: Nat) -> List<&2, Ev>:
  match gf:
    case True{}:
      [Dc{1n+c}]
    case False{}:
      []

def old_it_ev(it: It, +s: Nat, +c: Nat) -> List<&2, Ev>:
  match it:
    case Iter{rs, +fw, +lp, +cp, en, +gf, +rq, re}:
      Bg{} <> List.append(&2, Ev, old_rd(rs),
        List.append(&2, Ev, old_pf(fw, lp, s, c),
          Ck0{} <> Ck{} <> List.append(&2, Ev, old_put(cp, old_pf_s(fw, lp, s), old_fwd(fw, c)),
            List.append(&2, Ev, old_dc(gf, old_fwd(fw, c)),
              List.append(&2, Ev, old_put(rq, old_ns(cp, old_pf_s(fw, lp, s)), old_fwd(gf, old_fwd(fw, c))), old_rd(re))))))

def old_it_s(it: It, +s: Nat) -> Nat:
  match it:
    case Iter{rs, fw, lp, cp, en, gf, rq, re}:
      old_ns(rq, old_ns(cp, old_pf_s(fw, lp, s)))

def old_it_c(it: It, +c: Nat) -> Nat:
  match it:
    case Iter{rs, fw, lp, cp, en, gf, rq, re}:
      old_fwd(gf, old_fwd(fw, c))

def old_trace(its: List<&2, It>, +s: Nat, +c: Nat) -> List<&2, Ev>:
  match its:
    case Nil{}:
      []
    case +it <> rest:
      List.append(&2, Ev, old_it_ev(it, s, c), old_trace(rest, old_it_s(it, s), old_it_c(it, c)))

# ---------------------------------------------------------------------------------------------
# Law 6's state predicate over (current round, rounds of the pending deferred stashes): every pending entry was
# deferred in the current or the previous round.
def inv(+rnd: Nat, rs: List<&2, Nat>) -> Bool:
  match rs:
    case Nil{}:
      True{}
    case +r <> rest:
      Bool.and(Nat.is_le(rnd, 1n+r), Bool.and(Nat.is_le(r, rnd), inv(rnd, rest)))

# every pending entry was deferred in the current round (or later): what holds when an iterate ends
def all_at_least(+rnd: Nat, rs: List<&2, Nat>) -> Bool:
  match rs:
    case Nil{}:
      True{}
    case +r <> rest:
      Bool.and(Nat.is_le(rnd, r), all_at_least(rnd, rest))

# EXL3_PREFILL_NOSYNC=0 in every iterate (the off switch)
def all_off(its: List<&2, It>) -> Bool:
  match its:
    case Nil{}:
      True{}
    case Iter{rs, fw, lp, cp, en, gf, rq, re} <> rest:
      Bool.and(Bool.not(en), all_off(rest))
