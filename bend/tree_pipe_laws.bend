# Pipelined tree verify round (ext 9601, bend/tree_pipe.bend) against the
# sequential round (bend/tree_pipe_spec.bend), and through it the tree
# speculation invariance of bend/spec_inv_tree_laws.bend.
#
# Domain: every verify ~f (the verify's ids as a function of the inputs it
# reads), every host check ~ok of the builder record, every embedding table
# ~emb, every record, every position base, and every staged input v the
# verify may have consumed (any device fault or race included): the laws
# quantify over v instead of assuming the stage kernel ran correctly.
import Base
import ./tree_pipe.bend as Impl
import ./tree_pipe_spec.bend as Spec
import ./spec_inv_tree.bend as Tree
import ./spec_inv_spec.bend as Greedy
import ./exl3_tree_accept_spec.bend as Accept

# What the round emits: the ids it commits, or the sequential round's failure.
def emitted(o: Impl.Outcome) -> Spec.Round:
  match o:
    case Impl.Kept{ids}:
      Spec.SeqOk{ids}
    case Impl.Redone{ids}:
      Spec.SeqOk{ids}
    case Impl.Failed{}:
      Spec.SeqFail{}

# Whether the round committed the speculative verify (no recompute).
def kept(o: Impl.Outcome) -> Bool:
  match o:
    case Impl.Kept{ids}:
      True{}
    case Impl.Redone{ids}:
      False{}
    case Impl.Failed{}:
      False{}

def ids_of(r: Spec.Round) -> List<&2, U32>:
  match r:
    case Spec.SeqOk{ids}:
      ids
    case Spec.SeqFail{}:
      Nil{}

# P1 (refinement): for every staged input v, the pipelined round emits exactly
# what the sequential round emits: the same ids, or the same failure.
law pipe_refines_seq:
  for ~f: Impl.Staged -> List<&2, Nat> -> List<&2, U32>
  for ~ok: Impl.Record -> Bool
  for ~emb: Nat -> Nat
  for ~clob: Impl.Staged -> List<&2, Nat> -> List<&2, Nat>
  for +rec: Impl.Record
  for +v: Impl.Staged
  for +base: Nat
  for +w: List<&2, Nat>
  {emitted(Impl.pipe_round(~f, ~ok, ~emb, ~clob, rec, v, base, w)) == Spec.seq_round(~f, ~ok, ~emb, rec, base, w) : Spec.Round}

# P2: the speculative verify is committed only when the record passed the
# host check and the consumed input is exactly the sequential round's upload.
law pipe_kept_checked:
  for ~f: Impl.Staged -> List<&2, Nat> -> List<&2, U32>
  for ~ok: Impl.Record -> Bool
  for ~emb: Nat -> Nat
  for ~clob: Impl.Staged -> List<&2, Nat> -> List<&2, Nat>
  for +rec: Impl.Record
  for +v: Impl.Staged
  for +base: Nat
  for +w: List<&2, Nat>
  for k: {kept(Impl.pipe_round(~f, ~ok, ~emb, ~clob, rec, v, base, w)) == True{} : Bool}
  {v == Spec.upload(~emb, rec, base) : Impl.Staged} & {ok(rec) == True{} : Bool}

# P3: every mismatch path discards: a consumed input that differs from the
# upload is never committed.
law pipe_mismatch_discards:
  for ~f: Impl.Staged -> List<&2, Nat> -> List<&2, U32>
  for ~ok: Impl.Record -> Bool
  for ~emb: Nat -> Nat
  for ~clob: Impl.Staged -> List<&2, Nat> -> List<&2, Nat>
  for +rec: Impl.Record
  for +v: Impl.Staged
  for +base: Nat
  for +w: List<&2, Nat>
  for differ: {v == Spec.upload(~emb, rec, base) : Impl.Staged} -> Empty
  {kept(Impl.pipe_round(~f, ~ok, ~emb, ~clob, rec, v, base, w)) == False{} : Bool}

# P4: the stage kernel writes exactly the sequential round's upload for every
# well-shaped record (8 row tokens, a 128-byte TreeDesc) and base.
law stage_agree:
  for ~emb: Nat -> Nat
  for +toks: List<&2, Nat>
  for +desc: List<&2, Nat>
  for +base: Nat
  for +t8: {List.length(&2, Nat, toks) == 8n : Nat}
  for +d128: {List.length(&2, Nat, desc) == 128n : Nat}
  {Impl.stage(~emb, Impl.Record{toks, desc}, base) == Spec.upload(~emb, Impl.Record{toks, desc}, base) : Impl.Staged}

# P5: a fault-free staged round whose record passes the check is never
# recomputed.
law stage_kept:
  for ~f: Impl.Staged -> List<&2, Nat> -> List<&2, U32>
  for ~ok: Impl.Record -> Bool
  for ~emb: Nat -> Nat
  for ~clob: Impl.Staged -> List<&2, Nat> -> List<&2, Nat>
  for +toks: List<&2, Nat>
  for +desc: List<&2, Nat>
  for +base: Nat
  for +w: List<&2, Nat>
  for +t8: {List.length(&2, Nat, toks) == 8n : Nat}
  for +d128: {List.length(&2, Nat, desc) == 128n : Nat}
  for accepted: {ok(Impl.Record{toks, desc}) == True{} : Bool}
  {kept(Impl.pipe_round(~f, ~ok, ~emb, ~clob, Impl.Record{toks, desc},
      Impl.stage(~emb, Impl.Record{toks, desc}, base), base, w)) == True{} : Bool}

# spec_inv_tree's T0 (tree_round_commits_greedy) for a round whose verify ids
# are ids: the committed ids followed by what greedy decoding still emits are
# the greedy output, and a round that does not end the job leaves budget.
def commits_greedy(~row: List<&2, U32> -> Nat -> U32, +seq: List<&2, U32>, +t: Tree.Draft,
    +stops: List<&2, U32>, +bp: Nat, +checkpoint: Nat, +ids: List<&2, U32>) -> Type:
  {Greedy.greedy(~row, 1n+bp, stops, seq) ==
    List.append(&2, U32,
      List.take(&2, U32, Tree.path_ids(ids, Tree.nodes(t)),
        Accept.count(Accept.accept_tree(Tree.nodes(t), ids, stops, 1n+bp, checkpoint))),
      Greedy.after(~row, Accept.eos(Accept.accept_tree(Tree.nodes(t), ids, stops, 1n+bp, checkpoint)),
        Nat.sub(1n+bp, Accept.count(Accept.accept_tree(Tree.nodes(t), ids, stops, 1n+bp, checkpoint))),
        stops,
        List.append(&2, U32, seq,
          List.take(&2, U32, Tree.path_ids(ids, Tree.nodes(t)),
            Accept.count(Accept.accept_tree(Tree.nodes(t), ids, stops, 1n+bp, checkpoint)))))) : List<&2, U32>} &
  {Bool.or(Accept.eos(Accept.accept_tree(Tree.nodes(t), ids, stops, 1n+bp, checkpoint)),
    Nat.is_lt(Accept.count(Accept.accept_tree(Tree.nodes(t), ids, stops, 1n+bp, checkpoint)), 1n+bp)) == True{} : Bool}

# P6 (refinement of spec_inv_tree): when the record passes the check and the
# verify of the sequential upload is the tree model's verify (the engine
# hypothesis of spec_inv_tree, under ~rinv), the ids the pipelined round emits,
# whatever input v the verify consumed, commit the greedy continuation.
law pipe_round_commits_greedy:
  for ~row: List<&2, U32> -> Nat -> U32
  for ~rinv: @p:List<&2, U32> -> @r:Nat -> {row(p, r) == row(p, 0n) : U32}
  for ~f: Impl.Staged -> List<&2, Nat> -> List<&2, U32>
  for ~ok: Impl.Record -> Bool
  for ~emb: Nat -> Nat
  for ~clob: Impl.Staged -> List<&2, Nat> -> List<&2, Nat>
  for +rec: Impl.Record
  for +v: Impl.Staged
  for +base: Nat
  for +w: List<&2, Nat>
  for +seq: List<&2, U32>
  for +t: Tree.Draft
  for +stops: List<&2, U32>
  for +bp: Nat
  for +checkpoint: Nat
  for legal: {Tree.legal(t) == True{} : Bool}
  for accepted: {ok(rec) == True{} : Bool}
  for conf: {f(Spec.upload(~emb, rec, base), w) == Tree.verify(~row, seq, Tree.nodes(t)) : List<&2, U32>}
  commits_greedy(~row, seq, t, stops, bp, checkpoint, ids_of(emitted(Impl.pipe_round(~f, ~ok, ~emb, ~clob, rec, v, base, w))))

# P7 (mutation witnesses). (a) The round without the readback comparison
# (commit the speculative verify whenever the record passes) is not a
# refinement: with a verify that reads the first staged byte and an input of
# one byte where the upload has none, it emits [1] while the sequential and the
# pipelined round emit [0]. (b) The round whose discard does not restore the
# conv window (the recompute reads the window the speculative verify
# clobbered) is not a refinement: with a verify that reads the window and a
# clobber that writes [1], it emits [1] while the sequential and the pipelined
# round emit [0].
def probe(s: Impl.Staged, w: List<&2, Nat>) -> List<&2, U32>:
  match s:
    case Impl.Staged{bs, rs}:
      match bs:
        case Nil{}:
          [0]
        case b <> rest:
          [1]

def peek(s: Impl.Staged, w: List<&2, Nat>) -> List<&2, U32>:
  match w:
    case Nil{}:
      [0]
    case x <> rest:
      [1]

def pass(r: Impl.Record) -> Bool:
  True{}

def ident(n: Nat) -> Nat:
  n

def keep_w(s: Impl.Staged, w: List<&2, Nat>) -> List<&2, Nat>:
  w

def smash(s: Impl.Staged, w: List<&2, Nat>) -> List<&2, Nat>:
  [1n]

def commit_if(checked: Bool, ids: List<&2, U32>) -> Impl.Outcome:
  match checked:
    case True{}:
      Impl.Kept{ids}
    case False{}:
      Impl.Failed{}

def unchecked(~f: Impl.Staged -> List<&2, Nat> -> List<&2, U32>, ~ok: Impl.Record -> Bool, +rec: Impl.Record,
    +v: Impl.Staged, +w: List<&2, Nat>) -> Impl.Outcome:
  commit_if(ok(rec), f(v, w))

def unrestored(~f: Impl.Staged -> List<&2, Nat> -> List<&2, U32>, ~ok: Impl.Record -> Bool, ~emb: Nat -> Nat,
    ~clob: Impl.Staged -> List<&2, Nat> -> List<&2, Nat>, +rec: Impl.Record, +v: Impl.Staged, +base: Nat,
    +w: List<&2, Nat>) -> Impl.Outcome:
  +e = Impl.expect(~emb, rec, base)
  Impl.decide(ok(rec), Impl.staged_eq(v, e), f(v, w), f(e, clob(v, w)))

law unchecked_differs:
  {emitted(unchecked(~probe, ~pass, Impl.Record{Nil{}, Nil{}}, Impl.Staged{[1n], Nil{}}, Nil{})) == Spec.SeqOk{[1]} : Spec.Round} &
  {Spec.seq_round(~probe, ~pass, ~ident, Impl.Record{Nil{}, Nil{}}, 0n, Nil{}) == Spec.SeqOk{[0]} : Spec.Round} &
  {emitted(Impl.pipe_round(~probe, ~pass, ~ident, ~keep_w, Impl.Record{Nil{}, Nil{}}, Impl.Staged{[1n], Nil{}}, 0n, Nil{})) == Spec.SeqOk{[0]} : Spec.Round}

law unrestored_differs:
  {emitted(unrestored(~peek, ~pass, ~ident, ~smash, Impl.Record{Nil{}, Nil{}}, Impl.Staged{[1n], Nil{}}, 0n, Nil{})) == Spec.SeqOk{[1]} : Spec.Round} &
  {Spec.seq_round(~peek, ~pass, ~ident, Impl.Record{Nil{}, Nil{}}, 0n, Nil{}) == Spec.SeqOk{[0]} : Spec.Round} &
  {emitted(Impl.pipe_round(~peek, ~pass, ~ident, ~smash, Impl.Record{Nil{}, Nil{}}, Impl.Staged{[1n], Nil{}}, 0n, Nil{})) == Spec.SeqOk{[0]} : Spec.Round}
