# The sequential tree verify round (exl3 0006-tree-verify, iterate_gen without
# ext 9601): the host waits for the draft, checks the builder record (~ok), and
# only then uploads the verify inputs (tree_round.verify_bytes: the TreeDesc,
# then the row positions base + depth[r], depth[r] = TreeDesc byte 16 + r; the
# host embedding of the row tokens) and runs the verify on them (~f). A record
# that fails the check fails the round before any verify runs. Independent of
# tree_pipe.bend's kernel indexing and readback logic; it shares only the data
# shapes. w is the committed GDN conv window the verify reads.
import Base
import ./tree_pipe.bend as Impl

type Round is Data:
  SeqFail{}
  SeqOk{ids: List<&2, U32>}

def depths(k: Nat, d: List<&2, Nat>) -> List<&2, Nat>:
  match k:
    case 0n:
      Nil{}
    case 1n+j:
      match d:
        case Nil{}:
          Nil{}
        case x <> rest:
          x <> depths(j, rest)

def skip(k: Nat, d: List<&2, Nat>) -> List<&2, Nat>:
  match k:
    case 0n:
      d
    case 1n+j:
      match d:
        case Nil{}:
          Nil{}
        case x <> rest:
          skip(j, rest)

def positions(+base: Nat, ds: List<&2, Nat>) -> List<&2, Nat>:
  match ds:
    case Nil{}:
      Nil{}
    case d <> rest:
      Nat.add(base, d) <> positions(base, rest)

def rows(~emb: Nat -> Nat, ts: List<&2, Nat>) -> List<&2, Nat>:
  match ts:
    case Nil{}:
      Nil{}
    case t <> rest:
      emb(t) <> rows(~emb, rest)

# The bytes and rows the sequential round uploads.
def upload(~emb: Nat -> Nat, +rec: Impl.Record, +base: Nat) -> Impl.Staged:
  match rec:
    case Impl.Record{+toks, +desc}:
      Impl.Staged{List.append(&2, Nat, desc, positions(base, depths(8n, skip(16n, desc)))), rows(~emb, toks)}

def run(checked: Bool, ids: List<&2, U32>) -> Round:
  match checked:
    case True{}:
      SeqOk{ids}
    case False{}:
      SeqFail{}

def seq_round(~f: Impl.Staged -> List<&2, Nat> -> List<&2, U32>, ~ok: Impl.Record -> Bool, ~emb: Nat -> Nat,
    +rec: Impl.Record, +base: Nat, +w: List<&2, Nat>) -> Round:
  run(ok(rec), f(upload(~emb, rec, base), w))
