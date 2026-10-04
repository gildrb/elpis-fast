# Pipelined tree verify round (ext 9601: exllamav3_ext/tree_pipe.cu and
# generator/tree_pipe.py + generator.py), the model the laws in
# tree_pipe_laws.bend read; the sequential round it refines is
# tree_pipe_spec.bend.
#
# The tree builder (ext 9008) writes a record: the 8 row tokens (row 0 the
# anchor) and the 128-byte TreeDesc. A verify reads Staged inputs: bytes = its
# static buffer as 136 entries (the 128 TreeDesc bytes, then the 8 int32 row
# positions) and rows = the 8 embedded input rows, each abstracted to the code
# emb(token) of the target embedding's row (~emb is the table; the widening to
# float32 is exact).
#
# stage: what tree_verify_stage_kernel writes, one entry per thread as the
# kernel indexes it (block (0, 0): thread i < 128 copies TreeDesc byte i,
# thread r < 8 writes base + depth byte 16 + r; block (bx, r): row r is the
# table row of token r).
#
# pipe_round: the generator's round. The verify runs on the device-staged
# inputs v; after its launch the host runs the sequential round's own check of
# the record (~ok, tree_round.check_output with the proven derive and the
# drafted-ids check) and compares v, read back, with the bytes the sequential
# round uploads and the rows its host embedding computes (expect). Check fails:
# the round fails (as the sequential round does). Readback equal: the
# speculative verify's ids are kept. Readback different: the speculative
# verify is discarded and the round recomputed from expect.
#
# Boundaries (not modelled): ~f, the verify's ids as a function of the inputs
# it reads (everything else fixed: the kernels are deterministic, the
# speculative verify leaves no state the recompute reads, generator/
# tree_pipe.py discard); readbacks are faithful copies taken in stream order
# after the stage (the host waits on the side stream's event); positions are
# Nat sums because the host refuses base + 255 >= 2^31; int32 and int64 byte
# encodings are those of the host's struct.pack.
import Base

type Record is Data:
  Record{toks: List<&2, Nat>, desc: List<&2, Nat>}

type Staged is Data:
  Staged{bytes: List<&2, Nat>, rows: List<&2, Nat>}

type Outcome is Data:
  Kept{ids: List<&2, U32>}
  Redone{ids: List<&2, U32>}
  Failed{}

# Entry i of a byte list (0 past the end, never read in range).
def at(xs: List<&2, Nat>, i: Nat) -> Nat:
  match xs:
    case Nil{}:
      0n
    case x <> rest:
      match i:
        case 0n:
          x
        case 1n+j:
          at(rest, j)

# Threads i .. i + n - 1 of the TreeDesc copy: dev[i] = desc[i].
def copy_desc(n: Nat, +i: Nat, +d: List<&2, Nat>) -> List<&2, Nat>:
  match n:
    case 0n:
      Nil{}
    case 1n+m:
      at(d, i) <> copy_desc(m, 1n+i, d)

# Threads r .. r + n - 1 of the position write: pos[r] = base + dp[r], dp the
# TreeDesc's depth bytes onward (rec + 64 + 16).
def write_pos(n: Nat, +r: Nat, +dp: List<&2, Nat>, +base: Nat) -> List<&2, Nat>:
  match n:
    case 0n:
      Nil{}
    case 1n+m:
      Nat.add(base, at(dp, r)) <> write_pos(m, 1n+r, dp, base)

# Grid rows r .. r + n - 1 of the gather: row r = the table row of token r.
def gather(~emb: Nat -> Nat, n: Nat, +r: Nat, +ts: List<&2, Nat>) -> List<&2, Nat>:
  match n:
    case 0n:
      Nil{}
    case 1n+m:
      emb(at(ts, r)) <> gather(~emb, m, 1n+r, ts)

# One launch of tree_verify_stage_kernel.
def stage(~emb: Nat -> Nat, +rec: Record, +base: Nat) -> Staged:
  match rec:
    case Record{+toks, +desc}:
      Staged{List.append(&2, Nat, copy_desc(128n, 0n, desc),
          write_pos(8n, 0n, List.drop(&2, Nat, desc, 16n), base)),
        gather(~emb, 8n, 0n, toks)}

# The host's expectation (tree_round.verify_bytes(desc, depth, base): the
# TreeDesc, then base + depth[r]; Embedding.forward of the row tokens).
def add_base(+base: Nat, ds: List<&2, Nat>) -> List<&2, Nat>:
  match ds:
    case Nil{}:
      Nil{}
    case d <> rest:
      Nat.add(base, d) <> add_base(base, rest)

def embed(~emb: Nat -> Nat, ts: List<&2, Nat>) -> List<&2, Nat>:
  match ts:
    case Nil{}:
      Nil{}
    case t <> rest:
      emb(t) <> embed(~emb, rest)

def expect(~emb: Nat -> Nat, +rec: Record, +base: Nat) -> Staged:
  match rec:
    case Record{+toks, +desc}:
      Staged{List.append(&2, Nat, desc,
          add_base(base, List.take(&2, Nat, List.drop(&2, Nat, desc, 16n), 8n))),
        embed(~emb, toks)}

# Bytewise comparison of the readback with the expectation.
def same(xs: List<&2, Nat>, ys: List<&2, Nat>) -> Bool:
  match xs ys:
    case Nil{} Nil{}:
      True{}
    case Nil{} y <> b:
      False{}
    case x <> a Nil{}:
      False{}
    case x <> a y <> b:
      Bool.and(Nat.is_eq(x, y), same(a, b))

def staged_eq(+v: Staged, +e: Staged) -> Bool:
  match v e:
    case Staged{vb, vr} Staged{eb, er}:
      Bool.and(same(vb, eb), same(vr, er))

# The decision after the launch: record check, then the readback comparison.
def decide(checked: Bool, equal: Bool, kept: List<&2, U32>, redone: List<&2, U32>) -> Outcome:
  match checked:
    case False{}:
      Failed{}
    case True{}:
      match equal:
        case True{}:
          Kept{kept}
        case False{}:
          Redone{redone}

def pipe_round(~f: Staged -> List<&2, U32>, ~ok: Record -> Bool, ~emb: Nat -> Nat,
    +rec: Record, +v: Staged, +base: Nat) -> Outcome:
  +e = expect(~emb, rec, base)
  decide(ok(rec), staged_eq(v, e), f(v), f(e))
