# Independent checker for a two-stage shared-memory pipeline, independent of the ext 3022 schedule it judges
# (pattn8i_pipe.bend supplies the epochs). State: what each of the two stages holds (tile + 1; 0 = nothing). An epoch
# (one barrier interval, Ep{rt, rs, wr, wt, ws}) is accepted iff
#   - its read names a real stage (rs < 2) and that stage holds exactly tile rt, i.e. tile rt was issued into it in an
#     earlier epoch (so it landed and is visible at this epoch's barrier) and nothing overwrote it since;
#   - its write, if any, names a real stage (ws < 2) other than the stage read in the same epoch (no barrier separates
#     the two: the cp.async may land while any warp still reads, a race);
# and the write takes effect for the following epochs.
import Base
import ./pattn8i_pipe.bend as P

def get(+s: Nat, +c0: Nat, +c1: Nat) -> Nat:
  match s:
    case 0n:
      c0
    case 1n+p:
      c1

def put0(+s: Nat, +t: Nat, +c0: Nat) -> Nat:
  match s:
    case 0n:
      t
    case 1n+p:
      c0

def put1(+s: Nat, +t: Nat, +c1: Nat) -> Nat:
  match s:
    case 0n:
      c1
    case 1n+p:
      t

def rd_ok(+rt: Nat, +rs: Nat, +c0: Nat, +c1: Nat) -> Bool:
  Bool.and(Nat.is_lt(rs, 2n), Nat.is_eq(get(rs, c0, c1), 1n+rt))

def wr_ok(+rs: Nat, +ws: Nat) -> Bool:
  Bool.and(Nat.is_lt(ws, 2n), Nat.is_ne(ws, rs))

# True iff every epoch of es is accepted, from stage contents (c0, c1)
def run(es: List<&2, P.Ep>, +c0: Nat, +c1: Nat) -> Bool:
  match es:
    case Nil{}:
      True{}
    case Con{e, rest}:
      match e:
        case P.Ep{+rt, +rs, wr, +wt, +ws}:
          match wr:
            case False{}:
              Bool.and(rd_ok(rt, rs, c0, c1), run(rest, c0, c1))
            case True{}:
              Bool.and(rd_ok(rt, rs, c0, c1),
                Bool.and(wr_ok(rs, ws), run(rest, put0(ws, 1n+wt, c0), put1(ws, 1n+wt, c1))))
