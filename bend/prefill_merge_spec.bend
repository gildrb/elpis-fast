# Independent reference for ext 5110g (prefill_merge.bend): the prefill() step it modifies, without the merge, i.e. the
# served 2048-row schedule (generator/job.py before 5110g; the lines below are unchanged by 5110g and cited in the
# served tree J). Same domain as prefill_merge.bend (text-only recurrent model, no prompt-cache page hits).
import Base

# J:1245-1247: the next piece ends at min(floor((kv_position + max_chunk_size) / PAGE_SIZE) * PAGE_SIZE, last_tok)
def piece_end(+k: Nat, +T: Nat, +mc: Nat, +P: Nat) -> Nat:
  Nat.min(Nat.mul(Nat.div(Nat.add(k, mc), P), P), T)

# J:1363-1368: the piece containing the last page boundary floor(last_tok / P) * P ends there (a separate forward for
# the last page, for the latest recurrent checkpoint)
def last_page(+T: Nat, +P: Nat) -> Nat:
  Nat.mul(Nat.div(T, P), P)

def stop_at.pick(c: Bool, +b: Nat, +e: Nat) -> Nat:
  match c:
    case True{}:
      b
    case False{}:
      e

def step(+k: Nat, +T: Nat, +mc: Nat, +P: Nat) -> Nat:
  stop_at.pick(Bool.and(Nat.is_lt(k, last_page(T, P)), Nat.is_le(last_page(T, P), piece_end(k, T, mc, P))), last_page(T, P),
    piece_end(k, T, mc, P))

def visits.go(fuel: Nat, done: Bool, +k: Nat, +T: Nat, +mc: Nat, +P: Nat) -> List<&2, Nat>:
  match fuel:
    case 0n:
      []
    case 1n+f:
      match done:
        case True{}:
          []
        case False{}:
          step(k, T, mc, P) <> visits.go(f, Nat.is_le(T, step(k, T, mc, P)), step(k, T, mc, P), T, mc, P)

def visits(fuel: Nat, +k: Nat, +T: Nat, +mc: Nat, +P: Nat) -> List<&2, Nat>:
  visits.go(fuel, Nat.is_le(T, k), k, T, mc, P)

# The rows a forward over [a, e) processes, in order
def rows(k: Nat, +a: Nat) -> List<&2, Nat>:
  match k:
    case 0n:
      []
    case 1n+j:
      a <> rows(j, 1n+a)

# J:1636-1640 is_checkpoint_boundary() in the prompt-prefill branch (seq_pos < prompt_len - 2 mc): the stash condition
# (seq_pos - cached_pages * PAGE_SIZE) % recurrent_checkpoint_interval_pp == 0
def boundary(+pos: Nat, +base: Nat, +pp: Nat) -> Bool:
  Nat.is_eq(Nat.mod(Nat.sub(pos, base), pp), 0n)

def any_boundary.keep(c: Bool, +x: Nat, rest: List<&2, Nat>) -> List<&2, Nat>:
  match c:
    case True{}:
      x <> rest
    case False{}:
      rest

def any_boundary(+xs: List<&2, Nat>, +base: Nat, +pp: Nat, +lim: Nat) -> List<&2, Nat>:
  match xs:
    case Nil{}:
      []
    case x <> rest:
      any_boundary.keep(Bool.and(boundary(x, base, pp), Nat.is_lt(x, lim)), x, any_boundary(rest, base, pp, lim))
