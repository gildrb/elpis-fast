# Independent reference for the ext 5108 replay loader (gdn_replay_gather.bend): the shared-memory cells
# gdn_rule_tokens<RULE_COMMIT, 1> reads in block (v chunk c, head, job), their owners, and the old loader's value
# of a cell, written without the kernel's element decode.
#   RULE_COMMIT reads, for every token s < S: m.k[s * 128 + kslice * 32 + j] for kslice < 4, j < 32 (k_rd of
#   gdn_rule_dot1 / gdn_rule_update) and m.v[s * 128 + c * 32 + lane], lane < 32 (the thread's vcol); m.q is
#   never read (v_out is dead in commit mode), m.gexp / m.beta come from gdn_rule_scalars, unchanged by 5108.
#   In 16-byte chunks: k chunk kc < 16 = cells 8 kc .. 8 kc + 7, v chunk vc < 4 = cells 32 c + 8 vc .. + 7.
import Base
import ./gdn_replay_gather.bend as Impl

def kchunk(+s: Nat, +kc: Nat) -> Impl.Loc:
  Impl.Loc{1n, s, Nat.mul(kc, 8n)}

def vchunk(+c: Nat, +s: Nat, +vc: Nat) -> Impl.Loc:
  Impl.Loc{2n, s, Nat.add(Nat.mul(c, 32n), Nat.mul(vc, 8n))}

# Owner of a chunk: token-major, the 16 k chunks, then the 4 v chunks of the block
def own_k(+s: Nat, +kc: Nat) -> Nat:
  Nat.add(Nat.mul(s, 20n), kc)

def own_v(+s: Nat, +vc: Nat) -> Nat:
  Nat.add(Nat.mul(s, 20n), Nat.add(16n, vc))

def loc_eq(a: Impl.Loc, b: Impl.Loc) -> Bool:
  match a b:
    case Impl.Loc{p1, s1, c1} Impl.Loc{+p2, +s2, +c2}:
      Bool.and(Nat.is_eq(p1, p2), Bool.and(Nat.is_eq(s1, s2), Nat.is_eq(c1, c2)))

# chunk l is read by block c at S tokens; own_loc = its owner (meaningful when read)
def in_read(+S: Nat, +c: Nat, l: Impl.Loc) -> Bool:
  match l:
    case Impl.Loc{+part, +s, +col}:
      Bool.and(Nat.is_lt(s, S),
        Bool.or(Bool.and(Nat.is_eq(part, 1n), Bool.and(Nat.is_lt(col, 128n), Nat.is_eq(Nat.mod(col, 8n), 0n))),
          Bool.and(Nat.is_eq(part, 2n),
            Bool.and(Nat.is_le(Nat.mul(c, 32n), col),
              Bool.and(Nat.is_lt(Nat.sub(col, Nat.mul(c, 32n)), 32n), Nat.is_eq(Nat.mod(Nat.sub(col, Nat.mul(c, 32n)), 8n), 0n))))))

def own_loc(+c: Nat, l: Impl.Loc) -> Nat:
  match l:
    case Impl.Loc{+part, +s, +col}:
      match part:
        case 0n:
          own_v(s, Nat.div(Nat.sub(col, Nat.mul(c, 32n)), 8n))
        case 1n+z:
          match z:
            case 0n:
              own_k(s, Nat.div(col, 8n))
            case 1n+w:
              own_v(s, Nat.div(Nat.sub(col, Nat.mul(c, 32n)), 8n))

# ---- finite checks (S tokens, block c), decided by evaluation
# into + injective: every active element stores to a read chunk and is that chunk's owner
def into.go(n: Nat, +e: Nat, +S: Nat, +c: Nat) -> Bool:
  match n:
    case 0n:
      True{}
    case 1n+m:
      Bool.and(Bool.and(in_read(S, c, Impl.dst(c, e)), Nat.is_eq(own_loc(c, Impl.dst(c, e)), e)),
        into.go(m, 1n+e, S, c))

def into(+S: Nat, +c: Nat) -> Bool:
  into.go(Nat.mul(S, Impl.parts()), 0n, S, c)

# onto: every read chunk's owner is active and stores exactly that chunk
def onto_k.go(n: Nat, +kc: Nat, +s: Nat, +S: Nat, +c: Nat) -> Bool:
  match n:
    case 0n:
      True{}
    case 1n+m:
      Bool.and(Bool.and(Impl.active(S, own_k(s, kc)), loc_eq(Impl.dst(c, own_k(s, kc)), kchunk(s, kc))),
        onto_k.go(m, 1n+kc, s, S, c))

def onto_v.go(n: Nat, +vc: Nat, +s: Nat, +S: Nat, +c: Nat) -> Bool:
  match n:
    case 0n:
      True{}
    case 1n+m:
      Bool.and(Bool.and(Impl.active(S, own_v(s, vc)), loc_eq(Impl.dst(c, own_v(s, vc)), vchunk(c, s, vc))),
        onto_v.go(m, 1n+vc, s, S, c))

def onto.go(n: Nat, +s: Nat, +S: Nat, +c: Nat) -> Bool:
  match n:
    case 0n:
      True{}
    case 1n+m:
      Bool.and(Bool.and(onto_k.go(16n, 0n, s, S, c), onto_v.go(4n, 0n, s, S, c)), onto.go(m, 1n+s, S, c))

def onto(+S: Nat, +c: Nat) -> Bool:
  onto.go(S, 0n, S, c)

# every S in 1..16 and every block c < 4
def all_c(n: Nat, +c: Nat, +S: Nat) -> Bool:
  match n:
    case 0n:
      True{}
    case 1n+m:
      Bool.and(Bool.and(into(S, c), onto(S, c)), all_c(m, 1n+c, S))

# the reference: all_c at every S' in S .. S + n - 1
def all_S_ref(n: Nat, +S: Nat) -> Bool:
  match n:
    case 0n:
      True{}
    case 1n+m:
      Bool.and(all_c(4n, 0n, S), all_S_ref(m, 1n+S))

# what cover() evaluates: all_c at the largest S' = S + n - 1 only. The checks at S' + 1 tokens imply those at S'
# (an element e < 20 S' that is its chunk's owner lies in a token < S'; every owner 20 s + p, s < S', p < 20, is
# active at S'), so this equals all_S_ref for every n and S (law gather_all_S, proven in
# gdn_replay_gather_proof.bend) and costs one S' instead of n, within the checker's evaluation budget.
def all_S(n: Nat, +S: Nat) -> Bool:
  match n:
    case 0n:
      True{}
    case 1n+m:
      all_c(4n, 0n, Nat.add(S, m))

def cover() -> Bool:
  all_S(Impl.gr_max_s(), 1n)

# threads: every element e < GR_MAX_S * 20 is thread e % 128's slot e / 128 < REPLAY_GI, and every
# (t < 128, i < REPLAY_GI) names a distinct element (decode back)
def thr_e.go(n: Nat, +e: Nat) -> Bool:
  match n:
    case 0n:
      True{}
    case 1n+m:
      Bool.and(Bool.and(Nat.is_lt(Nat.div(e, 128n), Impl.gi()), Nat.is_eq(Impl.elem(Nat.mod(e, 128n), Nat.div(e, 128n)), e)),
        thr_e.go(m, 1n+e))

def thr_t.go(n: Nat, +t: Nat, +i: Nat) -> Bool:
  match n:
    case 0n:
      True{}
    case 1n+m:
      Bool.and(Bool.and(Nat.is_eq(Nat.mod(Impl.elem(t, i), 128n), t), Nat.is_eq(Nat.div(Impl.elem(t, i), 128n), i)),
        thr_t.go(m, 1n+t, i))

def thr_i.go(n: Nat, +i: Nat) -> Bool:
  match n:
    case 0n:
      True{}
    case 1n+m:
      Bool.and(thr_t.go(128n, 0n, i), thr_i.go(m, 1n+i))

def threads() -> Bool:
  Bool.and(Nat.is_eq(Impl.gi(), 3n),
    Bool.and(thr_e.go(Nat.mul(Impl.gr_max_s(), Impl.parts()), 0n), thr_i.go(Impl.gi(), 0n)))

# ---- the old path's widened value of bf16 bits b: __bfloat162float(b) has fp32 bits b << 16
def old_widen(+b: Nat) -> Impl.W:
  Impl.W{0n, b}
