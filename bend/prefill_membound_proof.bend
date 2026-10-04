# Proofs of prefill_membound_laws.bend (ext 5110h). Within the stage bound the 5110h decision is 5110g's, so the merged
# step reuses prefill_merge_proof.bend's law 1; past it no step merges. Nat lemmas from gemm_m16_partition_proof.bend (Ar).
import Base
import ./prefill_membound.bend as Impl
import ./prefill_merge.bend as M
import ./prefill_merge_spec.bend as Spec
import ./prefill_membound_laws.bend as Laws
import ./prefill_merge_laws.bend as PML
import ./prefill_merge_proof.bend as PM
import ./gemm_m16_partition_proof.bend as Ar

# ---------------------------------------------------------------------------------------------
# Bridges to 5110g

# with s <= B the decision is 5110g's
def mb_ok_eq(+en: Bool, +k: Nat, +T: Nat, +mc: Nat, +P: Nat, +G: Nat, +s: Nat, +B: Nat, w: {Nat.is_le(s, B) == True{} : Bool})
  -> {Impl.merge_ok(en, k, T, mc, P, G, s, B) == M.merge_ok(en, k, T, mc, P, G) : Bool}:
  %Equal.sym(Bool, Nat.is_le(s, B), True{}, w)
    : {Bool.and(en, Bool.and(Nat.is_le(T, G), Bool.and(_, PM.pm_rest(k, T, mc, P)))) == M.merge_ok(en, k, T, mc, P, G) : Bool}
  {==}

def mb_step_eq(+en: Bool, +k: Nat, +T: Nat, +mc: Nat, +P: Nat, +G: Nat, +s: Nat, +B: Nat, w: {Nat.is_le(s, B) == True{} : Bool})
  -> {Impl.step(en, k, T, mc, P, G, s, B) == M.step(en, k, T, mc, P, G) : Nat}:
  Equal.cong(Bool, Nat, z => M.cut(k, T, P, M.widen(z, M.base_end(k, T, mc, P), mc)),
    Impl.merge_ok(en, k, T, mc, P, G, s, B), M.merge_ok(en, k, T, mc, P, G), mb_ok_eq(en, k, T, mc, P, G, s, B, w))

def mb_go_eq(fuel: Nat, done: Bool, +en: Bool, +k: Nat, +T: Nat, +mc: Nat, +P: Nat, +G: Nat, +s: Nat, +B: Nat,
    +w: {Nat.is_le(s, B) == True{} : Bool})
  -> {Impl.visits.go(fuel, done, en, k, T, mc, P, G, s, B) == M.visits.go(fuel, done, en, k, T, mc, P, G) : List<&2, Nat>}:
  match fuel:
    case 0n:
      {==}
    case 1n+f:
      match done:
        case True{}:
          {==}
        case False{}:
          +x = Impl.step(en, k, T, mc, P, G, s, B)
          +y = M.step(en, k, T, mc, P, G)
          +se = mb_step_eq(en, k, T, mc, P, G, s, B, w)
          Equal.trans(List<&2, Nat>, Con{x, Impl.visits.go(f, Nat.is_le(T, x), en, x, T, mc, P, G, s, B)},
            Con{x, M.visits.go(f, Nat.is_le(T, x), en, x, T, mc, P, G)},
            Con{y, M.visits.go(f, Nat.is_le(T, y), en, y, T, mc, P, G)},
            Equal.cong(List<&2, Nat>, List<&2, Nat>, z => Con{x, z}, Impl.visits.go(f, Nat.is_le(T, x), en, x, T, mc, P, G, s, B),
              M.visits.go(f, Nat.is_le(T, x), en, x, T, mc, P, G), mb_go_eq(f, Nat.is_le(T, x), en, x, T, mc, P, G, s, B, w)),
            Equal.cong(Nat, List<&2, Nat>, z => Con{z, M.visits.go(f, Nat.is_le(T, z), en, z, T, mc, P, G)}, x, y, se))

# x && (y && r) == (x && y) && r
def mb_and_assoc(x: Bool, -y: Bool, -r: Bool) -> {Bool.and(x, Bool.and(y, r)) == Bool.and(Bool.and(x, y), r) : Bool}:
  match x:
    case True{}:
      {==}
    case False{}:
      {==}

# no merge once (T <= G) && (s <= B) is False
def mb_nm(+en: Bool, +k: Nat, +T: Nat, +mc: Nat, +P: Nat, +G: Nat, +s: Nat, +B: Nat,
    h: {Bool.and(Nat.is_le(T, G), Nat.is_le(s, B)) == False{} : Bool})
  -> {Impl.merge_ok(en, k, T, mc, P, G, s, B) == False{} : Bool}:
  +R = PM.pm_rest(k, T, mc, P)
  +a1 = mb_and_assoc(Nat.is_le(T, G), Nat.is_le(s, B), R)
  +a2 = Equal.trans(Bool, Bool.and(Bool.and(Nat.is_le(T, G), Nat.is_le(s, B)), R), Bool.and(False{}, R), False{},
    Equal.cong(Bool, Bool, z => Bool.and(z, R), Bool.and(Nat.is_le(T, G), Nat.is_le(s, B)), False{}, h), {==})
  Equal.trans(Bool, Impl.merge_ok(en, k, T, mc, P, G, s, B), Bool.and(en, False{}), False{},
    Equal.cong(Bool, Bool, z => Bool.and(en, z), Bool.and(Nat.is_le(T, G), Bool.and(Nat.is_le(s, B), R)), False{},
      Equal.trans(Bool, Bool.and(Nat.is_le(T, G), Bool.and(Nat.is_le(s, B), R)), Bool.and(Bool.and(Nat.is_le(T, G), Nat.is_le(s, B)), R), False{}, a1, a2)),
    Ar.p6_and_false(en))

def Laws.unmerged_same(en, k, T, mc, P, G, s, B, no):
  %Equal.sym(Bool, Impl.merge_ok(en, k, T, mc, P, G, s, B), False{}, no)
    : {M.cut(k, T, P, M.widen(_, M.base_end(k, T, mc, P), mc)) == Spec.step(k, T, mc, P) : Nat}
  PM.pm_cut_eq(Bool.and(Nat.is_lt(k, Nat.mul(Nat.div(T, P), P)), Nat.is_le(Nat.mul(Nat.div(T, P), P), M.base_end(k, T, mc, P))),
    Nat.mul(Nat.div(T, P), P), M.base_end(k, T, mc, P))

def mb_go(fuel: Nat, done: Bool, +en: Bool, +k: Nat, +T: Nat, +mc: Nat, +P: Nat, +G: Nat, +s: Nat, +B: Nat,
    +h: {Bool.and(Nat.is_le(T, G), Nat.is_le(s, B)) == False{} : Bool})
  -> {Impl.visits.go(fuel, done, en, k, T, mc, P, G, s, B) == Spec.visits.go(fuel, done, k, T, mc, P) : List<&2, Nat>}:
  match fuel:
    case 0n:
      {==}
    case 1n+f:
      match done:
        case True{}:
          {==}
        case False{}:
          +x = Impl.step(en, k, T, mc, P, G, s, B)
          +y = Spec.step(k, T, mc, P)
          +se = Laws.unmerged_same(en, k, T, mc, P, G, s, B, mb_nm(en, k, T, mc, P, G, s, B, h))
          Equal.trans(List<&2, Nat>, Con{x, Impl.visits.go(f, Nat.is_le(T, x), en, x, T, mc, P, G, s, B)},
            Con{x, Spec.visits.go(f, Nat.is_le(T, x), x, T, mc, P)},
            Con{y, Spec.visits.go(f, Nat.is_le(T, y), y, T, mc, P)},
            Equal.cong(List<&2, Nat>, List<&2, Nat>, z => Con{x, z}, Impl.visits.go(f, Nat.is_le(T, x), en, x, T, mc, P, G, s, B),
              Spec.visits.go(f, Nat.is_le(T, x), x, T, mc, P), mb_go(f, Nat.is_le(T, x), en, x, T, mc, P, G, s, B, h)),
            Equal.cong(Nat, List<&2, Nat>, z => Con{z, Spec.visits.go(f, Nat.is_le(T, z), z, T, mc, P)}, x, y, se))

# ---------------------------------------------------------------------------------------------
# Laws

def Laws.guard(en, k, T, mc, P, G, s, B, merged):
  +R = PM.pm_rest(k, T, mc, P)
  +r1 = Ar.ar_and_r(en, Bool.and(Nat.is_le(T, G), Bool.and(Nat.is_le(s, B), R)), merged)
  +r2 = Ar.ar_and_r(Nat.is_le(T, G), Bool.and(Nat.is_le(s, B), R), r1)
  (Ar.ar_and_l(Nat.is_le(T, G), Bool.and(Nat.is_le(s, B), R), r1), Ar.ar_and_l(Nat.is_le(s, B), R, r2))

# rewrite the third component of a 5110g law-1 triple from M.step to Impl.step
def mb_last(-e1: Nat, -e2: Nat, -sa: Nat, -sb: Nat, -ms: Nat, -hs: Nat, he: {hs == ms : Nat},
    p: {sa == e1 : Nat} & {sb == e2 : Nat} & {ms == e2 : Nat})
  -> {sa == e1 : Nat} & {sb == e2 : Nat} & {hs == e2 : Nat}:
  (x, r) = p
  (y, z) = r
  (x, y, Equal.trans(Nat, hs, ms, e2, he, z))

def Laws.merge_two_steps(a, c, T, P, G, s, B, pp, cp, merged):
  +m = merged
  +k = Nat.mul(a, P)
  +mc = Nat.mul(c, P)
  +sb = Ar.ard_snd({Nat.is_le(T, G) == True{} : Bool}, {Nat.is_le(s, B) == True{} : Bool}, Laws.guard(True{}, k, T, mc, P, G, s, B, m))
  +mm = Equal.trans(Bool, M.merge_ok(True{}, k, T, mc, P, G), Impl.merge_ok(True{}, k, T, mc, P, G, s, B), True{},
    Equal.sym(Bool, Impl.merge_ok(True{}, k, T, mc, P, G, s, B), M.merge_ok(True{}, k, T, mc, P, G), mb_ok_eq(True{}, k, T, mc, P, G, s, B, sb)), m)
  mb_last(Nat.add(Nat.mul(a, P), Nat.mul(c, P)), Nat.add(Nat.add(k, mc), mc), Spec.step(k, T, mc, P), Spec.step(Spec.step(k, T, mc, P), T, mc, P),
    M.step(True{}, k, T, mc, P, G), Impl.step(True{}, k, T, mc, P, G, s, B), mb_step_eq(True{}, k, T, mc, P, G, s, B, sb),
    PML.merge_two_steps(a, c, T, P, G, pp, cp, mm))

def Laws.above_stage_schedule(fuel, en, k, T, mc, P, G, s, B, above):
  +h = Equal.trans(Bool, Bool.and(Nat.is_le(T, G), Nat.is_le(s, B)), Bool.and(Nat.is_le(T, G), False{}), False{},
    Equal.cong(Bool, Bool, z => Bool.and(Nat.is_le(T, G), z), Nat.is_le(s, B), False{}, above), Ar.p6_and_false(Nat.is_le(T, G)))
  mb_go(fuel, Nat.is_le(T, k), en, k, T, mc, P, G, s, B, h)

def Laws.above_guard_schedule(fuel, en, k, T, mc, P, G, s, B, above):
  +h = Equal.cong(Bool, Bool, z => Bool.and(z, Nat.is_le(s, B)), Nat.is_le(T, G), False{}, above)
  mb_go(fuel, Nat.is_le(T, k), en, k, T, mc, P, G, s, B, h)

def Laws.within_bound_is_5110g(fuel, en, k, T, mc, P, G, s, B, within):
  +w = within
  (mb_ok_eq(en, k, T, mc, P, G, s, B, w), mb_step_eq(en, k, T, mc, P, G, s, B, w), mb_go_eq(fuel, Nat.is_le(T, k), en, k, T, mc, P, G, s, B, w))

def Laws.stage_pow2_bound():
  ({==}, {==}, {==}, {==}, {==}, {==})

def Laws.stage_1025_unmerged(fuel, en, k, T, mc, P):
  Laws.above_stage_schedule(fuel, en, k, T, mc, P, Impl.MAX_PROMPT(), Impl.stage_pages(1025n), Impl.MAX_STAGE_PAGES(), {==})
