# Proofs of the ext 8202 layer-tail schedule laws (tail_m16_sched_laws.bend) about the Bend model
# tail_m16_sched.bend. Structural evidence only: no unsafe definitions, axioms, holes, foreign proof
# code or answer tables; the patch 2001 v2 partition facts come from the proven gemm_m16_partition
# laws and lemmas, the embedded MLP facts from the proven ext 8201 laws. Every def precedes its uses.
import Base
import ./gemm_m16_partition.bend as Part
import ./gemm_m16_partition_spec.bend as PSpec
import ./gemm_m16_partition_laws.bend as PLaws
import ./gemm_m16_partition_proof.bend as PP
import ./mlp_m16_sched.bend as Mlp
import ./mlp_m16_sched_spec.bend as MSpec
import ./mlp_m16_sched_laws.bend as MLaws
import ./mlp_m16_sched_proof.bend as MP
import ./tail_m16_sched.bend as Impl
import ./tail_m16_sched_spec.bend as Spec
import ./tail_m16_sched_laws.bend as Laws

# ==== (1) phase 0 is v2's partition of T0 ====

def end0(+G: Nat, +A: Nat, +H: Nat, +b: Nat, gpos: {Nat.is_lt(0n, G) == True{} : Bool},
    fits: {Nat.is_le(G, Impl.total0(A, H)) == True{} : Bool})
  -> {Nat.add(Impl.s0(G, A, H, b), Impl.n0(G, A, H, b)) == Impl.e0(G, A, H, b) : Nat}:
  PP.ar_sub_add(Impl.e0(G, A, H, b), Impl.s0(G, A, H, b),
    PP.ar_lt_le(Impl.s0(G, A, H, b), Impl.e0(G, A, H, b),
      PP.p2_nonempty(b, Impl.total0(A, H), G, gpos, fits)))

def Laws.phase0_slice_owned(G, A, H, b, x, grid, fits):
  +gp = grid
  %Equal.sym(Nat, Nat.add(Impl.s0(G, A, H, b), Impl.n0(G, A, H, b)), Impl.e0(G, A, H, b),
      end0(G, A, H, b, gp, fits))
    : {Bool.and(Nat.is_le(Impl.s0(G, A, H, b), x), Nat.is_lt(x, _)) ==
      PSpec.owns(b, x, Impl.total0(A, H), G) : Bool}
  PLaws.range_is_owned(1n, H, A, G, b, x, gp)

def Laws.phase0_exactly_once(G, A, H, x, grid, inside):
  PLaws.owners_exactly_one(1n, H, A, G, x, grid, inside)

def Laws.phase0_v2(G, A, H, b, grid, shaped, fits):
  +gp = grid
  +fp = fits
  (end0(G, A, H, b, gp, fp),
    %PP.ar_add_sub(Impl.s0(G, A, H, b), Impl.n0(G, A, H, b))
      : {Part.mainloop(Impl.s0(G, A, H, b), Nat.add(Impl.s0(G, A, H, b), Impl.n0(G, A, H, b)),
          Impl.kt0(A)) ==
        PSpec.trace(_, Impl.s0(G, A, H, b), Nat.add(Impl.s0(G, A, H, b), Impl.n0(G, A, H, b)),
          Impl.kt0(A)) : List<&2, PSpec.Ev>}
    PLaws.mainloop_trace(Impl.s0(G, A, H, b), Nat.add(Impl.s0(G, A, H, b), Impl.n0(G, A, H, b)),
      Impl.kt0(A), PP.ar_le_add_r(Impl.s0(G, A, H, b), Impl.n0(G, A, H, b)),
      PP.p2_kt_pos(A, shaped)))

# ==== (2) workspace: gemm_m16_partition's slot laws at mats = 1 ====

def Laws.phase0_slots(G, A, H, g, b, grid, shaped, fits, group, contributor):
  PLaws.slot_in_range(1n, H, A, G, g, b, grid, shaped, fits, group, contributor)

def Laws.phase0_slot_injective(G, A, H, g, b, h, c, grid, shaped, fits, group_g, group_h, touch_b,
    touch_c, same):
  PLaws.slot_injective(1n, H, A, G, g, b, h, c, grid, shaped, fits, group_g, group_h, touch_b,
    touch_c, same)

# ==== (5) event projections through the tagged segments ====

def bars_work(evs: List<&2, PSpec.Ev>, +rest: List<&2, Spec.Ev>)
  -> {Spec.bars(rest) == Spec.bars(Impl.tag_work(evs, rest)) : List<&2, Nat>}:
  match evs:
    case Nil{}:
      {==}
    case ev <> more:
      bars_work(more, rest)

def bars_fin(xs: List<&2, Nat>, +rest: List<&2, Spec.Ev>)
  -> {Spec.bars(rest) == Spec.bars(Impl.tag_fin(xs, rest)) : List<&2, Nat>}:
  match xs:
    case Nil{}:
      {==}
    case x <> more:
      bars_fin(more, rest)

def bars_norm(xs: List<&2, Nat>, +rest: List<&2, Spec.Ev>)
  -> {Spec.bars(rest) == Spec.bars(Impl.tag_norm(xs, rest)) : List<&2, Nat>}:
  match xs:
    case Nil{}:
      {==}
    case x <> more:
      bars_norm(more, rest)

def bars_sync(evs: List<&2, MSpec.Ev>, +rest: List<&2, Spec.Ev>)
  -> {Spec.bars(rest) == Spec.bars(Impl.tag_sync(evs, rest)) : List<&2, Nat>}:
  match evs:
    case Nil{}:
      {==}
    case ev <> more:
      bars_sync(more, rest)

def Laws.same_barriers(G, A, H, I, m, b):
  +ML = Part.mainloop(Impl.s0(G, A, H, b), Impl.e0(G, A, H, b), Impl.kt0(A))
  +FJ = Impl.warp_jobs(G, Impl.jobs0(H, m), b)
  +NJ = Impl.warp_jobs(G, Impl.jobsn(H, m), b)
  +EV = Mlp.events(G, H, I, b)
  +R4 = {[Spec.Bar{4n}] : List<&2, Spec.Ev>}
  +R3 = {Spec.Bar{3n} <> Impl.tag_sync(EV, R4) : List<&2, Spec.Ev>}
  +R2 = {Spec.Bar{2n} <> Impl.tag_norm(NJ, R3) : List<&2, Spec.Ev>}
  +R1 = {Spec.Bar{1n} <> Impl.tag_fin(FJ, R2) : List<&2, Spec.Ev>}
  %bars_work(ML, R1) : {0n <> _ == Spec.bar_seq() : List<&2, Nat>}
  %bars_fin(FJ, R2) : {0n <> 1n <> _ == Spec.bar_seq() : List<&2, Nat>}
  %bars_norm(NJ, R3) : {0n <> 1n <> 2n <> _ == Spec.bar_seq() : List<&2, Nat>}
  %bars_sync(EV, R4) : {0n <> 1n <> 2n <> 3n <> _ == Spec.bar_seq() : List<&2, Nat>}
  {==}

# Shared let-bound pieces of events(G, A, H, I, m, b) are spelled out in each proof below:
# ML = phase-0 main loop, FJ / NJ = the block's F0 / N job lists, EV = 8201's events, R1 .. R4 the
# tails after barriers A .. D.

def st_work(evs: List<&2, PSpec.Ev>, +rest: List<&2, Spec.Ev>)
  -> {Spec.staged(rest, 1n) == Spec.staged(Impl.tag_work(evs, rest), 1n) : Bool}:
  match evs:
    case Nil{}:
      {==}
    case ev <> more:
      st_work(more, rest)

def st_fin(xs: List<&2, Nat>, +rest: List<&2, Spec.Ev>)
  -> {Spec.staged(rest, 2n) == Spec.staged(Impl.tag_fin(xs, rest), 2n) : Bool}:
  match xs:
    case Nil{}:
      {==}
    case x <> more:
      st_fin(more, rest)

def st_norm(xs: List<&2, Nat>, +rest: List<&2, Spec.Ev>)
  -> {Spec.staged(rest, 3n) == Spec.staged(Impl.tag_norm(xs, rest), 3n) : Bool}:
  match xs:
    case Nil{}:
      {==}
    case x <> more:
      st_norm(more, rest)

def st_sync(evs: List<&2, MSpec.Ev>, +rest: List<&2, Spec.Ev>)
  -> {Spec.staged(rest, 4n) == Spec.staged(Impl.tag_sync(evs, rest), 4n) : Bool}:
  match evs:
    case Nil{}:
      {==}
    case ev <> more:
      st_sync(more, rest)

def Laws.staged(G, A, H, I, m, b):
  +ML = Part.mainloop(Impl.s0(G, A, H, b), Impl.e0(G, A, H, b), Impl.kt0(A))
  +FJ = Impl.warp_jobs(G, Impl.jobs0(H, m), b)
  +NJ = Impl.warp_jobs(G, Impl.jobsn(H, m), b)
  +EV = Mlp.events(G, H, I, b)
  +R4 = {[Spec.Bar{4n}] : List<&2, Spec.Ev>}
  +R3 = {Spec.Bar{3n} <> Impl.tag_sync(EV, R4) : List<&2, Spec.Ev>}
  +R2 = {Spec.Bar{2n} <> Impl.tag_norm(NJ, R3) : List<&2, Spec.Ev>}
  +R1 = {Spec.Bar{1n} <> Impl.tag_fin(FJ, R2) : List<&2, Spec.Ev>}
  %st_work(ML, R1) : {_ == True{} : Bool}
  %st_fin(FJ, R2) : {_ == True{} : Bool}
  %st_norm(NJ, R3) : {_ == True{} : Bool}
  %st_sync(EV, R4) : {_ == True{} : Bool}
  {==}

def sy_work(evs: List<&2, PSpec.Ev>, +rest: List<&2, Spec.Ev>)
  -> {Spec.syncs(rest) == Spec.syncs(Impl.tag_work(evs, rest)) : List<&2, MSpec.Ev>}:
  match evs:
    case Nil{}:
      {==}
    case ev <> more:
      sy_work(more, rest)

def sy_fin(xs: List<&2, Nat>, +rest: List<&2, Spec.Ev>)
  -> {Spec.syncs(rest) == Spec.syncs(Impl.tag_fin(xs, rest)) : List<&2, MSpec.Ev>}:
  match xs:
    case Nil{}:
      {==}
    case x <> more:
      sy_fin(more, rest)

def sy_norm(xs: List<&2, Nat>, +rest: List<&2, Spec.Ev>)
  -> {Spec.syncs(rest) == Spec.syncs(Impl.tag_norm(xs, rest)) : List<&2, MSpec.Ev>}:
  match xs:
    case Nil{}:
      {==}
    case x <> more:
      sy_norm(more, rest)

def sy_sync(+evs: List<&2, MSpec.Ev>)
  -> {Spec.syncs(Impl.tag_sync(evs, [Spec.Bar{4n}])) == evs : List<&2, MSpec.Ev>}:
  match evs:
    case Nil{}:
      {==}
    case ev <> more:
      %sy_sync(more)
        : {ev <> Spec.syncs(Impl.tag_sync(more, [Spec.Bar{4n}])) == ev <> _ : List<&2, MSpec.Ev>}
      {==}

def Laws.embeds_8201(G, A, H, I, m, b):
  +ML = Part.mainloop(Impl.s0(G, A, H, b), Impl.e0(G, A, H, b), Impl.kt0(A))
  +FJ = Impl.warp_jobs(G, Impl.jobs0(H, m), b)
  +NJ = Impl.warp_jobs(G, Impl.jobsn(H, m), b)
  +EV = Mlp.events(G, H, I, b)
  +R4 = {[Spec.Bar{4n}] : List<&2, Spec.Ev>}
  +R3 = {Spec.Bar{3n} <> Impl.tag_sync(EV, R4) : List<&2, Spec.Ev>}
  +R2 = {Spec.Bar{2n} <> Impl.tag_norm(NJ, R3) : List<&2, Spec.Ev>}
  +R1 = {Spec.Bar{1n} <> Impl.tag_fin(FJ, R2) : List<&2, Spec.Ev>}
  %sy_work(ML, R1) : {_ == EV : List<&2, MSpec.Ev>}
  %sy_fin(FJ, R2) : {_ == EV : List<&2, MSpec.Ev>}
  %sy_norm(NJ, R3) : {_ == EV : List<&2, MSpec.Ev>}
  sy_sync(EV)

def emb_sym(+G: Nat, +A: Nat, +H: Nat, +I: Nat, +m: Nat, +b: Nat)
  -> {Mlp.events(G, H, I, b) == Spec.syncs(Impl.events(G, A, H, I, m, b)) : List<&2, MSpec.Ev>}:
  Equal.sym(List<&2, MSpec.Ev>, Spec.syncs(Impl.events(G, A, H, I, m, b)), Mlp.events(G, H, I, b),
    Laws.embeds_8201(G, A, H, I, m, b))

def Laws.sync_levels(G, A, H, I, m, b, d, c):
  +gl = MLaws.gate_level(c)
  (%emb_sym(G, A, H, I, m, b) : {MSpec.ordered(_, 0n) == True{} : Bool}
    MLaws.events_ordered(G, H, I, b),
    %emb_sym(G, A, H, I, m, d) : {MSpec.sigs_below(_, c, gl) == True{} : Bool}
    PP.ard_fst({MSpec.sigs_below(Mlp.events(G, H, I, d), c, gl) == True{} : Bool},
      {MSpec.waits_from(Mlp.events(G, H, I, b), c, gl) == True{} : Bool},
      MLaws.signals_before_waits(G, H, I, b, d, c)),
    %emb_sym(G, A, H, I, m, b) : {MSpec.waits_from(_, c, gl) == True{} : Bool}
    PP.ard_snd({MSpec.sigs_below(Mlp.events(G, H, I, d), c, gl) == True{} : Bool},
      {MSpec.waits_from(Mlp.events(G, H, I, b), c, gl) == True{} : Bool},
      MLaws.signals_before_waits(G, H, I, b, d, c)))

def wk_work(evs: List<&2, PSpec.Ev>, +rest: List<&2, Spec.Ev>,
    +h: {Spec.works(rest) == Nil{} : List<&2, PSpec.Ev>})
  -> {Spec.works(Impl.tag_work(evs, rest)) == evs : List<&2, PSpec.Ev>}:
  match evs:
    case Nil{}:
      h
    case ev <> more:
      %wk_work(more, rest, h)
        : {ev <> Spec.works(Impl.tag_work(more, rest)) == ev <> _ : List<&2, PSpec.Ev>}
      {==}

def wk_fin(xs: List<&2, Nat>, +rest: List<&2, Spec.Ev>)
  -> {Spec.works(rest) == Spec.works(Impl.tag_fin(xs, rest)) : List<&2, PSpec.Ev>}:
  match xs:
    case Nil{}:
      {==}
    case x <> more:
      wk_fin(more, rest)

def wk_norm(xs: List<&2, Nat>, +rest: List<&2, Spec.Ev>)
  -> {Spec.works(rest) == Spec.works(Impl.tag_norm(xs, rest)) : List<&2, PSpec.Ev>}:
  match xs:
    case Nil{}:
      {==}
    case x <> more:
      wk_norm(more, rest)

def wk_sync(evs: List<&2, MSpec.Ev>, +rest: List<&2, Spec.Ev>)
  -> {Spec.works(rest) == Spec.works(Impl.tag_sync(evs, rest)) : List<&2, PSpec.Ev>}:
  match evs:
    case Nil{}:
      {==}
    case ev <> more:
      wk_sync(more, rest)

def wk_tail(+G: Nat, +A: Nat, +H: Nat, +I: Nat, +m: Nat, +b: Nat)
  -> {Spec.works(Spec.Bar{1n} <> Impl.tag_fin(Impl.warp_jobs(G, Impl.jobs0(H, m), b),
      Spec.Bar{2n} <> Impl.tag_norm(Impl.warp_jobs(G, Impl.jobsn(H, m), b),
        Spec.Bar{3n} <> Impl.tag_sync(Mlp.events(G, H, I, b), [Spec.Bar{4n}])))) == Nil{}
    : List<&2, PSpec.Ev>}:
  +FJ = Impl.warp_jobs(G, Impl.jobs0(H, m), b)
  +NJ = Impl.warp_jobs(G, Impl.jobsn(H, m), b)
  +EV = Mlp.events(G, H, I, b)
  +R4 = {[Spec.Bar{4n}] : List<&2, Spec.Ev>}
  +R3 = {Spec.Bar{3n} <> Impl.tag_sync(EV, R4) : List<&2, Spec.Ev>}
  +R2 = {Spec.Bar{2n} <> Impl.tag_norm(NJ, R3) : List<&2, Spec.Ev>}
  %wk_fin(FJ, R2) : {_ == Nil{} : List<&2, PSpec.Ev>}
  %wk_norm(NJ, R3) : {_ == Nil{} : List<&2, PSpec.Ev>}
  %wk_sync(EV, R4) : {_ == Nil{} : List<&2, PSpec.Ev>}
  {==}

def Laws.phase0_events(G, A, H, I, m, b, grid, shaped, fits):
  +ML = Part.mainloop(Impl.s0(G, A, H, b), Impl.e0(G, A, H, b), Impl.kt0(A))
  +FJ = Impl.warp_jobs(G, Impl.jobs0(H, m), b)
  +NJ = Impl.warp_jobs(G, Impl.jobsn(H, m), b)
  +EV = Mlp.events(G, H, I, b)
  +R1 = {Spec.Bar{1n} <> Impl.tag_fin(FJ, Spec.Bar{2n} <> Impl.tag_norm(NJ,
    Spec.Bar{3n} <> Impl.tag_sync(EV, [Spec.Bar{4n}]))) : List<&2, Spec.Ev>}
  Equal.trans(List<&2, PSpec.Ev>, Spec.works(Impl.events(G, A, H, I, m, b)), ML,
    PSpec.trace(Impl.n0(G, A, H, b), Impl.s0(G, A, H, b), Impl.e0(G, A, H, b), Impl.kt0(A)),
    wk_work(ML, R1, wk_tail(G, A, H, I, m, b)),
    PLaws.mainloop_trace(Impl.s0(G, A, H, b), Impl.e0(G, A, H, b), Impl.kt0(A),
      PP.ar_lt_le(Impl.s0(G, A, H, b), Impl.e0(G, A, H, b),
        PP.p2_nonempty(b, Impl.total0(A, H), G, grid, fits)),
      PP.p2_kt_pos(A, shaped)))

# ==== (4) job ownership: warp w of block b runs job w G + b ====

def Laws.job_owner(G, J, x, grid, fits, inside):
  +gp = grid
  (PP.ard_dm_lt(x, G, gp),
    PP.ard_rt(Nat.is_lt(x, Nat.mul(8n, G)), Nat.is_lt(Nat.div(x, G), 8n),
      Equal.sym(Bool, Nat.is_lt(Nat.div(x, G), 8n), Nat.is_lt(x, Nat.mul(8n, G)),
        PP.ar_div_lt(8n, x, G, gp)),
      PP.ar_lt_le_trans(x, J, Nat.mul(8n, G), inside, fits)),
    PP.ard_dm_eq(x, G, gp))

# y >= J > x: y != x.
def ne_ge(+y: Nat, +x: Nat, +J: Nat, ge: {Nat.is_lt(y, J) == False{} : Bool},
    xj: {Nat.is_lt(x, J) == True{} : Bool}) -> {Nat.is_eq(y, x) == False{} : Bool}:
  PP.ar_eq_false(y, x, e => PP.p5_eq_lt_absurd(x, y, Equal.sym(Nat, y, x, e),
    PP.ar_lt_le_trans(x, J, y, xj, PP.p6_nlt_le(y, J, ge))))

# For x < J, keeping or dropping y adds [y == x] (a dropped y >= J is not x).
def occ_keep(c: Bool, +y: Nat, +rest: List<&2, Nat>, +x: Nat, +J: Nat,
    ev: {Nat.is_lt(y, J) == c : Bool}, xj: {Nat.is_lt(x, J) == True{} : Bool})
  -> {MSpec.occurs(Impl.warp_jobs.keep(c, y, rest), x) ==
    Nat.add(PSpec.bit(Nat.is_eq(y, x)), MSpec.occurs(rest, x)) : Nat}:
  match c:
    case True{}:
      {==}
    case False{}:
      %Equal.sym(Bool, Nat.is_eq(y, x), False{}, ne_ge(y, x, J, ev, xj))
        : {MSpec.occurs(rest, x) == Nat.add(PSpec.bit(_), MSpec.occurs(rest, x)) : Nat}
      {==}

# x >= J is never kept.
def occ_keep_out(c: Bool, +y: Nat, +rest: List<&2, Nat>, +x: Nat, +J: Nat,
    ev: {Nat.is_lt(y, J) == c : Bool}, xo: {Nat.is_lt(x, J) == False{} : Bool})
  -> {MSpec.occurs(Impl.warp_jobs.keep(c, y, rest), x) == MSpec.occurs(rest, x) : Nat}:
  match c:
    case True{}:
      %Equal.sym(Bool, Nat.is_eq(y, x), False{},
          PP.p6_lt_ne(y, x, PP.ar_lt_le_trans(y, J, x, ev, PP.p6_nlt_le(x, J, xo))))
        : {Nat.add(PSpec.bit(_), MSpec.occurs(rest, x)) == MSpec.occurs(rest, x) : Nat}
      {==}
    case False{}:
      {==}

def wj_fwd(+G: Nat, +b: Nat, +w: Nat, +x: Nat, +gpos: {Nat.is_lt(0n, G) == True{} : Bool},
    +bg: {Nat.is_lt(b, G) == True{} : Bool},
    h: {Nat.is_eq(Impl.warp_job(G, b, w), x) == True{} : Bool})
  -> {Bool.and(Nat.is_eq(w, Nat.div(x, G)), Nat.is_eq(b, Nat.mod(x, G))) == True{} : Bool}:
  +q = Nat.div(x, G)
  +r = Nat.mod(x, G)
  +wb = Nat.add(Nat.mul(w, G), b)
  +qr = Nat.add(Nat.mul(q, G), r)
  +same = Equal.trans(Nat, wb, x, qr, PP.ar_eq_true(wb, x, h),
    Equal.sym(Nat, qr, x, PP.ard_dm_eq(x, G, gpos)))
  PP.ar_and_intro(Nat.is_eq(w, q), Nat.is_eq(b, r),
    PP.p8_eq_of(w, q, PP.ard_fst({w == q : Nat}, {b == r : Nat},
      PP.ar_divmod_inj(w, b, q, r, G, bg, PP.ard_dm_lt(x, G, gpos), same))),
    PP.p8_eq_of(b, r, PP.ard_snd({w == q : Nat}, {b == r : Nat},
      PP.ar_divmod_inj(w, b, q, r, G, bg, PP.ard_dm_lt(x, G, gpos), same))))

def wj_bwd(+G: Nat, +b: Nat, +w: Nat, +x: Nat, +gpos: {Nat.is_lt(0n, G) == True{} : Bool},
    +h: {Bool.and(Nat.is_eq(w, Nat.div(x, G)), Nat.is_eq(b, Nat.mod(x, G))) == True{} : Bool})
  -> {Nat.is_eq(Impl.warp_job(G, b, w), x) == True{} : Bool}:
  +q = Nat.div(x, G)
  +r = Nat.mod(x, G)
  %Equal.sym(Nat, w, q, PP.ar_eq_true(w, q, PP.ar_and_l(Nat.is_eq(w, q), Nat.is_eq(b, r), h)))
    : {Nat.is_eq(Nat.add(Nat.mul(_, G), b), x) == True{} : Bool}
  %Equal.sym(Nat, b, r, PP.ar_eq_true(b, r, PP.ar_and_r(Nat.is_eq(w, q), Nat.is_eq(b, r), h)))
    : {Nat.is_eq(Nat.add(Nat.mul(q, G), _), x) == True{} : Bool}
  PP.p8_eq_of(Nat.add(Nat.mul(q, G), r), x, PP.ard_dm_eq(x, G, gpos))

# w G + b == x iff w == x div G and b == x mod G (b < G).
def wj_eq(+G: Nat, +b: Nat, +w: Nat, +x: Nat, +gpos: {Nat.is_lt(0n, G) == True{} : Bool},
    +bg: {Nat.is_lt(b, G) == True{} : Bool})
  -> {Nat.is_eq(Impl.warp_job(G, b, w), x) ==
    Bool.and(Nat.is_eq(w, Nat.div(x, G)), Nat.is_eq(b, Nat.mod(x, G))) : Bool}:
  PP.ar_bool_ext(Nat.is_eq(Impl.warp_job(G, b, w), x),
    Bool.and(Nat.is_eq(w, Nat.div(x, G)), Nat.is_eq(b, Nat.mod(x, G))),
    h => wj_fwd(G, b, w, x, gpos, bg, h),
    h => wj_bwd(G, b, w, x, gpos, h))

# [w == q and v] + [v and q < w] == [v and q < w + 1]
def occ_step(v: Bool, +w: Nat, +q: Nat)
  -> {Nat.add(PSpec.bit(Bool.and(Nat.is_eq(w, q), v)), PSpec.bit(Bool.and(v, Nat.is_lt(q, w)))) ==
    PSpec.bit(Bool.and(v, Nat.is_lt(q, 1n+w))) : Nat}:
  match v:
    case True{}:
      %Equal.sym(Bool, Bool.and(Nat.is_eq(w, q), True{}), Nat.is_eq(w, q),
          PP.p6_and_true(Nat.is_eq(w, q)))
        : {Nat.add(PSpec.bit(_), PSpec.bit(Nat.is_lt(q, w))) == PSpec.bit(Nat.is_lt(q, 1n+w)) : Nat}
      %PP.ar_add_comm(PSpec.bit(Nat.is_lt(q, w)), PSpec.bit(Nat.is_eq(w, q)))
        : {_ == PSpec.bit(Nat.is_lt(q, 1n+w)) : Nat}
      MP.bit_step(q, w)
    case False{}:
      %Equal.sym(Bool, Bool.and(Nat.is_eq(w, q), False{}), False{}, PP.p6_and_false(Nat.is_eq(w, q)))
        : {Nat.add(PSpec.bit(_), 0n) == 0n : Nat}
      {==}

# Occurrences of x < J among the jobs of warps n - 1 .. 0 of block b < G: one iff b == x mod G and
# x div G < n.
def occ_go(+n: Nat, +G: Nat, +J: Nat, +b: Nat, +x: Nat,
    +gpos: {Nat.is_lt(0n, G) == True{} : Bool}, +bg: {Nat.is_lt(b, G) == True{} : Bool},
    +xj: {Nat.is_lt(x, J) == True{} : Bool})
  -> {MSpec.occurs(Impl.warp_jobs.go(n, G, J, b), x) ==
    PSpec.bit(Bool.and(Nat.is_eq(b, Nat.mod(x, G)), Nat.is_lt(Nat.div(x, G), n))) : Nat}:
  match n:
    case 0n:
      +q = Nat.div(x, G)
      +E = Nat.is_eq(b, Nat.mod(x, G))
      %Equal.sym(Bool, Nat.is_lt(q, 0n), False{}, PP.ar_h_lt_zero(q))
        : {0n == PSpec.bit(Bool.and(E, _)) : Nat}
      %Equal.sym(Bool, Bool.and(E, False{}), False{}, PP.p6_and_false(E))
        : {0n == PSpec.bit(_) : Nat}
      {==}
    case 1n+w:
      +q = Nat.div(x, G)
      +E = Nat.is_eq(b, Nat.mod(x, G))
      +xw = Impl.warp_job(G, b, w)
      +rest = Impl.warp_jobs.go(w, G, J, b)
      %Equal.sym(Nat, MSpec.occurs(Impl.warp_jobs.keep(Nat.is_lt(xw, J), xw, rest), x),
          Nat.add(PSpec.bit(Nat.is_eq(xw, x)), MSpec.occurs(rest, x)),
          occ_keep(Nat.is_lt(xw, J), xw, rest, x, J, {==}, xj))
        : {_ == PSpec.bit(Bool.and(E, Nat.is_lt(q, 1n+w))) : Nat}
      %Equal.sym(Bool, Nat.is_eq(xw, x), Bool.and(Nat.is_eq(w, q), E), wj_eq(G, b, w, x, gpos, bg))
        : {Nat.add(PSpec.bit(_), MSpec.occurs(rest, x)) ==
          PSpec.bit(Bool.and(E, Nat.is_lt(q, 1n+w))) : Nat}
      %Equal.sym(Nat, MSpec.occurs(rest, x), PSpec.bit(Bool.and(E, Nat.is_lt(q, w))),
          occ_go(w, G, J, b, x, gpos, bg, xj))
        : {Nat.add(PSpec.bit(Bool.and(Nat.is_eq(w, q), E)), _) ==
          PSpec.bit(Bool.and(E, Nat.is_lt(q, 1n+w))) : Nat}
      occ_step(E, w, q)

def occ_go_out(+n: Nat, +G: Nat, +J: Nat, +b: Nat, +x: Nat,
    +xo: {Nat.is_lt(x, J) == False{} : Bool})
  -> {MSpec.occurs(Impl.warp_jobs.go(n, G, J, b), x) == 0n : Nat}:
  match n:
    case 0n:
      {==}
    case 1n+w:
      +xw = Impl.warp_job(G, b, w)
      +rest = Impl.warp_jobs.go(w, G, J, b)
      %Equal.sym(Nat, MSpec.occurs(Impl.warp_jobs.keep(Nat.is_lt(xw, J), xw, rest), x),
          MSpec.occurs(rest, x), occ_keep_out(Nat.is_lt(xw, J), xw, rest, x, J, {==}, xo))
        : {_ == 0n : Nat}
      occ_go_out(w, G, J, b, x, xo)

# The 8 warps of block b < G run job x < J <= 8 G once iff b == x mod G.
def occ_block(+G: Nat, +J: Nat, +b: Nat, +x: Nat, +gpos: {Nat.is_lt(0n, G) == True{} : Bool},
    +bg: {Nat.is_lt(b, G) == True{} : Bool}, +fits: {Nat.is_le(J, Nat.mul(8n, G)) == True{} : Bool},
    +xj: {Nat.is_lt(x, J) == True{} : Bool})
  -> {MSpec.occurs(Impl.warp_jobs(G, J, b), x) == PSpec.bit(Nat.is_eq(b, Nat.mod(x, G))) : Nat}:
  +E = Nat.is_eq(b, Nat.mod(x, G))
  +q8 = PP.ard_rt(Nat.is_lt(x, Nat.mul(8n, G)), Nat.is_lt(Nat.div(x, G), 8n),
    Equal.sym(Bool, Nat.is_lt(Nat.div(x, G), 8n), Nat.is_lt(x, Nat.mul(8n, G)),
      PP.ar_div_lt(8n, x, G, gpos)),
    PP.ar_lt_le_trans(x, J, Nat.mul(8n, G), xj, fits))
  %Equal.sym(Nat, MSpec.occurs(Impl.warp_jobs(G, J, b), x),
      PSpec.bit(Bool.and(E, Nat.is_lt(Nat.div(x, G), 8n))), occ_go(8n, G, J, b, x, gpos, bg, xj))
    : {_ == PSpec.bit(E) : Nat}
  %Equal.sym(Bool, Nat.is_lt(Nat.div(x, G), 8n), True{}, q8)
    : {PSpec.bit(Bool.and(E, _)) == PSpec.bit(E) : Nat}
  %Equal.sym(Bool, Bool.and(E, True{}), E, PP.p6_and_true(E)) : {PSpec.bit(_) == PSpec.bit(E) : Nat}
  {==}

# fin_hits / norm_hits through the tagged segments.
def fh_work(evs: List<&2, PSpec.Ev>, +rest: List<&2, Spec.Ev>, +x: Nat)
  -> {Spec.fin_hits(rest, x) == Spec.fin_hits(Impl.tag_work(evs, rest), x) : Nat}:
  match evs:
    case Nil{}:
      {==}
    case ev <> more:
      fh_work(more, rest, x)

def fh_fin(+xs: List<&2, Nat>, +rest: List<&2, Spec.Ev>, +x: Nat)
  -> {Nat.add(MSpec.occurs(xs, x), Spec.fin_hits(rest, x)) ==
    Spec.fin_hits(Impl.tag_fin(xs, rest), x) : Nat}:
  match xs:
    case Nil{}:
      {==}
    case y <> more:
      %fh_fin(more, rest, x)
        : {Nat.add(Nat.add(PSpec.bit(Nat.is_eq(y, x)), MSpec.occurs(more, x)),
            Spec.fin_hits(rest, x)) == Nat.add(PSpec.bit(Nat.is_eq(y, x)), _) : Nat}
      PP.ar_add_assoc(PSpec.bit(Nat.is_eq(y, x)), MSpec.occurs(more, x), Spec.fin_hits(rest, x))

def fh_norm(xs: List<&2, Nat>, +rest: List<&2, Spec.Ev>, +x: Nat)
  -> {Spec.fin_hits(rest, x) == Spec.fin_hits(Impl.tag_norm(xs, rest), x) : Nat}:
  match xs:
    case Nil{}:
      {==}
    case y <> more:
      fh_norm(more, rest, x)

def fh_sync(evs: List<&2, MSpec.Ev>, +rest: List<&2, Spec.Ev>, +x: Nat)
  -> {Spec.fin_hits(rest, x) == Spec.fin_hits(Impl.tag_sync(evs, rest), x) : Nat}:
  match evs:
    case Nil{}:
      {==}
    case ev <> more:
      fh_sync(more, rest, x)

def nh_work(evs: List<&2, PSpec.Ev>, +rest: List<&2, Spec.Ev>, +x: Nat)
  -> {Spec.norm_hits(rest, x) == Spec.norm_hits(Impl.tag_work(evs, rest), x) : Nat}:
  match evs:
    case Nil{}:
      {==}
    case ev <> more:
      nh_work(more, rest, x)

def nh_fin(xs: List<&2, Nat>, +rest: List<&2, Spec.Ev>, +x: Nat)
  -> {Spec.norm_hits(rest, x) == Spec.norm_hits(Impl.tag_fin(xs, rest), x) : Nat}:
  match xs:
    case Nil{}:
      {==}
    case y <> more:
      nh_fin(more, rest, x)

def nh_norm(+xs: List<&2, Nat>, +rest: List<&2, Spec.Ev>, +x: Nat)
  -> {Nat.add(MSpec.occurs(xs, x), Spec.norm_hits(rest, x)) ==
    Spec.norm_hits(Impl.tag_norm(xs, rest), x) : Nat}:
  match xs:
    case Nil{}:
      {==}
    case y <> more:
      %nh_norm(more, rest, x)
        : {Nat.add(Nat.add(PSpec.bit(Nat.is_eq(y, x)), MSpec.occurs(more, x)),
            Spec.norm_hits(rest, x)) == Nat.add(PSpec.bit(Nat.is_eq(y, x)), _) : Nat}
      PP.ar_add_assoc(PSpec.bit(Nat.is_eq(y, x)), MSpec.occurs(more, x), Spec.norm_hits(rest, x))

def nh_sync(evs: List<&2, MSpec.Ev>, +rest: List<&2, Spec.Ev>, +x: Nat)
  -> {Spec.norm_hits(rest, x) == Spec.norm_hits(Impl.tag_sync(evs, rest), x) : Nat}:
  match evs:
    case Nil{}:
      {==}
    case ev <> more:
      nh_sync(more, rest, x)

# Block b's F0 / N runs of job x are the occurrences of x in its warps' job lists.
def fin_block(+G: Nat, +A: Nat, +H: Nat, +I: Nat, +m: Nat, +b: Nat, +x: Nat)
  -> {Spec.fin_hits(Impl.events(G, A, H, I, m, b), x) ==
    MSpec.occurs(Impl.warp_jobs(G, Impl.jobs0(H, m), b), x) : Nat}:
  +ML = Part.mainloop(Impl.s0(G, A, H, b), Impl.e0(G, A, H, b), Impl.kt0(A))
  +FJ = Impl.warp_jobs(G, Impl.jobs0(H, m), b)
  +NJ = Impl.warp_jobs(G, Impl.jobsn(H, m), b)
  +EV = Mlp.events(G, H, I, b)
  +R4 = {[Spec.Bar{4n}] : List<&2, Spec.Ev>}
  +R3 = {Spec.Bar{3n} <> Impl.tag_sync(EV, R4) : List<&2, Spec.Ev>}
  +R2 = {Spec.Bar{2n} <> Impl.tag_norm(NJ, R3) : List<&2, Spec.Ev>}
  +R1 = {Spec.Bar{1n} <> Impl.tag_fin(FJ, R2) : List<&2, Spec.Ev>}
  %fh_work(ML, R1, x) : {_ == MSpec.occurs(FJ, x) : Nat}
  %fh_fin(FJ, R2, x) : {_ == MSpec.occurs(FJ, x) : Nat}
  %fh_norm(NJ, R3, x) : {Nat.add(MSpec.occurs(FJ, x), _) == MSpec.occurs(FJ, x) : Nat}
  %fh_sync(EV, R4, x) : {Nat.add(MSpec.occurs(FJ, x), _) == MSpec.occurs(FJ, x) : Nat}
  PP.ar_add_zero_r(MSpec.occurs(FJ, x))

def norm_block(+G: Nat, +A: Nat, +H: Nat, +I: Nat, +m: Nat, +b: Nat, +x: Nat)
  -> {Spec.norm_hits(Impl.events(G, A, H, I, m, b), x) ==
    MSpec.occurs(Impl.warp_jobs(G, Impl.jobsn(H, m), b), x) : Nat}:
  +ML = Part.mainloop(Impl.s0(G, A, H, b), Impl.e0(G, A, H, b), Impl.kt0(A))
  +FJ = Impl.warp_jobs(G, Impl.jobs0(H, m), b)
  +NJ = Impl.warp_jobs(G, Impl.jobsn(H, m), b)
  +EV = Mlp.events(G, H, I, b)
  +R4 = {[Spec.Bar{4n}] : List<&2, Spec.Ev>}
  +R3 = {Spec.Bar{3n} <> Impl.tag_sync(EV, R4) : List<&2, Spec.Ev>}
  +R2 = {Spec.Bar{2n} <> Impl.tag_norm(NJ, R3) : List<&2, Spec.Ev>}
  +R1 = {Spec.Bar{1n} <> Impl.tag_fin(FJ, R2) : List<&2, Spec.Ev>}
  %nh_work(ML, R1, x) : {_ == MSpec.occurs(NJ, x) : Nat}
  %nh_fin(FJ, R2, x) : {_ == MSpec.occurs(NJ, x) : Nat}
  %nh_norm(NJ, R3, x) : {_ == MSpec.occurs(NJ, x) : Nat}
  %nh_sync(EV, R4, x) : {Nat.add(MSpec.occurs(NJ, x), _) == MSpec.occurs(NJ, x) : Nat}
  PP.ar_add_zero_r(MSpec.occurs(NJ, x))

# k + 1 <= G gives k < G and k <= G.
def lt_of_succ_le(+k: Nat, +G: Nat, h: {Nat.is_le(1n+k, G) == True{} : Bool})
  -> {Nat.is_lt(k, G) == True{} : Bool}:
  PP.ard_tr(Nat.is_lt(k, G), Nat.is_le(1n+k, G), PP.ar_lt_succ_le(k, G), h)

def le_of_succ_le(+k: Nat, +G: Nat, h: {Nat.is_le(1n+k, G) == True{} : Bool})
  -> {Nat.is_le(k, G) == True{} : Bool}:
  PP.ar_le_trans(k, 1n+k, G, PP.ar_h_le_succ(k), h)

# Blocks b < n <= G run job x < J0 [x mod G < n] times.
def ft_in(+n: Nat, +G: Nat, +A: Nat, +H: Nat, +I: Nat, +m: Nat, +x: Nat,
    +gpos: {Nat.is_lt(0n, G) == True{} : Bool}, +nle: {Nat.is_le(n, G) == True{} : Bool},
    +fits: {Nat.is_le(Impl.jobs0(H, m), Nat.mul(8n, G)) == True{} : Bool},
    +xj: {Nat.is_lt(x, Impl.jobs0(H, m)) == True{} : Bool})
  -> {Laws.fin_total(G, A, H, I, m, n, x) == PSpec.bit(Nat.is_lt(Nat.mod(x, G), n)) : Nat}:
  match n:
    case 0n:
      +r = Nat.mod(x, G)
      %Equal.sym(Bool, Nat.is_lt(r, 0n), False{}, PP.ar_h_lt_zero(r)) : {0n == PSpec.bit(_) : Nat}
      {==}
    case 1n+k:
      +r = Nat.mod(x, G)
      %Equal.sym(Nat, Laws.fin_total(G, A, H, I, m, k, x), PSpec.bit(Nat.is_lt(r, k)),
          ft_in(k, G, A, H, I, m, x, gpos, le_of_succ_le(k, G, nle), fits, xj))
        : {Nat.add(_, Spec.fin_hits(Impl.events(G, A, H, I, m, k), x)) ==
          PSpec.bit(Nat.is_lt(r, 1n+k)) : Nat}
      %Equal.sym(Nat, Spec.fin_hits(Impl.events(G, A, H, I, m, k), x), PSpec.bit(Nat.is_eq(k, r)),
          Equal.trans(Nat, Spec.fin_hits(Impl.events(G, A, H, I, m, k), x),
            MSpec.occurs(Impl.warp_jobs(G, Impl.jobs0(H, m), k), x), PSpec.bit(Nat.is_eq(k, r)),
            fin_block(G, A, H, I, m, k, x),
            occ_block(G, Impl.jobs0(H, m), k, x, gpos, lt_of_succ_le(k, G, nle), fits, xj)))
        : {Nat.add(PSpec.bit(Nat.is_lt(r, k)), _) == PSpec.bit(Nat.is_lt(r, 1n+k)) : Nat}
      MP.bit_step(r, k)

def ft_out(+n: Nat, +G: Nat, +A: Nat, +H: Nat, +I: Nat, +m: Nat, +x: Nat,
    +xo: {Nat.is_lt(x, Impl.jobs0(H, m)) == False{} : Bool})
  -> {Laws.fin_total(G, A, H, I, m, n, x) == 0n : Nat}:
  match n:
    case 0n:
      {==}
    case 1n+k:
      %Equal.sym(Nat, Laws.fin_total(G, A, H, I, m, k, x), 0n, ft_out(k, G, A, H, I, m, x, xo))
        : {Nat.add(_, Spec.fin_hits(Impl.events(G, A, H, I, m, k), x)) == 0n : Nat}
      %Equal.sym(Nat, Spec.fin_hits(Impl.events(G, A, H, I, m, k), x),
          MSpec.occurs(Impl.warp_jobs(G, Impl.jobs0(H, m), k), x), fin_block(G, A, H, I, m, k, x))
        : {Nat.add(0n, _) == 0n : Nat}
      occ_go_out(8n, G, Impl.jobs0(H, m), k, x, xo)

def fe_split(v: Bool, +G: Nat, +A: Nat, +H: Nat, +I: Nat, +m: Nat, +x: Nat,
    +gpos: {Nat.is_lt(0n, G) == True{} : Bool},
    +fits: {Nat.is_le(Impl.jobs0(H, m), Nat.mul(8n, G)) == True{} : Bool},
    ev: {Nat.is_lt(x, Impl.jobs0(H, m)) == v : Bool})
  -> {Laws.fin_total(G, A, H, I, m, G, x) == PSpec.bit(v) : Nat}:
  match v:
    case True{}:
      %Equal.sym(Nat, Laws.fin_total(G, A, H, I, m, G, x), PSpec.bit(Nat.is_lt(Nat.mod(x, G), G)),
          ft_in(G, G, A, H, I, m, x, gpos, PP.ar_le_refl(G), fits, ev))
        : {_ == 1n : Nat}
      %Equal.sym(Bool, Nat.is_lt(Nat.mod(x, G), G), True{}, PP.ard_dm_lt(x, G, gpos))
        : {PSpec.bit(_) == 1n : Nat}
      {==}
    case False{}:
      ft_out(G, G, A, H, I, m, x, ev)

def Laws.fin_exactly_once(G, A, H, I, m, x, grid, fits):
  fe_split(Nat.is_lt(x, Impl.jobs0(H, m)), G, A, H, I, m, x, grid, fits, {==})

def nt_in(+n: Nat, +G: Nat, +A: Nat, +H: Nat, +I: Nat, +m: Nat, +x: Nat,
    +gpos: {Nat.is_lt(0n, G) == True{} : Bool}, +nle: {Nat.is_le(n, G) == True{} : Bool},
    +fits: {Nat.is_le(Impl.jobsn(H, m), Nat.mul(8n, G)) == True{} : Bool},
    +xj: {Nat.is_lt(x, Impl.jobsn(H, m)) == True{} : Bool})
  -> {Laws.norm_total(G, A, H, I, m, n, x) == PSpec.bit(Nat.is_lt(Nat.mod(x, G), n)) : Nat}:
  match n:
    case 0n:
      +r = Nat.mod(x, G)
      %Equal.sym(Bool, Nat.is_lt(r, 0n), False{}, PP.ar_h_lt_zero(r)) : {0n == PSpec.bit(_) : Nat}
      {==}
    case 1n+k:
      +r = Nat.mod(x, G)
      %Equal.sym(Nat, Laws.norm_total(G, A, H, I, m, k, x), PSpec.bit(Nat.is_lt(r, k)),
          nt_in(k, G, A, H, I, m, x, gpos, le_of_succ_le(k, G, nle), fits, xj))
        : {Nat.add(_, Spec.norm_hits(Impl.events(G, A, H, I, m, k), x)) ==
          PSpec.bit(Nat.is_lt(r, 1n+k)) : Nat}
      %Equal.sym(Nat, Spec.norm_hits(Impl.events(G, A, H, I, m, k), x), PSpec.bit(Nat.is_eq(k, r)),
          Equal.trans(Nat, Spec.norm_hits(Impl.events(G, A, H, I, m, k), x),
            MSpec.occurs(Impl.warp_jobs(G, Impl.jobsn(H, m), k), x), PSpec.bit(Nat.is_eq(k, r)),
            norm_block(G, A, H, I, m, k, x),
            occ_block(G, Impl.jobsn(H, m), k, x, gpos, lt_of_succ_le(k, G, nle), fits, xj)))
        : {Nat.add(PSpec.bit(Nat.is_lt(r, k)), _) == PSpec.bit(Nat.is_lt(r, 1n+k)) : Nat}
      MP.bit_step(r, k)

def nt_out(+n: Nat, +G: Nat, +A: Nat, +H: Nat, +I: Nat, +m: Nat, +x: Nat,
    +xo: {Nat.is_lt(x, Impl.jobsn(H, m)) == False{} : Bool})
  -> {Laws.norm_total(G, A, H, I, m, n, x) == 0n : Nat}:
  match n:
    case 0n:
      {==}
    case 1n+k:
      %Equal.sym(Nat, Laws.norm_total(G, A, H, I, m, k, x), 0n, nt_out(k, G, A, H, I, m, x, xo))
        : {Nat.add(_, Spec.norm_hits(Impl.events(G, A, H, I, m, k), x)) == 0n : Nat}
      %Equal.sym(Nat, Spec.norm_hits(Impl.events(G, A, H, I, m, k), x),
          MSpec.occurs(Impl.warp_jobs(G, Impl.jobsn(H, m), k), x), norm_block(G, A, H, I, m, k, x))
        : {Nat.add(0n, _) == 0n : Nat}
      occ_go_out(8n, G, Impl.jobsn(H, m), k, x, xo)

def ne_split(v: Bool, +G: Nat, +A: Nat, +H: Nat, +I: Nat, +m: Nat, +x: Nat,
    +gpos: {Nat.is_lt(0n, G) == True{} : Bool},
    +fits: {Nat.is_le(Impl.jobsn(H, m), Nat.mul(8n, G)) == True{} : Bool},
    ev: {Nat.is_lt(x, Impl.jobsn(H, m)) == v : Bool})
  -> {Laws.norm_total(G, A, H, I, m, G, x) == PSpec.bit(v) : Nat}:
  match v:
    case True{}:
      %Equal.sym(Nat, Laws.norm_total(G, A, H, I, m, G, x), PSpec.bit(Nat.is_lt(Nat.mod(x, G), G)),
          nt_in(G, G, A, H, I, m, x, gpos, PP.ar_le_refl(G), fits, ev))
        : {_ == 1n : Nat}
      %Equal.sym(Bool, Nat.is_lt(Nat.mod(x, G), G), True{}, PP.ard_dm_lt(x, G, gpos))
        : {PSpec.bit(_) == 1n : Nat}
      {==}
    case False{}:
      nt_out(G, G, A, H, I, m, x, ev)

def Laws.norm_exactly_once(G, A, H, I, m, x, grid, fits):
  ne_split(Nat.is_lt(x, Impl.jobsn(H, m)), G, A, H, I, m, x, grid, fits, {==})

# ==== (3) the weight ring ====

# (a mod p + t) mod p == (a + t) mod p   (p >= 1)
def mod_shift(+a: Nat, +t: Nat, +p: Nat, +pos: {Nat.is_lt(0n, p) == True{} : Bool})
  -> {Nat.mod(Nat.add(Nat.mod(a, p), t), p) == Nat.mod(Nat.add(a, t), p) : Nat}:
  +q = Nat.div(a, p)
  +r = Nat.mod(a, p)
  +S = Nat.add(r, t)
  +Q = Nat.div(S, p)
  +R = Nat.mod(S, p)
  +qp = Nat.mul(q, p)
  +Qp = Nat.mul(Q, p)
  +s1 = Equal.cong(Nat, Nat, z => Nat.add(z, t), a, Nat.add(qp, r),
    Equal.sym(Nat, Nat.add(qp, r), a, PP.ard_dm_eq(a, p, pos)))
  +s2 = PP.ar_add_assoc(qp, r, t)
  +s3 = Equal.cong(Nat, Nat, z => Nat.add(qp, z), S, Nat.add(Qp, R),
    Equal.sym(Nat, Nat.add(Qp, R), S, PP.ard_dm_eq(S, p, pos)))
  +s4 = Equal.sym(Nat, Nat.add(Nat.add(qp, Qp), R), Nat.add(qp, Nat.add(Qp, R)),
    PP.ar_add_assoc(qp, Qp, R))
  +s5 = Equal.cong(Nat, Nat, z => Nat.add(z, R), Nat.add(qp, Qp), Nat.mul(Nat.add(q, Q), p),
    Equal.sym(Nat, Nat.mul(Nat.add(q, Q), p), Nat.add(qp, Qp), PP.ar_mul_add_l(q, Q, p)))
  +eq = Equal.trans(Nat, Nat.add(a, t), Nat.add(Nat.add(qp, r), t),
    Nat.add(Nat.mul(Nat.add(q, Q), p), R), s1,
    Equal.trans(Nat, Nat.add(Nat.add(qp, r), t), Nat.add(qp, S),
      Nat.add(Nat.mul(Nat.add(q, Q), p), R), s2,
      Equal.trans(Nat, Nat.add(qp, S), Nat.add(qp, Nat.add(Qp, R)),
        Nat.add(Nat.mul(Nat.add(q, Q), p), R), s3,
        Equal.trans(Nat, Nat.add(qp, Nat.add(Qp, R)), Nat.add(Nat.add(qp, Qp), R),
          Nat.add(Nat.mul(Nat.add(q, Q), p), R), s4, s5))))
  %Equal.sym(Nat, Nat.add(a, t), Nat.add(Nat.mul(Nat.add(q, Q), p), R), eq)
    : {R == Nat.mod(_, p) : Nat}
  Equal.sym(Nat, Nat.mod(Nat.add(Nat.mul(Nat.add(q, Q), p), R), p), R,
    PP.p7_mod_r(Nat.add(q, Q), R, p, PP.ard_dm_lt(S, p, pos)))

def Laws.ring_slots(G, A, H, I, b, t):
  +n0 = Impl.n0(G, A, H, b)
  (mod_shift(n0, t, 8n, {==}), mod_shift(Nat.add(n0, Mlp.n1(G, H, I, b)), t, 8n, {==}))

def Laws.ring_stream(i):
  (MLaws.ring_issue_in_time(i, Impl.pf()),
    PP.ard_fst({Mlp.slot(Nat.add(i, Impl.pf()), Impl.pf()) == Mlp.slot(i, Impl.pf()) : Nat},
      {Mlp.issue_step(Nat.add(i, Impl.pf()), Impl.pf()) == 1n+i : Nat},
      MLaws.ring_reuse(i, Impl.pf(), {==})),
    PP.ard_snd({Mlp.slot(Nat.add(i, Impl.pf()), Impl.pf()) == Mlp.slot(i, Impl.pf()) : Nat},
      {Mlp.issue_step(Nat.add(i, Impl.pf()), Impl.pf()) == 1n+i : Nat},
      MLaws.ring_reuse(i, Impl.pf(), {==})))

# r < r + d for d > 0, as a disequality.
def ne_add_pos(+r: Nat, +d: Nat, pos: {Nat.is_lt(0n, d) == True{} : Bool})
  -> {Nat.is_eq(Nat.add(r, d), r) == False{} : Bool}:
  +lt = PP.ard_tr(Nat.is_lt(r, Nat.add(r, d)), Nat.is_lt(0n, d),
    Equal.trans(Bool, Nat.is_lt(r, Nat.add(r, d)), Nat.is_lt(Nat.add(r, 0n), Nat.add(r, d)),
      Nat.is_lt(0n, d),
      Equal.cong(Nat, Bool, z => Nat.is_lt(z, Nat.add(r, d)), r, Nat.add(r, 0n),
        Equal.sym(Nat, Nat.add(r, 0n), r, PP.ar_add_zero_r(r))),
      PP.ar_lt_add_cancel(r, 0n, d)),
    pos)
  PP.ar_eq_false(Nat.add(r, d), r, e => PP.p5_eq_lt_absurd(r, Nat.add(r, d),
    Equal.sym(Nat, Nat.add(r, d), r, e), lt))

# a < c and b < c give a + b < c + c.
def add_lt2(+a: Nat, +b: Nat, +c: Nat, ac: {Nat.is_lt(a, c) == True{} : Bool},
    bc: {Nat.is_lt(b, c) == True{} : Bool})
  -> {Nat.is_lt(Nat.add(a, b), Nat.add(c, c)) == True{} : Bool}:
  PP.ar_lt_trans(Nat.add(a, b), Nat.add(a, c), Nat.add(c, c),
    PP.ard_tr(Nat.is_lt(Nat.add(a, b), Nat.add(a, c)), Nat.is_lt(b, c),
      PP.ar_lt_add_cancel(a, b, c), bc),
    %PP.ar_add_comm(c, a) : {Nat.is_lt(_, Nat.add(c, c)) == True{} : Bool}
    PP.ard_tr(Nat.is_lt(Nat.add(c, a), Nat.add(c, c)), Nat.is_lt(a, c),
      PP.ar_lt_add_cancel(c, a, c), ac))

# (r + d) - p == r with p <= r + d and d < p is absurd: r + d < r + p == p + r == r + d.
def ifd_absurd(+r: Nat, +d: Nat, +p: Nat, near: {Nat.is_lt(d, p) == True{} : Bool},
    Su: {Nat.add(p, Nat.sub(Nat.add(r, d), p)) == Nat.add(r, d) : Nat},
    ur: {Nat.sub(Nat.add(r, d), p) == r : Nat})
  -> Empty:
  +S = Nat.add(r, d)
  +e1 = Equal.trans(Nat, Nat.add(p, r), Nat.add(p, Nat.sub(S, p)), S,
    Equal.cong(Nat, Nat, z => Nat.add(p, z), r, Nat.sub(S, p), Equal.sym(Nat, Nat.sub(S, p), r, ur)),
    Su)
  +eqS = Equal.sym(Nat, Nat.add(r, p), S,
    Equal.trans(Nat, Nat.add(r, p), Nat.add(p, r), S, PP.ar_add_comm(r, p), e1))
  PP.ard_lt_self(S, Equal.trans(Bool, Nat.is_lt(S, S), Nat.is_lt(S, Nat.add(r, p)), True{},
    Equal.cong(Nat, Bool, z => Nat.is_lt(S, z), S, Nat.add(r, p), eqS),
    PP.ard_tr(Nat.is_lt(S, Nat.add(r, p)), Nat.is_lt(d, p), PP.ar_lt_add_cancel(r, d, p), near)))

# For r < p and 0 < d < p: (r + d) mod p != r, split on r + d < p.
def ifd_split(v: Bool, +r: Nat, +d: Nat, +p: Nat, +rp: {Nat.is_lt(r, p) == True{} : Bool},
    +pos: {Nat.is_lt(0n, d) == True{} : Bool}, +near: {Nat.is_lt(d, p) == True{} : Bool},
    ev: {Nat.is_lt(Nat.add(r, d), p) == v : Bool})
  -> {Nat.is_eq(Nat.mod(Nat.add(r, d), p), r) == False{} : Bool}:
  match v:
    case True{}:
      %Equal.sym(Nat, Nat.mod(Nat.add(r, d), p), Nat.add(r, d),
          PP.p7_mod_r(0n, Nat.add(r, d), p, ev))
        : {Nat.is_eq(_, r) == False{} : Bool}
      ne_add_pos(r, d, pos)
    case False{}:
      +S = Nat.add(r, d)
      +u = Nat.sub(S, p)
      +le = PP.p6_nlt_le(S, p, ev)
      +Su = PP.ar_sub_add(S, p, le)
      +ulp = PP.ard_tr(Nat.is_lt(u, p), Nat.is_lt(S, Nat.add(p, p)), PP.ar_sub_lt_iff(S, p, p,
        PP.ard_pos(r, p, rp)), add_lt2(r, d, p, rp, near))
      +m1 = PP.p7_mod_r(1n, u, p, ulp)
      +m2 = Equal.trans(Nat, Nat.mod(S, p), Nat.mod(Nat.add(Nat.mul(1n, p), u), p), u,
        Equal.cong(Nat, Nat, z => Nat.mod(z, p), S, Nat.add(Nat.mul(1n, p), u),
          Equal.trans(Nat, S, Nat.add(p, u), Nat.add(Nat.mul(1n, p), u),
            Equal.sym(Nat, Nat.add(p, u), S, Su),
            Equal.cong(Nat, Nat, z => Nat.add(z, u), p, Nat.mul(1n, p),
              Equal.sym(Nat, Nat.mul(1n, p), p, PP.ar_mul_one_l(p))))),
        m1)
      %Equal.sym(Nat, Nat.mod(S, p), u, m2) : {Nat.is_eq(_, r) == False{} : Bool}
      PP.ar_eq_false(u, r, e => ifd_absurd(r, d, p, near, Su, e))

def Laws.ring_in_flight_distinct(i, d, pf, pos, near):
  +ps = pos
  +nr = near
  +pp = PP.ar_lt_trans(0n, d, pf, ps, nr)
  +r = Nat.mod(i, pf)
  %mod_shift(i, d, pf, pp) : {Nat.is_eq(_, r) == False{} : Bool}
  ifd_split(Nat.is_lt(Nat.add(r, d), pf), r, d, pf, PP.ard_dm_lt(i, pf, pp), ps, nr, {==})

# t < P1 and 0 < off give (off + t) - P1 < off.
def pre_lt(+off: Nat, +t: Nat, +P1: Nat, opos: {Nat.is_lt(0n, off) == True{} : Bool},
    tl: {Nat.is_lt(t, P1) == True{} : Bool})
  -> {Nat.is_lt(Nat.sub(Nat.add(off, t), P1), off) == True{} : Bool}:
  %Equal.sym(Bool, Nat.is_lt(Nat.sub(Nat.add(off, t), P1), off),
      Nat.is_lt(Nat.add(off, t), Nat.add(P1, off)), PP.ar_sub_lt_iff(Nat.add(off, t), P1, off, opos))
    : {_ == True{} : Bool}
  %PP.ar_add_comm(off, P1) : {Nat.is_lt(Nat.add(off, t), _) == True{} : Bool}
  PP.ard_tr(Nat.is_lt(Nat.add(off, t), Nat.add(off, P1)), Nat.is_lt(t, P1),
    PP.ar_lt_add_cancel(off, t, P1), tl)

# a < b gives 0 < b - a.
def sub_pos(+a: Nat, +b: Nat, lt: {Nat.is_lt(a, b) == True{} : Bool})
  -> {Nat.is_lt(0n, Nat.sub(b, a)) == True{} : Bool}:
  %Equal.sym(Bool, Nat.is_lt(0n, Nat.sub(b, a)), Nat.is_lt(Nat.add(a, 0n), b), PP.ar_lt_sub_iff(b, a, 0n))
    : {_ == True{} : Bool}
  %Equal.sym(Nat, Nat.add(a, 0n), a, PP.ar_add_zero_r(a)) : {Nat.is_lt(_, b) == True{} : Bool}
  lt

def Laws.ring_prefetch(G, A, H, I, b, t, grid, fits, early):
  +n0 = Impl.n0(G, A, H, b)
  +n1 = Mlp.n1(G, H, I, b)
  +np = sub_pos(Impl.s0(G, A, H, b), Impl.e0(G, A, H, b),
    PP.p2_nonempty(b, Impl.total0(A, H), G, grid, fits))
  +el = early
  (pre_lt(n0, t, 7n, np, el),
    pre_lt(Nat.add(n0, n1), t, 7n, PP.ar_lt_le_trans(0n, n0, Nat.add(n0, n1), np,
      PP.ar_le_add_r(n0, n1)), el))

# ==== (4) served job counts, chunk cover, F0 decode ====

def fit_m(+m: Nat, rows: {Nat.is_le(m, 8n) == True{} : Bool})
  -> {Nat.is_le(Impl.jobs0(Impl.hidden(), m), Nat.mul(8n, Impl.grid())) == True{} : Bool} &
    {Nat.is_le(Impl.jobsn(Impl.hidden(), m), Nat.mul(8n, Impl.grid())) == True{} : Bool}:
  match m:
    case 0n:
      ({==}, {==})
    case 1n+m1:
      match m1:
        case 0n:
          ({==}, {==})
        case 1n+m2:
          match m2:
            case 0n:
              ({==}, {==})
            case 1n+m3:
              match m3:
                case 0n:
                  ({==}, {==})
                case 1n+m4:
                  match m4:
                    case 0n:
                      ({==}, {==})
                    case 1n+m5:
                      match m5:
                        case 0n:
                          ({==}, {==})
                        case 1n+m6:
                          match m6:
                            case 0n:
                              ({==}, {==})
                            case 1n+m7:
                              match m7:
                                case 0n:
                                  ({==}, {==})
                                case 1n+m8:
                                  match m8:
                                    case 0n:
                                      ({==}, {==})
                                    case 1n+m9:
                                      Empty.absurd(
                                        {Nat.is_le(Impl.jobs0(Impl.hidden(), 9n+m9),
                                            Nat.mul(8n, Impl.grid())) == True{} : Bool} &
                                          {Nat.is_le(Impl.jobsn(Impl.hidden(), 9n+m9),
                                            Nat.mul(8n, Impl.grid())) == True{} : Bool},
                                        PP.ar_false_true(rows))

def Laws.jobs_fit_served(G, H, m, served_g, served_h, rows):
  %Equal.sym(Nat, G, Impl.grid(), served_g)
    : {Nat.is_le(Impl.jobs0(H, m), Nat.mul(8n, _)) == True{} : Bool} &
      {Nat.is_le(Impl.jobsn(H, m), Nat.mul(8n, _)) == True{} : Bool}
  %Equal.sym(Nat, H, Impl.hidden(), served_h)
    : {Nat.is_le(Impl.jobs0(_, m), Nat.mul(8n, Impl.grid())) == True{} : Bool} &
      {Nat.is_le(Impl.jobsn(_, m), Nat.mul(8n, Impl.grid())) == True{} : Bool}
  fit_m(m, rows)

def Laws.nch_covers(H, groups):
  +q = Nat.div(H, 512n)
  +qm = Nat.mul(q, 512n)
  Equal.trans(Nat, Nat.mul(Nat.mul(q, 4n), 128n), qm, H, PP.ar_mul_assoc(q, 4n, 128n),
    Equal.trans(Nat, qm, Nat.add(qm, 0n), H, Equal.sym(Nat, Nat.add(qm, 0n), qm, PP.ar_add_zero_r(qm)),
      Equal.trans(Nat, Nat.add(qm, 0n), Nat.add(qm, Nat.mod(H, 512n)), H,
        Equal.cong(Nat, Nat, z => Nat.add(qm, z), 0n, Nat.mod(H, 512n),
          Equal.sym(Nat, Nat.mod(H, 512n), 0n, groups)),
        PP.ard_dm_eq(H, 512n, {==}))))

# The contract's F0 decode (x div 4m, (x mod 4m) div 4, x mod 4) is v2's finisher decode
# (gemm_m16_partition job, K:360-363) whenever 4 m > 0.
def f0_eq_job(+x: Nat, +m: Nat, +mpos: {Nat.is_lt(0n, Nat.mul(m, 4n)) == True{} : Bool})
  -> {Impl.f0_job(x, m) == Part.job(x, m) : PSpec.Job}:
  +M = Nat.mul(m, 4n)
  +g = Nat.div(x, M)
  +rem = Nat.sub(x, Nat.mul(Nat.mul(g, m), 4n))
  +R = Nat.mod(x, M)
  +er = Equal.trans(Nat, R, Nat.sub(x, Nat.mul(g, M)), rem,
    Equal.sym(Nat, Nat.sub(x, Nat.mul(g, M)), R, PP.p9_sub_mod(x, M, mpos)),
    Equal.cong(Nat, Nat, z => Nat.sub(x, z), Nat.mul(g, M), Nat.mul(Nat.mul(g, m), 4n),
      Equal.sym(Nat, Nat.mul(Nat.mul(g, m), 4n), Nat.mul(g, M), PP.ar_mul_assoc(g, m, 4n))))
  %er : {PSpec.Job{g, Nat.div(R, 4n), Nat.mod(x, 4n)} ==
      PSpec.Job{g, Nat.div(_, 4n), Nat.sub(_, Nat.mul(Nat.div(_, 4n), 4n))} : PSpec.Job}
  %Equal.sym(Nat, Nat.sub(R, Nat.mul(Nat.div(R, 4n), 4n)), Nat.mod(R, 4n), PP.p9_sub_mod(R, 4n, {==}))
    : {PSpec.Job{g, Nat.div(R, 4n), Nat.mod(x, 4n)} == PSpec.Job{g, Nat.div(R, 4n), _} : PSpec.Job}
  %Equal.sym(Nat, Nat.mod(R, 4n), Nat.mod(x, 4n), PP.p7_mod_nest(x, m, 4n, {==}, mpos))
    : {PSpec.Job{g, Nat.div(R, 4n), Nat.mod(x, 4n)} == PSpec.Job{g, Nat.div(R, 4n), _} : PSpec.Job}
  {==}

def Laws.f0_decode(H, m, x, inside):
  +ins = inside
  %Equal.sym(PSpec.Job, Impl.f0_job(x, m), Part.job(x, m),
      f0_eq_job(x, m, PP.p9_dec_pos(Impl.ng0(H), m, x, ins)))
    : {PSpec.job_ok(_, Impl.ng0(H), m) == True{} : Bool} & {PSpec.encode(_, m) == x : Nat}
  PLaws.finish_job_decode(Impl.ng0(H), m, x, ins)

def Laws.f0_encode(H, m, j, ok):
  match j:
    case PSpec.Job{g, r, ch}:
      +okp = ok
      +rm = PP.ar_and_l(Nat.is_lt(r, m), Nat.is_lt(ch, 4n),
        PP.ar_and_r(Nat.is_lt(g, Impl.ng0(H)), Bool.and(Nat.is_lt(r, m), Nat.is_lt(ch, 4n)), okp))
      +E = PSpec.encode(PSpec.Job{g, r, ch}, m)
      %Equal.sym(PSpec.Job, Impl.f0_job(E, m), Part.job(E, m),
          f0_eq_job(E, m, PP.p9_m4_pos(m, PP.ard_pos(r, m, rm))))
        : {_ == PSpec.Job{g, r, ch} : PSpec.Job} &
          {Nat.is_lt(E, Impl.jobs0(H, m)) == True{} : Bool}
      PLaws.finish_job_encode(Impl.ng0(H), m, PSpec.Job{g, r, ch}, okp)

# ==== (2) MC0 at the served instance ====

# v2's owner (and so max_contrib) is invariant under scaling total and G by one k >= 1; the served
# MC0 (total 3840 = 960 * 4, G = 164 = 41 * 4) is then a closed computation at total 960, G 41.

# (1 + z) - 1 == z.
def sub_succ1(+z: Nat) -> {Nat.sub(1n+z, 1n) == z : Nat}:
  PP.ar_sub_zero_r(z)

# (k + y) - 1 == (k - 1) + y for k >= 1.
def sub1_add(+k: Nat, +y: Nat, kpos: {Nat.is_lt(0n, k) == True{} : Bool})
  -> {Nat.sub(Nat.add(k, y), 1n) == Nat.add(Nat.sub(k, 1n), y) : Nat}:
  match k:
    case 0n:
      Empty.absurd({Nat.sub(Nat.add(0n, y), 1n) == Nat.add(Nat.sub(0n, 1n), y) : Nat},
        PP.ar_false_true(kpos))
    case 1n+j:
      %Equal.sym(Nat, Nat.sub(1n+Nat.add(j, y), 1n), Nat.add(j, y), sub_succ1(Nat.add(j, y)))
        : {_ == Nat.add(Nat.sub(1n+j, 1n), y) : Nat}
      %Equal.sym(Nat, Nat.sub(1n+j, 1n), j, sub_succ1(j)) : {Nat.add(j, y) == Nat.add(_, y) : Nat}
      {==}

# ((1 + S1) k - 1) div (T k) == S1 div T for k, T >= 1: with S1 = q T + r,
# (1 + S1) k - 1 = q (T k) + ((k - 1) + r k) and (k - 1) + r k = (1 + r) k - 1 < T k.
def own_div(+S1: Nat, +T: Nat, +k: Nat, +kpos: {Nat.is_lt(0n, k) == True{} : Bool},
    +tpos: {Nat.is_lt(0n, T) == True{} : Bool})
  -> {Nat.div(Nat.sub(Nat.mul(1n+S1, k), 1n), Nat.mul(T, k)) == Nat.div(S1, T) : Nat}:
  +q = Nat.div(S1, T)
  +r = Nat.mod(S1, T)
  +Tk = Nat.mul(T, k)
  +km = Nat.sub(k, 1n)
  +R = Nat.add(km, Nat.mul(r, k))
  +Y = Nat.mul(1n+r, k)
  +YT = PP.ar_le_mul_mono(1n+r, T, k,
    PP.ard_rt(Nat.is_lt(r, T), Nat.is_le(1n+r, T), PP.ar_lt_succ_le(r, T), PP.ard_dm_lt(S1, T, tpos)))
  +tkpos = PP.ard_tr(Nat.is_lt(Nat.mul(0n, k), Tk), Nat.is_lt(0n, T), PP.ar_mul_lt_cancel(0n, T, k, kpos),
    tpos)
  +subY = PP.ard_tr(Nat.is_lt(Nat.sub(Y, 1n), Tk), Nat.is_lt(Y, Nat.add(1n, Tk)),
    PP.ar_sub_lt_iff(Y, 1n, Tk, tkpos),
    PP.ard_rt(Nat.is_le(Y, Tk), Nat.is_lt(Y, 1n+Tk), PP.p1_le_lt_succ(Y, Tk), YT))
  +Rlt = PP.p5_lt_rw(Nat.sub(Y, 1n), R, Tk, sub1_add(k, Nat.mul(r, k), kpos), subY)
  %Equal.sym(Nat, Nat.sub(Nat.add(k, Nat.mul(S1, k)), 1n), Nat.add(km, Nat.mul(S1, k)),
      sub1_add(k, Nat.mul(S1, k), kpos))
    : {Nat.div(_, Tk) == q : Nat}
  %Equal.sym(Nat, S1, Nat.add(Nat.mul(q, T), r),
      Equal.sym(Nat, Nat.add(Nat.mul(q, T), r), S1, PP.ard_dm_eq(S1, T, tpos)))
    : {Nat.div(Nat.add(km, Nat.mul(_, k)), Tk) == q : Nat}
  %Equal.sym(Nat, Nat.mul(Nat.add(Nat.mul(q, T), r), k), Nat.add(Nat.mul(Nat.mul(q, T), k), Nat.mul(r, k)),
      PP.ar_mul_add_l(Nat.mul(q, T), r, k))
    : {Nat.div(Nat.add(km, _), Tk) == q : Nat}
  %Equal.sym(Nat, Nat.mul(Nat.mul(q, T), k), Nat.mul(q, Tk), PP.ar_mul_assoc(q, T, k))
    : {Nat.div(Nat.add(km, Nat.add(_, Nat.mul(r, k))), Tk) == q : Nat}
  %Equal.sym(Nat, Nat.add(km, Nat.add(Nat.mul(q, Tk), Nat.mul(r, k))), Nat.add(Nat.mul(q, Tk), R),
      PP.ar_h_add_swap(km, Nat.mul(q, Tk), Nat.mul(r, k)))
    : {Nat.div(_, Tk) == q : Nat}
  PP.ard_fst({Nat.div(Nat.add(Nat.mul(q, Tk), R), Tk) == q : Nat},
    {Nat.mod(Nat.add(Nat.mul(q, Tk), R), Tk) == R : Nat}, PP.ar_div_unique(q, R, Tk, Rlt))

# owner(x, T k, G k) == owner(x, T, G) for k, T >= 1.
def own_scale(+x: Nat, +T: Nat, +G: Nat, +k: Nat, +kpos: {Nat.is_lt(0n, k) == True{} : Bool},
    tpos: {Nat.is_lt(0n, T) == True{} : Bool})
  -> {Part.owner(x, Nat.mul(T, k), Nat.mul(G, k)) == Part.owner(x, T, G) : Nat}:
  match T:
    case 0n:
      Empty.absurd({Part.owner(x, Nat.mul(0n, k), Nat.mul(G, k)) == Part.owner(x, 0n, G) : Nat},
        PP.ar_false_true(tpos))
    case 1n+T1:
      +a = Nat.mul(1n+x, G)
      +S1 = Nat.add(a, T1)
      +Tk = Nat.mul(1n+T1, k)
      %PP.ar_mul_assoc(1n+x, G, k)
        : {Nat.sub(Nat.div(Nat.sub(Nat.add(_, Tk), 1n), Tk), 1n) ==
          Nat.sub(Nat.div(Nat.sub(Nat.add(a, 1n+T1), 1n), 1n+T1), 1n) : Nat}
      %PP.ar_mul_add_l(a, 1n+T1, k)
        : {Nat.sub(Nat.div(Nat.sub(_, 1n), Tk), 1n) ==
          Nat.sub(Nat.div(Nat.sub(Nat.add(a, 1n+T1), 1n), 1n+T1), 1n) : Nat}
      %Equal.sym(Nat, Nat.add(a, 1n+T1), 1n+S1, PP.ar_add_succ_r(a, T1))
        : {Nat.sub(Nat.div(Nat.sub(Nat.mul(_, k), 1n), Tk), 1n) ==
          Nat.sub(Nat.div(Nat.sub(_, 1n), 1n+T1), 1n) : Nat}
      %Equal.sym(Nat, Nat.sub(1n+S1, 1n), S1, sub_succ1(S1))
        : {Nat.sub(Nat.div(Nat.sub(Nat.mul(1n+S1, k), 1n), Tk), 1n) ==
          Nat.sub(Nat.div(_, 1n+T1), 1n) : Nat}
      %Equal.sym(Nat, Nat.div(Nat.sub(Nat.mul(1n+S1, k), 1n), Tk), Nat.div(S1, 1n+T1),
          own_div(S1, 1n+T1, k, kpos, tpos))
        : {Nat.sub(_, 1n) == Nat.sub(Nat.div(S1, 1n+T1), 1n) : Nat}
      {==}

# max_contrib.go(left, g, KT, T k, G k, mc) == max_contrib.go(left, g, KT, T, G, mc).
def mc_go_scale(left: Nat, +g: Nat, +KT: Nat, +T: Nat, +G: Nat, +k: Nat, +mc: Nat,
    +kpos: {Nat.is_lt(0n, k) == True{} : Bool}, +tpos: {Nat.is_lt(0n, T) == True{} : Bool})
  -> {Part.max_contrib.go(left, g, KT, Nat.mul(T, k), Nat.mul(G, k), mc) ==
    Part.max_contrib.go(left, g, KT, T, G, mc) : Nat}:
  match left:
    case 0n:
      {==}
    case 1n+rest:
      +hi = Nat.sub(Nat.mul(1n+g, KT), 1n)
      +lo = Nat.mul(g, KT)
      +Tk = Nat.mul(T, k)
      +Gk = Nat.mul(G, k)
      +n = Nat.add(Nat.sub(Part.owner(hi, T, G), Part.owner(lo, T, G)), 1n)
      %Equal.sym(Nat, Part.owner(hi, Tk, Gk), Part.owner(hi, T, G), own_scale(hi, T, G, k, kpos, tpos))
        : {Part.max_contrib.go(rest, 1n+g, KT, Tk, Gk,
            Part.max_contrib.pick(Nat.is_gt(Nat.add(Nat.sub(_, Part.owner(lo, Tk, Gk)), 1n), mc),
              Nat.add(Nat.sub(_, Part.owner(lo, Tk, Gk)), 1n), mc)) ==
          Part.max_contrib.go(rest, 1n+g, KT, T, G, Part.max_contrib.pick(Nat.is_gt(n, mc), n, mc)) : Nat}
      %Equal.sym(Nat, Part.owner(lo, Tk, Gk), Part.owner(lo, T, G), own_scale(lo, T, G, k, kpos, tpos))
        : {Part.max_contrib.go(rest, 1n+g, KT, Tk, Gk,
            Part.max_contrib.pick(Nat.is_gt(Nat.add(Nat.sub(Part.owner(hi, T, G), _), 1n), mc),
              Nat.add(Nat.sub(Part.owner(hi, T, G), _), 1n), mc)) ==
          Part.max_contrib.go(rest, 1n+g, KT, T, G, Part.max_contrib.pick(Nat.is_gt(n, mc), n, mc)) : Nat}
      mc_go_scale(rest, 1n+g, KT, T, G, k, Part.max_contrib.pick(Nat.is_gt(n, mc), n, mc), kpos, tpos)

# max_contrib(Gr, KT, T k, G k) == max_contrib(Gr, KT, T, G).
def mc_scale(+Gr: Nat, +KT: Nat, +T: Nat, +G: Nat, +k: Nat, +kpos: {Nat.is_lt(0n, k) == True{} : Bool},
    +tpos: {Nat.is_lt(0n, T) == True{} : Bool})
  -> {Part.max_contrib(Gr, KT, Nat.mul(T, k), Nat.mul(G, k)) == Part.max_contrib(Gr, KT, T, G) : Nat}:
  mc_go_scale(Gr, 0n, KT, T, G, k, 1n, kpos, tpos)

# T0 = 3840 = 960 * 4 at the served A, H.
def total0_served(+A: Nat, +H: Nat, sa: {A == Impl.kin() : Nat}, sh: {H == Impl.hidden() : Nat})
  -> {Impl.total0(A, H) == Nat.mul(960n, 4n) : Nat}:
  %Equal.sym(Nat, A, Impl.kin(), sa) : {Impl.total0(_, H) == Nat.mul(960n, 4n) : Nat}
  %Equal.sym(Nat, H, Impl.hidden(), sh) : {Impl.total0(Impl.kin(), _) == Nat.mul(960n, 4n) : Nat}
  {==}

# owner(x, T, G) == q - 1 from the division identity (x + 1) G + T - 1 == q T + R with R < T, so
# the checker never runs the unary division.
def own_at(+x: Nat, +T: Nat, +G: Nat, +q: Nat, +R: Nat,
    e: {Nat.sub(Nat.add(Nat.mul(1n+x, G), T), 1n) == Nat.add(Nat.mul(q, T), R) : Nat},
    rlt: {Nat.is_lt(R, T) == True{} : Bool})
  -> {Part.owner(x, T, G) == Nat.sub(q, 1n) : Nat}:
  %Equal.sym(Nat, Nat.sub(Nat.add(Nat.mul(1n+x, G), T), 1n), Nat.add(Nat.mul(q, T), R), e)
    : {Nat.sub(Nat.div(_, T), 1n) == Nat.sub(q, 1n) : Nat}
  %Equal.sym(Nat, Nat.div(Nat.add(Nat.mul(q, T), R), T), q,
      PP.ard_fst({Nat.div(Nat.add(Nat.mul(q, T), R), T) == q : Nat},
        {Nat.mod(Nat.add(Nat.mul(q, T), R), T) == R : Nat}, PP.ar_div_unique(q, R, T, rlt)))
    : {Nat.sub(_, 1n) == Nat.sub(q, 1n) : Nat}
  {==}

# One max_contrib.go step from the group's owner values h (last k-tile) and l (first k-tile).
def mc_step(+rest: Nat, +g: Nat, +KT: Nat, +T: Nat, +G: Nat, +mc: Nat, +h: Nat, +l: Nat, +mc2: Nat,
    eh: {Part.owner(Nat.sub(Nat.mul(1n+g, KT), 1n), T, G) == h : Nat},
    el: {Part.owner(Nat.mul(g, KT), T, G) == l : Nat},
    ep: {Part.max_contrib.pick(Nat.is_gt(Nat.add(Nat.sub(h, l), 1n), mc), Nat.add(Nat.sub(h, l), 1n), mc) ==
      mc2 : Nat})
  -> {Part.max_contrib.go(1n+rest, g, KT, T, G, mc) == Part.max_contrib.go(rest, 1n+g, KT, T, G, mc2) : Nat}:
  +OH = Part.owner(Nat.sub(Nat.mul(1n+g, KT), 1n), T, G)
  +OL = Part.owner(Nat.mul(g, KT), T, G)
  +P = Part.max_contrib.pick(Nat.is_gt(Nat.add(Nat.sub(h, l), 1n), mc), Nat.add(Nat.sub(h, l), 1n), mc)
  %Equal.sym(Nat, mc2, P, Equal.sym(Nat, P, mc2, ep))
    : {Part.max_contrib.go(1n+rest, g, KT, T, G, mc) == Part.max_contrib.go(rest, 1n+g, KT, T, G, _) : Nat}
  %Equal.sym(Nat, h, OH, Equal.sym(Nat, OH, h, eh))
    : {Part.max_contrib.go(1n+rest, g, KT, T, G, mc) == Part.max_contrib.go(rest, 1n+g, KT, T, G,
        Part.max_contrib.pick(Nat.is_gt(Nat.add(Nat.sub(_, l), 1n), mc), Nat.add(Nat.sub(_, l), 1n), mc)) : Nat}
  %Equal.sym(Nat, l, OL, Equal.sym(Nat, OL, l, el))
    : {Part.max_contrib.go(1n+rest, g, KT, T, G, mc) == Part.max_contrib.go(rest, 1n+g, KT, T, G,
        Part.max_contrib.pick(Nat.is_gt(Nat.add(Nat.sub(OH, _), 1n), mc), Nat.add(Nat.sub(OH, _), 1n), mc)) : Nat}
  {==}

# owner of the last / first k-tile of group g at KT = 384, T = 960, G = 41.
def own_hi(+g: Nat, +q: Nat, +R: Nat,
    e: {Nat.sub(Nat.add(Nat.mul(1n+Nat.sub(Nat.mul(1n+g, 384n), 1n), 41n), 960n), 1n) ==
      Nat.add(Nat.mul(q, 960n), R) : Nat},
    rlt: {Nat.is_lt(R, 960n) == True{} : Bool})
  -> {Part.owner(Nat.sub(Nat.mul(1n+g, 384n), 1n), 960n, 41n) == Nat.sub(q, 1n) : Nat}:
  own_at(Nat.sub(Nat.mul(1n+g, 384n), 1n), 960n, 41n, q, R, e, rlt)

def own_lo(+g: Nat, +q: Nat, +R: Nat,
    e: {Nat.sub(Nat.add(Nat.mul(1n+Nat.mul(g, 384n), 41n), 960n), 1n) == Nat.add(Nat.mul(q, 960n), R) : Nat},
    rlt: {Nat.is_lt(R, 960n) == True{} : Bool})
  -> {Part.owner(Nat.mul(g, 384n), 960n, 41n) == Nat.sub(q, 1n) : Nat}:
  own_at(Nat.mul(g, 384n), 960n, 41n, q, R, e, rlt)

# The ten groups at KT = 384, T = 960, G = 41: per-group contributor counts 17, 17, 18, 17, 17,
# 17, 17, 18, 17, 17; one def per group so each step gets its own checker budget.
def mc_s0() -> {Part.max_contrib.go(10n, 0n, 384n, 960n, 41n, 1n) ==
    Part.max_contrib.go(9n, 1n, 384n, 960n, 41n, 17n) : Nat}:
  mc_step(9n, 0n, 384n, 960n, 41n, 1n, Nat.sub(17n, 1n), Nat.sub(1n, 1n), 17n,
    own_hi(0n, 17n, 383n, {==}, {==}), own_lo(0n, 1n, 40n, {==}, {==}), {==})

def mc_s1() -> {Part.max_contrib.go(9n, 1n, 384n, 960n, 41n, 17n) ==
    Part.max_contrib.go(8n, 2n, 384n, 960n, 41n, 17n) : Nat}:
  mc_step(8n, 1n, 384n, 960n, 41n, 17n, Nat.sub(33n, 1n), Nat.sub(17n, 1n), 17n,
    own_hi(1n, 33n, 767n, {==}, {==}), own_lo(1n, 17n, 424n, {==}, {==}), {==})

def mc_s2() -> {Part.max_contrib.go(8n, 2n, 384n, 960n, 41n, 17n) ==
    Part.max_contrib.go(7n, 3n, 384n, 960n, 41n, 18n) : Nat}:
  mc_step(7n, 2n, 384n, 960n, 41n, 17n, Nat.sub(50n, 1n), Nat.sub(33n, 1n), 18n,
    own_hi(2n, 50n, 191n, {==}, {==}), own_lo(2n, 33n, 808n, {==}, {==}), {==})

def mc_s3() -> {Part.max_contrib.go(7n, 3n, 384n, 960n, 41n, 18n) ==
    Part.max_contrib.go(6n, 4n, 384n, 960n, 41n, 18n) : Nat}:
  mc_step(6n, 3n, 384n, 960n, 41n, 18n, Nat.sub(66n, 1n), Nat.sub(50n, 1n), 18n,
    own_hi(3n, 66n, 575n, {==}, {==}), own_lo(3n, 50n, 232n, {==}, {==}), {==})

def mc_s4() -> {Part.max_contrib.go(6n, 4n, 384n, 960n, 41n, 18n) ==
    Part.max_contrib.go(5n, 5n, 384n, 960n, 41n, 18n) : Nat}:
  mc_step(5n, 4n, 384n, 960n, 41n, 18n, Nat.sub(82n, 1n), Nat.sub(66n, 1n), 18n,
    own_hi(4n, 82n, 959n, {==}, {==}), own_lo(4n, 66n, 616n, {==}, {==}), {==})

def mc_s5() -> {Part.max_contrib.go(5n, 5n, 384n, 960n, 41n, 18n) ==
    Part.max_contrib.go(4n, 6n, 384n, 960n, 41n, 18n) : Nat}:
  mc_step(4n, 5n, 384n, 960n, 41n, 18n, Nat.sub(99n, 1n), Nat.sub(83n, 1n), 18n,
    own_hi(5n, 99n, 383n, {==}, {==}), own_lo(5n, 83n, 40n, {==}, {==}), {==})

def mc_s6() -> {Part.max_contrib.go(4n, 6n, 384n, 960n, 41n, 18n) ==
    Part.max_contrib.go(3n, 7n, 384n, 960n, 41n, 18n) : Nat}:
  mc_step(3n, 6n, 384n, 960n, 41n, 18n, Nat.sub(115n, 1n), Nat.sub(99n, 1n), 18n,
    own_hi(6n, 115n, 767n, {==}, {==}), own_lo(6n, 99n, 424n, {==}, {==}), {==})

def mc_s7() -> {Part.max_contrib.go(3n, 7n, 384n, 960n, 41n, 18n) ==
    Part.max_contrib.go(2n, 8n, 384n, 960n, 41n, 18n) : Nat}:
  mc_step(2n, 7n, 384n, 960n, 41n, 18n, Nat.sub(132n, 1n), Nat.sub(115n, 1n), 18n,
    own_hi(7n, 132n, 191n, {==}, {==}), own_lo(7n, 115n, 808n, {==}, {==}), {==})

def mc_s8() -> {Part.max_contrib.go(2n, 8n, 384n, 960n, 41n, 18n) ==
    Part.max_contrib.go(1n, 9n, 384n, 960n, 41n, 18n) : Nat}:
  mc_step(1n, 8n, 384n, 960n, 41n, 18n, Nat.sub(148n, 1n), Nat.sub(132n, 1n), 18n,
    own_hi(8n, 148n, 575n, {==}, {==}), own_lo(8n, 132n, 232n, {==}, {==}), {==})

def mc_s9() -> {Part.max_contrib.go(1n, 9n, 384n, 960n, 41n, 18n) ==
    Part.max_contrib.go(0n, 10n, 384n, 960n, 41n, 18n) : Nat}:
  mc_step(0n, 9n, 384n, 960n, 41n, 18n, Nat.sub(164n, 1n), Nat.sub(148n, 1n), 18n,
    own_hi(9n, 164n, 959n, {==}, {==}), own_lo(9n, 148n, 616n, {==}, {==}), {==})

def mc_served_go() -> {Part.max_contrib.go(10n, 0n, 384n, 960n, 41n, 1n) == 18n : Nat}:
  %Equal.sym(Nat, Part.max_contrib.go(10n, 0n, 384n, 960n, 41n, 1n),
      Part.max_contrib.go(9n, 1n, 384n, 960n, 41n, 17n), mc_s0())
    : {_ == 18n : Nat}
  %Equal.sym(Nat, Part.max_contrib.go(9n, 1n, 384n, 960n, 41n, 17n),
      Part.max_contrib.go(8n, 2n, 384n, 960n, 41n, 17n), mc_s1())
    : {_ == 18n : Nat}
  %Equal.sym(Nat, Part.max_contrib.go(8n, 2n, 384n, 960n, 41n, 17n),
      Part.max_contrib.go(7n, 3n, 384n, 960n, 41n, 18n), mc_s2())
    : {_ == 18n : Nat}
  %Equal.sym(Nat, Part.max_contrib.go(7n, 3n, 384n, 960n, 41n, 18n),
      Part.max_contrib.go(6n, 4n, 384n, 960n, 41n, 18n), mc_s3())
    : {_ == 18n : Nat}
  %Equal.sym(Nat, Part.max_contrib.go(6n, 4n, 384n, 960n, 41n, 18n),
      Part.max_contrib.go(5n, 5n, 384n, 960n, 41n, 18n), mc_s4())
    : {_ == 18n : Nat}
  %Equal.sym(Nat, Part.max_contrib.go(5n, 5n, 384n, 960n, 41n, 18n),
      Part.max_contrib.go(4n, 6n, 384n, 960n, 41n, 18n), mc_s5())
    : {_ == 18n : Nat}
  %Equal.sym(Nat, Part.max_contrib.go(4n, 6n, 384n, 960n, 41n, 18n),
      Part.max_contrib.go(3n, 7n, 384n, 960n, 41n, 18n), mc_s6())
    : {_ == 18n : Nat}
  %Equal.sym(Nat, Part.max_contrib.go(3n, 7n, 384n, 960n, 41n, 18n),
      Part.max_contrib.go(2n, 8n, 384n, 960n, 41n, 18n), mc_s7())
    : {_ == 18n : Nat}
  %Equal.sym(Nat, Part.max_contrib.go(2n, 8n, 384n, 960n, 41n, 18n),
      Part.max_contrib.go(1n, 9n, 384n, 960n, 41n, 18n), mc_s8())
    : {_ == 18n : Nat}
  %Equal.sym(Nat, Part.max_contrib.go(1n, 9n, 384n, 960n, 41n, 18n),
      Part.max_contrib.go(0n, 10n, 384n, 960n, 41n, 18n), mc_s9())
    : {_ == 18n : Nat}
  {==}

# max_contrib's definition, generic so that the checker never evaluates a closed instance.
def mc_unfold(+Gr: Nat, +KT: Nat, +T: Nat, +G: Nat)
  -> {Part.max_contrib(Gr, KT, T, G) == Part.max_contrib.go(Gr, 0n, KT, T, G, 1n) : Nat}:
  {==}

def mc_served_full() -> {Part.max_contrib(Part.groups(1n, Impl.hidden()), Impl.kt0(Impl.kin()), 960n, 41n) ==
    18n : Nat}:
  %Equal.sym(Nat, Impl.kt0(Impl.kin()), 384n, {==})
    : {Part.max_contrib(Part.groups(1n, Impl.hidden()), _, 960n, 41n) == 18n : Nat}
  %Equal.sym(Nat, Part.groups(1n, Impl.hidden()), 10n, {==})
    : {Part.max_contrib(_, 384n, 960n, 41n) == 18n : Nat}
  %Equal.sym(Nat, Part.max_contrib(10n, 384n, 960n, 41n), Part.max_contrib.go(10n, 0n, 384n, 960n, 41n, 1n),
      mc_unfold(10n, 384n, 960n, 41n))
    : {_ == 18n : Nat}
  mc_served_go()

# At G = 164 = 41 * 4, A = 6144, H = 5120: MC0 = max_contrib over NG0 groups of KT0 tiles with
# T0 = 3840 = 960 * 4, evaluated at the scaled-down total 960 and grid 41. The served constants
# are substituted into NG0 / KT0 only after the scaling, and the owners are computed through
# division identities (mc_served_go), so the checker never normalizes an unscaled max_contrib.
def Laws.mc0_served(G, A, H, served_g, served_a, served_h):
  +sa = served_a
  +sh = served_h
  %Equal.sym(Nat, G, Nat.mul(41n, 4n), Equal.trans(Nat, G, Impl.grid(), Nat.mul(41n, 4n), served_g, {==}))
    : {Impl.mc0(_, A, H) == 18n : Nat}
  %Equal.sym(Nat, Impl.total0(A, H), Nat.mul(960n, 4n), total0_served(A, H, sa, sh))
    : {Part.max_contrib(Part.groups(1n, H), Impl.kt0(A), _, Nat.mul(41n, 4n)) == 18n : Nat}
  %Equal.sym(Nat, Part.max_contrib(Part.groups(1n, H), Impl.kt0(A), Nat.mul(960n, 4n), Nat.mul(41n, 4n)),
      Part.max_contrib(Part.groups(1n, H), Impl.kt0(A), 960n, 41n),
      mc_scale(Part.groups(1n, H), Impl.kt0(A), 960n, 41n, 4n, {==}, {==}))
    : {_ == 18n : Nat}
  %Equal.sym(Nat, A, Impl.kin(), sa)
    : {Part.max_contrib(Part.groups(1n, H), Impl.kt0(_), 960n, 41n) == 18n : Nat}
  %Equal.sym(Nat, H, Impl.hidden(), sh)
    : {Part.max_contrib(Part.groups(1n, _), Impl.kt0(Impl.kin()), 960n, 41n) == 18n : Nat}
  mc_served_full()

# ==== (4) sum-of-squares partials and norm cells ====

# NCH = 4 (H div 512) > 0 once H >= 512.
def nch_pos(+H: Nat, shaped: {Nat.is_le(512n, H) == True{} : Bool})
  -> {Nat.is_lt(0n, Impl.nch(H)) == True{} : Bool}:
  +q = Nat.div(H, 512n)
  +q1 = PP.ard_rt(Nat.is_le(Nat.mul(1n, 512n), H), Nat.is_le(1n, q), PP.ar_div_le(1n, H, 512n, {==}),
    shaped)
  PP.ard_tr(Nat.is_lt(Nat.mul(0n, 4n), Nat.mul(q, 4n)), Nat.is_lt(0n, q),
    PP.ar_mul_lt_cancel(0n, q, 4n, {==}),
    PP.ard_tr(Nat.is_lt(0n, q), Nat.is_le(1n, q), PP.ar_lt_succ_le(0n, q), q1))

# F0 job x < J0 decodes in range: group < NG0, row < m (chunk < 4 by x mod 4).
def f0_ok(+H: Nat, +m: Nat, +x: Nat, +xin: {Nat.is_lt(x, Impl.jobs0(H, m)) == True{} : Bool})
  -> {PSpec.job_ok(Impl.f0_job(x, m), Impl.ng0(H), m) == True{} : Bool}:
  PP.ard_fst({PSpec.job_ok(Impl.f0_job(x, m), Impl.ng0(H), m) == True{} : Bool},
    {PSpec.encode(Impl.f0_job(x, m), m) == x : Nat}, Laws.f0_decode(H, m, x, xin))

def f0_g_lt(+H: Nat, +m: Nat, +x: Nat, +xin: {Nat.is_lt(x, Impl.jobs0(H, m)) == True{} : Bool})
  -> {Nat.is_lt(Nat.div(x, Nat.mul(m, 4n)), Impl.ng0(H)) == True{} : Bool}:
  +r = Nat.div(Nat.mod(x, Nat.mul(m, 4n)), 4n)
  PP.ar_and_l(Nat.is_lt(Nat.div(x, Nat.mul(m, 4n)), Impl.ng0(H)),
    Bool.and(Nat.is_lt(r, m), Nat.is_lt(Nat.mod(x, 4n), 4n)), f0_ok(H, m, x, xin))

def f0_r_lt(+H: Nat, +m: Nat, +x: Nat, +xin: {Nat.is_lt(x, Impl.jobs0(H, m)) == True{} : Bool})
  -> {Nat.is_lt(Nat.div(Nat.mod(x, Nat.mul(m, 4n)), 4n), m) == True{} : Bool}:
  +r = Nat.div(Nat.mod(x, Nat.mul(m, 4n)), 4n)
  PP.ar_and_l(Nat.is_lt(r, m), Nat.is_lt(Nat.mod(x, 4n), 4n),
    PP.ar_and_r(Nat.is_lt(Nat.div(x, Nat.mul(m, 4n)), Impl.ng0(H)),
      Bool.and(Nat.is_lt(r, m), Nat.is_lt(Nat.mod(x, 4n), 4n)), f0_ok(H, m, x, xin)))

# The in-row chunk 4 g + ch of F0 job x < J0 is below NCH.
def f0_c_lt(+H: Nat, +m: Nat, +x: Nat, +xin: {Nat.is_lt(x, Impl.jobs0(H, m)) == True{} : Bool})
  -> {Nat.is_lt(Nat.add(Nat.mul(Nat.div(x, Nat.mul(m, 4n)), 4n), Nat.mod(x, 4n)), Impl.nch(H)) ==
    True{} : Bool}:
  PP.p7_mr_lt(Nat.div(x, Nat.mul(m, 4n)), Nat.mod(x, 4n), Impl.ng0(H), 4n, f0_g_lt(H, m, x, xin),
    PP.ard_dm_lt(x, 4n, {==}))

# The partial F0 job x < J0 writes is below m * NCH.
def f0_p_lt(+H: Nat, +m: Nat, +x: Nat, +xin: {Nat.is_lt(x, Impl.jobs0(H, m)) == True{} : Bool})
  -> {Nat.is_lt(Impl.f0_psum(H, Impl.f0_job(x, m)), Impl.jobsn(H, m)) == True{} : Bool}:
  PP.p7_mr_lt(Nat.div(Nat.mod(x, Nat.mul(m, 4n)), 4n),
    Nat.add(Nat.mul(Nat.div(x, Nat.mul(m, 4n)), 4n), Nat.mod(x, 4n)), m, Impl.nch(H),
    f0_r_lt(H, m, x, xin), f0_c_lt(H, m, x, xin))

# The (group, row, chunk) whose F0 job writes partial i: row i div NCH, chunk i mod NCH of the row.
def psum_job(+H: Nat, +i: Nat) -> PSpec.Job:
  PSpec.Job{Nat.div(Nat.mod(i, Impl.nch(H)), 4n), Nat.div(i, Impl.nch(H)),
    Nat.mod(Nat.mod(i, Impl.nch(H)), 4n)}

# psum_job inverts f0_psum on the F0 jobs x < J0.
def psum_inv(+H: Nat, +m: Nat, +x: Nat, +xin: {Nat.is_lt(x, Impl.jobs0(H, m)) == True{} : Bool})
  -> {psum_job(H, Impl.f0_psum(H, Impl.f0_job(x, m))) == Impl.f0_job(x, m) : PSpec.Job}:
  +N = Impl.nch(H)
  +g = Nat.div(x, Nat.mul(m, 4n))
  +r = Nat.div(Nat.mod(x, Nat.mul(m, 4n)), 4n)
  +ch = Nat.mod(x, 4n)
  +c = Nat.add(Nat.mul(g, 4n), ch)
  +P = Nat.add(Nat.mul(r, N), c)
  %Equal.sym(Nat, Nat.mod(P, N), c,
      PP.ard_snd({Nat.div(P, N) == r : Nat}, {Nat.mod(P, N) == c : Nat},
        PP.ar_div_unique(r, c, N, f0_c_lt(H, m, x, xin))))
    : {PSpec.Job{Nat.div(_, 4n), Nat.div(P, N), Nat.mod(_, 4n)} == PSpec.Job{g, r, ch} : PSpec.Job}
  %Equal.sym(Nat, Nat.div(P, N), r,
      PP.ard_fst({Nat.div(P, N) == r : Nat}, {Nat.mod(P, N) == c : Nat},
        PP.ar_div_unique(r, c, N, f0_c_lt(H, m, x, xin))))
    : {PSpec.Job{Nat.div(c, 4n), _, Nat.mod(c, 4n)} == PSpec.Job{g, r, ch} : PSpec.Job}
  %Equal.sym(Nat, Nat.div(c, 4n), g,
      PP.ard_fst({Nat.div(c, 4n) == g : Nat}, {Nat.mod(c, 4n) == ch : Nat},
        PP.ar_div_unique(g, ch, 4n, PP.ard_dm_lt(x, 4n, {==}))))
    : {PSpec.Job{_, r, Nat.mod(c, 4n)} == PSpec.Job{g, r, ch} : PSpec.Job}
  %Equal.sym(Nat, Nat.mod(c, 4n), ch,
      PP.ard_snd({Nat.div(c, 4n) == g : Nat}, {Nat.mod(c, 4n) == ch : Nat},
        PP.ar_div_unique(g, ch, 4n, PP.ard_dm_lt(x, 4n, {==}))))
    : {PSpec.Job{g, r, _} == PSpec.Job{g, r, ch} : PSpec.Job}
  {==}

# psum_job(i) is a real job for i < m * NCH.
def psum_job_ok(+H: Nat, +m: Nat, +i: Nat, +npos: {Nat.is_lt(0n, Impl.nch(H)) == True{} : Bool},
    iin: {Nat.is_lt(i, Impl.jobsn(H, m)) == True{} : Bool})
  -> {PSpec.job_ok(psum_job(H, i), Impl.ng0(H), m) == True{} : Bool}:
  +N = Impl.nch(H)
  +a = Nat.mod(i, N)
  PP.ar_and_intro(Nat.is_lt(Nat.div(a, 4n), Impl.ng0(H)),
    Bool.and(Nat.is_lt(Nat.div(i, N), m), Nat.is_lt(Nat.mod(a, 4n), 4n)),
    PP.ard_tr(Nat.is_lt(Nat.div(a, 4n), Impl.ng0(H)), Nat.is_lt(a, Nat.mul(Impl.ng0(H), 4n)),
      PP.ar_div_lt(Impl.ng0(H), a, 4n, {==}), PP.ard_dm_lt(i, N, npos)),
    PP.ar_and_intro(Nat.is_lt(Nat.div(i, N), m), Nat.is_lt(Nat.mod(a, 4n), 4n),
      PP.ard_tr(Nat.is_lt(Nat.div(i, N), m), Nat.is_lt(i, Nat.mul(m, N)), PP.ar_div_lt(m, i, N, npos),
        iin),
      PP.ard_dm_lt(a, 4n, {==})))

# F0 job x < J0 writes partial i only if x is psum_job(i)'s number ...
def pw_fwd(+H: Nat, +m: Nat, +i: Nat, +x: Nat,
    +xin: {Nat.is_lt(x, Impl.jobs0(H, m)) == True{} : Bool},
    h: {Nat.is_eq(Impl.f0_psum(H, Impl.f0_job(x, m)), i) == True{} : Bool})
  -> {Nat.is_eq(x, PSpec.encode(psum_job(H, i), m)) == True{} : Bool}:
  +P = Impl.f0_psum(H, Impl.f0_job(x, m))
  %PP.ar_eq_true(P, i, h) : {Nat.is_eq(x, PSpec.encode(psum_job(H, _), m)) == True{} : Bool}
  %Equal.sym(PSpec.Job, psum_job(H, P), Impl.f0_job(x, m), psum_inv(H, m, x, xin))
    : {Nat.is_eq(x, PSpec.encode(_, m)) == True{} : Bool}
  %Equal.sym(Nat, PSpec.encode(Impl.f0_job(x, m), m), x,
      PP.ard_snd({PSpec.job_ok(Impl.f0_job(x, m), Impl.ng0(H), m) == True{} : Bool},
        {PSpec.encode(Impl.f0_job(x, m), m) == x : Nat}, Laws.f0_decode(H, m, x, xin)))
    : {Nat.is_eq(x, _) == True{} : Bool}
  PP.ar_eq_refl(x)

# ... and that job does write partial i < m * NCH.
def pw_bwd(+H: Nat, +m: Nat, +i: Nat, +x: Nat, +npos: {Nat.is_lt(0n, Impl.nch(H)) == True{} : Bool},
    +iin: {Nat.is_lt(i, Impl.jobsn(H, m)) == True{} : Bool},
    h: {Nat.is_eq(x, PSpec.encode(psum_job(H, i), m)) == True{} : Bool})
  -> {Nat.is_eq(Impl.f0_psum(H, Impl.f0_job(x, m)), i) == True{} : Bool}:
  +N = Impl.nch(H)
  +a = Nat.mod(i, N)
  +y = PSpec.encode(psum_job(H, i), m)
  %Equal.sym(Nat, x, y, PP.ar_eq_true(x, y, h))
    : {Nat.is_eq(Impl.f0_psum(H, Impl.f0_job(_, m)), i) == True{} : Bool}
  %Equal.sym(PSpec.Job, Impl.f0_job(y, m), psum_job(H, i),
      PP.ard_fst({Impl.f0_job(y, m) == psum_job(H, i) : PSpec.Job},
        {Nat.is_lt(y, Impl.jobs0(H, m)) == True{} : Bool},
        Laws.f0_encode(H, m, psum_job(H, i), psum_job_ok(H, m, i, npos, iin))))
    : {Nat.is_eq(Impl.f0_psum(H, _), i) == True{} : Bool}
  %Equal.sym(Nat, Nat.add(Nat.mul(Nat.div(a, 4n), 4n), Nat.mod(a, 4n)), a, PP.ard_dm_eq(a, 4n, {==}))
    : {Nat.is_eq(Nat.add(Nat.mul(Nat.div(i, N), N), _), i) == True{} : Bool}
  PP.p8_eq_of(Nat.add(Nat.mul(Nat.div(i, N), N), a), i, PP.ard_dm_eq(i, N, npos))

# For x < J0 and i < m * NCH: F0 job x writes partial i iff x is psum_job(i)'s number.
def pw_pt(+H: Nat, +m: Nat, +i: Nat, +x: Nat, +npos: {Nat.is_lt(0n, Impl.nch(H)) == True{} : Bool},
    +xin: {Nat.is_lt(x, Impl.jobs0(H, m)) == True{} : Bool},
    +iin: {Nat.is_lt(i, Impl.jobsn(H, m)) == True{} : Bool})
  -> {Nat.is_eq(Impl.f0_psum(H, Impl.f0_job(x, m)), i) ==
    Nat.is_eq(x, PSpec.encode(psum_job(H, i), m)) : Bool}:
  PP.ar_bool_ext(Nat.is_eq(Impl.f0_psum(H, Impl.f0_job(x, m)), i),
    Nat.is_eq(x, PSpec.encode(psum_job(H, i), m)),
    h => pw_fwd(H, m, i, x, xin, h),
    h => pw_bwd(H, m, i, x, npos, iin, h))

# For x < J0 and i >= m * NCH: F0 job x does not write partial i.
def psum_ne(+H: Nat, +m: Nat, +i: Nat, +x: Nat, +xin: {Nat.is_lt(x, Impl.jobs0(H, m)) == True{} : Bool},
    iout: {Nat.is_lt(i, Impl.jobsn(H, m)) == False{} : Bool})
  -> {Nat.is_eq(Impl.f0_psum(H, Impl.f0_job(x, m)), i) == False{} : Bool}:
  +P = Impl.f0_psum(H, Impl.f0_job(x, m))
  PP.p6_lt_ne(P, i, PP.ar_lt_le_trans(P, Impl.jobsn(H, m), i, f0_p_lt(H, m, x, xin),
    PP.p6_nlt_le(i, Impl.jobsn(H, m), iout)))

# Writers of partial i < m * NCH among F0 jobs x < n <= J0: one iff psum_job(i)'s number is < n.
def pw_in(+n: Nat, +H: Nat, +m: Nat, +i: Nat, +npos: {Nat.is_lt(0n, Impl.nch(H)) == True{} : Bool},
    +nle: {Nat.is_le(n, Impl.jobs0(H, m)) == True{} : Bool},
    +iin: {Nat.is_lt(i, Impl.jobsn(H, m)) == True{} : Bool})
  -> {Laws.psum_writers(H, m, n, i) == PSpec.bit(Nat.is_lt(PSpec.encode(psum_job(H, i), m), n)) : Nat}:
  match n:
    case 0n:
      +y = PSpec.encode(psum_job(H, i), m)
      %Equal.sym(Bool, Nat.is_lt(y, 0n), False{}, PP.ar_h_lt_zero(y)) : {0n == PSpec.bit(_) : Nat}
      {==}
    case 1n+x:
      +y = PSpec.encode(psum_job(H, i), m)
      +J0 = Impl.jobs0(H, m)
      %Equal.sym(Nat, Laws.psum_writers(H, m, x, i), PSpec.bit(Nat.is_lt(y, x)),
          pw_in(x, H, m, i, npos, le_of_succ_le(x, J0, nle), iin))
        : {Nat.add(_, PSpec.bit(Nat.is_eq(Impl.f0_psum(H, Impl.f0_job(x, m)), i))) ==
          PSpec.bit(Nat.is_lt(y, 1n+x)) : Nat}
      %Equal.sym(Bool, Nat.is_eq(Impl.f0_psum(H, Impl.f0_job(x, m)), i), Nat.is_eq(x, y),
          pw_pt(H, m, i, x, npos, lt_of_succ_le(x, J0, nle), iin))
        : {Nat.add(PSpec.bit(Nat.is_lt(y, x)), PSpec.bit(_)) == PSpec.bit(Nat.is_lt(y, 1n+x)) : Nat}
      MP.bit_step(y, x)

# No F0 job x < n <= J0 writes a partial i >= m * NCH.
def pw_out(+n: Nat, +H: Nat, +m: Nat, +i: Nat, +nle: {Nat.is_le(n, Impl.jobs0(H, m)) == True{} : Bool},
    +iout: {Nat.is_lt(i, Impl.jobsn(H, m)) == False{} : Bool})
  -> {Laws.psum_writers(H, m, n, i) == 0n : Nat}:
  match n:
    case 0n:
      {==}
    case 1n+x:
      +J0 = Impl.jobs0(H, m)
      %Equal.sym(Nat, Laws.psum_writers(H, m, x, i), 0n,
          pw_out(x, H, m, i, le_of_succ_le(x, J0, nle), iout))
        : {Nat.add(_, PSpec.bit(Nat.is_eq(Impl.f0_psum(H, Impl.f0_job(x, m)), i))) == 0n : Nat}
      %Equal.sym(Bool, Nat.is_eq(Impl.f0_psum(H, Impl.f0_job(x, m)), i), False{},
          psum_ne(H, m, i, x, lt_of_succ_le(x, J0, nle), iout))
        : {Nat.add(0n, PSpec.bit(_)) == 0n : Nat}
      {==}

def pw_split(v: Bool, +H: Nat, +m: Nat, +i: Nat, +npos: {Nat.is_lt(0n, Impl.nch(H)) == True{} : Bool},
    +ev: {Nat.is_lt(i, Impl.jobsn(H, m)) == v : Bool})
  -> {Laws.psum_writers(H, m, Impl.jobs0(H, m), i) == PSpec.bit(v) : Nat}:
  match v:
    case True{}:
      +y = PSpec.encode(psum_job(H, i), m)
      +J0 = Impl.jobs0(H, m)
      %Equal.sym(Nat, Laws.psum_writers(H, m, J0, i), PSpec.bit(Nat.is_lt(y, J0)),
          pw_in(J0, H, m, i, npos, PP.ar_le_refl(J0), ev))
        : {_ == 1n : Nat}
      %Equal.sym(Bool, Nat.is_lt(y, J0), True{},
          PP.ard_snd({Impl.f0_job(y, m) == psum_job(H, i) : PSpec.Job}, {Nat.is_lt(y, J0) == True{} : Bool},
            Laws.f0_encode(H, m, psum_job(H, i), psum_job_ok(H, m, i, npos, ev))))
        : {PSpec.bit(_) == 1n : Nat}
      {==}
    case False{}:
      pw_out(Impl.jobs0(H, m), H, m, i, PP.ar_le_refl(Impl.jobs0(H, m)), ev)

def Laws.f0_partials_once(H, m, i, shaped):
  pw_split(Nat.is_lt(i, Impl.jobsn(H, m)), H, m, i, nch_pos(H, shaped), {==})

# a && !a is False.
def and_not_self(a: Bool) -> {Bool.and(a, Bool.not(a)) == False{} : Bool}:
  match a:
    case True{}:
      {==}
    case False{}:
      {==}

# One step of rng_mem: s <> go(k, s + 1).
def rng_step(e: Bool, +k: Nat, +s: Nat, +i: Nat, +ev: {Nat.is_eq(s, i) == e : Bool},
    ih: {MSpec.member(Impl.norm_reads.go(k, 1n+s), i) ==
      Bool.and(Nat.is_le(1n+s, i), Nat.is_lt(i, Nat.add(k, 1n+s))) : Bool})
  -> {Bool.or(Nat.is_eq(s, i), MSpec.member(Impl.norm_reads.go(k, 1n+s), i)) ==
    Bool.and(Nat.is_le(s, i), Nat.is_lt(i, 1n+Nat.add(k, s))) : Bool}:
  match e:
    case True{}:
      %Equal.sym(Bool, Nat.is_eq(s, i), True{}, ev)
        : {Bool.or(_, MSpec.member(Impl.norm_reads.go(k, 1n+s), i)) ==
          Bool.and(Nat.is_le(s, i), Nat.is_lt(i, 1n+Nat.add(k, s))) : Bool}
      %PP.ar_eq_true(s, i, ev)
        : {True{} == Bool.and(Nat.is_le(s, _), Nat.is_lt(_, 1n+Nat.add(k, s))) : Bool}
      %Equal.sym(Bool, Nat.is_le(s, s), True{}, PP.ar_le_refl(s))
        : {True{} == Bool.and(_, Nat.is_lt(s, 1n+Nat.add(k, s))) : Bool}
      Equal.sym(Bool, Nat.is_lt(s, 1n+Nat.add(k, s)), True{}, PP.p5_lt_succ_add(k, s))
    case False{}:
      %Equal.sym(Bool, Nat.is_eq(s, i), False{}, ev)
        : {Bool.or(_, MSpec.member(Impl.norm_reads.go(k, 1n+s), i)) ==
          Bool.and(Nat.is_le(s, i), Nat.is_lt(i, 1n+Nat.add(k, s))) : Bool}
      %Equal.sym(Bool, MSpec.member(Impl.norm_reads.go(k, 1n+s), i),
          Bool.and(Nat.is_le(1n+s, i), Nat.is_lt(i, Nat.add(k, 1n+s))), ih)
        : {_ == Bool.and(Nat.is_le(s, i), Nat.is_lt(i, 1n+Nat.add(k, s))) : Bool}
      %Equal.sym(Nat, Nat.add(k, 1n+s), 1n+Nat.add(k, s), PP.ar_add_succ_r(k, s))
        : {Bool.and(Nat.is_le(1n+s, i), Nat.is_lt(i, _)) ==
          Bool.and(Nat.is_le(s, i), Nat.is_lt(i, 1n+Nat.add(k, s))) : Bool}
      %Equal.sym(Bool, Nat.is_le(s, i), Bool.or(Nat.is_eq(s, i), Nat.is_le(1n+s, i)), PP.p8_le_split(s, i))
        : {Bool.and(Nat.is_le(1n+s, i), Nat.is_lt(i, 1n+Nat.add(k, s))) ==
          Bool.and(_, Nat.is_lt(i, 1n+Nat.add(k, s))) : Bool}
      %Equal.sym(Bool, Nat.is_eq(s, i), False{}, ev)
        : {Bool.and(Nat.is_le(1n+s, i), Nat.is_lt(i, 1n+Nat.add(k, s))) ==
          Bool.and(Bool.or(_, Nat.is_le(1n+s, i)), Nat.is_lt(i, 1n+Nat.add(k, s))) : Bool}
      {==}

# norm_reads.go(left, s) holds exactly the indices s .. s + left - 1.
def rng_mem(+left: Nat, +s: Nat, +i: Nat)
  -> {MSpec.member(Impl.norm_reads.go(left, s), i) ==
    Bool.and(Nat.is_le(s, i), Nat.is_lt(i, Nat.add(left, s))) : Bool}:
  match left:
    case 0n:
      %Equal.sym(Bool, Nat.is_lt(i, s), Bool.not(Nat.is_le(s, i)), PP.ar_lt_not_le(i, s))
        : {False{} == Bool.and(Nat.is_le(s, i), _) : Bool}
      Equal.sym(Bool, Bool.and(Nat.is_le(s, i), Bool.not(Nat.is_le(s, i))), False{},
        and_not_self(Nat.is_le(s, i)))
    case 1n+k:
      rng_step(Nat.is_eq(s, i), k, s, i, {==}, rng_mem(k, 1n+s, i))

# Norm job of row r reads partial i iff i div NCH == r.
def in_row(+H: Nat, +r: Nat, +i: Nat, +npos: {Nat.is_lt(0n, Impl.nch(H)) == True{} : Bool})
  -> {MSpec.member(Impl.norm_reads(H, r), i) == Nat.is_eq(r, Nat.div(i, Impl.nch(H))) : Bool}:
  +N = Impl.nch(H)
  +d = Nat.div(i, N)
  +lo = Nat.mul(r, N)
  %Equal.sym(Bool, MSpec.member(Impl.norm_reads(H, r), i),
      Bool.and(Nat.is_le(lo, i), Nat.is_lt(i, Nat.add(N, lo))), rng_mem(N, lo, i))
    : {_ == Nat.is_eq(r, d) : Bool}
  %Equal.sym(Bool, Nat.is_eq(r, d), Bool.and(Nat.is_le(r, d), Nat.is_lt(d, 1n+r)), PP.p8_eq_iv(r, d))
    : {Bool.and(Nat.is_le(lo, i), Nat.is_lt(i, Nat.add(N, lo))) == _ : Bool}
  %Equal.sym(Bool, Nat.is_le(r, d), Nat.is_le(lo, i),
      Equal.sym(Bool, Nat.is_le(lo, i), Nat.is_le(r, d), PP.ar_div_le(r, i, N, npos)))
    : {Bool.and(Nat.is_le(lo, i), Nat.is_lt(i, Nat.add(N, lo))) == Bool.and(_, Nat.is_lt(d, 1n+r)) : Bool}
  %Equal.sym(Bool, Nat.is_lt(d, 1n+r), Nat.is_lt(i, Nat.mul(1n+r, N)), PP.ar_div_lt(1n+r, i, N, npos))
    : {Bool.and(Nat.is_le(lo, i), Nat.is_lt(i, Nat.add(N, lo))) == Bool.and(Nat.is_le(lo, i), _) : Bool}
  {==}

# F0 job x < J0 writing partial i has row i div NCH.
def row_of_psum(+H: Nat, +m: Nat, +i: Nat, +x: Nat,
    +xin: {Nat.is_lt(x, Impl.jobs0(H, m)) == True{} : Bool},
    h: {Nat.is_eq(Impl.f0_psum(H, Impl.f0_job(x, m)), i) == True{} : Bool})
  -> {Nat.div(Nat.mod(x, Nat.mul(m, 4n)), 4n) == Nat.div(i, Impl.nch(H)) : Nat}:
  +N = Impl.nch(H)
  +rx = Nat.div(Nat.mod(x, Nat.mul(m, 4n)), 4n)
  +c = Nat.add(Nat.mul(Nat.div(x, Nat.mul(m, 4n)), 4n), Nat.mod(x, 4n))
  +P = Nat.add(Nat.mul(rx, N), c)
  Equal.trans(Nat, rx, Nat.div(P, N), Nat.div(i, N),
    Equal.sym(Nat, Nat.div(P, N), rx,
      PP.ard_fst({Nat.div(P, N) == rx : Nat}, {Nat.mod(P, N) == c : Nat},
        PP.ar_div_unique(rx, c, N, f0_c_lt(H, m, x, xin)))),
    Equal.cong(Nat, Nat, z => Nat.div(z, N), P, i, PP.ar_eq_true(P, i, h)))

# In row r (r == i div NCH), a row-r F0 job writing i is just a job writing i.
def rw_pt_in(+H: Nat, +m: Nat, +r: Nat, +i: Nat, +x: Nat,
    +npos: {Nat.is_lt(0n, Impl.nch(H)) == True{} : Bool},
    +xin: {Nat.is_lt(x, Impl.jobs0(H, m)) == True{} : Bool},
    +iin: {Nat.is_lt(i, Impl.jobsn(H, m)) == True{} : Bool},
    +w: {Nat.is_eq(r, Nat.div(i, Impl.nch(H))) == True{} : Bool})
  -> {Bool.and(Nat.is_eq(Spec.job_row(Impl.f0_job(x, m)), r),
      Nat.is_eq(Impl.f0_psum(H, Impl.f0_job(x, m)), i)) ==
    Nat.is_eq(x, PSpec.encode(psum_job(H, i), m)) : Bool}:
  +rx = Nat.div(Nat.mod(x, Nat.mul(m, 4n)), 4n)
  +E = Nat.is_eq(Impl.f0_psum(H, Impl.f0_job(x, m)), i)
  +d = Nat.div(i, Impl.nch(H))
  %Equal.sym(Bool, Bool.and(Nat.is_eq(rx, r), E), E,
      PP.p6_and_imp(Nat.is_eq(rx, r), E,
        h => PP.p8_eq_of(rx, r, Equal.trans(Nat, rx, d, r, row_of_psum(H, m, i, x, xin, h),
          Equal.sym(Nat, r, d, PP.ar_eq_true(r, d, w))))))
    : {_ == Nat.is_eq(x, PSpec.encode(psum_job(H, i), m)) : Bool}
  pw_pt(H, m, i, x, npos, xin, iin)

# Outside row r (r != i div NCH) no row-r F0 job writes i.
def rw_clash(+H: Nat, +m: Nat, +r: Nat, +i: Nat, +x: Nat,
    +xin: {Nat.is_lt(x, Impl.jobs0(H, m)) == True{} : Bool},
    w: {Nat.is_eq(r, Nat.div(i, Impl.nch(H))) == False{} : Bool},
    +h: {Bool.and(Nat.is_eq(Spec.job_row(Impl.f0_job(x, m)), r),
      Nat.is_eq(Impl.f0_psum(H, Impl.f0_job(x, m)), i)) == True{} : Bool})
  -> Empty:
  +rx = Nat.div(Nat.mod(x, Nat.mul(m, 4n)), 4n)
  +E = Nat.is_eq(Impl.f0_psum(H, Impl.f0_job(x, m)), i)
  +d = Nat.div(i, Impl.nch(H))
  PP.ar_false_ne(Nat.is_eq(r, d), w,
    PP.p8_eq_of(r, d, Equal.trans(Nat, r, rx, d,
      Equal.sym(Nat, rx, r, PP.ar_eq_true(rx, r, PP.ar_and_l(Nat.is_eq(rx, r), E, h))),
      row_of_psum(H, m, i, x, xin, PP.ar_and_r(Nat.is_eq(rx, r), E, h)))))

def rw_pt_out(+H: Nat, +m: Nat, +r: Nat, +i: Nat, +x: Nat,
    +xin: {Nat.is_lt(x, Impl.jobs0(H, m)) == True{} : Bool},
    +w: {Nat.is_eq(r, Nat.div(i, Impl.nch(H))) == False{} : Bool})
  -> {Bool.and(Nat.is_eq(Spec.job_row(Impl.f0_job(x, m)), r),
      Nat.is_eq(Impl.f0_psum(H, Impl.f0_job(x, m)), i)) == False{} : Bool}:
  PP.ar_not_true(Bool.and(Nat.is_eq(Spec.job_row(Impl.f0_job(x, m)), r),
      Nat.is_eq(Impl.f0_psum(H, Impl.f0_job(x, m)), i)),
    h => rw_clash(H, m, r, i, x, xin, w, h))

# Row-r writers of partial i (r == i div NCH) among F0 jobs x < n <= J0.
def rw_in(+n: Nat, +H: Nat, +m: Nat, +r: Nat, +i: Nat,
    +npos: {Nat.is_lt(0n, Impl.nch(H)) == True{} : Bool},
    +nle: {Nat.is_le(n, Impl.jobs0(H, m)) == True{} : Bool},
    +iin: {Nat.is_lt(i, Impl.jobsn(H, m)) == True{} : Bool},
    +w: {Nat.is_eq(r, Nat.div(i, Impl.nch(H))) == True{} : Bool})
  -> {Laws.row_writers(H, m, n, r, i) ==
    PSpec.bit(Nat.is_lt(PSpec.encode(psum_job(H, i), m), n)) : Nat}:
  match n:
    case 0n:
      +y = PSpec.encode(psum_job(H, i), m)
      %Equal.sym(Bool, Nat.is_lt(y, 0n), False{}, PP.ar_h_lt_zero(y)) : {0n == PSpec.bit(_) : Nat}
      {==}
    case 1n+x:
      +y = PSpec.encode(psum_job(H, i), m)
      +J0 = Impl.jobs0(H, m)
      +RW = Bool.and(Nat.is_eq(Spec.job_row(Impl.f0_job(x, m)), r),
        Nat.is_eq(Impl.f0_psum(H, Impl.f0_job(x, m)), i))
      %Equal.sym(Nat, Laws.row_writers(H, m, x, r, i), PSpec.bit(Nat.is_lt(y, x)),
          rw_in(x, H, m, r, i, npos, le_of_succ_le(x, J0, nle), iin, w))
        : {Nat.add(_, PSpec.bit(RW)) == PSpec.bit(Nat.is_lt(y, 1n+x)) : Nat}
      %Equal.sym(Bool, RW, Nat.is_eq(x, y),
          rw_pt_in(H, m, r, i, x, npos, lt_of_succ_le(x, J0, nle), iin, w))
        : {Nat.add(PSpec.bit(Nat.is_lt(y, x)), PSpec.bit(_)) == PSpec.bit(Nat.is_lt(y, 1n+x)) : Nat}
      MP.bit_step(y, x)

def rw_out(+n: Nat, +H: Nat, +m: Nat, +r: Nat, +i: Nat,
    +nle: {Nat.is_le(n, Impl.jobs0(H, m)) == True{} : Bool},
    +w: {Nat.is_eq(r, Nat.div(i, Impl.nch(H))) == False{} : Bool})
  -> {Laws.row_writers(H, m, n, r, i) == 0n : Nat}:
  match n:
    case 0n:
      {==}
    case 1n+x:
      +J0 = Impl.jobs0(H, m)
      +RW = Bool.and(Nat.is_eq(Spec.job_row(Impl.f0_job(x, m)), r),
        Nat.is_eq(Impl.f0_psum(H, Impl.f0_job(x, m)), i))
      %Equal.sym(Nat, Laws.row_writers(H, m, x, r, i), 0n,
          rw_out(x, H, m, r, i, le_of_succ_le(x, J0, nle), w))
        : {Nat.add(_, PSpec.bit(RW)) == 0n : Nat}
      %Equal.sym(Bool, RW, False{}, rw_pt_out(H, m, r, i, x, lt_of_succ_le(x, J0, nle), w))
        : {Nat.add(0n, PSpec.bit(_)) == 0n : Nat}
      {==}

# r < m and r == i div NCH give i < m * NCH.
def row_iin(+H: Nat, +m: Nat, +r: Nat, +i: Nat, +npos: {Nat.is_lt(0n, Impl.nch(H)) == True{} : Bool},
    rlt: {Nat.is_lt(r, m) == True{} : Bool},
    w: {Nat.is_eq(r, Nat.div(i, Impl.nch(H))) == True{} : Bool})
  -> {Nat.is_lt(i, Impl.jobsn(H, m)) == True{} : Bool}:
  +N = Impl.nch(H)
  +d = Nat.div(i, N)
  PP.ard_rt(Nat.is_lt(d, m), Nat.is_lt(i, Nat.mul(m, N)), PP.ar_div_lt(m, i, N, npos),
    PP.p5_lt_rw(r, d, m, PP.ar_eq_true(r, d, w), rlt))

def rw_split(v: Bool, +H: Nat, +m: Nat, +r: Nat, +i: Nat,
    +npos: {Nat.is_lt(0n, Impl.nch(H)) == True{} : Bool}, +rlt: {Nat.is_lt(r, m) == True{} : Bool},
    +ev: {Nat.is_eq(r, Nat.div(i, Impl.nch(H))) == v : Bool})
  -> {Laws.row_writers(H, m, Impl.jobs0(H, m), r, i) == PSpec.bit(v) : Nat}:
  match v:
    case True{}:
      +y = PSpec.encode(psum_job(H, i), m)
      +J0 = Impl.jobs0(H, m)
      +iin = row_iin(H, m, r, i, npos, rlt, ev)
      %Equal.sym(Nat, Laws.row_writers(H, m, J0, r, i), PSpec.bit(Nat.is_lt(y, J0)),
          rw_in(J0, H, m, r, i, npos, PP.ar_le_refl(J0), iin, ev))
        : {_ == 1n : Nat}
      %Equal.sym(Bool, Nat.is_lt(y, J0), True{},
          PP.ard_snd({Impl.f0_job(y, m) == psum_job(H, i) : PSpec.Job}, {Nat.is_lt(y, J0) == True{} : Bool},
            Laws.f0_encode(H, m, psum_job(H, i), psum_job_ok(H, m, i, npos, iin))))
        : {PSpec.bit(_) == 1n : Nat}
      {==}
    case False{}:
      rw_out(Impl.jobs0(H, m), H, m, r, i, PP.ar_le_refl(Impl.jobs0(H, m)), ev)

def Laws.norm_reads_partials(H, m, r, i, shaped, row):
  +npos = nch_pos(H, shaped)
  %Equal.sym(Bool, MSpec.member(Impl.norm_reads(H, r), i), Nat.is_eq(r, Nat.div(i, Impl.nch(H))),
      in_row(H, r, i, npos))
    : {Laws.row_writers(H, m, Impl.jobs0(H, m), r, i) == PSpec.bit(_) : Nat}
  rw_split(Nat.is_eq(r, Nat.div(i, Impl.nch(H))), H, m, r, i, npos, row, {==})

# is_eq is symmetric.
def eq_comm(+a: Nat, +b: Nat) -> {Nat.is_eq(a, b) == Nat.is_eq(b, a) : Bool}:
  PP.ar_bool_ext(Nat.is_eq(a, b), Nat.is_eq(b, a),
    h => PP.p8_eq_of(b, a, Equal.sym(Nat, a, b, PP.ar_eq_true(a, b, h))),
    h => PP.p8_eq_of(a, b, Equal.sym(Nat, b, a, PP.ar_eq_true(b, a, h))))

# For c < NCH, norm job x handles (row r, chunk c) iff x == r * NCH + c.
def nc_pt_in(+H: Nat, +r: Nat, +c: Nat, +x: Nat, +npos: {Nat.is_lt(0n, Impl.nch(H)) == True{} : Bool},
    +clt: {Nat.is_lt(c, Impl.nch(H)) == True{} : Bool})
  -> {Bool.and(Nat.is_eq(Impl.norm_row(H, x), r), Nat.is_eq(Impl.norm_col(H, x), c)) ==
    Nat.is_eq(x, Nat.add(Nat.mul(r, Impl.nch(H)), c)) : Bool}:
  +N = Impl.nch(H)
  +x0 = Nat.add(Nat.mul(r, N), c)
  +q = Nat.div(x, N)
  +u = Nat.mod(x, N)
  %Equal.sym(Bool, Nat.is_eq(q, r), Nat.is_eq(r, q), eq_comm(q, r))
    : {Bool.and(_, Nat.is_eq(u, c)) == Nat.is_eq(x, x0) : Bool}
  %Equal.sym(Bool, Nat.is_eq(u, c), Nat.is_eq(c, u), eq_comm(u, c))
    : {Bool.and(Nat.is_eq(r, q), _) == Nat.is_eq(x, x0) : Bool}
  %Equal.sym(Bool, Nat.is_eq(x, x0), Nat.is_eq(x0, x), eq_comm(x, x0))
    : {Bool.and(Nat.is_eq(r, q), Nat.is_eq(c, u)) == _ : Bool}
  Equal.sym(Bool, Nat.is_eq(x0, x), Bool.and(Nat.is_eq(r, q), Nat.is_eq(c, u)),
    PP.p8_eqA(x0, x, r, q, c, u, N, {==}, PP.ard_dm_eq(x, N, npos), clt, PP.ard_dm_lt(x, N, npos)))

# For c >= NCH no norm job handles chunk c.
def nc_pt_out(+H: Nat, +r: Nat, +c: Nat, +x: Nat, +npos: {Nat.is_lt(0n, Impl.nch(H)) == True{} : Bool},
    cout: {Nat.is_lt(c, Impl.nch(H)) == False{} : Bool})
  -> {Bool.and(Nat.is_eq(Impl.norm_row(H, x), r), Nat.is_eq(Impl.norm_col(H, x), c)) == False{} : Bool}:
  +N = Impl.nch(H)
  +u = Nat.mod(x, N)
  +A = Nat.is_eq(Nat.div(x, N), r)
  %Equal.sym(Bool, Nat.is_eq(u, c), False{},
      PP.p6_lt_ne(u, c, PP.ar_lt_le_trans(u, N, c, PP.ard_dm_lt(x, N, npos), PP.p6_nlt_le(c, N, cout))))
    : {Bool.and(A, _) == False{} : Bool}
  PP.p6_and_false(A)

# Norm jobs x < n handling (r, c) for c < NCH: one iff r * NCH + c < n.
def nc_in(+n: Nat, +H: Nat, +r: Nat, +c: Nat, +npos: {Nat.is_lt(0n, Impl.nch(H)) == True{} : Bool},
    +clt: {Nat.is_lt(c, Impl.nch(H)) == True{} : Bool})
  -> {Laws.norm_cells(H, n, r, c) == PSpec.bit(Nat.is_lt(Nat.add(Nat.mul(r, Impl.nch(H)), c), n)) : Nat}:
  match n:
    case 0n:
      +x0 = Nat.add(Nat.mul(r, Impl.nch(H)), c)
      %Equal.sym(Bool, Nat.is_lt(x0, 0n), False{}, PP.ar_h_lt_zero(x0)) : {0n == PSpec.bit(_) : Nat}
      {==}
    case 1n+x:
      +x0 = Nat.add(Nat.mul(r, Impl.nch(H)), c)
      +NC = Bool.and(Nat.is_eq(Impl.norm_row(H, x), r), Nat.is_eq(Impl.norm_col(H, x), c))
      %Equal.sym(Nat, Laws.norm_cells(H, x, r, c), PSpec.bit(Nat.is_lt(x0, x)),
          nc_in(x, H, r, c, npos, clt))
        : {Nat.add(_, PSpec.bit(NC)) == PSpec.bit(Nat.is_lt(x0, 1n+x)) : Nat}
      %Equal.sym(Bool, NC, Nat.is_eq(x, x0), nc_pt_in(H, r, c, x, npos, clt))
        : {Nat.add(PSpec.bit(Nat.is_lt(x0, x)), PSpec.bit(_)) == PSpec.bit(Nat.is_lt(x0, 1n+x)) : Nat}
      MP.bit_step(x0, x)

def nc_out(+n: Nat, +H: Nat, +r: Nat, +c: Nat, +npos: {Nat.is_lt(0n, Impl.nch(H)) == True{} : Bool},
    +cout: {Nat.is_lt(c, Impl.nch(H)) == False{} : Bool})
  -> {Laws.norm_cells(H, n, r, c) == 0n : Nat}:
  match n:
    case 0n:
      {==}
    case 1n+x:
      +NC = Bool.and(Nat.is_eq(Impl.norm_row(H, x), r), Nat.is_eq(Impl.norm_col(H, x), c))
      %Equal.sym(Nat, Laws.norm_cells(H, x, r, c), 0n, nc_out(x, H, r, c, npos, cout))
        : {Nat.add(_, PSpec.bit(NC)) == 0n : Nat}
      %Equal.sym(Bool, NC, False{}, nc_pt_out(H, r, c, x, npos, cout))
        : {Nat.add(0n, PSpec.bit(_)) == 0n : Nat}
      {==}

def nc_split(v: Bool, +H: Nat, +m: Nat, +r: Nat, +c: Nat,
    +npos: {Nat.is_lt(0n, Impl.nch(H)) == True{} : Bool},
    +ev: {Nat.is_lt(c, Impl.nch(H)) == v : Bool})
  -> {Laws.norm_cells(H, Impl.jobsn(H, m), r, c) == PSpec.bit(Bool.and(Nat.is_lt(r, m), v)) : Nat}:
  match v:
    case True{}:
      +N = Impl.nch(H)
      +x0 = Nat.add(Nat.mul(r, N), c)
      +JN = Impl.jobsn(H, m)
      %Equal.sym(Nat, Laws.norm_cells(H, JN, r, c), PSpec.bit(Nat.is_lt(x0, JN)),
          nc_in(JN, H, r, c, npos, ev))
        : {_ == PSpec.bit(Bool.and(Nat.is_lt(r, m), True{})) : Nat}
      %Equal.sym(Bool, Nat.is_lt(x0, JN), Nat.is_lt(Nat.div(x0, N), m),
          Equal.sym(Bool, Nat.is_lt(Nat.div(x0, N), m), Nat.is_lt(x0, Nat.mul(m, N)),
            PP.ar_div_lt(m, x0, N, npos)))
        : {PSpec.bit(_) == PSpec.bit(Bool.and(Nat.is_lt(r, m), True{})) : Nat}
      %Equal.sym(Nat, Nat.div(x0, N), r,
          PP.ard_fst({Nat.div(x0, N) == r : Nat}, {Nat.mod(x0, N) == c : Nat},
            PP.ar_div_unique(r, c, N, ev)))
        : {PSpec.bit(Nat.is_lt(_, m)) == PSpec.bit(Bool.and(Nat.is_lt(r, m), True{})) : Nat}
      %Equal.sym(Bool, Bool.and(Nat.is_lt(r, m), True{}), Nat.is_lt(r, m), PP.p6_and_true(Nat.is_lt(r, m)))
        : {PSpec.bit(Nat.is_lt(r, m)) == PSpec.bit(_) : Nat}
      {==}
    case False{}:
      %Equal.sym(Bool, Bool.and(Nat.is_lt(r, m), False{}), False{}, PP.p6_and_false(Nat.is_lt(r, m)))
        : {Laws.norm_cells(H, Impl.jobsn(H, m), r, c) == PSpec.bit(_) : Nat}
      nc_out(Impl.jobsn(H, m), H, r, c, npos, ev)

def Laws.norm_cells_once(H, m, r, c, shaped):
  nc_split(Nat.is_lt(c, Impl.nch(H)), H, m, r, c, nch_pos(H, shaped), {==})
