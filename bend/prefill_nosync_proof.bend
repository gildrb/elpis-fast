# Proofs of prefill_nosync_laws.bend (ext 9502 deferred checkpoint copies) about the model prefill_nosync.bend.
# Structural evidence only: no unsafe definitions, axioms, holes or foreign code. Every def precedes its uses.
#
# Idea: the observer state of the spec after any prefix of an iterate boundary is a function of the impl state:
# ob(st) = (round, forwards, no upload in flight, outstanding copies = the pending list, stash ids, no blocking copy,
# all checks true). Each impl step (begin_round, a reader, the prefill chunk with its put, recurrent_checkpoint, a put,
# a decode) maps ob(st) to ob(st') under the invariant "pending = entries of the previous round, then entries of the
# current round" (pv), and every iterate ends with all pending entries of the current round (cur).
import Base
import ./prefill_nosync.bend as Impl
import ./prefill_nosync_spec.bend as Spec
import ./prefill_nosync_laws.bend as Laws
import ./gemm_m16_partition_proof.bend as Ar

# ---------------------------------------------------------------------------------------------
# Abstraction and invariants

def O(+rnd: Nat, +nf: Nat, out: List<&2, Spec.Out>, +ns: Nat, fq: Bool, ks: Bool) -> Spec.Obs:
  Spec.Ob{rnd, nf, [], out, ns, 0n, fq, ks, False{}, True{}, True{}, True{}, True{}, True{}}

# the same inside a reader's settle
def ORd(+rnd: Nat, +nf: Nat, out: List<&2, Spec.Out>, +ns: Nat, fq: Bool, ks: Bool) -> Spec.Obs:
  Spec.Ob{rnd, nf, [], out, ns, 0n, fq, ks, True{}, True{}, True{}, True{}, True{}, True{}}

def outs(pend: List<&2, Impl.Pe>) -> List<&2, Spec.Out>:
  match pend:
    case Nil{}:
      []
    case Impl.Pe{r, s} <> t:
      Spec.Oc{s, r} <> outs(t)

def ob(st: Impl.St, fq: Bool, ks: Bool) -> Spec.Obs:
  match st:
    case Impl.St{rnd, pend, nxt, ver}:
      O(rnd, ver, outs(pend), nxt, fq, ks)

# every pending entry is of round rnd
def cur(+rnd: Nat, pend: List<&2, Impl.Pe>) -> Bool:
  match pend:
    case Nil{}:
      True{}
    case Impl.Pe{+r, s} <> t:
      Bool.and(Nat.is_eq(r, rnd), cur(rnd, t))

def pv.pick(k: Bool, a: Bool, b: Bool, c: Bool, d: Bool) -> Bool:
  match k:
    case True{}:
      Bool.and(a, b)
    case False{}:
      Bool.and(c, d)

# pending = entries of round rnd - 1, then entries of round rnd
def pv(+rnd: Nat, pend: List<&2, Impl.Pe>) -> Bool:
  match pend:
    case Nil{}:
      True{}
    case Impl.Pe{+r, s} <> +t:
      pv.pick(Nat.is_le(rnd, r), Nat.is_eq(r, rnd), cur(rnd, t), Nat.is_eq(1n+r, rnd), pv(rnd, t))

def cur_st(st: Impl.St) -> Bool:
  match st:
    case Impl.St{rnd, pend, nxt, ver}:
      cur(rnd, pend)

def pv_st(st: Impl.St) -> Bool:
  match st:
    case Impl.St{rnd, pend, nxt, ver}:
      pv(rnd, pend)

# ---------------------------------------------------------------------------------------------
# Generic lemmas

def pn_run_app(a: List<&2, Spec.Ev>, +b: List<&2, Spec.Ev>, +o: Spec.Obs)
  -> {Spec.run(List.append(&2, Spec.Ev, a, b), o) == Spec.run(b, Spec.run(a, o)) : Spec.Obs}:
  match a:
    case Nil{}:
      {==}
    case +e <> rest:
      pn_run_app(rest, b, Spec.step(o, e))

def pn_seq(+a: List<&2, Spec.Ev>, +b: List<&2, Spec.Ev>, +o: Spec.Obs, -o1: Spec.Obs, -o2: Spec.Obs,
    h1: {Spec.run(a, o) == o1 : Spec.Obs}, h2: {Spec.run(b, o1) == o2 : Spec.Obs})
  -> {Spec.run(List.append(&2, Spec.Ev, a, b), o) == o2 : Spec.Obs}:
  Equal.trans(Spec.Obs, Spec.run(List.append(&2, Spec.Ev, a, b), o), Spec.run(b, Spec.run(a, o)), o2,
    pn_run_app(a, b, o),
    Equal.trans(Spec.Obs, Spec.run(b, Spec.run(a, o)), Spec.run(b, o1), o2,
      Equal.cong(Spec.Obs, Spec.Obs, z => Spec.run(b, z), Spec.run(a, o), o1, h1), h2))

# r == rnd (as is_eq) gives rnd <= r
def pn_le_of_eq(+r: Nat, +rnd: Nat, he: {Nat.is_eq(r, rnd) == True{} : Bool}) -> {Nat.is_le(rnd, r) == True{} : Bool}:
  %Equal.sym(Nat, r, rnd, Ar.ar_eq_true(r, rnd, he)) : {Nat.is_le(rnd, _) == True{} : Bool}
  Ar.ar_le_refl(rnd)

# 1 + a <= a is false
def pn_le_succ(+a: Nat) -> {Nat.is_le(1n+a, a) == False{} : Bool}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      pn_le_succ(p)

# ---------------------------------------------------------------------------------------------
# Lists of pending entries

def pn_cur_ev(t: List<&2, Impl.Pe>, +rnd: Nat, +hc: {cur(rnd, t) == True{} : Bool})
  -> {Impl.settle_ev(t, True{}, rnd) == [] : List<&2, Spec.Ev>}:
  match t:
    case Nil{}:
      {==}
    case Impl.Pe{+r, +s} <> +t2:
      +le = pn_le_of_eq(r, rnd, Ar.ar_and_l(Nat.is_eq(r, rnd), cur(rnd, t2), hc))
      %Equal.sym(Bool, Nat.is_le(rnd, r), True{}, le) : {Impl.settle_ev.pick(_, s, Impl.settle_ev(t2, True{}, rnd)) == [] : List<&2, Spec.Ev>}
      pn_cur_ev(t2, rnd, Ar.ar_and_r(Nat.is_eq(r, rnd), cur(rnd, t2), hc))

def pn_cur_keep(t: List<&2, Impl.Pe>, +rnd: Nat, +hc: {cur(rnd, t) == True{} : Bool})
  -> {Impl.settle_keep(t, True{}, rnd) == t : List<&2, Impl.Pe>}:
  match t:
    case Nil{}:
      {==}
    case Impl.Pe{+r, +s} <> +t2:
      +le = pn_le_of_eq(r, rnd, Ar.ar_and_l(Nat.is_eq(r, rnd), cur(rnd, t2), hc))
      %Equal.sym(Bool, Nat.is_le(rnd, r), True{}, le) : {Impl.settle_keep.pick(_, Impl.Pe{r, s}, Impl.settle_keep(t2, True{}, rnd)) == Impl.Pe{r, s} <> t2 : List<&2, Impl.Pe>}
      Equal.cong(List<&2, Impl.Pe>, List<&2, Impl.Pe>, z => Impl.Pe{r, s} <> z, Impl.settle_keep(t2, True{}, rnd), t2,
        pn_cur_keep(t2, rnd, Ar.ar_and_r(Nat.is_eq(r, rnd), cur(rnd, t2), hc)))

def pn_cur_pv(+rnd: Nat, t: List<&2, Impl.Pe>, +hc: {cur(rnd, t) == True{} : Bool}) -> {pv(rnd, t) == True{} : Bool}:
  match t:
    case Nil{}:
      {==}
    case Impl.Pe{+r, +s} <> +t2:
      +le = pn_le_of_eq(r, rnd, Ar.ar_and_l(Nat.is_eq(r, rnd), cur(rnd, t2), hc))
      %Equal.sym(Bool, Nat.is_le(rnd, r), True{}, le) : {pv.pick(_, Nat.is_eq(r, rnd), cur(rnd, t2), Nat.is_eq(1n+r, rnd), pv(rnd, t2)) == True{} : Bool}
      hc

def pn_pv_tail(k: Bool, +r: Nat, +rnd: Nat, +t: List<&2, Impl.Pe>,
    +hp: {pv.pick(k, Nat.is_eq(r, rnd), cur(rnd, t), Nat.is_eq(1n+r, rnd), pv(rnd, t)) == True{} : Bool})
  -> {pv(rnd, t) == True{} : Bool}:
  match k:
    case True{}:
      pn_cur_pv(rnd, t, Ar.ar_and_r(Nat.is_eq(r, rnd), cur(rnd, t), hp))
    case False{}:
      Ar.ar_and_r(Nat.is_eq(1n+r, rnd), pv(rnd, t), hp)

# ---------------------------------------------------------------------------------------------
# Waits

# the wait of the oldest outstanding copy, in the next round, after a forward or inside recurrent_checkpoint
def pn_wt_head(+s: Nat, +r: Nat, +m: List<&2, Spec.Out>, +rnd: Nat, +nf: Nat, +ns: Nat, +fq: Bool, +ks: Bool,
    hf: {Bool.or(ks, fq) == True{} : Bool}, h1: {Nat.is_eq(1n+r, rnd) == True{} : Bool})
  -> {Spec.step(O(rnd, nf, Spec.Oc{s, r} <> m, ns, fq, ks), Spec.Wt{s}) == O(rnd, nf, m, ns, fq, ks) : Spec.Obs}:
  %Equal.sym(Bool, Nat.is_eq(s, s), True{}, Ar.ar_eq_refl(s)) :
    {Spec.Ob{rnd, nf, [], Spec.rm.pick(_, Spec.Oc{s, r}, m, Spec.rm(m, s)), ns, 0n, fq, ks, False{}, True{}, True{},
       Spec.w_ok.pick(_, r, rnd, fq, ks, False{}, Spec.w_ok(m, s, rnd, fq, ks, False{})), True{},
       Bool.or(_, Spec.has(m, s))} == O(rnd, nf, m, ns, fq, ks) : Spec.Obs}
  %Equal.sym(Bool, Bool.or(ks, fq), True{}, hf) :
    {Spec.Ob{rnd, nf, [], m, ns, 0n, fq, ks, False{}, True{}, True{}, Bool.and(_, Nat.is_eq(1n+r, rnd)), True{}, True{}}
      == O(rnd, nf, m, ns, fq, ks) : Spec.Obs}
  %Equal.sym(Bool, Nat.is_eq(1n+r, rnd), True{}, h1) :
    {Spec.Ob{rnd, nf, [], m, ns, 0n, fq, ks, False{}, True{}, True{}, _, True{}, True{}} == O(rnd, nf, m, ns, fq, ks) : Spec.Obs}
  {==}

# the wait of the oldest outstanding copy inside a reader's settle
def pn_wt_rd(+s: Nat, +r: Nat, +m: List<&2, Spec.Out>, +rnd: Nat, +nf: Nat, +ns: Nat, +fq: Bool, +ks: Bool)
  -> {Spec.step(ORd(rnd, nf, Spec.Oc{s, r} <> m, ns, fq, ks), Spec.Wt{s}) == ORd(rnd, nf, m, ns, fq, ks) : Spec.Obs}:
  %Equal.sym(Bool, Nat.is_eq(s, s), True{}, Ar.ar_eq_refl(s)) :
    {Spec.Ob{rnd, nf, [], Spec.rm.pick(_, Spec.Oc{s, r}, m, Spec.rm(m, s)), ns, 0n, fq, ks, True{}, True{}, True{},
       Spec.w_ok.pick(_, r, rnd, fq, ks, True{}, Spec.w_ok(m, s, rnd, fq, ks, True{})), True{},
       Bool.or(_, Spec.has(m, s))} == ORd(rnd, nf, m, ns, fq, ks) : Spec.Obs}
  {==}

# settle(round) over pv: the previous round's entries are waited (in order), the current round's are kept
def pn_settle_obs.go(k: Bool, +r: Nat, +s: Nat, +t: List<&2, Impl.Pe>, +rnd: Nat, +nf: Nat, +ns: Nat, +fq: Bool, +ks: Bool,
    +hp: {pv.pick(k, Nat.is_eq(r, rnd), cur(rnd, t), Nat.is_eq(1n+r, rnd), pv(rnd, t)) == True{} : Bool},
    +hf: {Bool.or(ks, fq) == True{} : Bool},
    ih: {Spec.run(Impl.settle_ev(t, True{}, rnd), O(rnd, nf, outs(t), ns, fq, ks)) ==
         O(rnd, nf, outs(Impl.settle_keep(t, True{}, rnd)), ns, fq, ks) : Spec.Obs})
  -> {Spec.run(Impl.settle_ev.pick(k, s, Impl.settle_ev(t, True{}, rnd)), O(rnd, nf, Spec.Oc{s, r} <> outs(t), ns, fq, ks)) ==
      O(rnd, nf, outs(Impl.settle_keep.pick(k, Impl.Pe{r, s}, Impl.settle_keep(t, True{}, rnd))), ns, fq, ks) : Spec.Obs}:
  match k:
    case True{}:
      +hc = Ar.ar_and_r(Nat.is_eq(r, rnd), cur(rnd, t), hp)
      %Equal.sym(List<&2, Spec.Ev>, Impl.settle_ev(t, True{}, rnd), [], pn_cur_ev(t, rnd, hc)) :
        {Spec.run(_, O(rnd, nf, Spec.Oc{s, r} <> outs(t), ns, fq, ks)) ==
         O(rnd, nf, Spec.Oc{s, r} <> outs(Impl.settle_keep(t, True{}, rnd)), ns, fq, ks) : Spec.Obs}
      %Equal.sym(List<&2, Impl.Pe>, Impl.settle_keep(t, True{}, rnd), t, pn_cur_keep(t, rnd, hc)) :
        {O(rnd, nf, Spec.Oc{s, r} <> outs(t), ns, fq, ks) == O(rnd, nf, Spec.Oc{s, r} <> outs(_), ns, fq, ks) : Spec.Obs}
      {==}
    case False{}:
      +h1 = Ar.ar_and_l(Nat.is_eq(1n+r, rnd), pv(rnd, t), hp)
      Equal.trans(Spec.Obs, Spec.run(Impl.settle_ev(t, True{}, rnd), Spec.step(O(rnd, nf, Spec.Oc{s, r} <> outs(t), ns, fq, ks), Spec.Wt{s})),
        Spec.run(Impl.settle_ev(t, True{}, rnd), O(rnd, nf, outs(t), ns, fq, ks)),
        O(rnd, nf, outs(Impl.settle_keep(t, True{}, rnd)), ns, fq, ks),
        Equal.cong(Spec.Obs, Spec.Obs, z => Spec.run(Impl.settle_ev(t, True{}, rnd), z),
          Spec.step(O(rnd, nf, Spec.Oc{s, r} <> outs(t), ns, fq, ks), Spec.Wt{s}), O(rnd, nf, outs(t), ns, fq, ks),
          pn_wt_head(s, r, outs(t), rnd, nf, ns, fq, ks, hf, h1)),
        ih)

def pn_settle_obs(pend: List<&2, Impl.Pe>, +rnd: Nat, +nf: Nat, +ns: Nat, +fq: Bool, +ks: Bool,
    +hp: {pv(rnd, pend) == True{} : Bool}, +hf: {Bool.or(ks, fq) == True{} : Bool})
  -> {Spec.run(Impl.settle_ev(pend, True{}, rnd), O(rnd, nf, outs(pend), ns, fq, ks)) ==
      O(rnd, nf, outs(Impl.settle_keep(pend, True{}, rnd)), ns, fq, ks) : Spec.Obs}:
  match pend:
    case Nil{}:
      {==}
    case Impl.Pe{+r, +s} <> +t:
      pn_settle_obs.go(Nat.is_le(rnd, r), r, s, t, rnd, nf, ns, fq, ks, hp, hf,
        pn_settle_obs(t, rnd, nf, ns, fq, ks, pn_pv_tail(Nat.is_le(rnd, r), r, rnd, t, hp), hf))

# settle(round) over pv keeps only current-round entries
def pn_settle_cur.go(k: Bool, +r: Nat, +s: Nat, +t: List<&2, Impl.Pe>, +rnd: Nat,
    +hp: {pv.pick(k, Nat.is_eq(r, rnd), cur(rnd, t), Nat.is_eq(1n+r, rnd), pv(rnd, t)) == True{} : Bool},
    ih: {cur(rnd, Impl.settle_keep(t, True{}, rnd)) == True{} : Bool})
  -> {cur(rnd, Impl.settle_keep.pick(k, Impl.Pe{r, s}, Impl.settle_keep(t, True{}, rnd))) == True{} : Bool}:
  match k:
    case True{}:
      Ar.ar_and_intro(Nat.is_eq(r, rnd), cur(rnd, Impl.settle_keep(t, True{}, rnd)),
        Ar.ar_and_l(Nat.is_eq(r, rnd), cur(rnd, t), hp), ih)
    case False{}:
      ih

def pn_settle_cur(pend: List<&2, Impl.Pe>, +rnd: Nat, +hp: {pv(rnd, pend) == True{} : Bool})
  -> {cur(rnd, Impl.settle_keep(pend, True{}, rnd)) == True{} : Bool}:
  match pend:
    case Nil{}:
      {==}
    case Impl.Pe{+r, +s} <> +t:
      pn_settle_cur.go(Nat.is_le(rnd, r), r, s, t, rnd, hp,
        pn_settle_cur(t, rnd, pn_pv_tail(Nat.is_le(rnd, r), r, rnd, t, hp)))

# a reader's settle() waits every outstanding copy and keeps nothing
def pn_settle_all(pend: List<&2, Impl.Pe>, +b: Nat, +rnd: Nat, +nf: Nat, +ns: Nat, +fq: Bool, +ks: Bool)
  -> {Spec.run(Impl.settle_ev(pend, False{}, b), ORd(rnd, nf, outs(pend), ns, fq, ks)) == ORd(rnd, nf, [], ns, fq, ks) : Spec.Obs}:
  match pend:
    case Nil{}:
      {==}
    case Impl.Pe{+r, +s} <> +t:
      Equal.trans(Spec.Obs, Spec.run(Impl.settle_ev(t, False{}, b), Spec.step(ORd(rnd, nf, Spec.Oc{s, r} <> outs(t), ns, fq, ks), Spec.Wt{s})),
        Spec.run(Impl.settle_ev(t, False{}, b), ORd(rnd, nf, outs(t), ns, fq, ks)),
        ORd(rnd, nf, [], ns, fq, ks),
        Equal.cong(Spec.Obs, Spec.Obs, z => Spec.run(Impl.settle_ev(t, False{}, b), z),
          Spec.step(ORd(rnd, nf, Spec.Oc{s, r} <> outs(t), ns, fq, ks), Spec.Wt{s}), ORd(rnd, nf, outs(t), ns, fq, ks),
          pn_wt_rd(s, r, outs(t), rnd, nf, ns, fq, ks)),
        pn_settle_all(t, b, rnd, nf, ns, fq, ks))

def pn_keep_all(pend: List<&2, Impl.Pe>, +b: Nat) -> {Impl.settle_keep(pend, False{}, b) == [] : List<&2, Impl.Pe>}:
  match pend:
    case Nil{}:
      {==}
    case Impl.Pe{r, s} <> t:
      pn_keep_all(t, b)

def pn_all_ge(pend: List<&2, Impl.Pe>, +rnd: Nat, +hc: {cur(rnd, pend) == True{} : Bool})
  -> {Spec.all_ge(outs(pend), rnd) == True{} : Bool}:
  match pend:
    case Nil{}:
      {==}
    case Impl.Pe{+r, +s} <> +t:
      Ar.ar_and_intro(Nat.is_le(rnd, r), Spec.all_ge(outs(t), rnd),
        pn_le_of_eq(r, rnd, Ar.ar_and_l(Nat.is_eq(r, rnd), cur(rnd, t), hc)),
        pn_all_ge(t, rnd, Ar.ar_and_r(Nat.is_eq(r, rnd), cur(rnd, t), hc)))

def pn_outs_app(xs: List<&2, Impl.Pe>, +ys: List<&2, Impl.Pe>)
  -> {outs(List.append(&2, Impl.Pe, xs, ys)) == List.append(&2, Spec.Out, outs(xs), outs(ys)) : List<&2, Spec.Out>}:
  match xs:
    case Nil{}:
      {==}
    case Impl.Pe{+r, +s} <> +t:
      Equal.cong(List<&2, Spec.Out>, List<&2, Spec.Out>, z => Spec.Oc{s, r} <> z,
        outs(List.append(&2, Impl.Pe, t, ys)), List.append(&2, Spec.Out, outs(t), outs(ys)), pn_outs_app(t, ys))

def pn_app_nil(xs: List<&2, Spec.Out>) -> {List.append(&2, Spec.Out, xs, []) == xs : List<&2, Spec.Out>}:
  match xs:
    case Nil{}:
      {==}
    case +x <> +t:
      Equal.cong(List<&2, Spec.Out>, List<&2, Spec.Out>, z => x <> z, List.append(&2, Spec.Out, t, []), t, pn_app_nil(t))

def pn_cur_app(xs: List<&2, Impl.Pe>, +ys: List<&2, Impl.Pe>, +rnd: Nat, +hx: {cur(rnd, xs) == True{} : Bool},
    +hy: {cur(rnd, ys) == True{} : Bool})
  -> {cur(rnd, List.append(&2, Impl.Pe, xs, ys)) == True{} : Bool}:
  match xs:
    case Nil{}:
      hy
    case Impl.Pe{+r, +s} <> +t:
      Ar.ar_and_intro(Nat.is_eq(r, rnd), cur(rnd, List.append(&2, Impl.Pe, t, ys)),
        Ar.ar_and_l(Nat.is_eq(r, rnd), cur(rnd, t), hx),
        pn_cur_app(t, ys, rnd, Ar.ar_and_r(Nat.is_eq(r, rnd), cur(rnd, t), hx), hy))

def pn_cur_add(d: Bool, +rnd: Nat, +s: Nat) -> {cur(rnd, Impl.pend_add(d, rnd, s)) == True{} : Bool}:
  match d:
    case True{}:
      Ar.ar_and_intro(Nat.is_eq(rnd, rnd), True{}, Ar.ar_eq_refl(rnd), {==})
    case False{}:
      {==}

# begin_round: the current round's entries become the previous round's
def pn_begin_pv.go(k: Bool, +r: Nat, +rnd: Nat, +t: List<&2, Impl.Pe>, +hk: {Nat.is_le(1n+rnd, r) == k : Bool},
    +he: {Nat.is_eq(r, rnd) == True{} : Bool}, ih: {pv(1n+rnd, t) == True{} : Bool})
  -> {pv.pick(k, Nat.is_eq(r, 1n+rnd), cur(1n+rnd, t), Nat.is_eq(1n+r, 1n+rnd), pv(1n+rnd, t)) == True{} : Bool}:
  match k:
    case True{}:
      +er = Ar.ar_eq_true(r, rnd, he)
      Empty.absurd({Bool.and(Nat.is_eq(r, 1n+rnd), cur(1n+rnd, t)) == True{} : Bool},
        Ar.ar_false_true(Equal.trans(Bool, False{}, Nat.is_le(1n+rnd, rnd), True{},
          Equal.sym(Bool, Nat.is_le(1n+rnd, rnd), False{}, pn_le_succ(rnd)),
          Equal.trans(Bool, Nat.is_le(1n+rnd, rnd), Nat.is_le(1n+rnd, r), True{},
            Equal.cong(Nat, Bool, z => Nat.is_le(1n+rnd, z), rnd, r, Equal.sym(Nat, r, rnd, er)), hk))))
    case False{}:
      Ar.ar_and_intro(Nat.is_eq(1n+r, 1n+rnd), pv(1n+rnd, t), he, ih)

def pn_begin_pv(+rnd: Nat, pend: List<&2, Impl.Pe>, +hc: {cur(rnd, pend) == True{} : Bool})
  -> {pv(1n+rnd, pend) == True{} : Bool}:
  match pend:
    case Nil{}:
      {==}
    case Impl.Pe{+r, +s} <> +t:
      pn_begin_pv.go(Nat.is_le(1n+rnd, r), r, rnd, t, {==}, Ar.ar_and_l(Nat.is_eq(r, rnd), cur(rnd, t), hc),
        pn_begin_pv(rnd, t, Ar.ar_and_r(Nat.is_eq(r, rnd), cur(rnd, t), hc)))

# ---------------------------------------------------------------------------------------------
# Segments of an iterate

def pn_rd_obs(r: Bool, st: Impl.St, +fq: Bool, +ks: Bool)
  -> {Spec.run(Impl.rd_ev(r, st), ob(st, fq, ks)) == ob(Impl.rd_st(r, st), fq, ks) : Spec.Obs}:
  match r:
    case True{}:
      match st:
        case Impl.St{+rnd, +pend, +nxt, +ver}:
          pn_seq(Impl.settle_ev(pend, False{}, 0n), [Spec.Rd{}], ORd(rnd, ver, outs(pend), nxt, fq, ks),
            ORd(rnd, ver, [], nxt, fq, ks), O(rnd, ver, outs(Impl.settle_keep(pend, False{}, 0n)), nxt, fq, ks),
            pn_settle_all(pend, 0n, rnd, ver, nxt, fq, ks),
            Equal.cong(List<&2, Impl.Pe>, Spec.Obs, z => O(rnd, ver, outs(z), nxt, fq, ks), [], Impl.settle_keep(pend, False{}, 0n),
              Equal.sym(List<&2, Impl.Pe>, Impl.settle_keep(pend, False{}, 0n), [], pn_keep_all(pend, 0n))))
    case False{}:
      {==}

def pn_rd_pv(r: Bool, st: Impl.St, +hp: {pv_st(st) == True{} : Bool}) -> {pv_st(Impl.rd_st(r, st)) == True{} : Bool}:
  match r:
    case True{}:
      match st:
        case Impl.St{+rnd, +pend, +nxt, +ver}:
          %Equal.sym(List<&2, Impl.Pe>, Impl.settle_keep(pend, False{}, 0n), [], pn_keep_all(pend, 0n)) : {pv(rnd, _) == True{} : Bool}
          {==}
    case False{}:
      hp

def pn_rd_cur(r: Bool, st: Impl.St, +hc: {cur_st(st) == True{} : Bool}) -> {cur_st(Impl.rd_st(r, st)) == True{} : Bool}:
  match r:
    case True{}:
      match st:
        case Impl.St{+rnd, +pend, +nxt, +ver}:
          %Equal.sym(List<&2, Impl.Pe>, Impl.settle_keep(pend, False{}, 0n), [], pn_keep_all(pend, 0n)) : {cur(rnd, _) == True{} : Bool}
          {==}
    case False{}:
      hc

# the stash of a put: copies (and the event record), waited at once when blocking
def pn_stash_obs(d: Bool, +rnd: Nat, +ver: Nat, +nxt: Nat, +keep: List<&2, Impl.Pe>, +fq: Bool, +ks: Bool,
    +hc: {cur(rnd, keep) == True{} : Bool})
  -> {Spec.run(Impl.stash_ev(d, nxt, ver), O(rnd, ver, outs(keep), nxt, fq, ks)) ==
      O(rnd, ver, outs(List.append(&2, Impl.Pe, keep, Impl.pend_add(d, rnd, nxt))), 1n+nxt, fq, ks) : Spec.Obs}:
  match d:
    case True{}:
      %Equal.sym(Bool, Nat.is_eq(ver, ver), True{}, Ar.ar_eq_refl(ver)) :
        {Spec.Ob{rnd, ver, [], List.append(&2, Spec.Out, outs(keep), [Spec.Oc{nxt, rnd}]), 1n+nxt, 0n, fq, ks, False{}, True{}, _,
           Spec.all_ge(outs(keep), rnd), True{}, Nat.is_eq(nxt, nxt)} ==
         O(rnd, ver, outs(List.append(&2, Impl.Pe, keep, [Impl.Pe{rnd, nxt}])), 1n+nxt, fq, ks) : Spec.Obs}
      %Equal.sym(Bool, Nat.is_eq(nxt, nxt), True{}, Ar.ar_eq_refl(nxt)) :
        {Spec.Ob{rnd, ver, [], List.append(&2, Spec.Out, outs(keep), [Spec.Oc{nxt, rnd}]), 1n+nxt, 0n, fq, ks, False{}, True{}, True{},
           Spec.all_ge(outs(keep), rnd), True{}, _} ==
         O(rnd, ver, outs(List.append(&2, Impl.Pe, keep, [Impl.Pe{rnd, nxt}])), 1n+nxt, fq, ks) : Spec.Obs}
      %Equal.sym(Bool, Spec.all_ge(outs(keep), rnd), True{}, pn_all_ge(keep, rnd, hc)) :
        {Spec.Ob{rnd, ver, [], List.append(&2, Spec.Out, outs(keep), [Spec.Oc{nxt, rnd}]), 1n+nxt, 0n, fq, ks, False{}, True{}, True{},
           _, True{}, True{}} ==
         O(rnd, ver, outs(List.append(&2, Impl.Pe, keep, [Impl.Pe{rnd, nxt}])), 1n+nxt, fq, ks) : Spec.Obs}
      %Equal.sym(List<&2, Spec.Out>, outs(List.append(&2, Impl.Pe, keep, [Impl.Pe{rnd, nxt}])),
          List.append(&2, Spec.Out, outs(keep), outs([Impl.Pe{rnd, nxt}])), pn_outs_app(keep, [Impl.Pe{rnd, nxt}])) :
        {O(rnd, ver, List.append(&2, Spec.Out, outs(keep), [Spec.Oc{nxt, rnd}]), 1n+nxt, fq, ks) == O(rnd, ver, _, 1n+nxt, fq, ks) : Spec.Obs}
      {==}
    case False{}:
      %Equal.sym(Bool, Nat.is_eq(ver, ver), True{}, Ar.ar_eq_refl(ver)) :
        {Spec.Ob{rnd, ver, [], outs(keep), 1n+nxt, 0n, fq, ks, False{}, True{}, _,
           Spec.all_ge(outs(keep), rnd), True{}, Bool.and(Nat.is_eq(nxt, nxt), Nat.is_eq(nxt, nxt))} ==
         O(rnd, ver, outs(List.append(&2, Impl.Pe, keep, [])), 1n+nxt, fq, ks) : Spec.Obs}
      %Equal.sym(Bool, Nat.is_eq(nxt, nxt), True{}, Ar.ar_eq_refl(nxt)) :
        {Spec.Ob{rnd, ver, [], outs(keep), 1n+nxt, 0n, fq, ks, False{}, True{}, True{},
           Spec.all_ge(outs(keep), rnd), True{}, Bool.and(_, _)} ==
         O(rnd, ver, outs(List.append(&2, Impl.Pe, keep, [])), 1n+nxt, fq, ks) : Spec.Obs}
      %Equal.sym(Bool, Spec.all_ge(outs(keep), rnd), True{}, pn_all_ge(keep, rnd, hc)) :
        {Spec.Ob{rnd, ver, [], outs(keep), 1n+nxt, 0n, fq, ks, False{}, True{}, True{}, _, True{}, True{}} ==
         O(rnd, ver, outs(List.append(&2, Impl.Pe, keep, [])), 1n+nxt, fq, ks) : Spec.Obs}
      %Equal.sym(List<&2, Spec.Out>, outs(List.append(&2, Impl.Pe, keep, [])), List.append(&2, Spec.Out, outs(keep), []), pn_outs_app(keep, [])) :
        {O(rnd, ver, outs(keep), 1n+nxt, fq, ks) == O(rnd, ver, _, 1n+nxt, fq, ks) : Spec.Obs}
      %Equal.sym(List<&2, Spec.Out>, List.append(&2, Spec.Out, outs(keep), []), outs(keep), pn_app_nil(outs(keep))) :
        {O(rnd, ver, outs(keep), 1n+nxt, fq, ks) == O(rnd, ver, _, 1n+nxt, fq, ks) : Spec.Obs}
      {==}

def pn_cur_pv_st(st: Impl.St, +hc: {cur_st(st) == True{} : Bool}) -> {pv_st(st) == True{} : Bool}:
  match st:
    case Impl.St{+rnd, +pend, +nxt, +ver}:
      pn_cur_pv(rnd, pend, hc)

# a put: settle(self.round), then the stash
def pn_put_obs(u: Spec.Pu, +d: Bool, st: Impl.St, +fq: Bool, +ks: Bool, +hp: {pv_st(st) == True{} : Bool},
    +hf: {Bool.or(ks, fq) == True{} : Bool})
  -> {Spec.run(Impl.put_ev(u, d, st), ob(st, fq, ks)) == ob(Impl.put_st(u, d, st), fq, ks) : Spec.Obs}:
  match u:
    case Spec.No{}:
      {==}
    case Spec.Hit{}:
      match st:
        case Impl.St{+rnd, +pend, +nxt, +ver}:
          pn_settle_obs(pend, rnd, ver, nxt, fq, ks, hp, hf)
    case Spec.New{}:
      match st:
        case Impl.St{+rnd, +pend, +nxt, +ver}:
          +keep = Impl.settle_keep(pend, True{}, rnd)
          pn_seq(Impl.settle_ev(pend, True{}, rnd), Impl.stash_ev(d, nxt, ver), O(rnd, ver, outs(pend), nxt, fq, ks),
            O(rnd, ver, outs(keep), nxt, fq, ks),
            O(rnd, ver, outs(List.append(&2, Impl.Pe, keep, Impl.pend_add(d, rnd, nxt))), 1n+nxt, fq, ks),
            pn_settle_obs(pend, rnd, ver, nxt, fq, ks, hp, hf),
            pn_stash_obs(d, rnd, ver, nxt, keep, fq, ks, pn_settle_cur(pend, rnd, hp)))

def pn_put_pv(u: Spec.Pu, +d: Bool, st: Impl.St, +hp: {pv_st(st) == True{} : Bool})
  -> {pv_st(Impl.put_st(u, d, st)) == True{} : Bool}:
  match u:
    case Spec.No{}:
      hp
    case Spec.Hit{}:
      match st:
        case Impl.St{+rnd, +pend, +nxt, +ver}:
          pn_cur_pv(rnd, Impl.settle_keep(pend, True{}, rnd), pn_settle_cur(pend, rnd, hp))
    case Spec.New{}:
      match st:
        case Impl.St{+rnd, +pend, +nxt, +ver}:
          pn_cur_pv(rnd, List.append(&2, Impl.Pe, Impl.settle_keep(pend, True{}, rnd), Impl.pend_add(d, rnd, nxt)),
            pn_cur_app(Impl.settle_keep(pend, True{}, rnd), Impl.pend_add(d, rnd, nxt), rnd, pn_settle_cur(pend, rnd, hp),
              pn_cur_add(d, rnd, nxt)))

def pn_put_cur(u: Spec.Pu, +d: Bool, st: Impl.St, +hc: {cur_st(st) == True{} : Bool})
  -> {cur_st(Impl.put_st(u, d, st)) == True{} : Bool}:
  match u:
    case Spec.No{}:
      hc
    case Spec.Hit{}:
      match st:
        case Impl.St{+rnd, +pend, +nxt, +ver}:
          pn_settle_cur(pend, rnd, pn_cur_pv(rnd, pend, hc))
    case Spec.New{}:
      match st:
        case Impl.St{+rnd, +pend, +nxt, +ver}:
          pn_cur_app(Impl.settle_keep(pend, True{}, rnd), Impl.pend_add(d, rnd, nxt), rnd,
            pn_settle_cur(pend, rnd, pn_cur_pv(rnd, pend, hc)), pn_cur_add(d, rnd, nxt))

def fq1(fw: Bool, fq: Bool) -> Bool:
  match fw:
    case True{}:
      True{}
    case False{}:
      fq

# the chunk's upload, then its forward
def pn_upfw(+rnd: Nat, +ver: Nat, +out: List<&2, Spec.Out>, +nxt: Nat, +fq: Bool)
  -> {Spec.step(Spec.step(O(rnd, ver, out, nxt, fq, False{}), Spec.Up{1n+ver}), Spec.Fw{1n+ver}) ==
      O(rnd, 1n+ver, out, nxt, True{}, False{}) : Spec.Obs}:
  %Equal.sym(Bool, Nat.is_eq(1n+ver, 1n+ver), True{}, Ar.ar_eq_refl(1n+ver)) :
    {Spec.Ob{rnd, 1n+ver, Spec.del.pick(_, 1n+ver, [], []), out, nxt, 0n, True{}, False{}, False{},
       Bool.or(_, False{}), True{}, True{}, True{}, True{}} == O(rnd, 1n+ver, out, nxt, True{}, False{}) : Spec.Obs}
  {==}

def pn_pf_obs(fw: Bool, lp: Spec.Pu, en: Bool, st: Impl.St, +fq: Bool, +hp: {pv_st(st) == True{} : Bool})
  -> {Spec.run(Impl.pf_ev(fw, lp, en, st), ob(st, fq, False{})) == ob(Impl.pf_st(fw, lp, en, st), fq1(fw, fq), False{}) : Spec.Obs}:
  match fw:
    case True{}:
      match st:
        case Impl.St{+rnd, +pend, +nxt, +ver}:
          +s1 = {Impl.St{rnd, pend, nxt, 1n+ver} : Impl.St}
          Equal.trans(Spec.Obs,
            Spec.run(Impl.put_ev(lp, en, s1), Spec.step(Spec.step(O(rnd, ver, outs(pend), nxt, fq, False{}), Spec.Up{1n+ver}), Spec.Fw{1n+ver})),
            Spec.run(Impl.put_ev(lp, en, s1), ob(s1, True{}, False{})),
            ob(Impl.put_st(lp, en, s1), True{}, False{}),
            Equal.cong(Spec.Obs, Spec.Obs, z => Spec.run(Impl.put_ev(lp, en, s1), z),
              Spec.step(Spec.step(O(rnd, ver, outs(pend), nxt, fq, False{}), Spec.Up{1n+ver}), Spec.Fw{1n+ver}),
              ob(s1, True{}, False{}), pn_upfw(rnd, ver, outs(pend), nxt, fq)),
            pn_put_obs(lp, en, s1, True{}, False{}, hp, {==}))
    case False{}:
      {==}

def pn_pf_pv(fw: Bool, lp: Spec.Pu, en: Bool, st: Impl.St, +hp: {pv_st(st) == True{} : Bool})
  -> {pv_st(Impl.pf_st(fw, lp, en, st)) == True{} : Bool}:
  match fw:
    case True{}:
      match st:
        case Impl.St{+rnd, +pend, +nxt, +ver}:
          pn_put_pv(lp, en, Impl.St{rnd, pend, nxt, 1n+ver}, hp)
    case False{}:
      hp

# recurrent_checkpoint: settle(round), then its stash loop starts
def pn_ck_obs(st: Impl.St, +fq: Bool, +ks: Bool, +hp: {pv_st(st) == True{} : Bool})
  -> {Spec.run(Impl.ck_ev(st), ob(st, fq, ks)) == ob(Impl.ck_st(st), fq, True{}) : Spec.Obs}:
  match st:
    case Impl.St{+rnd, +pend, +nxt, +ver}:
      +keep = Impl.settle_keep(pend, True{}, rnd)
      pn_seq(Impl.settle_ev(pend, True{}, rnd), [Spec.Ck{}], O(rnd, ver, outs(pend), nxt, fq, True{}),
        O(rnd, ver, outs(keep), nxt, fq, True{}), O(rnd, ver, outs(keep), nxt, fq, True{}),
        pn_settle_obs(pend, rnd, ver, nxt, fq, True{}, hp, {==}),
        Equal.cong(Bool, Spec.Obs, z => Spec.Ob{rnd, ver, [], outs(keep), nxt, 0n, fq, True{}, False{}, True{}, True{}, z, True{}, True{}},
          Spec.all_ge(outs(keep), rnd), True{}, pn_all_ge(keep, rnd, pn_settle_cur(pend, rnd, hp))))

def pn_ck_cur(st: Impl.St, +hp: {pv_st(st) == True{} : Bool}) -> {cur_st(Impl.ck_st(st)) == True{} : Bool}:
  match st:
    case Impl.St{+rnd, +pend, +nxt, +ver}:
      pn_settle_cur(pend, rnd, hp)

def pn_dc_obs(gf: Bool, st: Impl.St, +fq: Bool, +ks: Bool)
  -> {Spec.run(Impl.dc_ev(gf, st), ob(st, fq, ks)) == ob(Impl.dc_st(gf, st), fq, ks) : Spec.Obs}:
  match gf:
    case True{}:
      match st:
        case Impl.St{+rnd, +pend, +nxt, +ver}:
          {==}
    case False{}:
      {==}

def pn_dc_cur(gf: Bool, st: Impl.St, +hc: {cur_st(st) == True{} : Bool}) -> {cur_st(Impl.dc_st(gf, st)) == True{} : Bool}:
  match gf:
    case True{}:
      match st:
        case Impl.St{+rnd, +pend, +nxt, +ver}:
          hc
    case False{}:
      hc

# ---------------------------------------------------------------------------------------------
# Iterates

def it_fq(it: Spec.It) -> Bool:
  match it:
    case Spec.Iter{rs, fw, lp, cp, en, gf, rq, re}:
      fq1(fw, False{})

def pn_it_obs(it: Spec.It, st: Impl.St, +fq: Bool, +ks: Bool, +hc: {cur_st(st) == True{} : Bool})
  -> {Spec.run(Impl.it_ev(it, st), ob(st, fq, ks)) == ob(Impl.it_st(it, st), it_fq(it), True{}) : Spec.Obs}:
  match it:
    case Spec.Iter{+rs, +fw, +lp, +cp, +en, +gf, +rq, +re}:
      match st:
        case Impl.St{+rnd, +pend, +nxt, +ver}:
          +s1 = {Impl.St{1n+rnd, pend, nxt, ver} : Impl.St}
          +s2 = Impl.rd_st(rs, s1)
          +s3 = Impl.pf_st(fw, lp, en, s2)
          +s4 = Impl.ck_st(s3)
          +d = Impl.sd(fw, en)
          +s5 = Impl.put_st(cp, d, s4)
          +s6 = Impl.dc_st(gf, s5)
          +s7 = Impl.put_st(rq, d, s6)
          +f = fq1(fw, False{})
          +fin = ob(Impl.rd_st(re, s7), f, True{})
          +p2 = pn_rd_pv(rs, s1, pn_begin_pv(rnd, pend, hc))
          +p3 = pn_pf_pv(fw, lp, en, s2, p2)
          +c4 = pn_ck_cur(s3, p3)
          +c5 = pn_put_cur(cp, d, s4, c4)
          +c6 = pn_dc_cur(gf, s5, c5)
          +r6 = Impl.rd_ev(re, s7)
          +r5 = List.append(&2, Spec.Ev, Impl.put_ev(rq, d, s6), r6)
          +r4 = List.append(&2, Spec.Ev, Impl.dc_ev(gf, s5), r5)
          +r3 = List.append(&2, Spec.Ev, Impl.put_ev(cp, d, s4), r4)
          +r2 = List.append(&2, Spec.Ev, Impl.ck_ev(s3), r3)
          +r1 = List.append(&2, Spec.Ev, Impl.pf_ev(fw, lp, en, s2), r2)
          pn_seq(Impl.rd_ev(rs, s1), r1, ob(s1, False{}, False{}), ob(s2, False{}, False{}), fin,
            pn_rd_obs(rs, s1, False{}, False{}),
            pn_seq(Impl.pf_ev(fw, lp, en, s2), r2, ob(s2, False{}, False{}), ob(s3, f, False{}), fin,
              pn_pf_obs(fw, lp, en, s2, False{}, p2),
              pn_seq(Impl.ck_ev(s3), r3, ob(s3, f, False{}), ob(s4, f, True{}), fin,
                pn_ck_obs(s3, f, False{}, p3),
                pn_seq(Impl.put_ev(cp, d, s4), r4, ob(s4, f, True{}), ob(s5, f, True{}), fin,
                  pn_put_obs(cp, d, s4, f, True{}, pn_cur_pv_st(s4, c4), {==}),
                  pn_seq(Impl.dc_ev(gf, s5), r5, ob(s5, f, True{}), ob(s6, f, True{}), fin,
                    pn_dc_obs(gf, s5, f, True{}),
                    pn_seq(Impl.put_ev(rq, d, s6), r6, ob(s6, f, True{}), ob(s7, f, True{}), fin,
                      pn_put_obs(rq, d, s6, f, True{}, pn_cur_pv_st(s6, c6), {==}),
                      pn_rd_obs(re, s7, f, True{})))))))

def pn_it_cur(it: Spec.It, st: Impl.St, +hc: {cur_st(st) == True{} : Bool})
  -> {cur_st(Impl.it_st(it, st)) == True{} : Bool}:
  match it:
    case Spec.Iter{+rs, +fw, +lp, +cp, +en, +gf, +rq, +re}:
      match st:
        case Impl.St{+rnd, +pend, +nxt, +ver}:
          +s1 = {Impl.St{1n+rnd, pend, nxt, ver} : Impl.St}
          +s2 = Impl.rd_st(rs, s1)
          +s3 = Impl.pf_st(fw, lp, en, s2)
          +s4 = Impl.ck_st(s3)
          +d = Impl.sd(fw, en)
          +s5 = Impl.put_st(cp, d, s4)
          +s6 = Impl.dc_st(gf, s5)
          +p3 = pn_pf_pv(fw, lp, en, s2, pn_rd_pv(rs, s1, pn_begin_pv(rnd, pend, hc)))
          pn_rd_cur(re, Impl.put_st(rq, d, s6),
            pn_put_cur(rq, d, s6, pn_dc_cur(gf, s5, pn_put_cur(cp, d, s4, pn_ck_cur(s3, p3)))))

def ffq(its: List<&2, Spec.It>, fq: Bool) -> Bool:
  match its:
    case Nil{}:
      fq
    case +it <> rest:
      ffq(rest, it_fq(it))

def fks(its: List<&2, Spec.It>, ks: Bool) -> Bool:
  match its:
    case Nil{}:
      ks
    case it <> rest:
      fks(rest, True{})

def pn_trace(its: List<&2, Spec.It>, +st: Impl.St, +fq: Bool, +ks: Bool, +hc: {cur_st(st) == True{} : Bool})
  -> {Spec.run(Impl.trace(its, st), ob(st, fq, ks)) == ob(Impl.final(its, st), ffq(its, fq), fks(its, ks)) : Spec.Obs}:
  match its:
    case Nil{}:
      {==}
    case +it <> +rest:
      pn_seq(Impl.it_ev(it, st), Impl.trace(rest, Impl.it_st(it, st)), ob(st, fq, ks), ob(Impl.it_st(it, st), it_fq(it), True{}),
        ob(Impl.final(rest, Impl.it_st(it, st)), ffq(rest, it_fq(it)), fks(rest, True{})),
        pn_it_obs(it, st, fq, ks, hc),
        pn_trace(rest, Impl.it_st(it, st), it_fq(it), True{}, pn_it_cur(it, st, hc)))

def pn_final_cur(its: List<&2, Spec.It>, +st: Impl.St, +hc: {cur_st(st) == True{} : Bool})
  -> {cur_st(Impl.final(its, st)) == True{} : Bool}:
  match its:
    case Nil{}:
      hc
    case +it <> +rest:
      pn_final_cur(rest, Impl.it_st(it, st), pn_it_cur(it, st, hc))

def pn_main(+its: List<&2, Spec.It>)
  -> {Spec.run(Impl.trace(its, Impl.init()), Spec.init()) ==
      ob(Impl.final(its, Impl.init()), ffq(its, False{}), fks(its, False{})) : Spec.Obs}:
  pn_trace(its, Impl.init(), False{}, False{}, {==})

# ---------------------------------------------------------------------------------------------
# Laws 1-5 from the observer state at the end of any run

def pn_ok1(st: Impl.St, +fq: Bool, +ks: Bool) -> {Spec.ok1(ob(st, fq, ks)) == True{} : Bool}:
  match st:
    case Impl.St{rnd, pend, nxt, ver}:
      {==}

def pn_ok2(st: Impl.St, +fq: Bool, +ks: Bool) -> {Spec.ok2(ob(st, fq, ks)) == True{} : Bool}:
  match st:
    case Impl.St{rnd, pend, nxt, ver}:
      {==}

def pn_ok3(st: Impl.St, +fq: Bool, +ks: Bool) -> {Spec.ok3(ob(st, fq, ks)) == True{} : Bool}:
  match st:
    case Impl.St{rnd, pend, nxt, ver}:
      {==}

def pn_ok4(st: Impl.St, +fq: Bool, +ks: Bool) -> {Spec.ok4(ob(st, fq, ks)) == True{} : Bool}:
  match st:
    case Impl.St{rnd, pend, nxt, ver}:
      {==}

def pn_ok5(st: Impl.St, +fq: Bool, +ks: Bool) -> {Spec.ok5(ob(st, fq, ks)) == True{} : Bool}:
  match st:
    case Impl.St{rnd, pend, nxt, ver}:
      {==}

def pn_blk(st: Impl.St, +fq: Bool, +ks: Bool) -> {Spec.blk(ob(st, fq, ks)) == 0n : Nat}:
  match st:
    case Impl.St{rnd, pend, nxt, ver}:
      {==}

def pn_ids.go(pend: List<&2, Impl.Pe>) -> {Spec.out_ids.go(outs(pend)) == Impl.ids.go(pend) : List<&2, Nat>}:
  match pend:
    case Nil{}:
      {==}
    case Impl.Pe{+r, +s} <> +t:
      Equal.cong(List<&2, Nat>, List<&2, Nat>, z => s <> z, Spec.out_ids.go(outs(t)), Impl.ids.go(t), pn_ids.go(t))

def pn_ids(st: Impl.St, +fq: Bool, +ks: Bool) -> {Spec.out_ids(ob(st, fq, ks)) == Impl.pend_ids(st) : List<&2, Nat>}:
  match st:
    case Impl.St{rnd, +pend, nxt, ver}:
      pn_ids.go(pend)

def Laws.uploads_before_forwards(its):
  +fin = Impl.final(its, Impl.init())
  Equal.trans(Bool, Spec.ok1(Spec.run(Impl.trace(its, Impl.init()), Spec.init())), Spec.ok1(ob(fin, ffq(its, False{}), fks(its, False{}))), True{},
    Equal.cong(Spec.Obs, Bool, z => Spec.ok1(z), Spec.run(Impl.trace(its, Impl.init()), Spec.init()), ob(fin, ffq(its, False{}), fks(its, False{})), pn_main(its)),
    pn_ok1(fin, ffq(its, False{}), fks(its, False{})))

def Laws.copy_reads_its_chunk(its):
  +fin = Impl.final(its, Impl.init())
  Equal.trans(Bool, Spec.ok2(Spec.run(Impl.trace(its, Impl.init()), Spec.init())), Spec.ok2(ob(fin, ffq(its, False{}), fks(its, False{}))), True{},
    Equal.cong(Spec.Obs, Bool, z => Spec.ok2(z), Spec.run(Impl.trace(its, Impl.init()), Spec.init()), ob(fin, ffq(its, False{}), fks(its, False{})), pn_main(its)),
    pn_ok2(fin, ffq(its, False{}), fks(its, False{})))

def Laws.deferred_wait_window(its):
  +fin = Impl.final(its, Impl.init())
  Equal.trans(Bool, Spec.ok3(Spec.run(Impl.trace(its, Impl.init()), Spec.init())), Spec.ok3(ob(fin, ffq(its, False{}), fks(its, False{}))), True{},
    Equal.cong(Spec.Obs, Bool, z => Spec.ok3(z), Spec.run(Impl.trace(its, Impl.init()), Spec.init()), ob(fin, ffq(its, False{}), fks(its, False{})), pn_main(its)),
    pn_ok3(fin, ffq(its, False{}), fks(its, False{})))

def Laws.start_reader_waits_early():
  ({==}, {==})

def Laws.readers_see_no_pending(its):
  +fin = Impl.final(its, Impl.init())
  Equal.trans(Bool, Spec.ok4(Spec.run(Impl.trace(its, Impl.init()), Spec.init())), Spec.ok4(ob(fin, ffq(its, False{}), fks(its, False{}))), True{},
    Equal.cong(Spec.Obs, Bool, z => Spec.ok4(z), Spec.run(Impl.trace(its, Impl.init()), Spec.init()), ob(fin, ffq(its, False{}), fks(its, False{})), pn_main(its)),
    pn_ok4(fin, ffq(its, False{}), fks(its, False{})))

def Laws.persist_require(pend, b):
  pn_keep_all(pend, b)

def Laws.conservation(its):
  +fin = Impl.final(its, Impl.init())
  +run = Spec.run(Impl.trace(its, Impl.init()), Spec.init())
  +o = ob(fin, ffq(its, False{}), fks(its, False{}))
  +m = pn_main(its)
  (Equal.trans(Bool, Spec.ok5(run), Spec.ok5(o), True{}, Equal.cong(Spec.Obs, Bool, z => Spec.ok5(z), run, o, m),
     pn_ok5(fin, ffq(its, False{}), fks(its, False{}))),
   Equal.trans(List<&2, Nat>, Spec.out_ids(run), Spec.out_ids(o), Impl.pend_ids(fin),
     Equal.cong(Spec.Obs, List<&2, Nat>, z => Spec.out_ids(z), run, o, m), pn_ids(fin, ffq(its, False{}), fks(its, False{}))),
   Equal.trans(Nat, Spec.blk(run), Spec.blk(o), 0n, Equal.cong(Spec.Obs, Nat, z => Spec.blk(z), run, o, m),
     pn_blk(fin, ffq(its, False{}), fks(its, False{}))))

# ---------------------------------------------------------------------------------------------
# Law 6: the state invariant

def pn_le_succ_r(+a: Nat) -> {Nat.is_le(a, 1n+a) == True{} : Bool}:
  match a:
    case 0n:
      {==}
    case 1n+p:
      pn_le_succ_r(p)

def pn_eq_le(+r: Nat, +rnd: Nat, he: {Nat.is_eq(r, rnd) == True{} : Bool}) -> {Nat.is_le(r, rnd) == True{} : Bool}:
  %Equal.sym(Nat, r, rnd, Ar.ar_eq_true(r, rnd, he)) : {Nat.is_le(_, rnd) == True{} : Bool}
  Ar.ar_le_refl(rnd)

def pn_inv_settle.go(k: Bool, +r: Nat, +s: Nat, +t: List<&2, Impl.Pe>, +rnd: Nat, +hasb: Bool, +b: Nat,
    +h: {Spec.inv(rnd, r <> Impl.rounds.go(t)) == True{} : Bool},
    ih: {Spec.inv(rnd, Impl.rounds.go(Impl.settle_keep(t, hasb, b))) == True{} : Bool})
  -> {Spec.inv(rnd, Impl.rounds.go(Impl.settle_keep.pick(k, Impl.Pe{r, s}, Impl.settle_keep(t, hasb, b)))) == True{} : Bool}:
  match k:
    case True{}:
      +h2 = Ar.ar_and_r(Nat.is_le(rnd, 1n+r), Bool.and(Nat.is_le(r, rnd), Spec.inv(rnd, Impl.rounds.go(t))), h)
      Ar.ar_and_intro(Nat.is_le(rnd, 1n+r), Bool.and(Nat.is_le(r, rnd), Spec.inv(rnd, Impl.rounds.go(Impl.settle_keep(t, hasb, b)))),
        Ar.ar_and_l(Nat.is_le(rnd, 1n+r), Bool.and(Nat.is_le(r, rnd), Spec.inv(rnd, Impl.rounds.go(t))), h),
        Ar.ar_and_intro(Nat.is_le(r, rnd), Spec.inv(rnd, Impl.rounds.go(Impl.settle_keep(t, hasb, b))),
          Ar.ar_and_l(Nat.is_le(r, rnd), Spec.inv(rnd, Impl.rounds.go(t)), h2), ih))
    case False{}:
      ih

def pn_inv_settle(+rnd: Nat, pend: List<&2, Impl.Pe>, +hasb: Bool, +b: Nat,
    +h: {Spec.inv(rnd, Impl.rounds.go(pend)) == True{} : Bool})
  -> {Spec.inv(rnd, Impl.rounds.go(Impl.settle_keep(pend, hasb, b))) == True{} : Bool}:
  match pend:
    case Nil{}:
      {==}
    case Impl.Pe{+r, +s} <> +t:
      +h2 = Ar.ar_and_r(Nat.is_le(rnd, 1n+r), Bool.and(Nat.is_le(r, rnd), Spec.inv(rnd, Impl.rounds.go(t))), h)
      pn_inv_settle.go(Impl.keep_t(hasb, b, r), r, s, t, rnd, hasb, b, h,
        pn_inv_settle(rnd, t, hasb, b, Ar.ar_and_r(Nat.is_le(r, rnd), Spec.inv(rnd, Impl.rounds.go(t)), h2)))

def pn_inv_app(+rnd: Nat, xs: List<&2, Impl.Pe>, +ys: List<&2, Impl.Pe>, +hx: {Spec.inv(rnd, Impl.rounds.go(xs)) == True{} : Bool},
    +hy: {Spec.inv(rnd, Impl.rounds.go(ys)) == True{} : Bool})
  -> {Spec.inv(rnd, Impl.rounds.go(List.append(&2, Impl.Pe, xs, ys))) == True{} : Bool}:
  match xs:
    case Nil{}:
      hy
    case Impl.Pe{+r, +s} <> +t:
      +h2 = Ar.ar_and_r(Nat.is_le(rnd, 1n+r), Bool.and(Nat.is_le(r, rnd), Spec.inv(rnd, Impl.rounds.go(t))), hx)
      Ar.ar_and_intro(Nat.is_le(rnd, 1n+r), Bool.and(Nat.is_le(r, rnd), Spec.inv(rnd, Impl.rounds.go(List.append(&2, Impl.Pe, t, ys)))),
        Ar.ar_and_l(Nat.is_le(rnd, 1n+r), Bool.and(Nat.is_le(r, rnd), Spec.inv(rnd, Impl.rounds.go(t))), hx),
        Ar.ar_and_intro(Nat.is_le(r, rnd), Spec.inv(rnd, Impl.rounds.go(List.append(&2, Impl.Pe, t, ys))),
          Ar.ar_and_l(Nat.is_le(r, rnd), Spec.inv(rnd, Impl.rounds.go(t)), h2),
          pn_inv_app(rnd, t, ys, Ar.ar_and_r(Nat.is_le(r, rnd), Spec.inv(rnd, Impl.rounds.go(t)), h2), hy)))

def pn_inv_add(d: Bool, +rnd: Nat, +s: Nat) -> {Spec.inv(rnd, Impl.rounds.go(Impl.pend_add(d, rnd, s))) == True{} : Bool}:
  match d:
    case True{}:
      Ar.ar_and_intro(Nat.is_le(rnd, 1n+rnd), Bool.and(Nat.is_le(rnd, rnd), True{}), pn_le_succ_r(rnd),
        Ar.ar_and_intro(Nat.is_le(rnd, rnd), True{}, Ar.ar_le_refl(rnd), {==}))
    case False{}:
      {==}

def pn_inv_put(u: Spec.Pu, +d: Bool, st: Impl.St, +h: {Spec.inv(Impl.round(st), Impl.rounds(st)) == True{} : Bool})
  -> {Spec.inv(Impl.round(Impl.put_st(u, d, st)), Impl.rounds(Impl.put_st(u, d, st))) == True{} : Bool}:
  match u:
    case Spec.No{}:
      h
    case Spec.Hit{}:
      match st:
        case Impl.St{+rnd, +pend, +nxt, +ver}:
          pn_inv_settle(rnd, pend, True{}, rnd, h)
    case Spec.New{}:
      match st:
        case Impl.St{+rnd, +pend, +nxt, +ver}:
          pn_inv_app(rnd, Impl.settle_keep(pend, True{}, rnd), Impl.pend_add(d, rnd, nxt),
            pn_inv_settle(rnd, pend, True{}, rnd, h), pn_inv_add(d, rnd, nxt))

def pn_inv_rd(r: Bool, st: Impl.St, +h: {Spec.inv(Impl.round(st), Impl.rounds(st)) == True{} : Bool})
  -> {Spec.inv(Impl.round(Impl.rd_st(r, st)), Impl.rounds(Impl.rd_st(r, st))) == True{} : Bool}:
  match r:
    case True{}:
      match st:
        case Impl.St{+rnd, +pend, +nxt, +ver}:
          %Equal.sym(List<&2, Impl.Pe>, Impl.settle_keep(pend, False{}, 0n), [], pn_keep_all(pend, 0n)) :
            {Spec.inv(rnd, Impl.rounds.go(_)) == True{} : Bool}
          {==}
    case False{}:
      h

def pn_inv_ck(st: Impl.St, +h: {Spec.inv(Impl.round(st), Impl.rounds(st)) == True{} : Bool})
  -> {Spec.inv(Impl.round(Impl.ck_st(st)), Impl.rounds(Impl.ck_st(st))) == True{} : Bool}:
  match st:
    case Impl.St{+rnd, +pend, +nxt, +ver}:
      pn_inv_settle(rnd, pend, True{}, rnd, h)

def pn_inv_fwd(st: Impl.St, +h: {Spec.inv(Impl.round(st), Impl.rounds(st)) == True{} : Bool})
  -> {Spec.inv(Impl.round(Impl.fwd(st)), Impl.rounds(Impl.fwd(st))) == True{} : Bool}:
  match st:
    case Impl.St{+rnd, +pend, +nxt, +ver}:
      h

def pn_inv_begin(+rnd: Nat, pend: List<&2, Impl.Pe>, +h0: {Spec.inv(rnd, Impl.rounds.go(pend)) == True{} : Bool},
    +h: {Spec.all_at_least(rnd, Impl.rounds.go(pend)) == True{} : Bool})
  -> {Spec.inv(1n+rnd, Impl.rounds.go(pend)) == True{} : Bool}:
  match pend:
    case Nil{}:
      {==}
    case Impl.Pe{+r, +s} <> +t:
      +h2 = Ar.ar_and_r(Nat.is_le(rnd, 1n+r), Bool.and(Nat.is_le(r, rnd), Spec.inv(rnd, Impl.rounds.go(t))), h0)
      Ar.ar_and_intro(Nat.is_le(1n+rnd, 1n+r), Bool.and(Nat.is_le(r, 1n+rnd), Spec.inv(1n+rnd, Impl.rounds.go(t))),
        Ar.ar_and_l(Nat.is_le(rnd, r), Spec.all_at_least(rnd, Impl.rounds.go(t)), h),
        Ar.ar_and_intro(Nat.is_le(r, 1n+rnd), Spec.inv(1n+rnd, Impl.rounds.go(t)),
          Ar.ar_le_trans(r, rnd, 1n+rnd, Ar.ar_and_l(Nat.is_le(r, rnd), Spec.inv(rnd, Impl.rounds.go(t)), h2), pn_le_succ_r(rnd)),
          pn_inv_begin(rnd, t, Ar.ar_and_r(Nat.is_le(r, rnd), Spec.inv(rnd, Impl.rounds.go(t)), h2),
            Ar.ar_and_r(Nat.is_le(rnd, r), Spec.all_at_least(rnd, Impl.rounds.go(t)), h))))

def pn_cur_ala(+rnd: Nat, pend: List<&2, Impl.Pe>, +hc: {cur(rnd, pend) == True{} : Bool})
  -> {Spec.all_at_least(rnd, Impl.rounds.go(pend)) == True{} : Bool}:
  match pend:
    case Nil{}:
      {==}
    case Impl.Pe{+r, +s} <> +t:
      Ar.ar_and_intro(Nat.is_le(rnd, r), Spec.all_at_least(rnd, Impl.rounds.go(t)),
        pn_le_of_eq(r, rnd, Ar.ar_and_l(Nat.is_eq(r, rnd), cur(rnd, t), hc)),
        pn_cur_ala(rnd, t, Ar.ar_and_r(Nat.is_eq(r, rnd), cur(rnd, t), hc)))

def pn_cur_inv(+rnd: Nat, pend: List<&2, Impl.Pe>, +hc: {cur(rnd, pend) == True{} : Bool})
  -> {Spec.inv(rnd, Impl.rounds.go(pend)) == True{} : Bool}:
  match pend:
    case Nil{}:
      {==}
    case Impl.Pe{+r, +s} <> +t:
      +he = Ar.ar_and_l(Nat.is_eq(r, rnd), cur(rnd, t), hc)
      Ar.ar_and_intro(Nat.is_le(rnd, 1n+r), Bool.and(Nat.is_le(r, rnd), Spec.inv(rnd, Impl.rounds.go(t))),
        Ar.ar_le_trans(rnd, r, 1n+r, pn_le_of_eq(r, rnd, he), pn_le_succ_r(r)),
        Ar.ar_and_intro(Nat.is_le(r, rnd), Spec.inv(rnd, Impl.rounds.go(t)), pn_eq_le(r, rnd, he),
          pn_cur_inv(rnd, t, Ar.ar_and_r(Nat.is_eq(r, rnd), cur(rnd, t), hc))))

def pn_final_ala(st: Impl.St, +hc: {cur_st(st) == True{} : Bool})
  -> {Spec.all_at_least(Impl.round(st), Impl.rounds(st)) == True{} : Bool}:
  match st:
    case Impl.St{+rnd, +pend, +nxt, +ver}:
      pn_cur_ala(rnd, pend, hc)

def pn_final_inv(st: Impl.St, +hc: {cur_st(st) == True{} : Bool})
  -> {Spec.inv(Impl.round(st), Impl.rounds(st)) == True{} : Bool}:
  match st:
    case Impl.St{+rnd, +pend, +nxt, +ver}:
      pn_cur_inv(rnd, pend, hc)

def Laws.inv_init():
  {==}

def Laws.inv_settle(rnd, pend, hasb, b, h):
  pn_inv_settle(rnd, pend, hasb, b, h)

def Laws.inv_ops(st, u, d, r, h):
  (pn_inv_put(u, d, st, h), pn_inv_rd(r, st, h), pn_inv_ck(st, h), pn_inv_fwd(st, h))

def pn_inv_begin_st(st: Impl.St, +h0: {Spec.inv(Impl.round(st), Impl.rounds(st)) == True{} : Bool},
    +h: {Spec.all_at_least(Impl.round(st), Impl.rounds(st)) == True{} : Bool})
  -> {Spec.inv(Impl.round(Impl.begin(st)), Impl.rounds(Impl.begin(st))) == True{} : Bool}:
  match st:
    case Impl.St{+rnd, +pend, +nxt, +ver}:
      pn_inv_begin(rnd, pend, h0, h)

def Laws.inv_begin(st, h0, h):
  pn_inv_begin_st(st, h0, h)

def Laws.inv_begin_needs_settle():
  ({==}, {==})

def Laws.inv_iterates(its):
  +hc = pn_final_cur(its, Impl.init(), {==})
  (pn_final_ala(Impl.final(its, Impl.init()), hc), pn_final_inv(Impl.final(its, Impl.init()), hc))

# ---------------------------------------------------------------------------------------------
# Law 5': with the switch off nothing is ever pending, so every settle() is empty and the trace is the old one

def pn_off_put(u: Spec.Pu, +rnd: Nat, +s: Nat, +c: Nat)
  -> {Impl.put_ev(u, False{}, Impl.St{rnd, [], s, c}) == Spec.old_put(u, s, c) : List<&2, Spec.Ev>}:
  match u:
    case Spec.No{}:
      {==}
    case Spec.Hit{}:
      {==}
    case Spec.New{}:
      {==}

def pn_off_put_st(u: Spec.Pu, +rnd: Nat, +s: Nat, +c: Nat)
  -> {Impl.put_st(u, False{}, Impl.St{rnd, [], s, c}) == Impl.St{rnd, [], Spec.old_ns(u, s), c} : Impl.St}:
  match u:
    case Spec.No{}:
      {==}
    case Spec.Hit{}:
      {==}
    case Spec.New{}:
      {==}

def pn_off_rd(r: Bool, +rnd: Nat, +s: Nat, +c: Nat)
  -> {Impl.rd_ev(r, Impl.St{rnd, [], s, c}) == Spec.old_rd(r) : List<&2, Spec.Ev>}:
  match r:
    case True{}:
      {==}
    case False{}:
      {==}

def pn_off_rd_st(r: Bool, +rnd: Nat, +s: Nat, +c: Nat)
  -> {Impl.rd_st(r, Impl.St{rnd, [], s, c}) == Impl.St{rnd, [], s, c} : Impl.St}:
  match r:
    case True{}:
      {==}
    case False{}:
      {==}

def pn_off_pf(fw: Bool, +lp: Spec.Pu, +rnd: Nat, +s: Nat, +c: Nat)
  -> {Impl.pf_ev(fw, lp, False{}, Impl.St{rnd, [], s, c}) == Spec.old_pf(fw, lp, s, c) : List<&2, Spec.Ev>}:
  match fw:
    case True{}:
      Equal.cong(List<&2, Spec.Ev>, List<&2, Spec.Ev>, z => Spec.Up{1n+c} <> Spec.Fw{1n+c} <> z,
        Impl.put_ev(lp, False{}, Impl.St{rnd, [], s, 1n+c}), Spec.old_put(lp, s, 1n+c), pn_off_put(lp, rnd, s, 1n+c))
    case False{}:
      {==}

def pn_off_pf_st(fw: Bool, +lp: Spec.Pu, +rnd: Nat, +s: Nat, +c: Nat)
  -> {Impl.pf_st(fw, lp, False{}, Impl.St{rnd, [], s, c}) == Impl.St{rnd, [], Spec.old_pf_s(fw, lp, s), Spec.old_fwd(fw, c)} : Impl.St}:
  match fw:
    case True{}:
      pn_off_put_st(lp, rnd, s, 1n+c)
    case False{}:
      {==}

def pn_off_dc(gf: Bool, +rnd: Nat, +s: Nat, +c: Nat)
  -> {Impl.dc_ev(gf, Impl.St{rnd, [], s, c}) == Spec.old_dc(gf, c) : List<&2, Spec.Ev>}:
  match gf:
    case True{}:
      {==}
    case False{}:
      {==}

def pn_off_dc_st(gf: Bool, +rnd: Nat, +s: Nat, +c: Nat)
  -> {Impl.dc_st(gf, Impl.St{rnd, [], s, c}) == Impl.St{rnd, [], s, Spec.old_fwd(gf, c)} : Impl.St}:
  match gf:
    case True{}:
      {==}
    case False{}:
      {==}

def pn_sd_off(fw: Bool) -> {Impl.sd(fw, False{}) == False{} : Bool}:
  match fw:
    case True{}:
      {==}
    case False{}:
      {==}

def shape(a1: List<&2, Spec.Ev>, a2: List<&2, Spec.Ev>, a3: List<&2, Spec.Ev>, a4: List<&2, Spec.Ev>, a5: List<&2, Spec.Ev>,
    a6: List<&2, Spec.Ev>, a7: List<&2, Spec.Ev>) -> List<&2, Spec.Ev>:
  Spec.Bg{} <> List.append(&2, Spec.Ev, a1, List.append(&2, Spec.Ev, a2, List.append(&2, Spec.Ev, a3,
    List.append(&2, Spec.Ev, a4, List.append(&2, Spec.Ev, a5, List.append(&2, Spec.Ev, a6, a7))))))

def pn_shape(+a1: List<&2, Spec.Ev>, +a2: List<&2, Spec.Ev>, +a3: List<&2, Spec.Ev>, +a4: List<&2, Spec.Ev>,
    +a5: List<&2, Spec.Ev>, +a6: List<&2, Spec.Ev>, +a7: List<&2, Spec.Ev>,
    +b1: List<&2, Spec.Ev>, +b2: List<&2, Spec.Ev>, +b3: List<&2, Spec.Ev>, +b4: List<&2, Spec.Ev>,
    +b5: List<&2, Spec.Ev>, +b6: List<&2, Spec.Ev>, +b7: List<&2, Spec.Ev>,
    e1: {a1 == b1 : List<&2, Spec.Ev>}, e2: {a2 == b2 : List<&2, Spec.Ev>}, e3: {a3 == b3 : List<&2, Spec.Ev>},
    e4: {a4 == b4 : List<&2, Spec.Ev>}, e5: {a5 == b5 : List<&2, Spec.Ev>}, e6: {a6 == b6 : List<&2, Spec.Ev>},
    e7: {a7 == b7 : List<&2, Spec.Ev>})
  -> {shape(a1, a2, a3, a4, a5, a6, a7) == shape(b1, b2, b3, b4, b5, b6, b7) : List<&2, Spec.Ev>}:
  %Equal.sym(List<&2, Spec.Ev>, a1, b1, e1) : {shape(_, a2, a3, a4, a5, a6, a7) == shape(b1, b2, b3, b4, b5, b6, b7) : List<&2, Spec.Ev>}
  %Equal.sym(List<&2, Spec.Ev>, a2, b2, e2) : {shape(b1, _, a3, a4, a5, a6, a7) == shape(b1, b2, b3, b4, b5, b6, b7) : List<&2, Spec.Ev>}
  %Equal.sym(List<&2, Spec.Ev>, a3, b3, e3) : {shape(b1, b2, _, a4, a5, a6, a7) == shape(b1, b2, b3, b4, b5, b6, b7) : List<&2, Spec.Ev>}
  %Equal.sym(List<&2, Spec.Ev>, a4, b4, e4) : {shape(b1, b2, b3, _, a5, a6, a7) == shape(b1, b2, b3, b4, b5, b6, b7) : List<&2, Spec.Ev>}
  %Equal.sym(List<&2, Spec.Ev>, a5, b5, e5) : {shape(b1, b2, b3, b4, _, a6, a7) == shape(b1, b2, b3, b4, b5, b6, b7) : List<&2, Spec.Ev>}
  %Equal.sym(List<&2, Spec.Ev>, a6, b6, e6) : {shape(b1, b2, b3, b4, b5, _, a7) == shape(b1, b2, b3, b4, b5, b6, b7) : List<&2, Spec.Ev>}
  %Equal.sym(List<&2, Spec.Ev>, a7, b7, e7) : {shape(b1, b2, b3, b4, b5, b6, _) == shape(b1, b2, b3, b4, b5, b6, b7) : List<&2, Spec.Ev>}
  {==}

def pn_off_it(+rs: Bool, +fw: Bool, +lp: Spec.Pu, +cp: Spec.Pu, +gf: Bool, +rq: Spec.Pu, +re: Bool, +rnd: Nat, +s: Nat, +c: Nat)
  -> {Impl.it_ev(Spec.Iter{rs, fw, lp, cp, False{}, gf, rq, re}, Impl.St{rnd, [], s, c}) ==
      Spec.old_it_ev(Spec.Iter{rs, fw, lp, cp, False{}, gf, rq, re}, s, c) : List<&2, Spec.Ev>} &
     {Impl.it_st(Spec.Iter{rs, fw, lp, cp, False{}, gf, rq, re}, Impl.St{rnd, [], s, c}) ==
      Impl.St{1n+rnd, [], Spec.old_it_s(Spec.Iter{rs, fw, lp, cp, False{}, gf, rq, re}, s), Spec.old_it_c(Spec.Iter{rs, fw, lp, cp, False{}, gf, rq, re}, c)} : Impl.St}:
  -L = List<&2, Spec.Ev>
  +S1 = {Impl.St{1n+rnd, [], s, c} : Impl.St}
  +S2 = Impl.rd_st(rs, S1)
  +S3 = Impl.pf_st(fw, lp, False{}, S2)
  +S4 = Impl.ck_st(S3)
  +d = Impl.sd(fw, False{})
  +S5 = Impl.put_st(cp, d, S4)
  +S6 = Impl.dc_st(gf, S5)
  +S7 = Impl.put_st(rq, d, S6)
  +s3 = Spec.old_pf_s(fw, lp, s)
  +c3 = Spec.old_fwd(fw, c)
  +s5 = Spec.old_ns(cp, s3)
  +c6 = Spec.old_fwd(gf, c3)
  +s7 = Spec.old_ns(rq, s5)
  +T3 = {Impl.St{1n+rnd, [], s3, c3} : Impl.St}
  +T5 = {Impl.St{1n+rnd, [], s5, c3} : Impl.St}
  +T6 = {Impl.St{1n+rnd, [], s5, c6} : Impl.St}
  +T7 = {Impl.St{1n+rnd, [], s7, c6} : Impl.St}
  +hd = pn_sd_off(fw)
  +st2 = pn_off_rd_st(rs, 1n+rnd, s, c)
  +st3 = Equal.trans(Impl.St, S3, Impl.pf_st(fw, lp, False{}, S1), T3,
    Equal.cong(Impl.St, Impl.St, z => Impl.pf_st(fw, lp, False{}, z), S2, S1, st2), pn_off_pf_st(fw, lp, 1n+rnd, s, c))
  +st4 = Equal.cong(Impl.St, Impl.St, z => Impl.ck_st(z), S3, T3, st3)
  +st5 = Equal.trans(Impl.St, S5, Impl.put_st(cp, False{}, S4), T5,
    Equal.cong(Bool, Impl.St, z => Impl.put_st(cp, z, S4), d, False{}, hd),
    Equal.trans(Impl.St, Impl.put_st(cp, False{}, S4), Impl.put_st(cp, False{}, T3), T5,
      Equal.cong(Impl.St, Impl.St, z => Impl.put_st(cp, False{}, z), S4, T3, st4), pn_off_put_st(cp, 1n+rnd, s3, c3)))
  +st6 = Equal.trans(Impl.St, S6, Impl.dc_st(gf, T5), T6,
    Equal.cong(Impl.St, Impl.St, z => Impl.dc_st(gf, z), S5, T5, st5), pn_off_dc_st(gf, 1n+rnd, s5, c3))
  +st7 = Equal.trans(Impl.St, S7, Impl.put_st(rq, False{}, S6), T7,
    Equal.cong(Bool, Impl.St, z => Impl.put_st(rq, z, S6), d, False{}, hd),
    Equal.trans(Impl.St, Impl.put_st(rq, False{}, S6), Impl.put_st(rq, False{}, T6), T7,
      Equal.cong(Impl.St, Impl.St, z => Impl.put_st(rq, False{}, z), S6, T6, st6), pn_off_put_st(rq, 1n+rnd, s5, c6)))
  +e2 = Equal.trans(L, Impl.pf_ev(fw, lp, False{}, S2), Impl.pf_ev(fw, lp, False{}, S1), Spec.old_pf(fw, lp, s, c),
    Equal.cong(Impl.St, L, z => Impl.pf_ev(fw, lp, False{}, z), S2, S1, st2), pn_off_pf(fw, lp, 1n+rnd, s, c))
  +e3 = Equal.cong(Impl.St, L, z => Impl.ck_ev(z), S3, T3, st3)
  +e4 = Equal.trans(L, Impl.put_ev(cp, d, S4), Impl.put_ev(cp, False{}, S4), Spec.old_put(cp, s3, c3),
    Equal.cong(Bool, L, z => Impl.put_ev(cp, z, S4), d, False{}, hd),
    Equal.trans(L, Impl.put_ev(cp, False{}, S4), Impl.put_ev(cp, False{}, T3), Spec.old_put(cp, s3, c3),
      Equal.cong(Impl.St, L, z => Impl.put_ev(cp, False{}, z), S4, T3, st4), pn_off_put(cp, 1n+rnd, s3, c3)))
  +e5 = Equal.trans(L, Impl.dc_ev(gf, S5), Impl.dc_ev(gf, T5), Spec.old_dc(gf, c3),
    Equal.cong(Impl.St, L, z => Impl.dc_ev(gf, z), S5, T5, st5), pn_off_dc(gf, 1n+rnd, s5, c3))
  +e6 = Equal.trans(L, Impl.put_ev(rq, d, S6), Impl.put_ev(rq, False{}, S6), Spec.old_put(rq, s5, c6),
    Equal.cong(Bool, L, z => Impl.put_ev(rq, z, S6), d, False{}, hd),
    Equal.trans(L, Impl.put_ev(rq, False{}, S6), Impl.put_ev(rq, False{}, T6), Spec.old_put(rq, s5, c6),
      Equal.cong(Impl.St, L, z => Impl.put_ev(rq, False{}, z), S6, T6, st6), pn_off_put(rq, 1n+rnd, s5, c6)))
  +e7 = Equal.trans(L, Impl.rd_ev(re, S7), Impl.rd_ev(re, T7), Spec.old_rd(re),
    Equal.cong(Impl.St, L, z => Impl.rd_ev(re, z), S7, T7, st7), pn_off_rd(re, 1n+rnd, s7, c6))
  (pn_shape(Impl.rd_ev(rs, S1), Impl.pf_ev(fw, lp, False{}, S2), Impl.ck_ev(S3), Impl.put_ev(cp, d, S4), Impl.dc_ev(gf, S5),
     Impl.put_ev(rq, d, S6), Impl.rd_ev(re, S7),
     Spec.old_rd(rs), Spec.old_pf(fw, lp, s, c), Impl.ck_ev(T3), Spec.old_put(cp, s3, c3), Spec.old_dc(gf, c3),
     Spec.old_put(rq, s5, c6), Spec.old_rd(re),
     pn_off_rd(rs, 1n+rnd, s, c), e2, e3, e4, e5, e6, e7),
   Equal.trans(Impl.St, Impl.rd_st(re, S7), Impl.rd_st(re, T7), T7,
     Equal.cong(Impl.St, Impl.St, z => Impl.rd_st(re, z), S7, T7, st7), pn_off_rd_st(re, 1n+rnd, s7, c6)))

def pn_off_trace(its: List<&2, Spec.It>, +rnd: Nat, +s: Nat, +c: Nat, +off: {Spec.all_off(its) == True{} : Bool})
  -> {Impl.trace(its, Impl.St{rnd, [], s, c}) == Spec.old_trace(its, s, c) : List<&2, Spec.Ev>}:
  match its:
    case Nil{}:
      {==}
    case Spec.Iter{+rs, +fw, +lp, +cp, +en, +gf, +rq, +re} <> +rest:
      match en:
        case True{}:
          Empty.absurd({Impl.trace(Spec.Iter{rs, fw, lp, cp, True{}, gf, rq, re} <> rest, Impl.St{rnd, [], s, c}) ==
            Spec.old_trace(Spec.Iter{rs, fw, lp, cp, True{}, gf, rq, re} <> rest, s, c) : List<&2, Spec.Ev>}, Ar.ar_false_true(off))
        case False{}:
          -L = List<&2, Spec.Ev>
          +it = {Spec.Iter{rs, fw, lp, cp, False{}, gf, rq, re} : Spec.It}
          +st = {Impl.St{rnd, [], s, c} : Impl.St}
          +T = {Impl.St{1n+rnd, [], Spec.old_it_s(it, s), Spec.old_it_c(it, c)} : Impl.St}
          +ee = Pair.fst({Impl.it_ev(it, st) == Spec.old_it_ev(it, s, c) : L}, {Impl.it_st(it, st) == T : Impl.St},
            pn_off_it(rs, fw, lp, cp, gf, rq, re, rnd, s, c))
          +es = Pair.snd({Impl.it_ev(it, st) == Spec.old_it_ev(it, s, c) : L}, {Impl.it_st(it, st) == T : Impl.St},
            pn_off_it(rs, fw, lp, cp, gf, rq, re, rnd, s, c))
          Equal.trans(L, List.append(&2, Spec.Ev, Impl.it_ev(it, st), Impl.trace(rest, Impl.it_st(it, st))),
            List.append(&2, Spec.Ev, Spec.old_it_ev(it, s, c), Impl.trace(rest, Impl.it_st(it, st))),
            List.append(&2, Spec.Ev, Spec.old_it_ev(it, s, c), Spec.old_trace(rest, Spec.old_it_s(it, s), Spec.old_it_c(it, c))),
            Equal.cong(L, L, z => List.append(&2, Spec.Ev, z, Impl.trace(rest, Impl.it_st(it, st))), Impl.it_ev(it, st), Spec.old_it_ev(it, s, c), ee),
            Equal.cong(L, L, z => List.append(&2, Spec.Ev, Spec.old_it_ev(it, s, c), z), Impl.trace(rest, Impl.it_st(it, st)),
              Spec.old_trace(rest, Spec.old_it_s(it, s), Spec.old_it_c(it, c)),
              Equal.trans(L, Impl.trace(rest, Impl.it_st(it, st)), Impl.trace(rest, T), Spec.old_trace(rest, Spec.old_it_s(it, s), Spec.old_it_c(it, c)),
                Equal.cong(Impl.St, L, z => Impl.trace(rest, z), Impl.it_st(it, st), T, es),
                pn_off_trace(rest, 1n+rnd, Spec.old_it_s(it, s), Spec.old_it_c(it, c), off))))

def Laws.defer_off_is_blocking(its, off):
  pn_off_trace(its, 0n, 0n, 0n, off)
