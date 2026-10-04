# Proofs of bend/tree_pipe_laws.bend. The decision is split by cases on the
# record check and the readback comparison (helpers taking both booleans with
# their defining equations); the comparison is sound and reflexive; the host
# expectation equals the sequential upload by structural induction; the stage
# kernel's indexed writes equal the upload by shift lemmas (thread i + 1 of a
# list is thread i of its tail) and the record shapes; P6 rewrites the emitted
# ids to the tree model's verify ids and applies spec_inv_tree's T0.
import Base
import ./tree_pipe.bend as Impl
import ./tree_pipe_spec.bend as Spec
import ./tree_pipe_laws.bend as Laws
import ./spec_inv_tree.bend as Tree
import ./spec_inv_tree_laws.bend as TreeLaws
import ./spec_inv_tree_proof.bend as TreeProof
import ./exl3_accept_proof.bend as ChainProof
import ./gemm_m16_partition_proof.bend as Ar

# ---------------------------------------------------------------------------
# The readback comparison

def cons_eq(-a: Nat, -b: Nat, -as: List<&2, Nat>, -bs: List<&2, Nat>, ha: {a == b : Nat},
    hs: {as == bs : List<&2, Nat>}) -> {a <> as == b <> bs : List<&2, Nat>}:
  %ha : {a <> as == _ <> bs : List<&2, Nat>}
  %hs : {a <> as == a <> _ : List<&2, Nat>}
  {==}

def same_sound(+xs: List<&2, Nat>, +ys: List<&2, Nat>, +h: {Impl.same(xs, ys) == True{} : Bool})
  -> {xs == ys : List<&2, Nat>}:
  match xs ys:
    case Nil{} Nil{}:
      {==}
    case Nil{} y <> b:
      Empty.absurd({Nil{} == y <> b : List<&2, Nat>}, ChainProof.false_true(h))
    case x <> a Nil{}:
      Empty.absurd({x <> a == Nil{} : List<&2, Nat>}, ChainProof.false_true(h))
    case x <> a y <> b:
      +hx = Ar.ar_and_l(Nat.is_eq(x, y), Impl.same(a, b), h)
      +hr = Ar.ar_and_r(Nat.is_eq(x, y), Impl.same(a, b), h)
      cons_eq(x, y, a, b, Ar.ar_eq_true(x, y, hx), same_sound(a, b, hr))

def same_refl(+xs: List<&2, Nat>) -> {Impl.same(xs, xs) == True{} : Bool}:
  match xs:
    case Nil{}:
      {==}
    case x <> a:
      Ar.ar_and_intro(Nat.is_eq(x, x), Impl.same(a, a), Ar.ar_eq_refl(x), same_refl(a))

def staged_cong(-a: List<&2, Nat>, -b: List<&2, Nat>, -c: List<&2, Nat>, -d: List<&2, Nat>,
    hab: {a == b : List<&2, Nat>}, hcd: {c == d : List<&2, Nat>})
  -> {Impl.Staged{a, c} == Impl.Staged{b, d} : Impl.Staged}:
  %hab : {Impl.Staged{a, c} == Impl.Staged{_, d} : Impl.Staged}
  %hcd : {Impl.Staged{a, c} == Impl.Staged{a, _} : Impl.Staged}
  {==}

def staged_sound(+v: Impl.Staged, +e: Impl.Staged, +h: {Impl.staged_eq(v, e) == True{} : Bool})
  -> {v == e : Impl.Staged}:
  match v e:
    case Impl.Staged{+vb, +vr} Impl.Staged{+eb, +er}:
      staged_cong(vb, eb, vr, er,
        same_sound(vb, eb, Ar.ar_and_l(Impl.same(vb, eb), Impl.same(vr, er), h)),
        same_sound(vr, er, Ar.ar_and_r(Impl.same(vb, eb), Impl.same(vr, er), h)))

def staged_refl(+v: Impl.Staged) -> {Impl.staged_eq(v, v) == True{} : Bool}:
  match v:
    case Impl.Staged{+vb, +vr}:
      Ar.ar_and_intro(Impl.same(vb, vb), Impl.same(vr, vr), same_refl(vb), same_refl(vr))

# ---------------------------------------------------------------------------
# The host expectation is the sequential upload

def add_pos(+base: Nat, +xs: List<&2, Nat>) -> {Impl.add_base(base, xs) == Spec.positions(base, xs) : List<&2, Nat>}:
  match xs:
    case Nil{}:
      {==}
    case x <> a:
      %add_pos(base, a) : {Nat.add(base, x) <> Impl.add_base(base, a) == Nat.add(base, x) <> _ : List<&2, Nat>}
      {==}

def embed_rows(~emb: Nat -> Nat, +ts: List<&2, Nat>) -> {Impl.embed(~emb, ts) == Spec.rows(~emb, ts) : List<&2, Nat>}:
  match ts:
    case Nil{}:
      {==}
    case t <> a:
      %embed_rows(~emb, a) : {emb(t) <> Impl.embed(~emb, a) == emb(t) <> _ : List<&2, Nat>}
      {==}

def take_depths(+xs: List<&2, Nat>, +n: Nat) -> {List.take(&2, Nat, xs, n) == Spec.depths(n, xs) : List<&2, Nat>}:
  match xs n:
    case Nil{} 0n:
      {==}
    case Nil{} 1n+m:
      {==}
    case x <> a 0n:
      {==}
    case x <> a 1n+m:
      %take_depths(a, m) : {x <> List.take(&2, Nat, a, m) == x <> _ : List<&2, Nat>}
      {==}

def drop_skip(+xs: List<&2, Nat>, +n: Nat) -> {List.drop(&2, Nat, xs, n) == Spec.skip(n, xs) : List<&2, Nat>}:
  match xs n:
    case Nil{} 0n:
      {==}
    case Nil{} 1n+m:
      {==}
    case x <> a 0n:
      {==}
    case x <> a 1n+m:
      drop_skip(a, m)

def expect_upload(~emb: Nat -> Nat, +rec: Impl.Record, +base: Nat)
  -> {Impl.expect(~emb, rec, base) == Spec.upload(~emb, rec, base) : Impl.Staged}:
  match rec:
    case Impl.Record{+toks, +desc}:
      +sk = Spec.skip(16n, desc)
      %Equal.sym(List<&2, Nat>, Impl.embed(~emb, toks), Spec.rows(~emb, toks), embed_rows(~emb, toks))
        : {Impl.Staged{List.append(&2, Nat, desc, Impl.add_base(base, List.take(&2, Nat, List.drop(&2, Nat, desc, 16n), 8n))), _} ==
           Impl.Staged{List.append(&2, Nat, desc, Spec.positions(base, Spec.depths(8n, sk))), Spec.rows(~emb, toks)} : Impl.Staged}
      %Equal.sym(List<&2, Nat>, List.drop(&2, Nat, desc, 16n), sk, drop_skip(desc, 16n))
        : {Impl.Staged{List.append(&2, Nat, desc, Impl.add_base(base, List.take(&2, Nat, _, 8n))), Spec.rows(~emb, toks)} ==
           Impl.Staged{List.append(&2, Nat, desc, Spec.positions(base, Spec.depths(8n, sk))), Spec.rows(~emb, toks)} : Impl.Staged}
      %Equal.sym(List<&2, Nat>, List.take(&2, Nat, sk, 8n), Spec.depths(8n, sk), take_depths(sk, 8n))
        : {Impl.Staged{List.append(&2, Nat, desc, Impl.add_base(base, _)), Spec.rows(~emb, toks)} ==
           Impl.Staged{List.append(&2, Nat, desc, Spec.positions(base, Spec.depths(8n, sk))), Spec.rows(~emb, toks)} : Impl.Staged}
      %Equal.sym(List<&2, Nat>, Impl.add_base(base, Spec.depths(8n, sk)), Spec.positions(base, Spec.depths(8n, sk)),
          add_pos(base, Spec.depths(8n, sk)))
        : {Impl.Staged{List.append(&2, Nat, desc, _), Spec.rows(~emb, toks)} ==
           Impl.Staged{List.append(&2, Nat, desc, Spec.positions(base, Spec.depths(8n, sk))), Spec.rows(~emb, toks)} : Impl.Staged}
      {==}

# ---------------------------------------------------------------------------
# P1 - P3: the decision

def p1_go(~f: Impl.Staged -> List<&2, U32>, +b: Bool, +q: Bool, +v: Impl.Staged, +u: Impl.Staged,
    hq: {Impl.staged_eq(v, u) == q : Bool})
  -> {Laws.emitted(Impl.decide(b, q, f(v), f(u))) == Spec.run(b, f(u)) : Spec.Round}:
  match b:
    case False{}:
      {==}
    case True{}:
      match q:
        case True{}:
          %Equal.sym(Impl.Staged, v, u, staged_sound(v, u, hq)) : {Spec.SeqOk{f(_)} == Spec.SeqOk{f(u)} : Spec.Round}
          {==}
        case False{}:
          {==}

def Laws.pipe_refines_seq(f, ok, emb, rec, v, base):
  +u = Spec.upload(~emb, rec, base)
  %Equal.sym(Impl.Staged, Impl.expect(~emb, rec, base), u, expect_upload(~emb, rec, base))
    : {Laws.emitted(Impl.decide(ok(rec), Impl.staged_eq(v, _), f(v), f(_))) == Spec.run(ok(rec), f(u)) : Spec.Round}
  p1_go(~f, ok(rec), Impl.staged_eq(v, u), v, u, {==})

def p2_go(~f: Impl.Staged -> List<&2, U32>, b: Bool, q: Bool, +v: Impl.Staged, +e: Impl.Staged, -u: Impl.Staged,
    hq: {Impl.staged_eq(v, e) == q : Bool}, hk: {Laws.kept(Impl.decide(b, q, f(v), f(e))) == True{} : Bool},
    heu: {e == u : Impl.Staged})
  -> {v == u : Impl.Staged} & {b == True{} : Bool}:
  match b:
    case False{}:
      Empty.absurd({v == u : Impl.Staged} & {False{} == True{} : Bool}, ChainProof.false_true(hk))
    case True{}:
      match q:
        case True{}:
          (Equal.trans(Impl.Staged, v, e, u, staged_sound(v, e, hq), heu), {==})
        case False{}:
          Empty.absurd({v == u : Impl.Staged} & {True{} == True{} : Bool}, ChainProof.false_true(hk))

def Laws.pipe_kept_checked(f, ok, emb, rec, v, base, k):
  +e = Impl.expect(~emb, rec, base)
  p2_go(~f, ok(rec), Impl.staged_eq(v, e), v, e, Spec.upload(~emb, rec, base), {==}, k, expect_upload(~emb, rec, base))

def p3_go(~f: Impl.Staged -> List<&2, U32>, b: Bool, q: Bool, +v: Impl.Staged, +u: Impl.Staged,
    hq: {Impl.staged_eq(v, u) == q : Bool}, differ: {v == u : Impl.Staged} -> Empty)
  -> {Laws.kept(Impl.decide(b, q, f(v), f(u))) == False{} : Bool}:
  match b:
    case False{}:
      {==}
    case True{}:
      match q:
        case True{}:
          Empty.absurd({Laws.kept(Impl.Kept{f(v)}) == False{} : Bool}, differ(staged_sound(v, u, hq)))
        case False{}:
          {==}

def Laws.pipe_mismatch_discards(f, ok, emb, rec, v, base, differ):
  +u = Spec.upload(~emb, rec, base)
  %Equal.sym(Impl.Staged, Impl.expect(~emb, rec, base), u, expect_upload(~emb, rec, base))
    : {Laws.kept(Impl.decide(ok(rec), Impl.staged_eq(v, _), f(v), f(_))) == False{} : Bool}
  p3_go(~f, ok(rec), Impl.staged_eq(v, u), v, u, {==}, differ)

# ---------------------------------------------------------------------------
# P4: the stage kernel's indexed writes

def cd_shift(+n: Nat, +i: Nat, +x: Nat, +xs: List<&2, Nat>)
  -> {Impl.copy_desc(n, 1n+i, x <> xs) == Impl.copy_desc(n, i, xs) : List<&2, Nat>}:
  match n:
    case 0n:
      {==}
    case 1n+m:
      %cd_shift(m, 1n+i, x, xs) : {Impl.at(xs, i) <> Impl.copy_desc(m, 2n+i, x <> xs) == Impl.at(xs, i) <> _ : List<&2, Nat>}
      {==}

def cd_all(+d: List<&2, Nat>) -> {Impl.copy_desc(List.length(&2, Nat, d), 0n, d) == d : List<&2, Nat>}:
  match d:
    case Nil{}:
      {==}
    case x <> xs:
      %Equal.sym(List<&2, Nat>, Impl.copy_desc(List.length(&2, Nat, xs), 1n, x <> xs),
          Impl.copy_desc(List.length(&2, Nat, xs), 0n, xs), cd_shift(List.length(&2, Nat, xs), 0n, x, xs))
        : {x <> _ == x <> xs : List<&2, Nat>}
      %Equal.sym(List<&2, Nat>, Impl.copy_desc(List.length(&2, Nat, xs), 0n, xs), xs, cd_all(xs))
        : {x <> _ == x <> xs : List<&2, Nat>}
      {==}

def g_shift(~emb: Nat -> Nat, +n: Nat, +i: Nat, +x: Nat, +xs: List<&2, Nat>)
  -> {Impl.gather(~emb, n, 1n+i, x <> xs) == Impl.gather(~emb, n, i, xs) : List<&2, Nat>}:
  match n:
    case 0n:
      {==}
    case 1n+m:
      %g_shift(~emb, m, 1n+i, x, xs) : {emb(Impl.at(xs, i)) <> Impl.gather(~emb, m, 2n+i, x <> xs) == emb(Impl.at(xs, i)) <> _ : List<&2, Nat>}
      {==}

def g_all(~emb: Nat -> Nat, +ts: List<&2, Nat>)
  -> {Impl.gather(~emb, List.length(&2, Nat, ts), 0n, ts) == Spec.rows(~emb, ts) : List<&2, Nat>}:
  match ts:
    case Nil{}:
      {==}
    case x <> xs:
      %Equal.sym(List<&2, Nat>, Impl.gather(~emb, List.length(&2, Nat, xs), 1n, x <> xs),
          Impl.gather(~emb, List.length(&2, Nat, xs), 0n, xs), g_shift(~emb, List.length(&2, Nat, xs), 0n, x, xs))
        : {emb(x) <> _ == emb(x) <> Spec.rows(~emb, xs) : List<&2, Nat>}
      %Equal.sym(List<&2, Nat>, Impl.gather(~emb, List.length(&2, Nat, xs), 0n, xs), Spec.rows(~emb, xs), g_all(~emb, xs))
        : {emb(x) <> _ == emb(x) <> Spec.rows(~emb, xs) : List<&2, Nat>}
      {==}

def wp_shift(+n: Nat, +r: Nat, +x: Nat, +xs: List<&2, Nat>, +base: Nat)
  -> {Impl.write_pos(n, 1n+r, x <> xs, base) == Impl.write_pos(n, r, xs, base) : List<&2, Nat>}:
  match n:
    case 0n:
      {==}
    case 1n+m:
      %wp_shift(m, 1n+r, x, xs, base)
        : {Nat.add(base, Impl.at(xs, r)) <> Impl.write_pos(m, 2n+r, x <> xs, base) == Nat.add(base, Impl.at(xs, r)) <> _ : List<&2, Nat>}
      {==}

def wp_depths(+n: Nat, +dp: List<&2, Nat>, +base: Nat, hb: {Nat.is_le(n, List.length(&2, Nat, dp)) == True{} : Bool})
  -> {Impl.write_pos(n, 0n, dp, base) == Spec.positions(base, Spec.depths(n, dp)) : List<&2, Nat>}:
  match n:
    case 0n:
      {==}
    case 1n+m:
      match dp:
        case Nil{}:
          Empty.absurd({Impl.write_pos(1n+m, 0n, Nil{}, base) == Spec.positions(base, Spec.depths(1n+m, Nil{})) : List<&2, Nat>},
            ChainProof.false_true(hb))
        case x <> xs:
          %Equal.sym(List<&2, Nat>, Impl.write_pos(m, 1n, x <> xs, base), Impl.write_pos(m, 0n, xs, base),
              wp_shift(m, 0n, x, xs, base))
            : {Nat.add(base, x) <> _ == Nat.add(base, x) <> Spec.positions(base, Spec.depths(m, xs)) : List<&2, Nat>}
          %Equal.sym(List<&2, Nat>, Impl.write_pos(m, 0n, xs, base), Spec.positions(base, Spec.depths(m, xs)),
              wp_depths(m, xs, base, hb))
            : {Nat.add(base, x) <> _ == Nat.add(base, x) <> Spec.positions(base, Spec.depths(m, xs)) : List<&2, Nat>}
          {==}

def len_drop(+d: List<&2, Nat>, +k: Nat)
  -> {List.length(&2, Nat, List.drop(&2, Nat, d, k)) == Nat.sub(List.length(&2, Nat, d), k) : Nat}:
  match d k:
    case Nil{} 0n:
      {==}
    case Nil{} 1n+j:
      {==}
    case x <> xs 0n:
      {==}
    case x <> xs 1n+j:
      len_drop(xs, j)

def dp_bound(+desc: List<&2, Nat>, d128: {List.length(&2, Nat, desc) == 128n : Nat})
  -> {Nat.is_le(8n, List.length(&2, Nat, List.drop(&2, Nat, desc, 16n))) == True{} : Bool}:
  %Equal.sym(Nat, List.length(&2, Nat, List.drop(&2, Nat, desc, 16n)), Nat.sub(List.length(&2, Nat, desc), 16n),
      len_drop(desc, 16n))
    : {Nat.is_le(8n, _) == True{} : Bool}
  %Equal.sym(Nat, List.length(&2, Nat, desc), 128n, d128) : {Nat.is_le(8n, Nat.sub(_, 16n)) == True{} : Bool}
  {==}

def cd128(+desc: List<&2, Nat>, d128: {List.length(&2, Nat, desc) == 128n : Nat})
  -> {Impl.copy_desc(128n, 0n, desc) == desc : List<&2, Nat>}:
  %d128 : {Impl.copy_desc(_, 0n, desc) == desc : List<&2, Nat>}
  cd_all(desc)

def g8(~emb: Nat -> Nat, +toks: List<&2, Nat>, t8: {List.length(&2, Nat, toks) == 8n : Nat})
  -> {Impl.gather(~emb, 8n, 0n, toks) == Spec.rows(~emb, toks) : List<&2, Nat>}:
  %t8 : {Impl.gather(~emb, _, 0n, toks) == Spec.rows(~emb, toks) : List<&2, Nat>}
  g_all(~emb, toks)

def wp8(+desc: List<&2, Nat>, +base: Nat, d128: {List.length(&2, Nat, desc) == 128n : Nat})
  -> {Impl.write_pos(8n, 0n, List.drop(&2, Nat, desc, 16n), base) ==
      Spec.positions(base, Spec.depths(8n, Spec.skip(16n, desc))) : List<&2, Nat>}:
  %drop_skip(desc, 16n)
    : {Impl.write_pos(8n, 0n, List.drop(&2, Nat, desc, 16n), base) == Spec.positions(base, Spec.depths(8n, _)) : List<&2, Nat>}
  wp_depths(8n, List.drop(&2, Nat, desc, 16n), base, dp_bound(desc, d128))

def Laws.stage_agree(emb, toks, desc, base, t8, d128):
  +ws = Spec.positions(base, Spec.depths(8n, Spec.skip(16n, desc)))
  %Equal.sym(List<&2, Nat>, Impl.copy_desc(128n, 0n, desc), desc, cd128(desc, d128))
    : {Impl.Staged{List.append(&2, Nat, _, Impl.write_pos(8n, 0n, List.drop(&2, Nat, desc, 16n), base)),
         Impl.gather(~emb, 8n, 0n, toks)} == Impl.Staged{List.append(&2, Nat, desc, ws), Spec.rows(~emb, toks)} : Impl.Staged}
  %Equal.sym(List<&2, Nat>, Impl.write_pos(8n, 0n, List.drop(&2, Nat, desc, 16n), base), ws, wp8(desc, base, d128))
    : {Impl.Staged{List.append(&2, Nat, desc, _), Impl.gather(~emb, 8n, 0n, toks)} ==
       Impl.Staged{List.append(&2, Nat, desc, ws), Spec.rows(~emb, toks)} : Impl.Staged}
  %Equal.sym(List<&2, Nat>, Impl.gather(~emb, 8n, 0n, toks), Spec.rows(~emb, toks), g8(~emb, toks, t8))
    : {Impl.Staged{List.append(&2, Nat, desc, ws), _} == Impl.Staged{List.append(&2, Nat, desc, ws), Spec.rows(~emb, toks)} : Impl.Staged}
  {==}

# ---------------------------------------------------------------------------
# P5: a fault-free round is kept

def p5_go(b: Bool, q: Bool, +x: List<&2, U32>, +y: List<&2, U32>, hb: {b == True{} : Bool}, hq: {q == True{} : Bool})
  -> {Laws.kept(Impl.decide(b, q, x, y)) == True{} : Bool}:
  match b:
    case False{}:
      Empty.absurd({Laws.kept(Impl.decide(False{}, q, x, y)) == True{} : Bool}, ChainProof.false_true(hb))
    case True{}:
      match q:
        case True{}:
          {==}
        case False{}:
          Empty.absurd({Laws.kept(Impl.decide(True{}, False{}, x, y)) == True{} : Bool}, ChainProof.false_true(hq))

def Laws.stage_kept(f, ok, emb, toks, desc, base, t8, d128, accepted):
  +rec = {Impl.Record{toks, desc} : Impl.Record}
  +u = Spec.upload(~emb, rec, base)
  %Equal.sym(Impl.Staged, Impl.stage(~emb, rec, base), u, Laws.stage_agree(~emb, toks, desc, base, t8, d128))
    : {Laws.kept(Impl.decide(ok(rec), Impl.staged_eq(_, Impl.expect(~emb, rec, base)), f(_), f(Impl.expect(~emb, rec, base)))) == True{} : Bool}
  %Equal.sym(Impl.Staged, Impl.expect(~emb, rec, base), u, expect_upload(~emb, rec, base))
    : {Laws.kept(Impl.decide(ok(rec), Impl.staged_eq(u, _), f(u), f(_))) == True{} : Bool}
  p5_go(ok(rec), Impl.staged_eq(u, u), f(u), f(u), accepted, staged_refl(u))

# ---------------------------------------------------------------------------
# P6: refinement of spec_inv_tree

def run_true(b: Bool, +ids: List<&2, U32>, hb: {b == True{} : Bool}) -> {Spec.run(b, ids) == Spec.SeqOk{ids} : Spec.Round}:
  match b:
    case True{}:
      {==}
    case False{}:
      Empty.absurd({Spec.run(False{}, ids) == Spec.SeqOk{ids} : Spec.Round}, ChainProof.false_true(hb))

def Laws.pipe_round_commits_greedy(row, rinv, f, ok, emb, rec, v, base, seq, t, stops, bp, checkpoint, legal, accepted, conf):
  +u = Spec.upload(~emb, rec, base)
  +out = Laws.emitted(Impl.pipe_round(~f, ~ok, ~emb, rec, v, base))
  +tv = Tree.verify(~row, seq, Tree.nodes(t))
  +s3 = Equal.trans(Spec.Round, out, Spec.seq_round(~f, ~ok, ~emb, rec, base), Spec.SeqOk{f(u)},
    Laws.pipe_refines_seq(~f, ~ok, ~emb, rec, v, base), run_true(ok(rec), f(u), accepted))
  +s4 = Equal.cong(Spec.Round, List<&2, U32>, r => Laws.ids_of(r), out, Spec.SeqOk{f(u)}, s3)
  +s5 = Equal.trans(List<&2, U32>, Laws.ids_of(out), f(u), tv, s4, conf)
  %Equal.sym(List<&2, U32>, Laws.ids_of(out), tv, s5) : Laws.commits_greedy(~row, seq, t, stops, bp, checkpoint, _)
  TreeLaws.tree_round_commits_greedy(~row, ~rinv, seq, t, stops, bp, checkpoint, legal)

# ---------------------------------------------------------------------------
# P7: mutation witness

def Laws.unchecked_differs():
  ({==}, {==}, {==})
