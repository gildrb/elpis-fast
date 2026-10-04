# Obligations for the ext 3020 8-warp prefill attention kernel (pattn8_sched.bend) against the ext 3010 6-warp kernel it
# replaces (pattn_sched.bend, per-warp trace in pattn8_sched_spec.bend). Together they say: every 3010 (kv head, warp,
# query block) work unit is run by exactly one 3020 (kv head, pair, chunk, warp); that warp sees the same q head, the
# same 16 positions, the same absolute row bounds, the same tile iterations in the same order, the same mask state
# and the same kept keys per tile as the 3010 warp; so, with the per-warp code below the warp mapping textually equal
# (pattn8_sched_diff.py), every output element combines the same values in the same order with the same rounding
# points (bit-identical outputs). Domain: q_len rows over L >= q_len keys; pair < 3, warp < 8 (grid K8:40, block
# K8:80). Floating point is outside this model: values are opaque, only which / where / in what order is modelled.
import Base
import ./pattn8_sched.bend as P8
import ./pattn8_sched_spec.bend as Spec
import ./pattn_sched.bend as P6
import ./pattn_sched_spec.bend as S6

# 1. Warp map, forward: warp w < 8 of pair < 3 is 3010 warp 2 pair + w / 4 (< 6) of the same kv head, computing the same
# q head, on 3010's query block 4 qc + w % 4 (its first position is that block's first position).
law warp_map:
  for +kvh: Nat
  for +pair: Nat
  for +qc: Nat
  for +w: Nat
  for pb: {Nat.is_lt(pair, 3n) == True{} : Bool}
  for wlt: {Nat.is_lt(w, 8n) == True{} : Bool}
  {Nat.is_lt(Nat.add(Nat.mul(pair, 2n), P8.wh(w)), 6n) == True{} : Bool} &
  {P8.q_head(kvh, pair, w) == P6.q_head(kvh, Nat.add(Nat.mul(pair, 2n), P8.wh(w))) : Nat} &
  {P8.p0(qc, P8.wb(w)) == P6.q_pos(Nat.add(Nat.mul(qc, 4n), P8.wb(w)), 0n) : Nat}

# 2. Warp map, injective: (pair, w, qc) is recovered from its 3010 image (w6, qb) = (2 pair + w / 4, 4 qc + w % 4).
law warp_map_inj:
  for +pair: Nat
  for +qc: Nat
  for +w: Nat
  for wlt: {Nat.is_lt(w, 8n) == True{} : Bool}
  {Nat.div(Nat.add(Nat.mul(pair, 2n), P8.wh(w)), 2n) == pair : Nat} &
  {Nat.add(Nat.mul(Nat.mod(Nat.add(Nat.mul(pair, 2n), P8.wh(w)), 2n), 4n), Nat.mod(Nat.add(Nat.mul(qc, 4n), P8.wb(w)), 4n)) == w : Nat} &
  {Nat.div(Nat.add(Nat.mul(qc, 4n), P8.wb(w)), 4n) == qc : Nat}

# 3. Warp map, onto: every 3010 warp w6 < 6 of every block qb is the image of pair w6 / 2 (< 3), warp
# 4 (w6 % 2) + qb % 4 (< 8), chunk qb / 4.
law warp_map_onto:
  for +w6: Nat
  for +qb: Nat
  for w6b: {Nat.is_lt(w6, 6n) == True{} : Bool}
  {Nat.is_lt(Nat.div(w6, 2n), 3n) == True{} : Bool} &
  {Nat.is_lt(Nat.add(Nat.mul(Nat.mod(w6, 2n), 4n), Nat.mod(qb, 4n)), 8n) == True{} : Bool} &
  {Nat.add(Nat.mul(Nat.div(w6, 2n), 2n), P8.wh(Nat.add(Nat.mul(Nat.mod(w6, 2n), 4n), Nat.mod(qb, 4n)))) == w6 : Nat} &
  {Nat.add(Nat.mul(Nat.div(qb, 4n), 4n), P8.wb(Nat.add(Nat.mul(Nat.mod(w6, 2n), 4n), Nat.mod(qb, 4n)))) == qb : Nat}

# 4. Grid: every block 3010 launches (16 qb < q_len, 3010 K:60 grid x = ceil(q_len / 16)) lies in a chunk 3020 launches
# (qb / 4 < ceil(q_len / 64), K8:38-40).
law grid_cover:
  for +q: Nat
  for +qb: Nat
  for live: {Nat.is_lt(Nat.mul(qb, 16n), q) == True{} : Bool}
  {Nat.is_lt(Nat.div(qb, 4n), P8.nqc(q)) == True{} : Bool}

# 5. A warp whose 16 positions all lie at or past q_len (no 3010 block) computes nothing and stores nothing (K8:123, 326).
law idle_warp:
  for +L: Nat
  for +q: Nat
  for +qc: Nat
  for +b: Nat
  for dead: {Nat.is_lt(P8.p0(qc, b), q) == False{} : Bool}
  {P8.ntiles(L, q, qc, b) == 0n : Nat}

# 6. Same absolute row bounds: the warp's q_abs0 is 3010's for block 4 qc + b.
law q_abs_eq:
  for +L: Nat
  for +q: Nat
  for +qc: Nat
  for +b: Nat
  {P8.q_abs0(L, q, qc, b) == P6.q_abs0(L, q, Nat.add(Nat.mul(qc, 4n), b)) : Nat}

# 7. Same mask state in every iteration.
law masked_eq:
  for +L: Nat
  for +q: Nat
  for +qc: Nat
  for +b: Nat
  for +it: Nat
  {P8.masked(L, q, qc, b, it) == P6.masked(L, q, Nat.add(Nat.mul(qc, 4n), b), it) : Bool}

# 8. A live warp's own tile count is 3010's block's, and the CTA loop runs at least that many iterations.
law ntiles_eq:
  for +L: Nat
  for +q: Nat
  for +qc: Nat
  for +b: Nat
  for live: {Nat.is_lt(P8.p0(qc, b), q) == True{} : Bool}
  {P8.ntiles(L, q, qc, b) == P6.ntiles(L, q, Nat.add(Nat.mul(qc, 4n), b)) : Nat}

law ntiles_fit:
  for +L: Nat
  for +q: Nat
  for +qc: Nat
  for +b: Nat
  for blt: {Nat.is_lt(b, 4n) == True{} : Bool}
  {Nat.is_le(P8.ntiles(L, q, qc, b), P8.ntiles_cta(L, q, qc)) == True{} : Bool}

# 9. Same iterations in the same order: the iterations of the CTA loop in which a live warp computes are exactly 3010's
# loop 0 .. ntiles - 1 of block 4 qc + b.
law trace_eq:
  for +L: Nat
  for +q: Nat
  for +qc: Nat
  for +b: Nat
  for blt: {Nat.is_lt(b, 4n) == True{} : Bool}
  for live: {Nat.is_lt(P8.p0(qc, b), q) == True{} : Bool}
  {P8.trace(L, q, qc, b) == Spec.trace(L, q, Nat.add(Nat.mul(qc, 4n), b)) : List<&2, Nat>}

# 10. Same kept keys: in every iteration it, key p enters row r of the live warp exactly when it enters row r of 3010's
# block 4 qc + b.
law visits_eq:
  for +L: Nat
  for +q: Nat
  for +qc: Nat
  for +b: Nat
  for +r: Nat
  for +it: Nat
  for +p: Nat
  for blt: {Nat.is_lt(b, 4n) == True{} : Bool}
  for live: {Nat.is_lt(P8.p0(qc, b), q) == True{} : Bool}
  {P8.visits(L, q, qc, b, r, it, p) == P6.visits(L, q, Nat.add(Nat.mul(qc, 4n), b), r, it, p) : Bool}

# 11. Causal coverage of the 8-warp kernel itself: every key a row sees enters it in tile p / 32, and in no other tile.
law attend_once8:
  for +L: Nat
  for +q: Nat
  for +qc: Nat
  for +b: Nat
  for +r: Nat
  for +p: Nat
  for blt: {Nat.is_lt(b, 4n) == True{} : Bool}
  for rb: {Nat.is_lt(r, 16n) == True{} : Bool}
  for row: {Nat.is_lt(Nat.add(P8.p0(qc, b), r), q) == True{} : Bool}
  for len: {Nat.is_le(q, L) == True{} : Bool}
  {S6.once(P8.visits(L, q, qc, b, r, Nat.div(p, 32n), p)) == S6.once(S6.sees(L, q, Nat.add(P8.p0(qc, b), r), p)) : Nat}

law tile_unique8:
  for +L: Nat
  for +q: Nat
  for +qc: Nat
  for +b: Nat
  for +r: Nat
  for +it: Nat
  for +p: Nat
  for blt: {Nat.is_lt(b, 4n) == True{} : Bool}
  for live: {Nat.is_lt(P8.p0(qc, b), q) == True{} : Bool}
  {Bool.and(P8.visits(L, q, qc, b, r, it, p), Nat.is_ne(it, Nat.div(p, 32n))) == False{} : Bool}

# 12. q prologue: smem row w * 16 + pos (the rows warp w's MMA reads, K8:175, 201) is filled from q head
# q_head(kvh, pair, w) at position p0(qc, w % 4) + pos, the warp's own head and positions.
law q_rows:
  for +kvh: Nat
  for +pair: Nat
  for +qc: Nat
  for +w: Nat
  for +pos: Nat
  for posb: {Nat.is_lt(pos, 16n) == True{} : Bool}
  {P8.prow_h(kvh, pair, P8.qrow(w, pos)) == P8.q_head(kvh, pair, w) : Nat} &
  {P8.prow_p(qc, P8.qrow(w, pos)) == Nat.add(P8.p0(qc, P8.wb(w)), pos) : Nat}

# 13. K / V tile issue: with THREADS = T > 0 (256 here, 192 in 3010), thread tid < T in step i issues chunk tid + i T,
# which names (tid, i) back; every chunk c < 1024 is issued by thread c % T in step c / T < ceil(1024 / T). The chunk's
# address depends only on (c, it) (K8:136-139 = 3010 K:228-231), so the staged tile is the same in both kernels.
law issue_once:
  for +T: Nat
  for +tid: Nat
  for +i: Nat
  for +c: Nat
  for tb: {Nat.is_lt(tid, T) == True{} : Bool}
  for cb: {Nat.is_lt(c, 1024n) == True{} : Bool}
  {Nat.div(P8.issue_c(T, tid, i), T) == i : Nat} &
  {Nat.mod(P8.issue_c(T, tid, i), T) == tid : Nat} &
  {P8.issue_c(T, Nat.mod(c, T), Nat.div(c, T)) == c : Nat} &
  {Nat.is_lt(Nat.div(c, T), Nat.div(Nat.sub(Nat.add(1024n, T), 1n), T)) == True{} : Bool}
