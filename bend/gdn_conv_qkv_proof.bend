# Proofs of gdn_conv_qkv_laws.bend (ext 5111 fused GDN prefill conv index maps against the stock transpose / split /
# contiguous path). Nat lemmas from gemm_m16_partition_proof.bend (Ar).
import Base
import ./gdn_conv_qkv.bend as Impl
import ./gdn_conv_qkv_spec.bend as Spec
import ./gdn_conv_qkv_laws.bend as Laws
import ./gemm_m16_partition_proof.bend as Ar

def Laws.x_src(S, dim, d, t, tb):
  +tl = tb
  +J = Nat.add(Nat.mul(d, S), t)
  Ar.p8_add_cong(Nat.mul(Nat.mod(J, S), dim), Nat.mul(t, dim), Nat.div(J, S), d,
    Equal.cong(Nat, Nat, z => Nat.mul(z, dim), Nat.mod(J, S), t, Ar.p7_mod_r(d, t, S, tl)),
    Ar.p7_div_q(d, t, S, tl))

def Laws.view_q(S, K, V, dim, s, j, jb):
  %Equal.sym(Bool, Nat.is_lt(j, K), True{}, jb)
    : {Impl.dst.pick(_, Nat.is_lt(j, Nat.mul(2n, K)), S, K, dim, s, j) == Nat.add(Nat.mul(s, K), j) : Nat}
  {==}

def Laws.view_k(S, K, V, dim, s, j, jb):
  +d = Nat.add(K, j)
  +K0 = Nat.add(K, 0n)
  +R = Nat.add(Nat.mul(S, K), Nat.add(Nat.mul(s, K), j))
  +f1 = Ar.p6_le_nlt(d, K, Ar.ar_le_add_r(K, j))
  +jb0 = Equal.trans(Bool, Nat.is_lt(j, K0), Nat.is_lt(j, K), True{}, Equal.cong(Nat, Bool, z => Nat.is_lt(j, z), K0, K, Ar.ar_add_zero_r(K)), jb)
  +f2 = Ar.ard_tr(Nat.is_lt(d, Nat.add(K, K0)), Nat.is_lt(j, K0), Ar.ar_lt_add_cancel(K, j, K0), jb0)
  %Equal.sym(Bool, Nat.is_lt(d, K), False{}, f1)
    : {Impl.dst.pick(_, Nat.is_lt(d, Nat.mul(2n, K)), S, K, dim, s, d) == R : Nat}
  %Equal.sym(Bool, Nat.is_lt(d, Nat.mul(2n, K)), True{}, f2)
    : {Impl.dst.pick(False{}, _, S, K, dim, s, d) == R : Nat}
  %Equal.sym(Nat, Nat.sub(d, K), j, Ar.ar_add_sub(K, j))
    : {Nat.add(Nat.add(Nat.mul(S, K), Nat.mul(s, K)), _) == R : Nat}
  Ar.ar_add_assoc(Nat.mul(S, K), Nat.mul(s, K), j)

def Laws.view_v(S, K, V, dim, s, j, shape):
  +K2 = Nat.mul(2n, K)
  +KK = Nat.add(K, K)
  +d = Nat.add(KK, j)
  +A = Nat.mul(Nat.mul(2n, S), K)
  +R = Nat.add(A, Nat.add(Nat.mul(s, V), j))
  +e2 = Equal.cong(Nat, Nat, z => Nat.add(K, z), K, Nat.add(K, 0n), Equal.sym(Nat, Nat.add(K, 0n), K, Ar.ar_add_zero_r(K)))
  +f1 = Ar.p6_le_nlt(d, K, Ar.ar_le_trans(K, KK, d, Ar.ar_le_add_r(K, K), Ar.ar_le_add_r(KK, j)))
  +le2 = Equal.trans(Bool, Nat.is_le(K2, d), Nat.is_le(KK, d), True{},
    Equal.cong(Nat, Bool, z => Nat.is_le(z, d), K2, KK, Equal.sym(Nat, KK, K2, e2)), Ar.ar_le_add_r(KK, j))
  +f2 = Ar.p6_le_nlt(d, K2, le2)
  +ev = Equal.trans(Nat, Impl.vd(dim, K), Nat.sub(Nat.add(K2, V), K2), V,
    Equal.cong(Nat, Nat, z => Nat.sub(z, K2), dim, Nat.add(K2, V), Equal.sym(Nat, Nat.add(K2, V), dim, shape)),
    Ar.ar_add_sub(K2, V))
  +ej = Equal.trans(Nat, Nat.sub(d, K2), Nat.sub(Nat.add(K2, j), K2), j,
    Equal.cong(Nat, Nat, z => Nat.sub(Nat.add(z, j), K2), KK, K2, e2),
    Ar.ar_add_sub(K2, j))
  %Equal.sym(Bool, Nat.is_lt(d, K), False{}, f1)
    : {Impl.dst.pick(_, Nat.is_lt(d, K2), S, K, dim, s, d) == R : Nat}
  %Equal.sym(Bool, Nat.is_lt(d, K2), False{}, f2)
    : {Impl.dst.pick(False{}, _, S, K, dim, s, d) == R : Nat}
  Equal.trans(Nat, Nat.add(Nat.add(A, Nat.mul(s, Impl.vd(dim, K))), Nat.sub(d, K2)), Nat.add(Nat.add(A, Nat.mul(s, V)), j), R,
    Ar.p8_add_cong(Nat.add(A, Nat.mul(s, Impl.vd(dim, K))), Nat.add(A, Nat.mul(s, V)), Nat.sub(d, K2), j,
      Equal.cong(Nat, Nat, z => Nat.add(A, Nat.mul(s, z)), Impl.vd(dim, K), V, ev), ej),
    Ar.ar_add_assoc(A, Nat.mul(s, V), j))

def Laws.views_tile(S, K, V, dim, shape):
  +K2 = Nat.mul(2n, K)
  +S0 = Nat.add(S, 0n)
  +p2 = Equal.trans(Nat, Nat.mul(Nat.mul(2n, S), K), Nat.add(Nat.mul(S, K), Nat.mul(S0, K)), Nat.add(Nat.mul(S, K), Nat.mul(S, K)),
    Ar.ar_mul_add_l(S, S0, K),
    Equal.cong(Nat, Nat, z => Nat.add(Nat.mul(S, K), Nat.mul(z, K)), S0, S, Ar.ar_add_zero_r(S)))
  +ea = Equal.trans(Nat, Nat.mul(Nat.mul(2n, S), K), Nat.mul(2n, Nat.mul(S, K)), Nat.mul(K2, S),
    Ar.ar_mul_assoc(2n, S, K),
    Equal.trans(Nat, Nat.mul(2n, Nat.mul(S, K)), Nat.mul(2n, Nat.mul(K, S)), Nat.mul(K2, S),
      Equal.cong(Nat, Nat, z => Nat.mul(2n, z), Nat.mul(S, K), Nat.mul(K, S), Ar.ar_mul_comm(S, K)),
      Equal.sym(Nat, Nat.mul(K2, S), Nat.mul(2n, Nat.mul(K, S)), Ar.ar_mul_assoc(2n, K, S))))
  +p3 = Equal.trans(Nat, Nat.add(Nat.mul(Nat.mul(2n, S), K), Nat.mul(S, V)), Nat.add(Nat.mul(K2, S), Nat.mul(V, S)), Nat.mul(S, dim),
    Ar.p8_add_cong(Nat.mul(Nat.mul(2n, S), K), Nat.mul(K2, S), Nat.mul(S, V), Nat.mul(V, S), ea, Ar.ar_mul_comm(S, V)),
    Equal.trans(Nat, Nat.add(Nat.mul(K2, S), Nat.mul(V, S)), Nat.mul(Nat.add(K2, V), S), Nat.mul(S, dim),
      Equal.sym(Nat, Nat.mul(Nat.add(K2, V), S), Nat.add(Nat.mul(K2, S), Nat.mul(V, S)), Ar.ar_mul_add_l(K2, V, S)),
      Equal.trans(Nat, Nat.mul(Nat.add(K2, V), S), Nat.mul(S, Nat.add(K2, V)), Nat.mul(S, dim),
        Ar.ar_mul_comm(Nat.add(K2, V), S),
        Equal.cong(Nat, Nat, z => Nat.mul(S, z), Nat.add(K2, V), dim, shape))))
  ({==}, p2, p3)

def Laws.contig_inj(K, V, b, s, j, jb):
  +jl = jb
  (Ar.p7_div_q(s, j, Spec.width(K, V, b), jl), Ar.p7_mod_r(s, j, Spec.width(K, V, b), jl))

def Laws.contig_onto(K, V, b, o, wpos):
  +wp = wpos
  (Ar.ard_dm_eq(o, Spec.width(K, V, b), wp), Ar.ard_dm_lt(o, Spec.width(K, V, b), wp))
