# Proofs of attn_fast_laws.bend (ext 3032 int8 QK^T operands). Finite facts are case splits evaluated by the checker;
# the IMMA and i2f facts reuse the generic lemmas of pattn8i_int_proof.bend (PI) and gemm_m16_partition_proof.bend (Ar).
# The bound P = 32 * (127 * 7) = 28448 is small enough to evaluate; 2^22 and 2^31 stay products (never evaluated).
import Base
import ./attn_fast.bend as Impl
import ./attn_fast_spec.bend as Spec
import ./attn_fast_laws.bend as Laws
import ./pattn8i_int.bend as PImpl
import ./pattn8i_int_spec.bend as Z
import ./pattn8i_int_proof.bend as PI
import ./gemm_m16_partition_proof.bend as Ar

# ---------------------------------------------------------------------------------------------
# 1. K operand (law k_operand): c = 0 .. 7 evaluated

def af_k_operand(c: Nat, lt: {Nat.is_lt(c, 8n) == True{} : Bool})
  -> {Nat.is_lt(Nat.add(Nat.mul(c, 2n), 121n), 256n) == True{} : Bool} & {Spec.s8(Impl.kbyte(c)) == Spec.kval(c) : Z.Z} & {Nat.is_le(Z.mass(Spec.kval(c)), 7n) == True{} : Bool}:
  match c:
    case 0n:
      ({==}, {==}, {==})
    case 1n+c1:
      match c1:
        case 0n:
          ({==}, {==}, {==})
        case 1n+c2:
          match c2:
            case 0n:
              ({==}, {==}, {==})
            case 1n+c3:
              match c3:
                case 0n:
                  ({==}, {==}, {==})
                case 1n+c4:
                  match c4:
                    case 0n:
                      ({==}, {==}, {==})
                    case 1n+c5:
                      match c5:
                        case 0n:
                          ({==}, {==}, {==})
                        case 1n+c6:
                          match c6:
                            case 0n:
                              ({==}, {==}, {==})
                            case 1n+c7:
                              match c7:
                                case 0n:
                                  ({==}, {==}, {==})
                                case 1n+c8:
                                  PI.pi_lt0_absurd({Nat.is_lt(Nat.add(Nat.mul(8n+c8, 2n), 121n), 256n) == True{} : Bool} & {Spec.s8(Impl.kbyte(8n+c8)) == Spec.kval(8n+c8) : Z.Z} & {Nat.is_le(Z.mass(Spec.kval(8n+c8)), 7n) == True{} : Bool}, c8, lt)

def Laws.k_operand(c, lt):
  af_k_operand(c, lt)

# ---------------------------------------------------------------------------------------------
# 2. Slot pairing (laws slot_pairing_q, slot_pairing_k): d, s = 0 .. 31 evaluated

def af_slot_q(d: Nat, lt: {Nat.is_lt(d, 32n) == True{} : Bool})
  -> {Nat.is_lt(Impl.qk8_slot(d), 32n) == True{} : Bool} & {Spec.kdim(Impl.qk8_slot(d)) == d : Nat}:
  match d:
    case 0n:
      ({==}, {==})
    case 1n+d1:
      match d1:
        case 0n:
          ({==}, {==})
        case 1n+d2:
          match d2:
            case 0n:
              ({==}, {==})
            case 1n+d3:
              match d3:
                case 0n:
                  ({==}, {==})
                case 1n+d4:
                  match d4:
                    case 0n:
                      ({==}, {==})
                    case 1n+d5:
                      match d5:
                        case 0n:
                          ({==}, {==})
                        case 1n+d6:
                          match d6:
                            case 0n:
                              ({==}, {==})
                            case 1n+d7:
                              match d7:
                                case 0n:
                                  ({==}, {==})
                                case 1n+d8:
                                  match d8:
                                    case 0n:
                                      ({==}, {==})
                                    case 1n+d9:
                                      match d9:
                                        case 0n:
                                          ({==}, {==})
                                        case 1n+d10:
                                          match d10:
                                            case 0n:
                                              ({==}, {==})
                                            case 1n+d11:
                                              match d11:
                                                case 0n:
                                                  ({==}, {==})
                                                case 1n+d12:
                                                  match d12:
                                                    case 0n:
                                                      ({==}, {==})
                                                    case 1n+d13:
                                                      match d13:
                                                        case 0n:
                                                          ({==}, {==})
                                                        case 1n+d14:
                                                          match d14:
                                                            case 0n:
                                                              ({==}, {==})
                                                            case 1n+d15:
                                                              match d15:
                                                                case 0n:
                                                                  ({==}, {==})
                                                                case 1n+d16:
                                                                  match d16:
                                                                    case 0n:
                                                                      ({==}, {==})
                                                                    case 1n+d17:
                                                                      match d17:
                                                                        case 0n:
                                                                          ({==}, {==})
                                                                        case 1n+d18:
                                                                          match d18:
                                                                            case 0n:
                                                                              ({==}, {==})
                                                                            case 1n+d19:
                                                                              match d19:
                                                                                case 0n:
                                                                                  ({==}, {==})
                                                                                case 1n+d20:
                                                                                  match d20:
                                                                                    case 0n:
                                                                                      ({==}, {==})
                                                                                    case 1n+d21:
                                                                                      match d21:
                                                                                        case 0n:
                                                                                          ({==}, {==})
                                                                                        case 1n+d22:
                                                                                          match d22:
                                                                                            case 0n:
                                                                                              ({==}, {==})
                                                                                            case 1n+d23:
                                                                                              match d23:
                                                                                                case 0n:
                                                                                                  ({==}, {==})
                                                                                                case 1n+d24:
                                                                                                  match d24:
                                                                                                    case 0n:
                                                                                                      ({==}, {==})
                                                                                                    case 1n+d25:
                                                                                                      match d25:
                                                                                                        case 0n:
                                                                                                          ({==}, {==})
                                                                                                        case 1n+d26:
                                                                                                          match d26:
                                                                                                            case 0n:
                                                                                                              ({==}, {==})
                                                                                                            case 1n+d27:
                                                                                                              match d27:
                                                                                                                case 0n:
                                                                                                                  ({==}, {==})
                                                                                                                case 1n+d28:
                                                                                                                  match d28:
                                                                                                                    case 0n:
                                                                                                                      ({==}, {==})
                                                                                                                    case 1n+d29:
                                                                                                                      match d29:
                                                                                                                        case 0n:
                                                                                                                          ({==}, {==})
                                                                                                                        case 1n+d30:
                                                                                                                          match d30:
                                                                                                                            case 0n:
                                                                                                                              ({==}, {==})
                                                                                                                            case 1n+d31:
                                                                                                                              match d31:
                                                                                                                                case 0n:
                                                                                                                                  ({==}, {==})
                                                                                                                                case 1n+d32:
                                                                                                                                  PI.pi_lt0_absurd({Nat.is_lt(Impl.qk8_slot(32n+d32), 32n) == True{} : Bool} & {Spec.kdim(Impl.qk8_slot(32n+d32)) == 32n+d32 : Nat}, d32, lt)

def af_slot_k(s: Nat, lt: {Nat.is_lt(s, 32n) == True{} : Bool})
  -> {Nat.is_lt(Spec.kdim(s), 32n) == True{} : Bool} & {Impl.qk8_slot(Spec.kdim(s)) == s : Nat}:
  match s:
    case 0n:
      ({==}, {==})
    case 1n+s1:
      match s1:
        case 0n:
          ({==}, {==})
        case 1n+s2:
          match s2:
            case 0n:
              ({==}, {==})
            case 1n+s3:
              match s3:
                case 0n:
                  ({==}, {==})
                case 1n+s4:
                  match s4:
                    case 0n:
                      ({==}, {==})
                    case 1n+s5:
                      match s5:
                        case 0n:
                          ({==}, {==})
                        case 1n+s6:
                          match s6:
                            case 0n:
                              ({==}, {==})
                            case 1n+s7:
                              match s7:
                                case 0n:
                                  ({==}, {==})
                                case 1n+s8:
                                  match s8:
                                    case 0n:
                                      ({==}, {==})
                                    case 1n+s9:
                                      match s9:
                                        case 0n:
                                          ({==}, {==})
                                        case 1n+s10:
                                          match s10:
                                            case 0n:
                                              ({==}, {==})
                                            case 1n+s11:
                                              match s11:
                                                case 0n:
                                                  ({==}, {==})
                                                case 1n+s12:
                                                  match s12:
                                                    case 0n:
                                                      ({==}, {==})
                                                    case 1n+s13:
                                                      match s13:
                                                        case 0n:
                                                          ({==}, {==})
                                                        case 1n+s14:
                                                          match s14:
                                                            case 0n:
                                                              ({==}, {==})
                                                            case 1n+s15:
                                                              match s15:
                                                                case 0n:
                                                                  ({==}, {==})
                                                                case 1n+s16:
                                                                  match s16:
                                                                    case 0n:
                                                                      ({==}, {==})
                                                                    case 1n+s17:
                                                                      match s17:
                                                                        case 0n:
                                                                          ({==}, {==})
                                                                        case 1n+s18:
                                                                          match s18:
                                                                            case 0n:
                                                                              ({==}, {==})
                                                                            case 1n+s19:
                                                                              match s19:
                                                                                case 0n:
                                                                                  ({==}, {==})
                                                                                case 1n+s20:
                                                                                  match s20:
                                                                                    case 0n:
                                                                                      ({==}, {==})
                                                                                    case 1n+s21:
                                                                                      match s21:
                                                                                        case 0n:
                                                                                          ({==}, {==})
                                                                                        case 1n+s22:
                                                                                          match s22:
                                                                                            case 0n:
                                                                                              ({==}, {==})
                                                                                            case 1n+s23:
                                                                                              match s23:
                                                                                                case 0n:
                                                                                                  ({==}, {==})
                                                                                                case 1n+s24:
                                                                                                  match s24:
                                                                                                    case 0n:
                                                                                                      ({==}, {==})
                                                                                                    case 1n+s25:
                                                                                                      match s25:
                                                                                                        case 0n:
                                                                                                          ({==}, {==})
                                                                                                        case 1n+s26:
                                                                                                          match s26:
                                                                                                            case 0n:
                                                                                                              ({==}, {==})
                                                                                                            case 1n+s27:
                                                                                                              match s27:
                                                                                                                case 0n:
                                                                                                                  ({==}, {==})
                                                                                                                case 1n+s28:
                                                                                                                  match s28:
                                                                                                                    case 0n:
                                                                                                                      ({==}, {==})
                                                                                                                    case 1n+s29:
                                                                                                                      match s29:
                                                                                                                        case 0n:
                                                                                                                          ({==}, {==})
                                                                                                                        case 1n+s30:
                                                                                                                          match s30:
                                                                                                                            case 0n:
                                                                                                                              ({==}, {==})
                                                                                                                            case 1n+s31:
                                                                                                                              match s31:
                                                                                                                                case 0n:
                                                                                                                                  ({==}, {==})
                                                                                                                                case 1n+s32:
                                                                                                                                  PI.pi_lt0_absurd({Nat.is_lt(Spec.kdim(32n+s32), 32n) == True{} : Bool} & {Impl.qk8_slot(Spec.kdim(32n+s32)) == 32n+s32 : Nat}, s32, lt)

def Laws.slot_pairing_q(d, lt):
  af_slot_q(d, lt)

def Laws.slot_pairing_k(s, lt):
  af_slot_k(s, lt)

# ---------------------------------------------------------------------------------------------
# 3, 4. The group IMMA and i2f_exact (laws group_dot, group_i2f)

# 32 bounded pairs (|q| <= 127, |k| <= 7): mass(dot) <= 32 * (127 * 7)
def af_group_mass(+q: List<&2, Z.Z>, +k: List<&2, Z.Z>, ok: {Z.zip_ok(q, k, 127n, 7n) == True{} : Bool},
    n32: {Nat.is_eq(Z.len(k), 32n) == True{} : Bool})
  -> {Nat.is_le(Z.mass(Z.dot(q, k)), Nat.mul(32n, Nat.mul(127n, 7n))) == True{} : Bool}:
  PI.pi_le_rw(Z.mass(Z.dot(q, k)), Z.mass(Z.dot(q, k)), Nat.mul(Z.len(k), Nat.mul(127n, 7n)), Nat.mul(32n, Nat.mul(127n, 7n)), {==},
    Equal.cong(Nat, Nat, z => Nat.mul(z, Nat.mul(127n, 7n)), Z.len(k), 32n, Ar.ar_eq_true(Z.len(k), 32n, n32)),
    PI.pi_dot_mass(q, k, 127n, 7n, ok))

def af_group_abs(+q: List<&2, Z.Z>, +k: List<&2, Z.Z>, ok: {Z.zip_ok(q, k, 127n, 7n) == True{} : Bool},
    n32: {Nat.is_eq(Z.len(k), 32n) == True{} : Bool})
  -> {Nat.is_le(Z.abs(Z.dot(q, k)), Nat.mul(32n, Nat.mul(127n, 7n))) == True{} : Bool}:
  Ar.ar_le_trans(Z.abs(Z.dot(q, k)), Z.mass(Z.dot(q, k)), Nat.mul(32n, Nat.mul(127n, 7n)), PI.pi_abs_le_mass_z(Z.dot(q, k)),
    af_group_mass(q, k, ok, n32))

# P < 16 * 2048 (28448 < 32768, evaluated) <= 2048 * 2048
def af_p_lt22() -> {Nat.is_lt(Nat.mul(32n, Nat.mul(127n, 7n)), Nat.mul(2048n, 2048n)) == True{} : Bool}:
  Ar.ar_lt_le_trans(Nat.mul(32n, Nat.mul(127n, 7n)), Nat.mul(16n, 2048n), Nat.mul(2048n, 2048n), {==},
    Ar.ar_le_mul_mono(16n, 2048n, 2048n, {==}))

# P <= INT32_MAX
def af_p_max() -> {Nat.is_le(Nat.mul(32n, Nat.mul(127n, 7n)), Nat.sub(Nat.mul(32768n, 65536n), 1n)) == True{} : Bool}:
  PI.pi_le_max(Nat.mul(32n, Nat.mul(127n, 7n)),
    Ar.ar_lt_trans(Nat.mul(32n, Nat.mul(127n, 7n)), Nat.mul(2048n, 2048n), Nat.mul(32768n, 65536n), af_p_lt22(), PI.pi_22_lt_31()))

def af_group_mma(+q: List<&2, Z.Z>, +k: List<&2, Z.Z>, ok: {Z.zip_ok(q, k, 127n, 7n) == True{} : Bool},
    n32: {Nat.is_eq(Z.len(k), 32n) == True{} : Bool})
  -> Z.zeq(PImpl.mma(Z.Z{0n, 0n}, q, k, Nat.sub(Nat.mul(32768n, 65536n), 1n)), Z.dot(q, k)):
  +m = Nat.sub(Nat.mul(32768n, 65536n), 1n)
  +P = Nat.mul(32n, Nat.mul(127n, 7n))
  +d = Z.dot(q, k)
  +ba = PI.pi_le_rw(Z.abs(d), Z.abs(Z.zadd(Z.Z{0n, 0n}, d)), m, m,
    Equal.cong(Z.Z, Nat, z => Z.abs(z), d, Z.zadd(Z.Z{0n, 0n}, d), Equal.sym(Z.Z, Z.zadd(Z.Z{0n, 0n}, d), d, PI.pi_zadd_zero_l(d))),
    {==}, Ar.ar_le_trans(Z.abs(d), P, m, af_group_abs(q, k, ok, n32), af_p_max()))
  PI.pi_zeq_eq_r(PImpl.mma(Z.Z{0n, 0n}, q, k, m), Z.zadd(Z.Z{0n, 0n}, d), d, PI.pi_zadd_zero_l(d),
    PI.pi_mma_wrap(Z.Z{0n, 0n}, q, k, m, PI.pi_lt_succ_of_le(Z.abs(Z.zadd(Z.Z{0n, 0n}, d)), m, ba)))

def Laws.group_dot(q, k, ok, n32):
  (af_group_mma(q, k, ok, n32), af_group_abs(q, k, ok, n32), af_p_lt22())

# |z| <= P < h gives n <= p + h and p < n + h, the i2f_exact preconditions
def Laws.group_i2f(p, n, small):
  +P = Nat.mul(32n, Nat.mul(127n, 7n))
  +h = Nat.mul(2048n, 2048n)
  +lo = Ar.ar_le_trans(n, Nat.add(p, P), Nat.add(p, h), PI.pi_abs_split2(p, n, P, small),
    Ar.ard_tr(Nat.is_le(Nat.add(p, P), Nat.add(p, h)), Nat.is_le(P, h), Ar.ar_le_add_cancel(p, P, h), Ar.ar_lt_le(P, h, af_p_lt22())))
  +hi = Ar.ar_le_lt_trans(p, Nat.add(n, P), Nat.add(n, h), PI.pi_abs_split1(p, n, P, small),
    Ar.ard_tr(Nat.is_lt(Nat.add(n, P), Nat.add(n, h)), Nat.is_lt(P, h), Ar.ar_lt_add_cancel(n, P, h), af_p_lt22()))
  PI.pi_i2f(p, n, h, {==}, lo, hi)

# ---------------------------------------------------------------------------------------------
# 5. Row-local q operands (laws q_operand_rows, q_scale_rows, q_region)

# g < 8 and qk8_slot(d) < 32 give 32 g + qk8_slot(d) <= 255
def af_inner_le(+g: Nat, +d: Nat, gl: {Nat.is_lt(g, 8n) == True{} : Bool}, dl: {Nat.is_lt(d, 32n) == True{} : Bool})
  -> {Nat.is_le(Nat.add(Nat.mul(g, 32n), Impl.qk8_slot(d)), Nat.add(Nat.mul(7n, 32n), 31n)) == True{} : Bool}:
  +sl = Ar.ard_fst({Nat.is_lt(Impl.qk8_slot(d), 32n) == True{} : Bool}, {Spec.kdim(Impl.qk8_slot(d)) == d : Nat}, af_slot_q(d, dl))
  PI.pi_le_add2(Nat.mul(g, 32n), Nat.mul(7n, 32n), Impl.qk8_slot(d), 31n,
    Ar.ar_le_mul_mono(g, 7n, 32n, PI.pi_le_of_lt_succ(g, 7n, gl)), PI.pi_le_of_lt_succ(Impl.qk8_slot(d), 31n, sl))

def af_inner_lt(+g: Nat, +d: Nat, gl: {Nat.is_lt(g, 8n) == True{} : Bool}, dl: {Nat.is_lt(d, 32n) == True{} : Bool})
  -> {Nat.is_lt(Nat.add(Nat.mul(g, 32n), Impl.qk8_slot(d)), 256n) == True{} : Bool}:
  PI.pi_lt_succ_of_le(Nat.add(Nat.mul(g, 32n), Impl.qk8_slot(d)), 255n, af_inner_le(g, d, gl, dl))

def Laws.q_operand_rows(r, g, d, rl, gl, dl):
  +I = Nat.add(Nat.mul(g, 32n), Impl.qk8_slot(d))
  +ilt = af_inner_lt(g, d, gl, dl)
  +small = Ar.ar_lt_le_trans(I, 256n, 272n, ilt, {==})
  +rle = Ar.ar_le_mul_mono(r, 47n, 272n, PI.pi_le_of_lt_succ(r, 47n, rl))
  +sum = PI.pi_le_add2(I, 255n, Nat.mul(r, 272n), Nat.mul(47n, 272n), af_inner_le(g, d, gl, dl), rle)
  (PI.pi_div_r(r, I, 272n, small), PI.pi_mod_r(r, I, 272n, small), ilt,
   Ar.ar_le_lt_trans(Impl.qaddr(r, g, d), Nat.add(255n, Nat.mul(47n, 272n)), Nat.mul(48n, 272n), sum, {==}))

def Laws.q_scale_rows(g, r, rl, gl):
  +gle = Ar.ar_le_mul_mono(g, 7n, 48n, PI.pi_le_of_lt_succ(g, 7n, gl))
  +sum = PI.pi_le_add2(r, 47n, Nat.mul(g, 48n), Nat.mul(7n, 48n), PI.pi_le_of_lt_succ(r, 47n, rl), gle)
  (PI.pi_mod_r(g, r, 48n, rl), PI.pi_div_r(g, r, 48n, rl),
   Ar.ar_le_lt_trans(Impl.qsidx(g, r), Nat.add(47n, Nat.mul(7n, 48n)), Nat.mul(8n, 48n), sum, {==}))

def Laws.q_region():
  {==}

# ---------------------------------------------------------------------------------------------
# 6. absmax partners (law amax_partners): t = 0 .. 3 evaluated, gid symbolic (pi_div_r)

def af_amax(+gid: Nat, t: Nat, tl: {Nat.is_lt(t, 4n) == True{} : Bool})
  -> {Nat.div(Impl.partner(gid, t, 1n), 4n) == gid : Nat} & {Nat.div(Impl.partner(gid, t, 2n), 4n) == gid : Nat} & {Spec.distinct4(t, Impl.txor(t, 1n), Impl.txor(t, 2n), Impl.txor(Impl.txor(t, 1n), 2n)) == True{} : Bool}:
  match t:
    case 0n:
      (PI.pi_div_r(gid, Impl.txor(0n, 1n), 4n, {==}), PI.pi_div_r(gid, Impl.txor(0n, 2n), 4n, {==}), {==})
    case 1n+t1:
      match t1:
        case 0n:
          (PI.pi_div_r(gid, Impl.txor(1n, 1n), 4n, {==}), PI.pi_div_r(gid, Impl.txor(1n, 2n), 4n, {==}), {==})
        case 1n+t2:
          match t2:
            case 0n:
              (PI.pi_div_r(gid, Impl.txor(2n, 1n), 4n, {==}), PI.pi_div_r(gid, Impl.txor(2n, 2n), 4n, {==}), {==})
            case 1n+t3:
              match t3:
                case 0n:
                  (PI.pi_div_r(gid, Impl.txor(3n, 1n), 4n, {==}), PI.pi_div_r(gid, Impl.txor(3n, 2n), 4n, {==}), {==})
                case 1n+t4:
                  PI.pi_lt0_absurd({Nat.div(Impl.partner(gid, 4n+t4, 1n), 4n) == gid : Nat} & {Nat.div(Impl.partner(gid, 4n+t4, 2n), 4n) == gid : Nat} & {Spec.distinct4(4n+t4, Impl.txor(4n+t4, 1n), Impl.txor(4n+t4, 2n), Impl.txor(Impl.txor(4n+t4, 1n), 2n)) == True{} : Bool}, t4, tl)

def Laws.amax_partners(gid, t, tl):
  af_amax(gid, t, tl)
