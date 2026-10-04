# Finite differential table of the grouped m16 GEMM model (bend/gemm_m16_group.bend, ext patch
# 2102), compared byte for byte by bend/gemm_m16_group_diff.py against the patch's index
# expressions compiled for the CPU. usage: GEMM_M16_GROUP_TABLE m k G n0 [n1 [n2 [n3]]]
# Rows (every value is an Impl def; the printer only enumerates indices):
#   S ns= k= G= m= groups= KT= total= mc= fits= shape= gbase= jobs= tw=
#   M g mat= gb= local= nmat= lo= hi= nc= base= fl= ff=     g < groups (column map, slots)
#   B b s= e= segs=                                        b < G (block range, main loop)
#   W b w first= count= last=                              b < G, warp < 8 (finisher loop)
#   J x g= r= ch= mat= col0= out=                          x < jobs (finisher task)
#   Q b w first= count= last=                              b < G, warp < 8 (prologue loop)
#   P w mat= r= xh= a= suh=                                w < total_warps (prologue job)
#   I b warp lane cs= end=g,kt,mat,off,rb                  b < G, (warp, lane) in {(0,0), (7,31)}:
#     weight issue stream, cs = sum_{j < n_iter} (mat_j * 1000003 + off_j) * (j + 1) over the
#     pointer before issue j, end = cursor and pointer after the block's n_iter issues
# Sections are balanced trees of parallel calls over index ranges [base, base + 2^d).
import Base
import ./gemm_m16_group.bend as Impl
import ./gemm_m16_partition.bend as V2
import ./gemm_m16_partition_spec.bend as V2Spec
import ./gemm_m16_group_spec.bend as Spec

def num(+n: Nat) -> String:
  Nat.show(n)

def bit(b: Bool) -> String:
  match b:
    case True{}:
      "1"
    case False{}:
      "0"

def join(xs: List<&2, Nat>) -> String:
  match xs:
    case Nil{}:
      ""
    case +x <> rest:
      match rest:
        case Nil{}:
          num(x)
        case _:
          num(x) ++ "," ++ join(rest)

type Cfg is Data:
  Cfg{ns: List<&2, Nat>, k: Nat, G: Nat, m: Nat, KT: Nat, total: Nat, mc: Nat, jobs: Nat, tw: Nat}

def c_ns(+c: Cfg) -> List<&2, Nat>:
  match c:
    case Cfg{ns, _, _, _, _, _, _, _, _}:
      ns

def c_k(+c: Cfg) -> Nat:
  match c:
    case Cfg{_, k, _, _, _, _, _, _, _}:
      k

def c_G(+c: Cfg) -> Nat:
  match c:
    case Cfg{_, _, G, _, _, _, _, _, _}:
      G

def c_m(+c: Cfg) -> Nat:
  match c:
    case Cfg{_, _, _, m, _, _, _, _, _}:
      m

def c_KT(+c: Cfg) -> Nat:
  match c:
    case Cfg{_, _, _, _, KT, _, _, _, _}:
      KT

def c_total(+c: Cfg) -> Nat:
  match c:
    case Cfg{_, _, _, _, _, total, _, _, _}:
      total

def c_mc(+c: Cfg) -> Nat:
  match c:
    case Cfg{_, _, _, _, _, _, mc, _, _}:
      mc

def c_jobs(+c: Cfg) -> Nat:
  match c:
    case Cfg{_, _, _, _, _, _, _, jobs, _}:
      jobs

def c_tw(+c: Cfg) -> Nat:
  match c:
    case Cfg{_, _, _, _, _, _, _, _, tw}:
      tw

# ---- main loop segments (as in the 2001 v2 table): tiles since the last flush form one run
# (group of its first tile, first it, last it, ok while every tile has that group and it = last + 1);
# Flush{h} prints "h:first-last" (ok and group == h), "h:?" (not), "h:-" (no tiles); tiles after
# the last flush print "~first-last". Items joined by ','.
type Run is Data:
  NoRun{}
  Run{g: Nat, first: Nat, last: Nat, ok: Bool}

def run_tile(run: Run, +h: Nat, +it: Nat) -> Run:
  match run:
    case NoRun{}:
      Run{h, it, it, True{}}
    case Run{+g, f, l, ok}:
      Run{g, f, it, Bool.and(ok, Bool.and(Nat.is_eq(h, g), Nat.is_eq(it, 1n+l)))}

def flush_item(good: Bool, +h: Nat, f: Nat, l: Nat) -> String:
  match good:
    case True{}:
      num(h) ++ ":" ++ num(f) ++ "-" ++ num(l)
    case False{}:
      num(h) ++ ":?"

def run_flush(run: Run, +h: Nat) -> String:
  match run:
    case NoRun{}:
      num(h) ++ ":-"
    case Run{g, f, l, ok}:
      flush_item(Bool.and(ok, Nat.is_eq(g, h)), h, f, l)

def run_tail(run: Run, sep: String) -> String:
  match run:
    case NoRun{}:
      ""
    case Run{_, f, l, _}:
      sep ++ "~" ++ num(f) ++ "-" ++ num(l)

def segs(evs: List<&2, V2Spec.Ev>, run: Run, sep: String) -> String:
  match evs:
    case Nil{}:
      run_tail(run, sep)
    case ev <> rest:
      match ev:
        case V2Spec.Tile{h, it}:
          segs(rest, run_tile(run, h, it), sep)
        case V2Spec.Flush{h}:
          sep ++ run_flush(run, h) ++ segs(rest, NoRun{}, ",")

# bits(m) = number of binary digits of m; depth(n) = bits(n - 1): smallest d with n <= 2^d
def bits.pick(zero: Bool, rest: Nat) -> Nat:
  match zero:
    case True{}:
      0n
    case False{}:
      1n+rest

def bits.go(fuel: Nat, +m: Nat) -> Nat:
  match fuel:
    case 0n:
      0n
    case 1n+f:
      bits.pick(Nat.is_eq(m, 0n), bits.go(f, Nat.div(m, 2n)))

def depth(+n: Nat) -> Nat:
  bits.go(n, Nat.sub(n, 1n))

def wgo(xs: List<&2, Nat>, cnt: Nat, last: Nat) -> String:
  match xs:
    case Nil{}:
      " count=" ++ num(cnt) ++ " last=" ++ num(last)
    case x <> rest:
      wgo(rest, 1n+cnt, x)

def wstats(xs: List<&2, Nat>) -> String:
  match xs:
    case Nil{}:
      "first=- count=0 last=-"
    case +x <> rest:
      "first=" ++ num(x) ++ wgo(rest, 1n, x)

# ---- rows
def row_m(+c: Cfg, +g: Nat) -> String:
  +ns = c_ns(c)
  +KT = c_KT(c)
  +total = c_total(c)
  +G = c_G(c)
  +mc = c_mc(c)
  +lo = Impl.b_lo(g, KT, total, G)
  +hi = Impl.b_hi(g, KT, total, G)
  +nc = Impl.nc(g, KT, total, G)
  +mr = Nat.sub(c_m(c), 1n)
  "M " ++ num(g) ++ " mat=" ++ num(Impl.mat(ns, g)) ++ " gb=" ++ num(Impl.gb(ns, g)) ++
    " local=" ++ num(Impl.local(ns, g)) ++ " nmat=" ++ num(Impl.nmat(ns, g)) ++ " lo=" ++ num(lo) ++
    " hi=" ++ num(hi) ++ " nc=" ++ num(nc) ++ " base=" ++ num(Impl.address(g, mc, lo, lo)) ++
    " fl=" ++ num(Impl.flush_float(Impl.address(g, mc, hi, lo), mr, V2.col(7n, 3n, 1n, 7n))) ++
    " ff=" ++ num(Impl.finish_float(g, mc, Nat.sub(nc, 1n), mr, 3n, 31n, 3n)) ++ "\n"

def row_b(+c: Cfg, +b: Nat) -> String:
  +total = c_total(c)
  +G = c_G(c)
  +s = Impl.start(b, total, G)
  +e = Impl.start(1n+b, total, G)
  "B " ++ num(b) ++ " s=" ++ num(s) ++ " e=" ++ num(e) ++ " segs=" ++
    segs(Impl.mainloop(s, e, c_KT(c)), NoRun{}, "") ++ "\n"

def row_w(+c: Cfg, +i: Nat) -> String:
  +b = Nat.div(i, 8n)
  +w = Nat.mod(i, 8n)
  "W " ++ num(b) ++ " " ++ num(w) ++ " " ++ wstats(Impl.jobloop(b, w, c_G(c), c_jobs(c))) ++ "\n"

def job_str(j: V2Spec.Job) -> String:
  match j:
    case V2Spec.Job{g, r, ch}:
      " g=" ++ num(g) ++ " r=" ++ num(r) ++ " ch=" ++ num(ch)

def place_str(p: Spec.Place) -> String:
  match p:
    case Spec.Place{mat, r, col0}:
      " mat=" ++ num(mat) ++ " col0=" ++ num(col0)

def out_str(j: V2Spec.Job, +ns: List<&2, Nat>) -> String:
  match j:
    case V2Spec.Job{g, r, ch}:
      " out=" ++ num(Impl.out_off(ns, g, r, ch))

def row_j(+c: Cfg, +x: Nat) -> String:
  +m = c_m(c)
  +ns = c_ns(c)
  "J " ++ num(x) ++ job_str(Impl.job(x, m)) ++ place_str(Impl.place(ns, x, m)) ++
    out_str(Impl.job(x, m), ns) ++ "\n"

def row_q(+c: Cfg, +i: Nat) -> String:
  +b = Nat.div(i, 8n)
  +w = Nat.mod(i, 8n)
  "Q " ++ num(b) ++ " " ++ num(w) ++ " " ++ wstats(Impl.pro_loop(b, w, c_G(c), c_tw(c))) ++ "\n"

def row_p(+c: Cfg, +w: Nat) -> String:
  +m = c_m(c)
  +k = c_k(c)
  +pm = Impl.per_mat(m, k)
  "P " ++ num(w) ++ " mat=" ++ num(Impl.pro_mat(w, pm)) ++ " r=" ++ num(Impl.pro_r(w, pm)) ++
    " xh=" ++ num(Impl.pro_xh(m, k, w)) ++ " a=" ++ num(Impl.pro_a(m, k, w)) ++
    " suh=" ++ num(Impl.pro_suh(m, k, w)) ++ "\n"

def iss_term(st: Impl.Iss, +j: Nat) -> Nat:
  match st:
    case Impl.Iss{c, p}:
      match p:
        case Impl.WPtr{mat, off, rb}:
          Nat.mul(Nat.add(Nat.mul(mat, 1000003n), off), 1n+j)

def iss_end(st: Impl.Iss) -> String:
  match st:
    case Impl.Iss{c, p}:
      match c p:
        case V2.Cursor{g, kt} Impl.WPtr{mat, off, rb}:
          num(g) ++ "," ++ num(kt) ++ "," ++ num(mat) ++ "," ++ num(off) ++ "," ++ num(rb)

def cs_leaf(inside: Bool, +j: Nat, +c: Cfg, +s: Nat, +n: Nat, +warp: Nat, +lane: Nat) -> Nat:
  match inside:
    case True{}:
      iss_term(Impl.issued(j, s, n, c_KT(c), c_ns(c), warp, lane), j)
    case False{}:
      0n

def cs_tree(+d: Nat, +base: Nat, +c: Cfg, +s: Nat, +n: Nat, +warp: Nat, +lane: Nat) -> Nat:
  match d:
    case 0n:
      cs_leaf(Nat.is_lt(base, n), base, c, s, n, warp, lane)
    case 1n+p:
      +half = Nat.pow(2n, p)
      l r = cs_tree(p, base, c, s, n, warp, lane) cs_tree(p, Nat.add(base, half), c, s, n, warp, lane)
      Nat.add(l, r)

def row_i(+c: Cfg, +i: Nat) -> String:
  +b = Nat.div(i, 2n)
  +sel = Nat.mod(i, 2n)
  +warp = Nat.mul(sel, 7n)
  +lane = Nat.mul(sel, 31n)
  +s = Impl.start(b, c_total(c), c_G(c))
  +n = Nat.sub(Impl.start(1n+b, c_total(c), c_G(c)), s)
  "I " ++ num(b) ++ " " ++ num(warp) ++ " " ++ num(lane) ++ " cs=" ++
    num(cs_tree(depth(n), 0n, c, s, n, warp, lane)) ++ " end=" ++
    iss_end(Impl.issued(n, s, n, c_KT(c), c_ns(c), warp, lane)) ++ "\n"

type Sec is Data:
  KM{}
  KB{}
  KW{}
  KJ{}
  KQ{}
  KP{}
  KI{}

def row(kind: Sec, +c: Cfg, +i: Nat) -> String:
  match kind:
    case KM{}:
      row_m(c, i)
    case KB{}:
      row_b(c, i)
    case KW{}:
      row_w(c, i)
    case KJ{}:
      row_j(c, i)
    case KQ{}:
      row_q(c, i)
    case KP{}:
      row_p(c, i)
    case KI{}:
      row_i(c, i)

def leaf(inside: Bool, +kind: Sec, +c: Cfg, +i: Nat) -> String:
  match inside:
    case True{}:
      row(kind, c, i)
    case False{}:
      ""

def tree(+kind: Sec, +c: Cfg, +d: Nat, +base: Nat, +n: Nat) -> String:
  match d:
    case 0n:
      leaf(Nat.is_lt(base, n), kind, c, base)
    case 1n+p:
      +half = Nat.pow(2n, p)
      left right = tree(kind, c, p, base, n) tree(kind, c, p, Nat.add(base, half), n)
      String.append(left, right)

def section(+kind: Sec, +c: Cfg, +n: Nat) -> String:
  tree(kind, c, depth(n), 0n, n)

def rows(+c: Cfg) -> String:
  section(KM{}, c, Impl.groups(c_ns(c))) ++ section(KB{}, c, c_G(c)) ++
    section(KW{}, c, Nat.mul(c_G(c), 8n)) ++ section(KJ{}, c, c_jobs(c)) ++
    section(KQ{}, c, Nat.mul(c_G(c), 8n)) ++ section(KP{}, c, c_tw(c)) ++
    section(KI{}, c, Nat.mul(c_G(c), 2n))

def table(+m: Nat, +k: Nat, +G: Nat, +ns: List<&2, Nat>) -> IO(Unit):
  +KT = Impl.kt(k)
  +groups = Impl.groups(ns)
  +total = Impl.total(ns, k)
  +mc = Impl.max_contrib(groups, KT, G)
  +jobs = Impl.jobs(groups, m)
  +tw = Impl.total_warps(Impl.num_mats(ns), m, k)
  do IO<Unit>:
    IO.print("S ns=" ++ join(ns) ++ " k=" ++ num(k) ++ " G=" ++ num(G) ++ " m=" ++ num(m) ++
      " groups=" ++ num(groups) ++ " KT=" ++ num(KT) ++ " total=" ++ num(total) ++ " mc=" ++ num(mc) ++
      " fits=" ++ bit(Impl.launch_fits(groups, KT, G, Impl.slot_bytes())) ++
      " shape=" ++ bit(Impl.shape_ok(m, k, ns, Impl.xh_bytes())) ++
      " gbase=" ++ num(Impl.gbase(ns, 0n)) ++ "," ++ num(Impl.gbase(ns, 1n)) ++ "," ++
      num(Impl.gbase(ns, 2n)) ++ "," ++ num(Impl.gbase(ns, 3n)) ++ "," ++ num(Impl.gbase(ns, 4n)) ++
      " jobs=" ++ num(jobs) ++ " tw=" ++ num(tw))
    IO.print(rows(Cfg{ns, k, G, m, KT, total, mc, jobs, tw}))

def nats.cons(h: Maybe<&2, Nat>, t: Maybe<&2, List<&2, Nat>>) -> Maybe<&2, List<&2, Nat>>:
  match h t:
    case Some{a} Some{r}:
      Some{a <> r}
    case _ _:
      None{}

def nats(xs: List<String>) -> Maybe<&2, List<&2, Nat>>:
  match xs:
    case Nil{}:
      Some{Nil{}}
    case x <> rest:
      nats.cons(Nat.read(x), nats(rest))

def run(xs: Maybe<&2, List<&2, Nat>>) -> IO(Unit):
  match xs:
    case Some{m <> k <> G <> ns}:
      table(m, k, G, ns)
    case _:
      IO.die(Unit, 2, "usage: GEMM_M16_GROUP_TABLE m k G n0 [n1 [n2 [n3]]] (naturals)")

# IO.args() lists the program path first (Bend 2.0.34 and 2.0.35 both do).
def args.after_program(xs: List<String>) -> List<String>:
  match xs:
    case Nil{}:
      Nil{}
    case _ <> rest:
      rest

def main() -> IO(Unit):
  do IO<Unit>:
    xs : List<String> <- IO.args()
    run(nats(args.after_program(xs)))
