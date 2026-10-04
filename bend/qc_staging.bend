# Ext 9503f + 9503b: the quantized-cache prefill staging scratch size qc_staging_pages (modules/attention_fn/
# triton_paged.py, T) and the 5110h merge bound that now uses it (generator/job.py, J), transcribed over Nat. T and J are
# the files of P2Baseline's pfast3 tree (main + 9503f-qc-staging-round64-fast.patch + 9503b-prefill-m4096-stage1088.patch);
# line numbers are that tree's. n = pages = the block-table span bsz * npps_w (T:2280) or len(seq.allocated_pages)
# (J:44-49). The environment variable is the input env (spec type QcEnv). Reference: qc_staging_spec.bend; laws:
# qc_staging_laws.bend.
import Base
import ./prefill_membound.bend as MB
import ./qc_staging_spec.bend as S

# T:714 QC_STAGING_ROUND_PAGES = 64
def ROUND_PAGES() -> Nat:
  64n

# T:719 pow2 = max(1, 1 << (pages - 1).bit_length())   ((n - 1).bit_length() as in prefill_membound.bend: 1 at n = 0)
def pow2(n: Nat) -> Nat:
  Nat.max(1n, Nat.pow(2n, MB.bit_length_dec(n)))

# T:722 -(-pages // QC_STAGING_ROUND_PAGES) * QC_STAGING_ROUND_PAGES: for pages >= 0, -(-p // 64) = (p + 63) // 64
# (Python floor division; checked against the engine function for p = 0..4096 by qc_staging_diff.py)
def ceil64(+n: Nat) -> Nat:
  Nat.mul(Nat.div(Nat.add(n, 63n), 64n), 64n)

# T:722 return min(pow2, -(-pages // QC_STAGING_ROUND_PAGES) * QC_STAGING_ROUND_PAGES)
def round64(+n: Nat) -> Nat:
  Nat.min(pow2(n), ceil64(n))

# T:717-725 qc_staging_pages: env = os.environ.get("EXL3_QC_STAGING_ROUND64"); if env is None or env == "1": round64;
# if env == "0": pow2; else raise ValueError
def stage(env: S.QcEnv, +n: Nat) -> S.QcOut:
  match env:
    case S.QcUnset{}:
      S.QcPages{round64(n)}
    case S.QcOne{}:
      S.QcPages{round64(n)}
    case S.QcZero{}:
      S.QcPages{pow2(n)}
    case S.QcOther{}:
      S.QcError{}

# J:40-41 (9503b) M4096_MAX_STAGE_PAGES = 1088, M4096_MAX_PROMPT = 262143; J:44-49 m4096_stage_pages(block_table_pages)
# = qc_staging_pages(block_table_pages), compared with M4096_MAX_STAGE_PAGES in the 5110h merge condition (J:1336,
# prefill_membound.bend merge_ok with s = the staged pages, B = 1088)
def MAX_STAGE_PAGES() -> Nat:
  1088n

def MAX_PROMPT() -> Nat:
  262143n
