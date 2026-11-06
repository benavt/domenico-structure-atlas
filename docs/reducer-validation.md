# Bounded reducer validation adapter

`repro64` is an explicitly approved diagnostic phase over a frozen small cohort.
It uses the existing numerical wheel and a separately bound portability helper.
Each host/profile job runs `umap1`, `tsne1`, `umap2`, `tsne2` in fresh children:
six projection fits including the two internal t-SNE PCA fits, no HDBSCAN.
Four host/profile jobs total 24 fits. New runs have fresh output/cache roots;
completed production stages are preserved.

Before fitting, the controller verifies the frozen installed-package payload
once. Children check file identities before imports and after fitting. Explicit
`-B -X pycache_prefix=...` flags avoid inherited bytecode; isolated Python ignores
environment-only Python startup flags. Profile keys survive sanitization and
Numba/native thread-pool state is checked before and after each fit. Every child
also rechecks installed-library integrity and resource containment.

Closed receipts bind graph/init/RNG traces, source/runtime/profile, projection
outputs and exact counts. Missing/extra hooks or cached fit bypasses fail. Generic
jobs require both native jobs' closed proofs plus their own native predecessor;
this gate does not demand numerical agreement before the diagnostic experiment.

`reducer_gate.py` assembles closed native evidence without fits.
`reducer_trace_compare.py` compares four closed runs, same-host repeats first and
then cross-host pairs, identifying the first different recorded boundary. It
reuses frozen coordinate tolerances and does not validate cluster stability or
large-workload capacity. Later replay, labels and scaling remain separate scopes.

Use the monitored scheduler front door and private user scopes remotely; use the
reviewed strict pressure guard and a private unit for the reference host. Keep
RAM, zero-swap, PID, CPU, disk and deadline controls. Preparing plans or passing
source tests never starts a job or grants execution approval.

## Targeted follow-up: `isolate64`

The separately approved follow-up runs only `umap1` and `tsne1` on each host:
2 UMAP + 2 t-SNE + 2 internal PCA fits in total, zero HDBSCAN. UMAP adds bounded
scalar/epoch-0/epoch-1 observation without replacing its optimizer. t-SNE explicitly
substitutes the same hash-bound float32 gradient-entry vector from a closed native
reference trial on both hosts. Native PCA still runs and is counted. Every receipt
records this intervention; no normal analysis default or numerical wheel changes.

The initialization must bind its source plan, closed completion, stage receipt,
trace and exact NPY artifact. The controller validates cohort/order/residue mask,
source host and closed provenance before creating a run. Actual children record
native and substituted starting vectors and prove exactly one substitution. The
same fresh-process, runtime integrity, smoke, CPU/RAM/swap/PID/disk/deadline and
reference-host pressure checks apply. There is no retry or generic-target trial.

`reducer_isolation_compare.py` reads only two closed trials and the previous native
references. It checks the intervention and native recipe, compares the extended
cross-host boundaries and compares final coordinates with same-host native results.
It distinguishes a discarded native t-SNE start from effective optimizer inputs.
It reports failed numerical gates honestly; it never certifies labels/capacity,
submits fits or changes production stages. Further strict-math/label/scaling tests
remain separate scopes.

Before requesting execution approval, run the exact isolated CLI against each
staged host plan: `python -I -B scripts/mimas_cpu.py preflight PLAN`. This reads
frozen source/input/provenance bindings without requiring or granting approval,
submitting work, importing estimators or creating a run. It is distinct from
`verify`, `run`, `child`, `setup-check` and `smoke`, which retain exact approval
checks. Source tests launch this real `-I` entry point so the test runner's own
module path cannot hide bootstrap import errors.

## Matched UMAP curve: `curve64`

One UMAP per host, two fits total, holds the public `a`/`b` constructor parameters
constant alongside their `spread`/`min_dist` recipe. The pair comes from a closed
reference-host trace; its source plan, completion, stage receipt, trace and scalar
artifact must all verify before any run. It is an explicit diagnostic intervention.
The existing numerical wheel, graph/init/seed/precision and resource guards remain.

The closed-only `umap_curve_compare.py` requires matching array shape, dtype and
coordinate bytes, matching recorded checkpoints, and bitwise preservation of the
reference host's original native coordinates. Approximate geometric agreement is
reported but cannot pass this gate. No cache bypass or additional fit is allowed.
Passing this test is evidence for that cohort; it does not authorize or certify
larger cohorts, clustering labels, a changed residue selection or full analysis.
The next paired case should freeze a deterministic, method-stratified 1,000-model
cohort using the same accepted terminal mask. Full-cohort work needs its own
resource and scientific acceptance plan; at least one case must exercise the
approximate-neighbor path used by the actual full workload.

## Larger paired UMAP case: `curve1000`

This separately approved phase keeps the exact matched curve recipe and terminal
mask while binding 1,000 selected stored feature rows and their existing original
coordinate paths. No duplicate model copies, new distances or scoring are needed.
One UMAP per host remains the entire fit budget. The input order, rejection flags
and nested 64-model membership must verify before execution.

`saved_curve_evidence.py` rechecks the closed accepted reference-host UMAP64 and
the matched-curve destination result: input identity, native source proofs, every
output hash, finite saved coordinates and all 35 shared checkpoints must agree.
It explicitly records that a new controlled reference-host confirmation was not
run. This is completed-artifact reuse, not a fabricated completion of that launch.
The larger source/approval scope must name this reference policy deliberately.

The reference host may have an explicit bounded wait before its first workload
launch. Consecutive quiet samples use the same frozen admission thresholds; the
unchanged guard checks them again before starting the child. The wait never
changes a host setting, restarts a refused/started fit, or extends the unit's hard
deadline. Timeout creates a no-launch receipt. Omit the option to retain the
existing immediate-refusal behavior. The destination trial follows only after a
closed reference-host result, avoiding unused destination fits on a busy host.

`umap1000_compare.py` verifies two closed 1,000-point trials and their saved64
gate, then requires exact coordinate bytes and effective checkpoints. No
full-cohort execution or cluster-stability claim follows automatically. The
large-workload approximate-neighbor path remains a separate gate.
