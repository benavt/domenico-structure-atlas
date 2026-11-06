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
