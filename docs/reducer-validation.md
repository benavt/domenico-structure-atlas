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
