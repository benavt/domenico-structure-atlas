# Mimas CPU validation and deployment

The consumer can run CPU diagnostics on Mimas without changing the pinned
Structure_Clustering commit or the Pax analysis. `scripts/mimas_cpu.py` controls
fresh child processes; `mimas_contract.py` checks the actual allocation and
sanitizes native writer/cache paths. `mimas_smoke.py` checks the installed wheel,
its complete payload, imports, packaged assets and analysis CLI. `mimas_pilot.py`
calls the installed native reducers and cluster selection without monkeypatches.
No module fits models, installs dependencies or submits jobs on import.

Private machine paths, input registries, resource limits and approval receipts
belong in ignored `local/` plans. Fleet owns SSH and monitored submission. This
port adds no host configuration, scheduler changes, daemon or automatic retries.
The existing Fe94 library pin and scientific defaults remain unchanged.

## Start with the packaged installation

Build a wheel from a source-only export of the exact reviewed library commit;
do not build inside a frozen scientific checkout. `mimas_package.wheel_contract`
binds every installed payload file. Freeze the wheelhouse and a hash-checked,
fully pinned requirements file. Install offline into a new Python 3.11 venv;
never modify an existing folding/report environment. The existing interpreter is
only a hash-bound bootstrap and base-stdlib dependency, not a promise of a fully
independent Python distribution. Retain that base installation while using the venv.

The generated smoke batch verifies the approved plan and frozen files, then
checks the existing bootstrap and actual Slurm RAM/CPU limits **before** creating
an environment. It refuses an existing destination. Imports run outside source
checkouts under isolated Python, with the user site and inherited native hooks
removed. All temporary arrays, native caches and result paths are run-owned.
The installed library must match the wheel bytes, not just its version string.

## Pilot ladder

Each row needs its own exact approved plan, immutable attempt and whole-job Fleet
CPU/RSS monitor. These are provisional budgets to validate, not measured peaks.

| Phase | Work | Initial CPU / RAM / wall budget | Fit ceiling |
|---|---|---|---|
| Setup + smoke | Fresh offline environment; installed-package, CLI and tiny NPY checks; no study models | 1 / 8 GiB / 1 hour | 0 |
| SERCA-64 | Deterministic frozen 64-model subset, accepted 894-residue core (canonical 58–951), PCA/UMAP/t-SNE, primary label checks, one nonrecursive density diagnostic | 2 / 8 GiB / 1 hour | 9 projection fits, 6 HDBSCAN extractions |
| Full SERCA UMAP | All 5,072 accepted models and the same core, UMAP only, compared with the saved Pax result | 4 / 64 GiB / 2 hours | 1 |
| Production continuation | Verified saved stages, recursive density, model selection and standard reports | Set from pilot telemetry | Freeze separately |

The 64-model test exercises packaging and APIs but cannot validate the full-size
UMAP algorithm branch or prove its peak memory. Density-0 is diagnostic: it is
not recursive HDBSCAN and must not become the published default labels. Energy
remains energy, original scores/physicality flags are preserved, and input
coordinates are unchanged. A subset projection is not expected to reproduce a
filtered full-cohort projection.

Before the numerical pilot, freeze the exact selected feature NPY, raw float16
value hash, residue-mask hash, model order, original-file registry/flags and
configuration. Keep the source archive and accepted Pax outputs read-only.
Cross-machine numerical comparisons must bind the reference and tolerances
before execution; bitwise coordinate identity is not assumed across runtimes.
No study-input transfer or subset extraction is part of the zero-model smoke.

## Admission and results

Mimas advertised 96 CPUs, 1,024,000 MB node RAM and about 3.5 TB free scratch in the
2026-11-05 discovery. These are node totals, not an allocation or a guaranteed
available budget. Live Slurm uses CR_Core and has accounting disabled; requested
`--mem` alone therefore proves neither memory scheduling nor enforcement.
Observe the job cgroup. Numerical fits require verified finite CPU/RAM/PID limits,
independent zero-swap containment, the exact interpreter and actual sanitized
child environment. Missing PID/swap controls may permit an import smoke with an
explicit `smoke_complete_fit_blocked` receipt, but never a fit. Report missing
controls to the administrator; do not alter Slurm, global cgroups or UID limits.

Every phase writes owned logs, resource observations, input/source bindings and
closed receipts. Failed or interrupted work stays available; no implicit resume
or retry occurs. A successful smoke or pilot is not completion of the standard
scientific deliverables. Keep final scientific outputs on Mimas and transfer
only verified selected plots, receipts and report assets as needed.

This initial environment validates analysis only. PyMOL, US-align, Bonsai and the
presentation renderer require their own pinned runtime checks before a complete
report can be generated on Mimas. Existing approved Pax results and camera work
remain available for later report assembly. No publication follows automatically.

## Unprivileged job limits

Mimas's user manager delegates the memory and process controllers. A transient
`systemd-run --user --scope` therefore supplies a private kernel RAM limit,
zero-swap limit and task limit without sudo or a scheduler reconfiguration.
The no-fit Slurm migration probe 20478 verified a 128 MiB / zero-swap / 16-task
scope and preserved its original process parent and Slurm CPU affinity.

The corrected smoke uses `mimas_scope.py`: a supervisor remains in the Slurm
batch task, and a scopekeeper runs the bound setup workload as its descendant.
Fleet's process-tree monitor retains that ancestry. The keeper checks a kernel
pidfd for supervisor death, arms a parent-death signal, and stops only its exact
scope on cancellation or exit. A separate systemd runtime deadline provides a
fallback; the private scope enables group-wide OOM killing. Actual workload
return codes are retained separately from the unit's status.

CPU, RAM and process controls have different meanings. `maximum_cpus` remains
the Slurm request and each native worker remains single-threaded. An explicit
one-physical-core/two-logical-CPU allowance accepts SMT siblings only after
reading their topology. The user manager does not delegate the CPU controller:
the workload uses inherited kernel affinity, checked against its live Slurm
allocation and supervisor. This is recorded as allocation-bound affinity, not a
private cgroup CPU quota. Missing or changed bindings block admission.

The supervisor and whole-job monitor remain outside the workload scope; their
small memory overhead must be recorded separately. The private cap does not
make memory a scheduler-consumable resource on this CR_Core site, so concurrent
production admission still needs a host-memory budget. No ancestor cgroup, UID
limit, shared service or other job is modified. The earlier administrator-helper
prototype is not used. Every new environment/fixture run still needs its exact
approved plan; the migration probe grants no scientific execution authority.


## Python base and native C++ ABI

The first scoped environment installed successfully but its import smoke found
that a copied Conda interpreter changed the meaning of `$ORIGIN/../lib` in the
ELF RPATH. A wheel loaded the older system C++ runtime before the base Python's
ICU extension, which requires CXXABI_1.3.15. This is independent of job RAM.

For this base, set `runtime_interpreter: base_symlink`. The venv then links its
interpreter to the exact hash-bound base binary, retaining its original native
library search directory. Prefix/bin parents must remain real directories;
only the declared interpreter link is permitted, and its resolved target must
match the bound base binary. Bind the base C++/GCC library hashes as runtime
closure evidence. Do not use a broad LD_LIBRARY_PATH or preload workaround.
The base installation remains read-only and must be retained. Preserve failed
copied environments; a successor approved plan creates a fresh linked venv.

## Matched SERCA-64 comparison

The first numerical pilot uses the **same 64 originals, row order and raw F16
values on Mimas and Pax**, with the accepted 894-residue core (canonical 58–951).
Selection is deterministic and balances available methods; rejection flags stay
visible. Its 64 × 399,171 input matrix is about 49 MiB. Full-cohort projections
are context, not a matched reference.

Both hosts use the same pinned library wheel, dependency wheels, configuration,
seeds and archived density provider. Pax gets a separate linked environment;
Mimas reuses its accepted linked environment. Each host has an 8 GiB kernel RAM
cap, zero swap, 64 tasks, single-threaded workers and a one-hour deadline. Pax
also retains host-pressure admission and sustained-pressure stopping. No shared
runtime, scheduler setting or ongoing full-cohort analysis changes.

`pilot_compare.py` reads closed saved outputs only. It checks raw features,
model identities, flags, native source hashes and masks, then reports PCA
geometry, nonlinear neighborhood diagnostics and label agreement. Nonlinear
coordinate differences require review; degenerate all-noise/one-group results
are not structural validation. Runtime facts, timings and resource receipts
remain separate from numerical agreement. This small pilot establishes neither
a full-cohort memory ceiling nor completion of standard scientific deliverables.

Freeze both host plans and the comparison contract before approval. Never infer
approval from an earlier smoke. Run Mimas first, collect only verified closed
receipts/selected arrays, and run the identical Pax reference when its admission
checks permit. A failure stops once and preserves evidence; it does not trigger
an automatic retry or resume.

## Full accepted SERCA continuation

The separately approved `serca-full` phase keeps all 5,072 accepted models and
canonical residues 58–951, including rejection flags. A closed paired 1,000-model
proof must reverify exact coordinates and all recorded checkpoints before launch.
The three accepted Pax projections are imported into owned sidecars with identical
coordinate values/dtypes and explicitly translated native model paths. Originals
and the accepted source artifacts remain unchanged.

One fixed-curve full-cohort UMAP fit exercises the approximate-neighbor path.
It must match the accepted full Pax coordinate bytes before either primary label
validation or recursive HDBSCAN can start. A mismatch retains the diagnostic trace
and stops once. No standalone PCA or t-SNE is refitted. The primary repeats use
the same public 2D curve recipe; the CASP17 10D density settings, including its
native zero-minimum-distance curve, are preserved. This does not claim a new
cross-host density-label comparison.

The complete budget is at most 18 UMAP fits (one full-cohort check, two primary
repeat fits and at most fifteen density fits), thirty HDBSCAN extractions and
zero PCA/t-SNE fits. Children run serially under verified job-specific RAM,
zero-swap, task, CPU-affinity, deadline and disk controls. Longer batch deadlines
are supported only for this explicit production phase. HDBSCAN is the default
label view; every model and explicit noise remain in the saved outputs.

Source tests, successful pilots and this adapter do not authorize the full run.
Freeze the actual source, installed runtime, originals/feature hashes, saved
projection proofs, destinations and resource budgets before exact execution
approval. Model selection, molecular reports and publication remain later stages.

Production admission requires at least 128 GiB of currently available node RAM.
During each child, a 32 GiB host reserve and the actual job-memory PSI interval
are checked every five seconds; three consecutive violations stop this job.
The memory-full threshold is 0.5% per measured interval, without using a decaying
PSI average as an interval counter. I/O counters remain explicitly unavailable
when the kernel does not expose them; this is never reported as zero I/O.
