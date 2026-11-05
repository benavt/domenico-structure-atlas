# 5NTase and SERCA continuation

The accepted cohorts retain every model and physicality flag. Confidence is
genuine confidence; ProteinEBM energy remains a separate measurement. Originals
and native chain/residue labels stay unchanged, including the zero-based native
numbering in some 5NTase models. Analysis positions are sequence ordinals.

First apply the pinned shared terminal-variance default to the verified full
float16 feature caches. The adapter reproduces the native float64 population
variance reduction in blocks of 512 features. It verifies the stored NPY layout,
CRC, complete file hash, model order, sequence, native reference axes and original
metadata signature; it never loads object arrays or parses original coordinates.
Saved HTML variance is float32 and is not substituted for this calculation.

The standard combines the robust distant-pair rule (at least 2 Å SD, median plus
3 × 1.4826 MAD, at least 60% of eligible comparisons) with empirical rigid-core
anchors (1 Å average-linkage distance SD, at least 20 residues per core). Only
contiguous actual N/C ends qualify. The native segment, chain-length and combined
25% exclusion safeguards remain. Internal unassigned residues stay included.
No qualifying tail is reported as no trim; a cut is never forced. These are
geometric selections, not biological disorder or domain annotations.

The earlier project-specific conservative screen retained the full constructs.
Its direct rule found no qualifying terminal run; it did not execute the shared
default's additional core-boundary rule. Old results remain valid for their
recorded full-axis scope and are preserved as history.

`scripts/terminal_rescreen.py` needs its exact approved private plan and existing
protected runner. It saves variance, explicit masks, native decisions and input
lineage. Mask indices are zero-based; human-facing canonical positions are
one-based. Native labels describe the reference model, not a renumbering of every
model. Four residue-level linkage calculations are disclosed for the two
single-chain targets; this stage fits no model projections and runs no native TM,
scoring, molecular export or public deployment.

The accepted 2026-11-04 screen selects canonical positions 4–438 for 5NTase
(435 of 523 residues) and 58–951 for SERCA (894 of 993). Both cuts come from the
standard core-anchor boundary rule; the direct 60% distant-pair rule flags no
residues. Internal residues remain included. These changed masks require fresh
projections; the previous full-construct 5NTase projections remain preserved.

The continuation adapter stages only exact selected float16 pairs from the
verified stored caches and uses public native projection workers. It verifies
metadata assignments before any worker can invoke its legacy coordinate-rebuild
fallback. The first continuation stage computes primary projections/labels and
recursive HDBSCAN labels, retaining all points and explicit noise. Its maximum
is 42 projection fits and 60 HDBSCAN extractions across both targets; it performs
no native TM calls, coordinate parsing, model exports or deployment. Actual
cluster counts determine the following representative/state/report plan.

After masks are accepted, calculate the actual feature/RAM envelope and freeze
the clustering/model-selection continuation. Reuse 5NTase projections only with
the identical ordered feature mask; a changed mask needs new projections.
The original full-axis SERCA workflow had no completed projections. All seven saved 5NTase checkpoint stages
verify in the pinned native runtime. SQLite dump formatting differs between
runtime versions, so metrics digests must use the runtime recorded by the run;
a different runtime's digest is not evidence that scores changed. Preserve the
original checkpoint identity. If the new mask is identical, bind an explicit
feature-equivalence proof and the verified saved stages; if it differs, create
a new scientific identity and metrics seed from verified saved confidence.

Selected-state exports fit on the accepted comparison core while retaining the
complete protein's atoms, sequence correspondence and native residue labels.

Targets and large fits run serially. The old RBP state-only 16 GiB profile and
300-second progress deadline do not apply to wide, uninterrupted estimators.
Memory/no-swap/reserve controls remain; active fits are not declared pausable
checkpoints. Freeze the fit-stage policy and actual resource estimates before
launch. Representatives use the native confidence/medoid policy; subsequent
recursive HDBSCAN and supported-state MCMC50/DISCO50 work retain their own exact
budgets and provenance. Final camera reports and public replacement keep their
separate review gates.

## SERCA recovery after the selected-mask attempt

The accepted-mask attempt completed all primary and recursive HDBSCAN stages
for 5NTase. SERCA completed PCA, then the guard stopped its first UMAP fit at
28.49 GiB of charged memory after sustained memory pressure. No OOM or job swap
was recorded. Preserve these outputs and charge the aborted fit; do not repeat
5NTase or the accepted SERCA PCA.

`scripts/serca_recovery.py` authenticates the selected cache and accepted PCA,
copies the closed metrics/PCA artifacts to a new run, and executes UMAP, t-SNE,
primary-label validation and recursive HDBSCAN in separate fresh processes.
`scripts/saved_feature_cache.py` maps the exact stored float16 numeric member
read-only, after archive/CRC/numeric hashes and cohort/mask verification. It
neither reads the full-axis numerical cache nor rebuilds selected features.
Density's explicit stored-float16 option casts only the selected fit rows to
float32, retaining the native engines, values, order, parameters and seeds.

Each child's native temporary workspace uses a separate owned SSD directory.
Native UMAP still makes two full float32 copies internally; the SSD workspace
can be reclaimed instead of occupying RAM-backed scratch. A finished child's
workspace is retired only after its process and descendants have exited.
Unit-stop cleanup checks ownership and quiescence and removes only disposable
scratch. Saved inputs, checkpoints, figures and receipts remain intact.
Memory category and per-child I/O observations are recorded separately from
scientific progress. A live fit never produces a fabricated checkpoint.

Freeze a new exact recovery plan before execution. Its scope is SERCA only,
20 new projection fits at most (including t-SNE's PCA initialization), and
30 HDBSCAN extractions. There are zero repeated standalone PCA fits, TM calls,
coordinate parses, scoring calls, molecular exports or deployment. The 23
previous attempts plus this allowance total 43, including the aborted UMAP.
Keep the existing memory/no-swap/CPU/pressure controls and 50 GiB SSD reserve;
the reviewed plan must also bind the additional SSD workspace allowance.
Final model selection and reports follow a separate plan based on actual groups.
