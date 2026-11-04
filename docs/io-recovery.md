# I/O diagnosis and remaining RBP recovery

The RBP parity attempt completed density clustering, representative selection,
both Bonsai trees and both medoid TM grids. It then stopped during canonical
TM-input persistence. The three final host I/O-full intervals were 2.838%,
3.556% and 3.316%: 0.486 seconds cumulative full stalls over 15 seconds.
Host available memory was about 49.5 GiB; job peak was 6.76 GiB, with no swap,
memory.high/max or OOM events. This stop was the configured I/O policy acting
on synchronous persistence, rather than evidence of imminent memory exhaustion.
It does not establish the cause of the earlier unguarded machine freeze.

Earlier attempts exposed distinct problems: large file-backed PCA staging,
cold model/distance reads, a decaying-average guard and source/checker errors.
Those failures must not all be attributed to the same cause. The current guard
uses actual interval stalls, but still terminates after three violating samples
with a shared counter across resource types. A sustained 2% I/O-only warning
therefore has the same response as memory pressure.

Read-only host inspection found no matching logged NVMe/ext4/OOM/hung-task
faults today, no cached NVMe critical warning, normal temperature and a quiet
idle baseline. These observations do not certify drive wear or loaded latency.
The intended IOWeight=10 was not demonstrated effective: every RBP io.stat
sample was missing and the I/O controller was disabled below the root cgroup.
A relative I/O weight is not a bandwidth or latency cap. A 29 ms control-only
user-service probe confirmed that adding IOAccounting=yes alone still produced
no io.stat. A system-manager transient service running as the original user is
the narrower candidate to verify, avoiding a restart or reconfiguration of the
desktop user manager. It must refuse science unless actual accounting works.

## Source preparation

`rbp_native_tm.py` preserves the pinned native canonicalization and TM math.
Canonical correspondence records use batches of at most 50 rows. Native
pair work uses waves of at most 25 unordered pairs, with one durable intent
transaction before any calls and one atomic proof/direction/query/completion
transaction afterwards. Both databases use rollback journals with synchronous
EXTRA. An interrupted intent stays charged and cannot trigger an automatic retry.
Group commits reduce sync frequency; they do not make the sole evidence ledger
volatile. Mock tests exercise failed directions, failed transaction boundaries,
exact budgets, immutable records and partial final groups.

The old driver also recursively scanned and statted the growing output tree
on every cancellation/resource check, including checks around individual TM
input records. The successor audits total output size and free-space reserves
at bounded 30-second intervals while the independent guard continues sampling
memory/CPU/pressure. It retains the same disk limits without repeatedly walking
all prior outputs for each small operation.

The successor adapter `rbp_state_recovery.py` verifies independent acceptance
of all saved density, Bonsai, representative and TM-grid artifacts. It requests
zero additional UMAP, HDBSCAN, Bonsai or medoid-grid fits. It renders the saved
variance annotations before the remaining state search and panel exports.
The 873 committed canonical input rows may be reused only after exact registry,
native mapping, sequence/core, source and file-hash checks. Originals, all old
outputs and the scientific library pin remain preserved.

A source-only instrumented comparison used 50 fake input identities and 29
fake pairs per arm. Both produced 58 fake directional calls and the same scalar
result hash. Durable write transactions fell from 257 to 9 and fsync syscalls
from 1,100 to 91. Mock work took 0.180 versus 0.172 seconds, which does not
establish a meaningful scientific speedup. Peak RSS was about 158 MiB. An abrupt
process-exit probe left a fully charged pending intent with no reusable column;
legacy five-column canonical evidence imported without modifying its files or
recanonicalizing. No real models, alignments or scientific fits ran in these checks.

## Proposed execution policy

Keep hard memory/CPU caps, zero job swap, disk reserves, wall-time limits and
memory-pressure stopping. Retain the conservative busy-host admission check.
Treat sustained I/O-only pressure as a warning and cooperative backoff request,
with explicit grace, progress-freshness and checkpoint deadlines. The worker
responds only at a durable batch or stage boundary; do not SIGSTOP a transaction
or leave a large disposable workspace resident while waiting. Escalate to a
checkpointed stop when the approved bounds expire. Keep per-resource counters
so an I/O warning does not accumulate into an unrelated fatal memory counter.

The next named service must request IOAccounting=yes and verify real io.stat
availability before scientific work. Record job and host pressure, completed
units, checkpoint age and telemetry-write latency. Missing accounting must not
be interpreted as zero I/O. Do not rely on IOWeight or select a new universal
I/O percentage without measurements. Any rate/IOPS cap or host configuration
change needs a concrete reviewed configuration.

The cooperative guard candidate and recovery adapter need source review and
an exact new execution plan before use. No recovery run, controller setting,
threshold, environment or public deployment is authorized by this document.
Synthetic SQLite timings describe persistence overhead only; they are not
full-cohort TM-search performance measurements.

## Preserved scientific and publication status

The saved result has 108 assigned HDBSCAN groups, 6,570 unassigned models and
all 10,926 plotted identities/scores. Repeat-seed diagnostics remain exploratory.
Both Bonsai trees and complete 108 by 108 / 4 by 4 medoid grids are available.
No TM-state native calls were made before the stop; TM-state MCMC50/DISCO50
panels and final camera-dependent figures remain pending. The representative-only
Pages update was independently verified and published as
`atlas-rbp-hdbscan-2026-11-04-001`, independently of the state search. The live
[RBP atlas](https://benavt.github.io/domenico-structure-atlas/) keeps the unfinished
components visibly pending.

Private evidence receipts are stored with the original parity plan: its exact
plan, guard telemetry, independently verified completed-stage acceptance, host
storage audit and per-attempt I/O review. A new recovery must reference those
immutable records and write to a fresh output directory.

The policy interpretation follows the [Linux PSI documentation](https://docs.kernel.org/accounting/psi.html),
[cgroup v2 I/O controls](https://docs.kernel.org/admin-guide/cgroup-v2.html), and
[SQLite synchronization guarantees](https://www.sqlite.org/pragma.html#pragma_synchronous).
