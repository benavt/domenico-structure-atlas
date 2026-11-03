# Domenico structure atlas

Independent consumer of Structure_Clustering for RBP (P02925), 5NTase (P07024)
and SERCA (P04191). This project tests the complete report workflow against a
fixed, accepted set of existing structures and ProteinEBM scores. It does not
generate structures or recalculate energies.

The selected snapshot contains **10,926 RBP, 11,000 5NTase and 5,072 SERCA models**.
RBP and SERCA are partial method panels; later outputs belong to a successor
snapshot. `config/study.json` records target identities, coverage and exact source
hashes. Private source locators and execution receipts stay in ignored `local/`.

The library pin is the reviewed workflow candidate under validation, rather than
a claim that the candidate has already been merged or released. The shared
workflow is pinned at `external/research-workflows`; source installation and that
private submodule require repository access. Published static results will not.

## Workflow

1. Verify the accepted score manifest and every native coordinate hash. Reserve
   storage and collect only those models and their explicitly associated scores.
2. Approve the frozen terminal-screening plan. `scripts/workflow.py screen`
   uses the library's native intake, Cα distograms, conservative terminal policy
   and diagnostic figures. Review exact retained ranges and abstentions before
   approving clustering. Geometric flexibility does not establish disorder.
3. `scripts/workflow.py analyze` reuses verified distograms and the frozen mask
   through `StructureWorkflowWorker`. PCA, UMAP and t-SNE share the standard
   component-preserving partition and its validation evidence. Complete reports
   remain the default. Genuine confidence ranks eligible representatives;
   clusters without eligible predictions use an explicitly recorded medoid.
4. Approve the prepared molecular camera, then finish PowerPoint, PyMOL/ensembles,
   confidence figures, medoid TM grids, Bonsai and qualifying grouped exports.
5. Verify saved deliverables and prepare a privacy-checked, hashed publication
   manifest. `structure_clustering.publication.export` supplies the shared explorer.
   The manual Pages workflow deploys only a reviewed static bundle from a release;
   it cannot run scientific calculations. Bulk coordinates and reports are kept
   outside source Git. Final download placement depends on measured bundle sizes.

ProteinEBM uses existing Model6 expert results. Lower **t=0.10** energy is better
within a target; these learned units are neither physical kcal/mol nor confidence.
Energy coloring never substitutes for pLDDT or changes cluster memberships.

No scientific result or live site is claimed until its saved outputs pass review.

Run source-only checks with `python3 -m unittest discover -s tests`.
