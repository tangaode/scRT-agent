# scRNA-scTCR Workflow Skill

Use this skill when analyzing paired scRNA-seq and scTCR-seq datasets.

## Operating Rules

1. RNA leads the biological story.
2. TCR evidence supports lineage, persistence, sharing, replacement, or
   clone-state coupling.
3. Clone expansion alone is not antigen specificity.
4. Shared clonotypes alone are not migration.
5. UMAP proximity alone is not lineage transition.
6. CDR3 similarity alone is not mechanism.
7. Patient, sample, tissue, timepoint, and clone-size structure must be treated
   as possible confounders.

## Minimum First Run

- Build a dataset profile for RNA and TCR.
- Verify barcode joins and matched fraction.
- Scope clone identity by patient in cohort data; do not merge local labels
  such as `clonotype1` across patients.
- Identify RNA state/group columns.
- Score reusable T cell programs when genes are available.
- Summarize clone expansion and clone-state occupancy.
- Rank hypotheses only after a skeptic audit.

## Preferred Hypothesis Shapes

- RNA state/program with clone-lineage support.
- Treatment response as clone reinvigoration or replacement.
- Tissue compartmentalization of shared clones.
- Non-clonal RNA biology explaining apparent clone differences.

## Required Artifacts

- `dataset_profile.md`
- `literature_context.md`
- `agent_rna_analyst.md`
- `agent_tcr_analyst.md`
- `agent_integrator.md`
- `agent_skeptic.md`
- `scripts/scrna_sctcr_joint_analysis.py`
- `final_report.md`
