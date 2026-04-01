"""Validation helpers for scRT-agent v2."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .utils import load_tcr_table, normalize_tcr_columns


RAW_CLONOTYPE_RE = re.compile(r"^clonotype\d+$", re.IGNORECASE)


@dataclass
class ValidationSummary:
    """Validation summary for inputs or notebook step outputs."""

    strengths: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    guardrails: list[str] = field(default_factory=list)
    metrics: dict[str, object] = field(default_factory=dict)

    def to_prompt_text(self) -> str:
        lines: list[str] = []
        if self.metrics:
            lines.append("Metrics:")
            lines.extend(f"- {key}: {value}" for key, value in self.metrics.items())
        if self.strengths:
            lines.append("Strengths:")
            lines.extend(f"- {item}" for item in self.strengths)
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {item}" for item in self.warnings)
        if self.guardrails:
            lines.append("Guardrails:")
            lines.extend(f"- {item}" for item in self.guardrails)
        return "\n".join(lines).strip() or "No validation notes."

    def to_markdown(self) -> str:
        return self.to_prompt_text()


class DatasetValidator:
    """Checks whether the paired scRNA + scTCR inputs are analysis-ready."""

    def _normalize_barcode(self, value: object) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return ""
        return text.split("-")[0]

    def inspect_inputs(self, rna_h5ad_path: str, tcr_path: str) -> ValidationSummary:
        import anndata as ad

        summary = ValidationSummary()
        adata = ad.read_h5ad(rna_h5ad_path, backed="r")
        try:
            obs_names = [str(idx) for idx in adata.obs_names]
            obs_core = {self._normalize_barcode(idx) for idx in obs_names}
            obs_columns = {str(col).lower(): str(col) for col in adata.obs.columns}
            tcr_df = normalize_tcr_columns(load_tcr_table(tcr_path))

            tcr_barcodes = set()
            tcr_core = set()
            if "barcode" in tcr_df.columns:
                tcr_barcodes = set(tcr_df["barcode"].dropna().astype(str))
                tcr_core = {self._normalize_barcode(item) for item in tcr_barcodes}

            exact_overlap = sum(1 for barcode in obs_names if barcode in tcr_barcodes)
            core_overlap = sum(1 for barcode in obs_core if barcode and barcode in tcr_core)
            coverage = max(exact_overlap, core_overlap) / max(len(obs_names), 1)

            summary.metrics.update(
                {
                    "rna_cells": adata.n_obs,
                    "rna_genes": adata.n_vars,
                    "tcr_rows": len(tcr_df),
                    "tcr_unique_barcodes": len(tcr_barcodes),
                    "exact_overlap_cells": exact_overlap,
                    "core_overlap_cells": core_overlap,
                    "paired_coverage": f"{coverage:.3f}",
                }
            )

            if "x_umap" in {key.lower() for key in adata.obsm.keys()}:
                summary.strengths.append("RNA object already contains a UMAP embedding.")
            if "sample_id" in obs_columns:
                summary.strengths.append("RNA metadata contains sample_id, enabling sample-aware analyses.")
            elif "sample" in obs_columns:
                summary.strengths.append("RNA metadata contains a sample-like column.")
            else:
                summary.warnings.append("RNA metadata lacks a clear sample identifier.")

            if coverage >= 0.25:
                summary.strengths.append("RNA/TCR overlap is high enough for joint analyses.")
            elif coverage >= 0.10:
                summary.strengths.append("RNA/TCR overlap is usable but limited.")
            else:
                summary.warnings.append("RNA/TCR overlap is low; prioritize descriptive analyses.")

            sample_col = None
            for candidate in ("sample_key", "sample_id"):
                if candidate in tcr_df.columns:
                    sample_col = candidate
                    break

            if "clonotype_id" in tcr_df.columns:
                clonotypes = tcr_df["clonotype_id"].dropna().astype(str)
                summary.metrics["unique_clonotypes"] = clonotypes.nunique()
                if sample_col:
                    multi_sample = (
                        tcr_df.loc[tcr_df["clonotype_id"].notna(), [sample_col, "clonotype_id"]]
                        .drop_duplicates()
                        .groupby("clonotype_id")[sample_col]
                        .nunique()
                    )
                    shared = multi_sample[multi_sample > 1]
                    if not shared.empty:
                        raw_like_fraction = float(sum(bool(RAW_CLONOTYPE_RE.match(value)) for value in shared.index)) / len(shared)
                        if raw_like_fraction >= 0.5:
                            summary.warnings.append(
                                "clonotype_id appears sample-local; apply sample-aware prefixing before clone-size analyses."
                            )
                            summary.guardrails.append(
                                "Never trust raw clonotype_id globally unless it is namespaced by sample_key or sample_id."
                            )
                        else:
                            summary.guardrails.append(
                                "Some clonotype IDs span multiple samples; verify whether this reflects public clones or naming collisions."
                            )
                else:
                    summary.warnings.append("TCR table lacks sample_key/sample_id, so global clonotype scope is ambiguous.")
            else:
                summary.warnings.append("TCR table lacks clonotype_id; clone-size analyses will be limited.")

            summary.guardrails.extend(
                [
                    "Prefer sample-aware or tissue-aware comparisons over pooled global claims.",
                    "Treat unadjusted differential expression as provisional until confounders are checked.",
                ]
            )
            return summary
        finally:
            if getattr(adata, "file", None) is not None:
                adata.file.close()

    def inspect_step_output(self, analysis, text_output: str, image_count: int, error_message: str | None = None) -> ValidationSummary:
        summary = ValidationSummary()
        summary.metrics["image_count"] = image_count
        summary.metrics["text_output_chars"] = len(text_output or "")
        lowered = (analysis.code_description + "\n" + analysis.first_step_code).lower()
        if error_message:
            summary.warnings.append(f"Execution error: {error_message}")
        if image_count > 0:
            summary.strengths.append("The notebook step produced at least one figure.")
        if text_output.strip():
            summary.strengths.append("The notebook step produced textual output that can support replanning.")
        if any(token in lowered for token in ("umap", "plot", "scatter", "heatmap", "violin")) and image_count == 0:
            summary.warnings.append("The step looked visualization-oriented but no figure was produced.")
        if "rank_genes_groups" in lowered and "rank_genes_groups" not in (text_output or ""):
            summary.guardrails.append("If DE was intended, verify that ranked genes were stored and surfaced before downstream reuse.")
        if "expanded_clone" in lowered and all(
            token not in lowered for token in ("has_tcr", "paired_tcr_subset", "paired_only")
        ):
            summary.guardrails.append(
                "Clone-expansion analyses should usually operate on the paired TCR subset, not all RNA cells."
            )
        if "rank_genes_groups" in lowered and all(
            token not in lowered for token in ("safe_rank_genes_groups", "tissue_stratified_expansion_de")
        ):
            summary.guardrails.append(
                "Prefer safe_rank_genes_groups or tissue_stratified_expansion_de over ad hoc DE code."
            )
        if "== 'tumor'" in lowered or '== "tumor"' in lowered:
            summary.guardrails.append(
                "Do not hardcode tissue == 'tumor' unless that label really exists; inspect tissue levels or use tumor_like_subset."
            )
        if "pd1" in lowered and "pdcd1" not in lowered:
            summary.guardrails.append(
                "Checkpoint markers may use canonical gene symbols like PDCD1 rather than PD1; resolve aliases before declaring genes missing."
            )
        if not text_output.strip() and image_count == 0 and not error_message:
            summary.warnings.append("The step produced no visible evidence; avoid treating it as support for the hypothesis.")
        return summary
