"""Publication-style figure builder for scRT-agent v2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.stats import pearsonr

from .notebook_tools import (
    expression_frame,
    paired_tcr_subset,
    resolve_gene_names,
    safe_rank_genes_groups,
    tissue_stratified_expansion_de,
    tumor_like_subset,
)
from .utils import load_tcr_table, normalize_tcr_columns


PANEL_COLORS = {
    "accent": "#C44E52",
    "teal": "#4C9A8A",
    "blue": "#4C72B0",
}


@dataclass
class FigureResult:
    png_path: Path
    pdf_path: Path
    summary_path: Path


def _global_expansion_de(
    adata,
    *,
    label: str,
    expansion_col: str = "expanded_clone",
    min_cells_per_group: int = 20,
    top_n: int = 12,
) -> pd.DataFrame:
    working = paired_tcr_subset(adata)
    if expansion_col not in working.obs.columns:
        raise KeyError(f"'{expansion_col}' is not present in adata.obs.")
    working.obs[expansion_col] = working.obs[expansion_col].fillna(False).astype(bool).map(
        {True: "expanded", False: "non_expanded"}
    )
    ranked = safe_rank_genes_groups(
        working,
        groupby=expansion_col,
        groups=["expanded"],
        reference="non_expanded",
        min_cells_per_group=min_cells_per_group,
        key_added=f"{label}_expanded_vs_non_expanded",
    )
    frame = sc.get.rank_genes_groups_df(
        ranked,
        group="expanded",
        key=f"{label}_expanded_vs_non_expanded",
    ).head(top_n)
    frame.insert(0, "tissue", label)
    frame.insert(1, "expanded_n", int((working.obs[expansion_col] == "expanded").sum()))
    frame.insert(2, "non_expanded_n", int((working.obs[expansion_col] == "non_expanded").sum()))
    return frame


def _normalize_barcode(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text.split("-")[0]


def _normalize_barcode_exact(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def _normalize_sample(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def _make_merge_key(barcode: object, sample: object = None, *, use_core: bool = False) -> str:
    barcode_value = _normalize_barcode(barcode) if use_core else _normalize_barcode_exact(barcode)
    sample_value = _normalize_sample(sample)
    if not barcode_value:
        return ""
    return f"{sample_value}::{barcode_value}" if sample_value else barcode_value


def _coerce_bool(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "t", "1", "yes", "y", "productive", "high"}


def _prepare_tcr_table(tcr_path: str | Path) -> pd.DataFrame:
    df = normalize_tcr_columns(load_tcr_table(tcr_path)).copy()
    if "barcode" not in df.columns:
        raise ValueError("TCR table must contain a barcode-like column.")
    df["barcode"] = df["barcode"].astype(str)
    df["barcode_core"] = df["barcode"].map(_normalize_barcode)
    if "productive" in df.columns:
        df["productive"] = df["productive"].map(_coerce_bool)
    else:
        df["productive"] = False
    return df


def _sample_scope_column(df: pd.DataFrame) -> str | None:
    for column in ("sample_key", "sample_id"):
        if column in df.columns:
            return column
    return None


def _needs_sample_prefixed_clonotypes(df: pd.DataFrame) -> tuple[bool, str | None]:
    if "clonotype_id" not in df.columns:
        return False, None
    sample_col = _sample_scope_column(df)
    if sample_col is None:
        return False, None
    scoped = df.loc[df["clonotype_id"].notna(), [sample_col, "clonotype_id"]].drop_duplicates()
    if scoped.empty:
        return False, sample_col
    spread = scoped.groupby("clonotype_id")[sample_col].nunique()
    risky = spread[spread > 1]
    if risky.empty:
        return False, sample_col
    raw_like = risky.index.to_series().astype(str).str.fullmatch(r"clonotype\d+", case=False, na=False)
    return float(raw_like.mean()) >= 0.5, sample_col


def _join_unique(series: pd.Series) -> object:
    values = [str(v) for v in series if pd.notna(v) and str(v).strip()]
    return "|".join(sorted(set(values))) if values else np.nan


def _aggregate_tcr_by_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = df.groupby(column, dropna=False)
    agg = pd.DataFrame(index=grouped.size().index)
    if "clonotype_id" in df.columns:
        agg["clonotype_id"] = grouped["clonotype_id"].agg(lambda s: next((v for v in s if pd.notna(v)), np.nan))
    if "chain" in df.columns:
        agg["chain"] = grouped["chain"].agg(_join_unique)
    if "cdr3" in df.columns:
        agg["cdr3"] = grouped["cdr3"].agg(_join_unique)
    if "v_gene" in df.columns:
        agg["v_gene"] = grouped["v_gene"].agg(_join_unique)
    if "j_gene" in df.columns:
        agg["j_gene"] = grouped["j_gene"].agg(_join_unique)
    agg["productive_any"] = grouped["productive"].agg("max")
    agg["tcr_chain_count"] = grouped.size()
    if "reads" in df.columns:
        reads = pd.to_numeric(df["reads"], errors="coerce")
        agg["tcr_reads"] = reads.groupby(df[column]).sum(min_count=1)
    return agg


def load_joint_adata(rna_h5ad_path: str | Path, tcr_path: str | Path) -> tuple[ad.AnnData, dict[str, object]]:
    adata = sc.read_h5ad(rna_h5ad_path)
    tcr_df = _prepare_tcr_table(tcr_path)

    needs_prefix, sample_scope = _needs_sample_prefixed_clonotypes(tcr_df)
    clonotype_scope = "as_provided"
    if needs_prefix and sample_scope is not None:
        mask = tcr_df["clonotype_id"].notna()
        tcr_df.loc[mask, "clonotype_id"] = (
            tcr_df.loc[mask, sample_scope].astype(str) + ":" + tcr_df.loc[mask, "clonotype_id"].astype(str)
        )
        clonotype_scope = f"prefixed_by_{sample_scope}"

    if "barcode" in adata.obs.columns:
        adata.obs["barcode"] = adata.obs["barcode"].astype(str)
    else:
        adata.obs["barcode"] = adata.obs_names.astype(str)
    adata.obs["barcode_exact"] = adata.obs["barcode"].map(_normalize_barcode_exact)
    adata.obs["barcode_core"] = adata.obs["barcode"].map(_normalize_barcode)

    rna_sample_scope = _sample_scope_column(adata.obs)
    tcr_sample_scope = _sample_scope_column(tcr_df)
    adata.obs["sample_merge_key"] = adata.obs.apply(
        lambda row: _make_merge_key(row["barcode"], row[rna_sample_scope] if rna_sample_scope else np.nan, use_core=False),
        axis=1,
    )
    adata.obs["sample_merge_key_core"] = adata.obs.apply(
        lambda row: _make_merge_key(row["barcode"], row[rna_sample_scope] if rna_sample_scope else np.nan, use_core=True),
        axis=1,
    )
    if tcr_sample_scope:
        tcr_df["sample_merge_key"] = [
            _make_merge_key(barcode, sample, use_core=False)
            for barcode, sample in zip(tcr_df["barcode"], tcr_df[tcr_sample_scope])
        ]
        tcr_df["sample_merge_key_core"] = [
            _make_merge_key(barcode, sample, use_core=True)
            for barcode, sample in zip(tcr_df["barcode"], tcr_df[tcr_sample_scope])
        ]

    tcr_cell_exact = _aggregate_tcr_by_column(tcr_df, "barcode")
    tcr_cell_core = _aggregate_tcr_by_column(tcr_df, "barcode_core")
    tcr_cell_sample_exact = _aggregate_tcr_by_column(tcr_df, "sample_merge_key") if "sample_merge_key" in tcr_df.columns else pd.DataFrame()
    tcr_cell_sample_core = _aggregate_tcr_by_column(tcr_df, "sample_merge_key_core") if "sample_merge_key_core" in tcr_df.columns else pd.DataFrame()
    exact_overlap = int(adata.obs["barcode_exact"].isin(tcr_cell_exact.index).sum())
    core_overlap = int(adata.obs["barcode_core"].isin(tcr_cell_core.index).sum())
    sample_exact_overlap = int(adata.obs["sample_merge_key"].isin(tcr_cell_sample_exact.index).sum()) if not tcr_cell_sample_exact.empty else 0
    sample_core_overlap = int(adata.obs["sample_merge_key_core"].isin(tcr_cell_sample_core.index).sum()) if not tcr_cell_sample_core.empty else 0
    overlap_modes = {
        "sample_exact": sample_exact_overlap,
        "sample_barcode_core": sample_core_overlap,
        "exact": exact_overlap,
        "barcode_core": core_overlap,
    }
    merge_mode = max(overlap_modes, key=overlap_modes.get)

    if merge_mode == "sample_exact":
        adata.obs = adata.obs.join(tcr_cell_sample_exact, on="sample_merge_key")
    elif merge_mode == "sample_barcode_core":
        adata.obs = adata.obs.join(tcr_cell_sample_core, on="sample_merge_key_core")
    elif merge_mode == "exact":
        adata.obs = adata.obs.join(tcr_cell_exact, on="barcode_exact")
    else:
        adata.obs = adata.obs.join(tcr_cell_core, on="barcode_core")

    adata.obs["has_tcr"] = adata.obs["clonotype_id"].notna()
    clone_sizes = adata.obs.loc[adata.obs["has_tcr"], "clonotype_id"].value_counts()
    adata.obs["clone_size"] = adata.obs["clonotype_id"].map(clone_sizes).fillna(0).astype(int)
    adata.obs["expanded_clone"] = adata.obs["clone_size"] >= 3
    adata.obs["clone_size_log1p"] = np.log1p(adata.obs["clone_size"].astype(float))

    for column in ("sample_id", "tissue", "sample_key", "leiden"):
        if column not in adata.obs.columns:
            continue
        adata.obs[column] = adata.obs[column].astype("object").where(adata.obs[column].notna(), "Unknown")
        adata.obs[column] = adata.obs[column].astype("category")

    meta = {
        "exact_overlap": exact_overlap,
        "core_overlap": core_overlap,
        "sample_exact_overlap": sample_exact_overlap,
        "sample_core_overlap": sample_core_overlap,
        "merge_mode": merge_mode,
        "clonotype_scope": clonotype_scope,
        "paired_cells": int(adata.obs["has_tcr"].sum()),
        "expanded_fraction": float(adata.obs.loc[adata.obs["has_tcr"], "expanded_clone"].mean()) if int(adata.obs["has_tcr"].sum()) else 0.0,
    }
    return adata, meta


def _panel_label(ax, label: str) -> None:
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        va="top",
        ha="right",
    )


def _sample_level_marker_frame(adata, genes: list[str]) -> pd.DataFrame:
    frame = expression_frame(adata, genes, obs_columns=["sample_id", "tissue"])
    grouped = (
        frame.groupby(["sample_id", "tissue"], observed=False)
        .mean(numeric_only=True)
        .reset_index()
    )
    grouped["sample_tissue"] = grouped["sample_id"].astype(str) + "|" + grouped["tissue"].astype(str)
    return grouped


def _scatter_with_fit(ax, df: pd.DataFrame, x: str, y: str, title: str) -> None:
    working = df[[x, y]].dropna()
    if working.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    sns.regplot(
        data=working,
        x=x,
        y=y,
        ax=ax,
        color=PANEL_COLORS["blue"],
        scatter_kws={"s": 28, "alpha": 0.85},
        line_kws={"lw": 1.5},
    )
    if len(working) >= 3 and working[x].nunique() > 1 and working[y].nunique() > 1:
        corr, pvalue = pearsonr(working[x], working[y])
        subtitle = f"r = {corr:.2f}, P = {pvalue:.2e}"
    else:
        subtitle = "r = NA, P = NA"
    ax.set_title(f"{title}\n{subtitle}", fontsize=11)
    ax.set_xlabel(x)
    ax.set_ylabel(y)


def _draw_umap(adata, *, ax, color: str, title: str, **kwargs) -> None:
    sc.pl.umap(
        adata,
        color=color,
        ax=ax,
        show=False,
        frameon=False,
        title=title,
        **kwargs,
    )
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")


def build_publication_figure(
    *,
    rna_h5ad_path: str | Path,
    tcr_path: str | Path,
    output_dir: str | Path,
    figure_name: str = "scrt_publication_figure",
) -> FigureResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="talk")
    sc.settings.set_figure_params(dpi=120, facecolor="white", frameon=False)

    adata, meta = load_joint_adata(rna_h5ad_path, tcr_path)
    paired = paired_tcr_subset(adata)
    tumor_like = tumor_like_subset(adata)
    tumor_like_paired = paired_tcr_subset(tumor_like)

    resolved = resolve_gene_names(
        adata,
        ["CCL5", "NKG7", "GZMB", "XBP1", "PDCD1", "TIGIT"],
    )
    marker_panel_gene = resolved.get("CCL5", next(iter(resolved.values()), None))
    hypothesis_gene = resolved.get("XBP1", marker_panel_gene)

    figure_notes: list[str] = []
    de_source = "tumor_like_stratified"
    de_title = "Top expanded-vs-non-expanded DE genes\nin tumor-like tissues"
    try:
        de_table = tissue_stratified_expansion_de(tumor_like, top_n=6)
    except Exception as exc:
        figure_notes.append(f"Stratified tumor-like DE fallback triggered: {exc}")
        try:
            de_table = _global_expansion_de(tumor_like, label="tumor_like_global", top_n=12)
            de_source = "tumor_like_global"
            de_title = "Top expanded-vs-non-expanded DE genes\nin pooled tumor-like cells"
        except Exception as inner_exc:
            figure_notes.append(f"Pooled tumor-like DE fallback triggered: {inner_exc}")
            de_table = _global_expansion_de(adata, label="all_paired_global", top_n=12)
            de_source = "all_paired_global"
            de_title = "Top expanded-vs-non-expanded DE genes\nin all paired cells"

    heatmap_table = (
        de_table.pivot_table(
            index="names",
            columns="tissue",
            values="logfoldchanges",
            aggfunc="mean",
        )
        .fillna(0.0)
    )
    if heatmap_table.shape[0] > 12:
        heatmap_table = heatmap_table.reindex(
            heatmap_table.abs().max(axis=1).sort_values(ascending=False).head(12).index
        )

    expanded_tumor = tumor_like_paired[tumor_like_paired.obs["expanded_clone"].fillna(False).astype(bool)].copy()
    correlation_markers = [gene for gene in ["CCL5", "NKG7", "GZMB", "XBP1", "PDCD1", "TIGIT"] if gene in resolved]
    sample_frame = _sample_level_marker_frame(expanded_tumor, correlation_markers)
    actual = {gene: resolved[gene] for gene in correlation_markers}

    candidate_pairs = [
        ("CCL5", "NKG7", "Effector correlation"),
        ("CCL5", "GZMB", "Cytotoxic program"),
        ("XBP1", "TIGIT", "ER stress vs checkpoint"),
        ("XBP1", "PDCD1", "ER stress vs PDCD1"),
    ]
    available_pairs = [
        (actual[x], actual[y], title)
        for x, y, title in candidate_pairs
        if x in actual and y in actual
    ]
    if len(available_pairs) < 4:
        numeric_columns = [col for col in sample_frame.columns if col not in {"sample_id", "tissue", "sample_tissue"}]
        while len(available_pairs) < 4 and len(numeric_columns) >= 2:
            x = numeric_columns[len(available_pairs) % len(numeric_columns)]
            y = numeric_columns[(len(available_pairs) + 1) % len(numeric_columns)]
            if x != y:
                available_pairs.append((x, y, f"{x} vs {y}"))
            else:
                break
    available_pairs = available_pairs[:4]

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 4, figure=fig, width_ratios=[1.2, 1.2, 1.0, 1.0], height_ratios=[1.0, 1.1, 1.2])

    ax_a = fig.add_subplot(gs[0, 0])
    clone_box = paired.obs[["tissue", "clone_size_log1p"]].copy()
    clone_box["tissue"] = clone_box["tissue"].astype(str)
    sns.boxplot(data=clone_box, x="tissue", y="clone_size_log1p", ax=ax_a, color=PANEL_COLORS["teal"], fliersize=1.5)
    ax_a.set_title("Clone size by tissue")
    ax_a.set_xlabel("")
    ax_a.set_ylabel("log1p(clone size)")
    ax_a.tick_params(axis="x", rotation=20)
    _panel_label(ax_a, "a")

    ax_b1 = fig.add_subplot(gs[0, 1])
    _draw_umap(paired, color="leiden", ax=ax_b1, title="UMAP by Leiden", legend_loc="on data", legend_fontsize=7)
    _panel_label(ax_b1, "b")

    ax_b2 = fig.add_subplot(gs[0, 2])
    _draw_umap(paired, color="tissue", ax=ax_b2, title="UMAP by tissue", legend_loc="right margin", legend_fontsize=8)

    ax_c1 = fig.add_subplot(gs[1, 0])
    if marker_panel_gene is not None:
        _draw_umap(paired, color=marker_panel_gene, ax=ax_c1, title=f"{marker_panel_gene} expression", color_map="magma")
    else:
        ax_c1.text(0.5, 0.5, "No marker gene available", ha="center", va="center", transform=ax_c1.transAxes)
    _panel_label(ax_c1, "c")

    ax_c2 = fig.add_subplot(gs[1, 1])
    if hypothesis_gene is not None:
        _draw_umap(paired, color=hypothesis_gene, ax=ax_c2, title=f"{hypothesis_gene} expression", color_map="rocket")
    else:
        ax_c2.text(0.5, 0.5, "No hypothesis gene available", ha="center", va="center", transform=ax_c2.transAxes)

    scatter_axes = [
        fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[1, 3]),
        fig.add_subplot(gs[2, 2]),
        fig.add_subplot(gs[2, 3]),
    ]
    for idx, ax in enumerate(scatter_axes):
        if idx < len(available_pairs):
            x, y, title = available_pairs[idx]
            _scatter_with_fit(ax, sample_frame, x, y, title)
        else:
            ax.axis("off")
    _panel_label(scatter_axes[0], "d")

    ax_e = fig.add_subplot(gs[2, 0:2])
    sns.heatmap(
        heatmap_table,
        cmap="coolwarm",
        center=0,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "log fold change"},
        ax=ax_e,
    )
    ax_e.set_title(de_title)
    ax_e.set_xlabel("")
    ax_e.set_ylabel("")
    _panel_label(ax_e, "e")

    fig.suptitle(
        "scRT-agent v2 Figure Mode: integrated scRNA + scTCR view of GSE201425",
        fontsize=18,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    png_path = output_dir / f"{figure_name}.png"
    pdf_path = output_dir / f"{figure_name}.pdf"
    summary_path = output_dir / f"{figure_name}_summary.txt"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary_lines = [
        f"Figure name: {figure_name}",
        f"RNA input: {Path(rna_h5ad_path).resolve()}",
        f"TCR input: {Path(tcr_path).resolve()}",
        f"Merge mode: {meta['merge_mode']}",
        f"Clonotype scope: {meta['clonotype_scope']}",
        f"Paired cells: {meta['paired_cells']}",
        f"Expanded fraction among paired cells: {meta['expanded_fraction']:.3f}",
        f"Tumor-like tissues: {', '.join(sorted(tumor_like.obs['tissue'].astype(str).unique()))}",
        f"Resolved markers: {', '.join(f'{src}->{dst}' for src, dst in resolved.items()) or 'none'}",
        f"DE source for panel e: {de_source}",
        f"Figure notes: {' | '.join(figure_notes) if figure_notes else 'none'}",
        "",
        "Top DE genes by tissue:",
        de_table[["tissue", "names", "logfoldchanges", "pvals_adj"]].to_string(index=False),
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    return FigureResult(png_path=png_path, pdf_path=pdf_path, summary_path=summary_path)
