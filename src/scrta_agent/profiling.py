from __future__ import annotations

import csv
import gzip
from pathlib import Path

from .schemas import DatasetProfile


COMMON_RNA_HINTS = (
    "sample",
    "sample_id",
    "patient",
    "donor",
    "tissue",
    "condition",
    "response",
    "timepoint",
    "cell_type",
    "annotation",
    "cluster",
    "leiden",
)

COMMON_TCR_HINTS = (
    "barcode",
    "cell_barcode",
    "clonotype",
    "clone",
    "cdr3",
    "cdr3_aa",
    "chain",
    "v_gene",
    "j_gene",
    "productive",
)


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _sniff_delimiter(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".csv") or name.endswith(".csv.gz"):
        return ","
    return "\t"


def _head(values, limit: int = 16) -> list[str]:
    return [str(v) for v in list(values)[:limit]]


def profile_h5ad(path: Path) -> tuple[list[str], list[str]]:
    try:
        import anndata as ad
    except Exception:
        return (
            [
                f"RNA file: {path}",
                "RNA profile detail unavailable because anndata is not installed.",
            ],
            [],
        )

    try:
        adata = ad.read_h5ad(path, backed="r")
    except Exception as exc:
        return (
            [
                f"RNA file: {path}",
                f"RNA profile detail unavailable because h5ad reading failed: {exc}",
            ],
            [],
        )
    obs_cols = [str(x) for x in adata.obs.columns]
    var_preview = [str(x) for x in adata.var_names[:12]]
    obsm_keys = list(adata.obsm.keys())
    layer_keys = list(adata.layers.keys())
    matched = [c for c in obs_cols if any(h in c.lower() for h in COMMON_RNA_HINTS)]
    summary = [
        f"RNA file: {path}",
        f"RNA shape: {adata.n_obs} cells x {adata.n_vars} genes/features",
        f"RNA gene preview: {', '.join(var_preview) or 'none'}",
        f"RNA obs columns: {', '.join(_head(obs_cols, 24)) or 'none'}",
        f"RNA matched metadata hints: {', '.join(_head(matched, 16)) or 'none'}",
        f"RNA embeddings: {', '.join(_head(obsm_keys, 12)) or 'none'}",
        f"RNA layers: {', '.join(_head(layer_keys, 12)) or 'none'}",
    ]
    try:
        adata.file.close()
    except Exception:
        pass
    inventory = [
        f"RNA obs columns: {', '.join(_head(obs_cols, 40)) or 'none'}",
        f"RNA matched metadata hints: {', '.join(_head(matched, 24)) or 'none'}",
    ]
    return summary, inventory


def profile_tcr_table(path: Path) -> tuple[list[str], list[str], list[str]]:
    sep = _sniff_delimiter(path)
    rows_seen = 0
    columns: list[str] = []
    matched: list[str] = []
    clonotype_like: list[str] = []
    with _open_text(path) as fh:
        reader = csv.DictReader(fh, delimiter=sep)
        columns = [str(c) for c in (reader.fieldnames or [])]
        for row in reader:
            rows_seen += 1
            if rows_seen >= 100000:
                break
    matched = [c for c in columns if any(h in c.lower() for h in COMMON_TCR_HINTS)]
    clonotype_like = [
        c
        for c in columns
        if any(h in c.lower() for h in ("barcode", "clonotype", "clone", "cdr3"))
    ]
    summary = [
        f"TCR file: {path}",
        f"TCR delimiter: {'comma' if sep == ',' else 'tab'}",
        f"TCR rows scanned: {rows_seen}",
        f"TCR columns: {', '.join(_head(columns, 30)) or 'none'}",
        f"TCR recognized fields: {', '.join(_head(matched, 20)) or 'none'}",
    ]
    inventory = [f"TCR columns: {', '.join(_head(columns, 50)) or 'none'}"]
    return summary, inventory, clonotype_like


def build_dataset_profile(rna_h5ad_path: str, tcr_path: str) -> DatasetProfile:
    rna_path = Path(rna_h5ad_path).resolve()
    tcr_file = Path(tcr_path).resolve()
    if not rna_path.exists():
        raise FileNotFoundError(f"RNA h5ad not found: {rna_path}")
    if not tcr_file.exists():
        raise FileNotFoundError(f"TCR table not found: {tcr_file}")

    rna_summary, rna_inventory = profile_h5ad(rna_path)
    tcr_summary, tcr_inventory, tcr_join = profile_tcr_table(tcr_file)
    inferred = []
    for col in tcr_join:
        low = col.lower()
        if "barcode" in low or "cell" in low:
            inferred.append(f"TCR column likely joins RNA cells: {col}")
        elif "clonotype" in low or "clone" in low:
            inferred.append(f"TCR column likely defines clone identity: {col}")
        elif "cdr3" in low:
            inferred.append(f"TCR column likely defines receptor sequence: {col}")

    guardrails = [
        "Lead with RNA-defined cell states, programs, response groups, tissues, or timepoints.",
        "Use TCR clonotype evidence as lineage support or discriminating evidence, not as the default endpoint.",
        "Control clone-size effects before claiming receptor-sequence mechanisms.",
        "Separate sample composition, tissue, patient, and timepoint effects from within-state biology.",
        "Prefer falsifiable branch screens before deep TCR mechanism audits.",
    ]
    return DatasetProfile(
        rna_path=str(rna_path),
        tcr_path=str(tcr_file),
        rna_summary=rna_summary,
        tcr_summary=tcr_summary,
        metadata_inventory=rna_inventory + tcr_inventory,
        inferred_join_keys=inferred,
        guardrails=guardrails,
    )
