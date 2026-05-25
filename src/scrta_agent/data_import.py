from __future__ import annotations

import csv
import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .llm import LLMClient
from .utils import ensure_dir, write_json, write_text


RNA_FILE_SUFFIXES = {
    ".h5ad",
    ".h5",
    ".hdf5",
    ".csv",
    ".tsv",
    ".txt",
    ".mtx",
    ".loom",
    ".zarr",
}
TCR_FILE_NAMES = {
    "filtered_contig_annotations.csv",
    "all_contig_annotations.csv",
    "contig_annotations.csv",
    "clonotypes.csv",
    "airr_rearrangement.tsv",
}
TCR_FILE_SUFFIXES = {".csv", ".tsv", ".txt"}


@dataclass
class InputFileSummary:
    path: str
    kind_hint: str
    exists: bool
    is_dir: bool
    size_bytes: int | None = None
    child_preview: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DataPreparationResult:
    rna_h5ad_path: str
    tcr_path: str
    output_dir: str
    plan_path: str
    manifest_path: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def split_user_paths(raw: str | list[str] | tuple[str, ...] | None) -> list[Path]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        values: list[str] = []
        for item in raw:
            values.extend(_split_path_string(str(item)))
    else:
        values = _split_path_string(str(raw))
    return [Path(value).expanduser().resolve() for value in values if value.strip()]


def prepare_inputs(
    rna_inputs: str | list[str] | tuple[str, ...],
    tcr_inputs: str | list[str] | tuple[str, ...],
    output_dir: str | Path,
    llm: LLMClient | None = None,
    analysis_name: str = "scrna_sctcr_case",
    require_llm_plan: bool = True,
) -> DataPreparationResult:
    """Prepare user-provided inputs for the scRTA workflow.

    The LLM is used to review the input inventory and produce a conversion
    plan. File conversion itself is deterministic and uses standard Python
    readers, so large expression matrices are not sent to the LLM.
    """

    out = ensure_dir(output_dir)
    rna_paths = split_user_paths(rna_inputs)
    tcr_paths = split_user_paths(tcr_inputs)
    if not rna_paths:
        raise ValueError("At least one RNA input path is required.")
    if not tcr_paths:
        raise ValueError("At least one TCR input path is required.")

    all_summaries = summarize_inputs(rna_paths, "rna") + summarize_inputs(tcr_paths, "tcr")
    plan_text, plan_json = build_llm_preparation_plan(
        all_summaries,
        llm=llm,
        analysis_name=analysis_name,
        require_llm=require_llm_plan,
    )
    plan_path = write_text(out / "data_preparation_plan.md", plan_text)
    write_json(out / "data_preparation_plan.json", plan_json)

    rna_source = choose_rna_source(rna_paths, plan_json)
    tcr_source = choose_tcr_source(tcr_paths, plan_json)
    h5ad_path, rna_notes = convert_rna_to_h5ad(rna_source, out / "prepared_rna.h5ad")
    tcr_path, tcr_notes = normalize_tcr_table(tcr_source, out / "prepared_tcr.csv", plan_json)

    manifest = {
        "analysis_name": analysis_name,
        "rna_inputs": [str(path) for path in rna_paths],
        "tcr_inputs": [str(path) for path in tcr_paths],
        "selected_rna_source": str(rna_source),
        "selected_tcr_source": str(tcr_source),
        "rna_h5ad_path": str(h5ad_path),
        "tcr_path": str(tcr_path),
        "llm_plan_path": str(plan_path),
        "notes": rna_notes + tcr_notes,
        "input_inventory": [summary.to_dict() for summary in all_summaries],
    }
    manifest_path = write_json(out / "prepared_inputs_manifest.json", manifest)
    return DataPreparationResult(
        rna_h5ad_path=str(h5ad_path),
        tcr_path=str(tcr_path),
        output_dir=str(out),
        plan_path=str(plan_path),
        manifest_path=str(manifest_path),
        notes=rna_notes + tcr_notes,
    )


def summarize_inputs(paths: list[Path], kind_hint: str) -> list[InputFileSummary]:
    return [_summarize_path(path, kind_hint) for path in paths]


def build_llm_preparation_plan(
    summaries: list[InputFileSummary],
    llm: LLMClient | None,
    analysis_name: str,
    require_llm: bool = True,
) -> tuple[str, dict[str, Any]]:
    inventory_json = json.dumps([summary.to_dict() for summary in summaries], indent=2)
    if require_llm:
        if llm is None:
            raise RuntimeError("An LLM client is required for interactive data preparation.")
        llm.require_ready()
        system_prompt = (
            "You are a data preparation planner for paired single-cell RNA-seq "
            "and single-cell TCR-seq analysis. You inspect file inventories and "
            "return a conservative machine-readable plan. Do not invent files."
        )
        user_prompt = f"""# Analysis Name
{analysis_name}

# Input Inventory
{inventory_json}

# Task
Choose the best RNA source and TCR source for a paired scRNA/scTCR workflow.
The RNA source must be convertible to an AnnData .h5ad file. The TCR source
should be a contig, clonotype, AIRR, or other table containing cell barcodes
and clonotype or receptor-sequence fields.

Return concise notes and this JSON block:

DATA_PREPARATION_PLAN_JSON
{{
  "rna_source_path": "",
  "rna_format": "",
  "tcr_source_path": "",
  "tcr_format": "",
  "barcode_column": "",
  "clonotype_column": "",
  "sample_column": "",
  "warnings": []
}}
END_DATA_PREPARATION_PLAN_JSON
"""
        plan_text = llm.complete(system_prompt, user_prompt, temperature=0)
        plan_json = _extract_json_block(plan_text, "DATA_PREPARATION_PLAN_JSON")
        if not plan_json:
            plan_json = _deterministic_plan(summaries)
            plan_text = (
                plan_text.rstrip()
                + "\n\n# Parser Note\n"
                + "The LLM response did not contain a parseable plan block; "
                + "a conservative deterministic plan was used for execution.\n"
            )
        return plan_text, plan_json

    plan_json = _deterministic_plan(summaries)
    plan_text = (
        "# Data Preparation Plan\n\n"
        "LLM planning was disabled. A deterministic source selection was used.\n\n"
        "DATA_PREPARATION_PLAN_JSON\n"
        + json.dumps(plan_json, indent=2)
        + "\nEND_DATA_PREPARATION_PLAN_JSON\n"
    )
    return plan_text, plan_json


def choose_rna_source(paths: list[Path], plan: dict[str, Any]) -> Path:
    planned = _path_from_plan(plan.get("rna_source_path"), paths)
    if planned:
        return planned
    candidates = _expand_rna_candidates(paths)
    if not candidates:
        raise FileNotFoundError("No supported RNA input source was found.")
    return candidates[0]


def choose_tcr_source(paths: list[Path], plan: dict[str, Any]) -> Path:
    planned = _path_from_plan(plan.get("tcr_source_path"), paths)
    if planned:
        return planned
    candidates = _expand_tcr_candidates(paths)
    if not candidates:
        raise FileNotFoundError("No supported TCR input table was found.")
    return candidates[0]


def convert_rna_to_h5ad(source: Path, destination: Path) -> tuple[Path, list[str]]:
    source = source.resolve()
    destination = destination.resolve()
    notes: list[str] = []
    if not source.exists():
        raise FileNotFoundError(f"RNA source not found: {source}")
    if source.suffix.lower() == ".h5ad":
        if source == destination:
            return destination, ["RNA source was already an h5ad file."]
        shutil.copy2(source, destination)
        return destination, ["RNA source was already h5ad and was copied into the prepared input directory."]

    try:
        import anndata as ad
        import pandas as pd
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError('Install analysis dependencies first: pip install -e ".[analysis]"') from exc

    suffix = source.suffix.lower()
    if source.is_dir():
        adata = _read_10x_directory(source)
        notes.append("RNA source was read as a 10x matrix directory.")
    elif suffix in {".h5", ".hdf5"}:
        try:
            import scanpy as sc
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError('10x HDF5 input requires scanpy: pip install -e ".[analysis]"') from exc
        adata = sc.read_10x_h5(str(source))
        notes.append("RNA source was read as a 10x HDF5 matrix.")
    elif suffix == ".loom":
        try:
            import scanpy as sc
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError('Loom input requires scanpy: pip install -e ".[analysis]"') from exc
        adata = sc.read_loom(str(source))
        notes.append("RNA source was read as a loom file.")
    elif suffix == ".zarr":
        adata = ad.read_zarr(str(source))
        notes.append("RNA source was read as an AnnData zarr store.")
    elif _is_text_table(source):
        matrix = pd.read_csv(source, sep=_sniff_delimiter(source), index_col=0)
        if _should_transpose_expression_matrix(matrix):
            matrix = matrix.T
            notes.append("RNA text matrix was transposed so observations are cells and variables are genes.")
        adata = ad.AnnData(matrix)
        notes.append("RNA source was read as a dense text expression matrix.")
    else:
        raise ValueError(f"Unsupported RNA input format: {source}")

    if adata.obs_names.has_duplicates:
        adata.obs_names_make_unique()
        notes.append("Duplicate RNA cell names were made unique.")
    if adata.var_names.has_duplicates:
        adata.var_names_make_unique()
        notes.append("Duplicate RNA feature names were made unique.")
    adata.write_h5ad(destination)
    return destination, notes


def normalize_tcr_table(source: Path, destination: Path, plan: dict[str, Any] | None = None) -> tuple[Path, list[str]]:
    source = _resolve_tcr_table(source)
    if not source.exists():
        raise FileNotFoundError(f"TCR source not found: {source}")
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError('TCR table preparation requires pandas: pip install -e ".[analysis]"') from exc

    sep = _sniff_delimiter(source)
    table = pd.read_csv(source, sep=sep)
    notes = [f"TCR table was read from {source.name}."]
    plan = plan or {}
    barcode_col = _choose_column(
        table.columns,
        str(plan.get("barcode_column") or ""),
        ["barcode", "cell_barcode", "cell_id", "cell", "barcode_id"],
    )
    clonotype_col = _choose_column(
        table.columns,
        str(plan.get("clonotype_column") or ""),
        [
            "raw_clonotype_id",
            "clonotype_id",
            "clone_id",
            "clonotype",
            "cdr3s_aa",
            "cdr3_aa",
            "cdr3",
        ],
    )
    if barcode_col and barcode_col != "barcode":
        table["barcode"] = table[barcode_col].astype(str)
        notes.append(f"Normalized barcode column from `{barcode_col}`.")
    elif "barcode" not in table.columns:
        raise ValueError("Could not identify a cell barcode column in the TCR table.")
    if clonotype_col and clonotype_col != "clonotype_id":
        table["clonotype_id"] = table[clonotype_col].astype(str)
        notes.append(f"Normalized clonotype column from `{clonotype_col}`.")
    elif "clonotype_id" not in table.columns:
        raise ValueError("Could not identify a clonotype or receptor sequence column in the TCR table.")

    table["barcode"] = table["barcode"].astype(str)
    table["clonotype_id"] = table["clonotype_id"].astype(str)
    clone_sizes = table.groupby("clonotype_id", dropna=False)["barcode"].transform("nunique")
    table["clone_size"] = clone_sizes.astype(int)
    table["clone_size_category"] = table["clone_size"].map(clone_size_category)
    table.to_csv(destination, index=False)
    notes.append("TCR table was written with normalized barcode, clonotype_id, clone_size, and clone_size_category fields.")
    return destination, notes


def clone_size_category(size: int) -> str:
    if size <= 1:
        return "Single"
    if size <= 5:
        return "Small"
    if size <= 20:
        return "Medium"
    if size <= 100:
        return "Large"
    if size <= 500:
        return "Hyperexpanded"
    return "Ultraexpanded"


def _split_path_string(raw: str) -> list[str]:
    return [part.strip().strip('"') for part in re.split(r";|\n", raw) if part.strip()]


def _summarize_path(path: Path, kind_hint: str) -> InputFileSummary:
    exists = path.exists()
    is_dir = path.is_dir() if exists else False
    summary = InputFileSummary(path=str(path), kind_hint=kind_hint, exists=exists, is_dir=is_dir)
    if not exists:
        return summary
    if is_dir:
        children = sorted(child.name for child in path.iterdir())[:40]
        summary.child_preview = children
        return summary
    summary.size_bytes = path.stat().st_size
    if _is_text_table(path):
        summary.columns = _read_table_columns(path)
    return summary


def _read_table_columns(path: Path) -> list[str]:
    try:
        if path.name.lower().endswith(".gz"):
            import gzip

            handle_cm = gzip.open(path, "rt", encoding="utf-8", errors="replace")
        else:
            handle_cm = path.open("r", encoding="utf-8", errors="replace")
        with handle_cm as handle:
            reader = csv.reader(handle, delimiter=_sniff_delimiter(path))
            return [str(value) for value in next(reader, [])]
    except Exception:
        return []


def _deterministic_plan(summaries: list[InputFileSummary]) -> dict[str, Any]:
    rna_paths = [Path(item.path) for item in summaries if item.kind_hint == "rna" and item.exists]
    tcr_paths = [Path(item.path) for item in summaries if item.kind_hint == "tcr" and item.exists]
    rna = _expand_rna_candidates(rna_paths)
    tcr = _expand_tcr_candidates(tcr_paths)
    return {
        "rna_source_path": str(rna[0]) if rna else "",
        "rna_format": _format_label(rna[0]) if rna else "",
        "tcr_source_path": str(tcr[0]) if tcr else "",
        "tcr_format": _format_label(tcr[0]) if tcr else "",
        "barcode_column": "",
        "clonotype_column": "",
        "sample_column": "",
        "warnings": [],
    }


def _expand_rna_candidates(paths: list[Path]) -> list[Path]:
    candidates: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        if path.is_dir():
            for nested in ["filtered_feature_bc_matrix", "raw_feature_bc_matrix"]:
                if _looks_like_10x_matrix_dir(path / nested):
                    candidates.append(path / nested)
            if _looks_like_10x_matrix_dir(path):
                candidates.append(path)
            candidates.extend(
                child for child in sorted(path.rglob("*")) if child.is_file() and _looks_like_rna_file(child)
            )
        elif _looks_like_rna_file(path):
            candidates.append(path)
    return _dedupe_paths(candidates)


def _expand_tcr_candidates(paths: list[Path]) -> list[Path]:
    candidates: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        if path.is_dir():
            for preferred in TCR_FILE_NAMES:
                candidates.extend(sorted(path.rglob(preferred)))
            candidates.extend(
                child for child in sorted(path.rglob("*")) if child.is_file() and _looks_like_tcr_file(child)
            )
        elif _is_text_table(path) or _looks_like_tcr_file(path):
            candidates.append(path)
    return _dedupe_paths(candidates)


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path.resolve()).lower()
        if key not in seen:
            seen.add(key)
            unique.append(path.resolve())
    return unique


def _looks_like_rna_file(path: Path) -> bool:
    name = path.name.lower()
    if name.endswith((".csv.gz", ".tsv.gz", ".txt.gz", ".mtx.gz")):
        return True
    return path.suffix.lower() in RNA_FILE_SUFFIXES


def _looks_like_tcr_file(path: Path) -> bool:
    name = path.name.lower()
    if name in TCR_FILE_NAMES:
        return True
    if name.endswith((".csv.gz", ".tsv.gz", ".txt.gz")):
        return True
    if path.suffix.lower() not in TCR_FILE_SUFFIXES:
        return False
    lowered = str(path).lower()
    return any(token in lowered for token in ("tcr", "vdj", "contig", "clonotype", "airr"))


def _looks_like_10x_matrix_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    names = {child.name.lower() for child in path.iterdir()}
    return (
        ("matrix.mtx" in names or "matrix.mtx.gz" in names)
        and ("barcodes.tsv" in names or "barcodes.tsv.gz" in names)
        and (
            "features.tsv" in names
            or "features.tsv.gz" in names
            or "genes.tsv" in names
            or "genes.tsv.gz" in names
        )
    )


def _read_10x_directory(path: Path):
    try:
        import scanpy as sc
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError('10x matrix directory input requires scanpy: pip install -e ".[analysis]"') from exc
    if not _looks_like_10x_matrix_dir(path):
        raise ValueError(f"Directory does not look like a 10x matrix folder: {path}")
    return sc.read_10x_mtx(str(path), var_names="gene_symbols", make_unique=True)


def _resolve_tcr_table(source: Path) -> Path:
    if source.is_dir():
        candidates = _expand_tcr_candidates([source])
        if not candidates:
            raise FileNotFoundError(f"No supported TCR table found under {source}")
        return candidates[0]
    return source


def _is_text_table(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith((".csv", ".tsv", ".txt", ".csv.gz", ".tsv.gz", ".txt.gz"))


def _sniff_delimiter(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".csv") or name.endswith(".csv.gz"):
        return ","
    return "\t"


def _should_transpose_expression_matrix(matrix) -> bool:
    index_names = [str(value) for value in list(matrix.index[:50])]
    column_names = [str(value) for value in list(matrix.columns[:50])]
    index_barcode_score = sum(_looks_like_cell_barcode(value) for value in index_names)
    column_barcode_score = sum(_looks_like_cell_barcode(value) for value in column_names)
    if column_barcode_score > index_barcode_score:
        return True
    if len(matrix.columns) > len(matrix.index) * 2:
        return True
    return False


def _looks_like_cell_barcode(value: str) -> bool:
    value = value.strip()
    if re.search(r"-\d+$", value) and len(value) >= 12:
        return True
    if len(value) >= 14 and re.fullmatch(r"[ACGTNacgtn]+", value):
        return True
    return False


def _choose_column(columns, planned: str, candidates: list[str]) -> str:
    col_list = [str(col) for col in columns]
    if planned and planned in col_list:
        return planned
    lowered = {col.lower(): col for col in col_list}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    for col in col_list:
        low = col.lower()
        if any(candidate.lower() in low for candidate in candidates):
            return col
    return ""


def _path_from_plan(value: Any, allowed_roots: list[Path]) -> Path | None:
    if not value:
        return None
    planned = Path(str(value)).expanduser().resolve()
    if planned.exists():
        return planned
    allowed = _expand_rna_candidates(allowed_roots) + _expand_tcr_candidates(allowed_roots)
    for path in allowed:
        if str(path).lower() == str(planned).lower() or path.name.lower() == planned.name.lower():
            return path
    return None


def _format_label(path: Path) -> str:
    if path.is_dir():
        return "10x_mtx_directory"
    name = path.name.lower()
    if name.endswith(".h5ad"):
        return "h5ad"
    if name.endswith((".h5", ".hdf5")):
        return "10x_hdf5_or_hdf5"
    if name.endswith(".loom"):
        return "loom"
    if name.endswith(".zarr"):
        return "zarr"
    if _is_text_table(path):
        return "text_table"
    return path.suffix.lower().lstrip(".")


def _extract_json_block(text: str, marker: str) -> dict[str, Any]:
    match = re.search(rf"{marker}\s*(\{{.*?\}})\s*END_{marker}", text or "", flags=re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(1))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}
