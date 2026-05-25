from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any


TEXT_EXTENSIONS = {".md", ".txt"}
TABLE_EXTENSIONS = {".csv", ".tsv"}
IMAGE_EXTENSIONS = {".png", ".pdf", ".svg"}
JSON_EXTENSIONS = {".json", ".jsonl"}


def _read_table_schema(path: Path, max_preview_rows: int = 3) -> dict[str, Any]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    opener = gzip.open if path.name.endswith(".gz") else open
    try:
        with opener(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            header = next(reader, [])
            preview = []
            for row in reader:
                preview.append(row[: min(len(row), 12)])
                if len(preview) >= max_preview_rows:
                    break
    except Exception as exc:
        return {"readable": False, "error": str(exc)}
    return {
        "readable": True,
        "columns": header,
        "n_columns": len(header),
        "preview_rows": preview,
    }


def _json_schema(path: Path) -> dict[str, Any]:
    try:
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                first = handle.readline().strip()
            data = json.loads(first) if first else {}
        else:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        return {"readable": False, "error": str(exc)}
    if isinstance(data, dict):
        return {"readable": True, "json_type": "object", "keys": sorted(map(str, data.keys()))[:80]}
    if isinstance(data, list):
        first = data[0] if data else {}
        keys = sorted(map(str, first.keys()))[:80] if isinstance(first, dict) else []
        return {"readable": True, "json_type": "array", "length": len(data), "first_item_keys": keys}
    return {"readable": True, "json_type": type(data).__name__}


def build_result_inventory(run_dir: Path, max_files: int = 260) -> dict[str, Any]:
    """Describe the executed result artifacts available for figure design.

    The visualizer should choose figures from this inventory after analyses have
    actually run. This prevents figure specs from referencing planned-but-missing
    outputs.
    """

    output_dir = run_dir / "analysis_outputs"
    files: list[dict[str, Any]] = []
    if not output_dir.exists():
        return {"analysis_outputs": str(output_dir), "files": []}

    all_files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    for path in all_files[:max_files]:
        rel = path.relative_to(output_dir).as_posix()
        suffix = path.suffix.lower()
        if path.name.endswith(".csv.gz"):
            suffix = ".csv"
        elif path.name.endswith(".tsv.gz"):
            suffix = ".tsv"
        entry: dict[str, Any] = {
            "path": rel,
            "name": path.name,
            "bytes": path.stat().st_size,
            "kind": "other",
        }
        if suffix in TABLE_EXTENSIONS:
            entry["kind"] = "table"
            entry.update(_read_table_schema(path))
        elif suffix in IMAGE_EXTENSIONS:
            entry["kind"] = "image"
        elif suffix in JSON_EXTENSIONS:
            entry["kind"] = "json"
            entry.update(_json_schema(path))
        elif suffix in TEXT_EXTENSIONS:
            entry["kind"] = "text"
        files.append(entry)

    return {
        "analysis_outputs": str(output_dir),
        "file_count": len(all_files),
        "listed_file_count": len(files),
        "files": files,
    }


def render_result_inventory_markdown(inventory: dict[str, Any], max_columns: int = 28) -> str:
    lines = [
        "# Available Result Inventory",
        "",
        "This inventory is generated after the executed analyses finish. The visualizer",
        "must choose publication panels from these existing files only. If a desired",
        "result table is absent, the corresponding figure panel must be omitted rather",
        "than represented as a text placeholder.",
        "",
        f"- Analysis outputs: `{inventory.get('analysis_outputs', '')}`",
        f"- Files listed: {inventory.get('listed_file_count', 0)} / {inventory.get('file_count', 0)}",
        "",
    ]

    for kind in ["table", "image", "json", "text", "other"]:
        entries = [f for f in inventory.get("files", []) if f.get("kind") == kind]
        if not entries:
            continue
        lines.extend([f"## {kind.title()} Files", ""])
        for entry in entries:
            path = entry.get("path", "")
            lines.append(f"- `{path}` ({entry.get('bytes', 0)} bytes)")
            if kind == "table" and entry.get("readable"):
                columns = entry.get("columns", [])
                preview = ", ".join(f"`{col}`" for col in columns[:max_columns])
                if len(columns) > max_columns:
                    preview += f", ... (+{len(columns) - max_columns})"
                lines.append(f"  - columns: {preview}")
            elif kind == "json" and entry.get("readable"):
                keys = entry.get("keys") or entry.get("first_item_keys") or []
                if keys:
                    lines.append("  - keys: " + ", ".join(f"`{key}`" for key in keys[:max_columns]))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
