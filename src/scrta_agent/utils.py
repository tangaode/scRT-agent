from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def utc_timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_text(path: str | Path | None) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="replace")


def write_text(path: str | Path, text: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def write_json(path: str | Path, data: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def slugify(value: str, default: str = "run") -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or default


def truncate_text(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]"


def bullet_list(items: list[str], empty: str = "none") -> str:
    if not items:
        return f"- {empty}"
    return "\n".join(f"- {item}" for item in items)


def compact_dict(data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if v not in (None, "", [], {})}
