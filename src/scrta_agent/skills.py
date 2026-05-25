from __future__ import annotations

from pathlib import Path


DEFAULT_SKILL_CANDIDATES = [
    Path("E:/scRTA/scrta-agent/skills/scRNA_scTCR_workflow.md"),
]


def load_skill_context(paths: list[str | Path] | None = None) -> str:
    """Load local operating rules that should be injected into agent context."""
    candidates = [Path(p) for p in paths] if paths else DEFAULT_SKILL_CANDIDATES
    parts = []
    for path in candidates:
        if path.exists():
            parts.append(f"# Skill: {path.name}\n\n{path.read_text(encoding='utf-8', errors='replace')}")
    return "\n\n".join(parts).strip()
