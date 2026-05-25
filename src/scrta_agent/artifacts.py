from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .utils import ensure_dir, utc_timestamp, write_json, write_text


class ArtifactStore:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = ensure_dir(run_dir)
        self.events_path = self.run_dir / "session_events.jsonl"
        self.timeline_path = self.run_dir / "session_timeline.md"
        self._timeline_lines: list[str] = ["# Session Timeline", ""]

    def event(self, kind: str, payload: dict[str, Any]) -> None:
        record = {"time": utc_timestamp(), "kind": kind, "payload": payload}
        with self.events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._timeline_lines.append(f"- `{record['time']}` **{kind}**: {payload.get('summary', '')}")
        write_text(self.timeline_path, "\n".join(self._timeline_lines).strip() + "\n")

    def write_markdown(self, name: str, text: str) -> Path:
        path = self.run_dir / f"{name}.md"
        write_text(path, text.strip() + "\n")
        self.event("artifact", {"summary": f"wrote {path.name}", "path": str(path)})
        return path

    def write_json(self, name: str, data: Any) -> Path:
        path = self.run_dir / f"{name}.json"
        write_json(path, data)
        self.event("artifact", {"summary": f"wrote {path.name}", "path": str(path)})
        return path

    def write_script(self, name: str, text: str) -> Path:
        scripts_dir = ensure_dir(self.run_dir / "scripts")
        path = scripts_dir / name
        write_text(path, text.rstrip() + "\n")
        self.event("artifact", {"summary": f"wrote scripts/{name}", "path": str(path)})
        return path
