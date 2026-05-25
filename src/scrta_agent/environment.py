from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path
from typing import Any


ANALYSIS_PACKAGES = [
    "anndata",
    "scanpy",
    "pandas",
    "numpy",
    "scipy",
    "matplotlib",
    "seaborn",
]


def collect_environment(packages: list[str] | None = None) -> dict[str, Any]:
    """Collect a compact environment inventory for reproducible analysis runs."""
    packages = packages or ANALYSIS_PACKAGES
    package_status = {}
    for package in packages:
        spec = importlib.util.find_spec(package)
        package_status[package] = {
            "available": spec is not None,
            "origin": str(spec.origin) if spec and spec.origin else "",
        }

    return {
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "working_directory": str(Path.cwd()),
        "packages": package_status,
    }


def render_environment_markdown(info: dict[str, Any]) -> str:
    lines = [
        "# Environment",
        "",
        f"- Python: {info.get('python', '')}",
        f"- Executable: {info.get('python_executable', '')}",
        f"- Platform: {info.get('platform', '')}",
        f"- Working directory: {info.get('working_directory', '')}",
        "",
        "## Packages",
        "",
    ]
    for name, status in sorted((info.get("packages") or {}).items()):
        marker = "available" if status.get("available") else "missing"
        origin = status.get("origin") or ""
        lines.append(f"- {name}: {marker}" + (f" ({origin})" if origin else ""))
    return "\n".join(lines).rstrip() + "\n"
