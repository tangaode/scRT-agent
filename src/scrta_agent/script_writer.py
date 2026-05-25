from __future__ import annotations

import json
import re
from importlib import resources
from pathlib import Path
from string import Template

from .schemas import AnalysisPlan, WorkflowConfig
from .deep_dive import DeepDiveSelection


def render_joint_analysis_script(
    config: WorkflowConfig,
    run_dir: Path,
    plan: AnalysisPlan | None = None,
) -> str:
    """Render a standalone RNA-first paired scRNA/scTCR analysis script."""
    template = (
        resources.files("scrta_agent.templates")
        .joinpath("joint_analysis_template.py.tmpl")
        .read_text(encoding="utf-8")
    )
    return Template(template).safe_substitute(
        rna_path=json.dumps(str(Path(config.rna_h5ad_path).resolve())),
        tcr_path=json.dumps(str(Path(config.tcr_path).resolve())),
        output_dir=json.dumps(str((run_dir / "analysis_outputs").resolve())),
        analysis_name=json.dumps(config.analysis_name),
        dataset_reconnaissance_contract=json.dumps(plan.to_dict() if plan else {}, ensure_ascii=False),
    )


def render_code_generation_note(plan: AnalysisPlan | None, script_path: Path) -> str:
    lines = [
        "# Code Generation",
        "",
        f"- Script: {script_path}",
        "- Mode: bounded template code generation for dataset reconnaissance and hypothesis-driven follow-up modules",
        "",
    ]
    if plan:
        lines.extend(
            [
                "## Enabled Modules",
                "",
                *[f"- {task.task_id}: {task.module} -> {task.title}" for task in plan.tasks],
                "",
            ]
        )
    lines.extend(
        [
            "## Safety",
            "",
            "- The generated code is standalone and writes all outputs under analysis_outputs/.",
            "- The first script is dataset reconnaissance only; selected-hypothesis deep-dive and downstream agents write later executable scripts.",
            "- LLM-authored follow-up scripts must write reviewable contracts, summaries, and manifests for reproducibility.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def render_publication_figure_script(run_dir: Path, visualizer_output: str = "") -> str:
    """Extract the LLM-authored publication plotting script.

    Publication figures are intentionally not rendered from a fixed template:
    the visualizer agent must inspect the executed result inventory and emit a
    hypothesis-specific standalone Python script.
    """
    script = extract_marked_python_script(
        visualizer_output,
        start_marker="PUBLICATION_FIGURE_PYTHON_SCRIPT",
        end_marker="END_PUBLICATION_FIGURE_PYTHON_SCRIPT",
    )
    analysis_outputs = str((run_dir / "analysis_outputs").resolve())
    run_dir_text = str(run_dir.resolve())
    header = "\n".join(
        [
            "# Auto-extracted from visualizer LLM output.",
            "# The workflow executes this script as the publication figure stage.",
            "from pathlib import Path",
            f"RUN_DIR = Path({json.dumps(run_dir_text)})",
            f"ANALYSIS_OUTPUTS = Path({json.dumps(analysis_outputs)})",
            "FIG_DIR = ANALYSIS_OUTPUTS / 'publication_figures'",
            "FIG_DIR.mkdir(parents=True, exist_ok=True)",
            "",
        ]
    )
    return header + script.rstrip() + "\n"


def render_deep_dive_script(run_dir: Path, selection: DeepDiveSelection, deep_dive_plan: str = "") -> str:
    """Extract the LLM-authored selected-hypothesis deep-dive script.

    Deep-dive validation is intentionally not rendered from a fixed template:
    the deep_planner agent must inspect the selected hypothesis and available
    outputs, then emit a hypothesis-specific standalone Python script.
    """
    try:
        script = extract_marked_python_script(
            deep_dive_plan,
            start_marker="DEEP_DIVE_PYTHON_SCRIPT",
            end_marker="END_DEEP_DIVE_PYTHON_SCRIPT",
        )
    except ValueError:
        script = extract_single_fenced_python_script(deep_dive_plan)
    analysis_outputs = str((run_dir / "analysis_outputs").resolve())
    run_dir_text = str(run_dir.resolve())
    selection_json = json.dumps(selection.to_dict(), ensure_ascii=False)
    header = "\n".join(
        [
            "# Auto-extracted from deep_planner LLM output.",
            "# The workflow executes this script as the selected-hypothesis deep-dive stage.",
            "from pathlib import Path",
            "import json",
            f"RUN_DIR = Path({json.dumps(run_dir_text)})",
            f"ANALYSIS_OUTPUTS = Path({json.dumps(analysis_outputs)})",
            "DEEP_DIVE_DIR = ANALYSIS_OUTPUTS / 'deep_dive'",
            "DEEP_DIVE_DIR.mkdir(parents=True, exist_ok=True)",
            f"SELECTED_HYPOTHESIS = json.loads({json.dumps(selection_json)})",
            "",
        ]
    )
    return header + script.rstrip() + "\n"


def render_biology_mechanism_script(run_dir: Path) -> str:
    """Render a biological interpretation and mechanism-mapping script."""
    template = (
        resources.files("scrta_agent.templates")
        .joinpath("biology_mechanism_template.py.tmpl")
        .read_text(encoding="utf-8")
    )
    return Template(template).safe_substitute(
        analysis_outputs=json.dumps(str((run_dir / "analysis_outputs").resolve())),
    )


def render_downstream_analysis_script(run_dir: Path, downstream_plan: str = "") -> str:
    """Extract the LLM-authored downstream script.

    Downstream analysis is intentionally not rendered from a fixed template:
    the downstream_analyst agent must emit a hypothesis-specific standalone
    Python script between the required markers.
    """
    try:
        script = extract_marked_python_script(
            downstream_plan,
            start_marker="DOWNSTREAM_PYTHON_SCRIPT",
            end_marker="END_DOWNSTREAM_PYTHON_SCRIPT",
        )
    except ValueError:
        script = extract_single_fenced_python_script(downstream_plan)
    analysis_outputs = str((run_dir / "analysis_outputs").resolve())
    header = "\n".join(
        [
            "# Auto-extracted from downstream_analyst LLM output.",
            "# The workflow executes this script as the selected-hypothesis downstream analysis.",
            f"# Expected analysis_outputs root: {analysis_outputs}",
            "",
        ]
    )
    return header + script.rstrip() + "\n"


def extract_marked_python_script(text: str, start_marker: str, end_marker: str) -> str:
    """Return Python code between explicit LLM output markers.

    The marker body may contain either raw Python or a fenced ```python block.
    We deliberately fail when the marker is missing so the workflow does not
    silently fall back to a generic deterministic analysis.
    """
    pattern = re.compile(
        rf"{re.escape(start_marker)}\s*(.*?)\s*{re.escape(end_marker)}",
        flags=re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text or "")
    if not match:
        raise ValueError(
            f"LLM output did not contain a `{start_marker}` ... `{end_marker}` Python script block."
        )
    body = _strip_optional_python_fence(match.group(1).strip())
    if not body:
        raise ValueError(f"LLM Python script block `{start_marker}` was empty.")
    compile(body, "llm_authored_script.py", "exec")
    return body


def extract_single_fenced_python_script(text: str) -> str:
    """Return the only fenced Python block from an LLM response.

    This is still LLM-authored code, not a deterministic template fallback.
    It exists because some models follow the instruction to write a complete
    Python block but omit the surrounding marker labels. We accept this only
    when there is exactly one fenced Python block, so ambiguity still fails.
    """
    blocks = re.findall(r"```(?:python|py)\s*(.*?)\s*```", text or "", flags=re.IGNORECASE | re.DOTALL)
    if len(blocks) != 1:
        raise ValueError(
            "LLM output did not contain the required markers and did not contain exactly one fenced Python block."
        )
    body = blocks[0].strip()
    body = _strip_optional_python_fence(body)
    if not body:
        raise ValueError("LLM fenced Python script block was empty.")
    compile(body, "llm_authored_script.py", "exec")
    return body


def _strip_optional_python_fence(body: str) -> str:
    """Strip an optional Markdown Python fence from LLM-authored script text.

    Some models put a fenced block inside the required marker labels and then
    add short notes after the closing fence. In that case the script is still
    unambiguous: the first fenced Python block is the executable body.
    """
    body = (body or "").strip()
    if not body.startswith("```"):
        return body
    fence = re.match(r"^```(?:python|py)?\s*(.*?)\s*```", body, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        return fence.group(1).strip()
    first_newline = body.find("\n")
    if first_newline != -1:
        return body[first_newline + 1 :].strip()
    return body
