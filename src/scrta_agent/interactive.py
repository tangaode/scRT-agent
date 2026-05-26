from __future__ import annotations

from pathlib import Path

from .data_import import prepare_inputs
from .llm import LLMClient
from .schemas import WorkflowConfig
from .utils import ensure_dir, slugify, utc_timestamp
from .workflow import ScRTAWorkflow


def run_interactive_wizard() -> int:
    print("scRT-agent interactive workflow")
    print("Provide input paths; the wizard will prepare an h5ad file and run the paired scRNA/scTCR workflow.")
    print("Use semicolons to provide multiple paths for the same input group.")
    print("")

    analysis_name = _ask("Analysis name", "scrna_sctcr_case")
    output_root = _ask("Output root", "runs")
    model = _ask("LLM model", "gpt-5.4")
    rna_inputs = _ask_required("RNA input path(s)")
    tcr_inputs = _ask_required("TCR input path(s)")
    research_brief = _ask("Short research brief", "")
    literature_cards = _ask("Literature cards CSV path (optional)", "")
    rag_index = _ask("RAG chunks JSONL path (optional)", "")
    execute = _ask_yes_no("Execute generated scripts", True)
    plan_review = _ask_yes_no("Review and edit selected-hypothesis plans before execution", True)
    repair_attempts = int(_ask("Script repair attempts", "1"))
    analysis_loops = int(_ask("Hypothesis refinement loop count", "6"))

    prepared_root = ensure_dir(Path(output_root) / "prepared_inputs")
    prepared_dir = prepared_root / f"{slugify(analysis_name, 'case')}_{utc_timestamp()}"
    llm = LLMClient(model=model, use_llm=True)

    print("")
    print("Preparing input data with LLM-assisted source selection...")
    prepared = prepare_inputs(
        rna_inputs=rna_inputs,
        tcr_inputs=tcr_inputs,
        output_dir=prepared_dir,
        llm=llm,
        analysis_name=analysis_name,
        require_llm_plan=True,
    )
    print(f"Prepared RNA h5ad: {prepared.rna_h5ad_path}")
    print(f"Prepared TCR table: {prepared.tcr_path}")
    print(f"Preparation manifest: {prepared.manifest_path}")

    config = WorkflowConfig(
        rna_h5ad_path=prepared.rna_h5ad_path,
        tcr_path=prepared.tcr_path,
        analysis_name=analysis_name,
        output_root=output_root,
        research_brief=research_brief,
        literature_cards_path=literature_cards or None,
        rag_index_path=rag_index or None,
        execute_script=execute,
        use_llm=True,
        model=model,
        analysis_loops=analysis_loops,
        repair_attempts=repair_attempts,
        interactive_plan_review=plan_review,
        interactive_hypothesis_selection=True,
    )
    print("")
    print("Starting workflow. You will be asked to select and optionally edit the hypothesis after candidates are generated.")
    state = ScRTAWorkflow(config, llm=llm).run()
    print("")
    print(f"Run directory: {state.run_dir}")
    print("Key artifacts:")
    for key in [
        "prepared_inputs_manifest",
        "rag_grounded_hypothesis_candidates",
        "selected_hypothesis",
        "selected_hypothesis_deep_dive_plan",
        "hypothesis_downstream_plan",
        "publication_figure_execution",
        "final_report",
    ]:
        path = state.artifacts.get(key)
        if path:
            print(f"- {key}: {path}")
    return 0


def _ask(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value if value else default


def _ask_required(label: str) -> str:
    while True:
        value = input(f"{label}: ").strip()
        if value:
            return value
        print("This value is required.")


def _ask_yes_no(label: str, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    value = input(f"{label} [{suffix}]: ").strip().lower()
    if not value:
        return default
    return value in {"y", "yes", "1", "true", "on"}
