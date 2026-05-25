from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .agents import ScRTATeam
from .llm import LLMClient
from .schemas import WorkflowConfig
from .workflow import ScRTAWorkflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scrta-agent",
        description="Focused multi-agent workflow for paired scRNA-seq and scTCR-seq analysis.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="create a scRNA-scTCR analysis run")
    run.add_argument("--config", help="optional JSON config file")
    run.add_argument("--rna", dest="rna_h5ad_path", help="RNA .h5ad path")
    run.add_argument("--tcr", dest="tcr_path", help="TCR contig/clonotype table path")
    run.add_argument("--analysis-name", default=None, help="run name")
    run.add_argument("--out", dest="output_root", default=None, help="output root directory")
    run.add_argument("--brief", dest="research_brief", default=None, help="short research brief")
    run.add_argument("--brief-file", dest="research_brief_path", default=None, help="research brief text file")
    run.add_argument("--literature-cards", dest="literature_cards_path", default=None, help="local literature cards CSV")
    run.add_argument("--rag-index", dest="rag_index_path", default=None, help="RAG chunks JSONL path")
    run.add_argument("--rag-top-k", dest="rag_top_k", type=int, default=None, help="RAG chunks per agent call")
    run.add_argument("--execute", action="store_true", help="execute the generated analysis script")
    run.add_argument("--model", default=None, help="LLM model name")
    run.add_argument("--analysis-loops", dest="analysis_loops", type=int, default=None, help="planned analysis loop count")
    run.add_argument("--repair-attempts", dest="repair_attempts", type=int, default=None, help="script rerun attempts after failure")
    run.add_argument("--script-timeout", dest="script_timeout_seconds", type=int, default=None, help="script timeout in seconds")
    run.add_argument("--no-deep-dive", dest="deep_dive_enabled", action="store_false", help="disable hypothesis deep-dive loop")
    run.add_argument(
        "--no-mechanism-loop",
        dest="mechanism_loop_enabled",
        action="store_false",
        help="disable biological interpretation, mechanism mapping, and next-test proposal loop",
    )
    run.add_argument(
        "--no-downstream-analysis",
        dest="downstream_analysis_enabled",
        action="store_false",
        help="disable RAG-grounded downstream analysis after hypothesis selection",
    )

    agents = subparsers.add_parser("agents", help="list fixed role agents")
    agents.add_argument("--json", action="store_true", help="print JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "agents":
        team = ScRTATeam(LLMClient(use_llm=False))
        agents = team.list_agents()
        if args.json:
            print(json.dumps(agents, indent=2, ensure_ascii=False))
        else:
            for agent in agents:
                caps = ", ".join(agent["capabilities"])
                print(f"{agent['name']}\t{agent['role']}\t{caps}")
        return 0

    if args.command == "run":
        config = _build_config(args, parser)
        state = ScRTAWorkflow(config).run()
        print(f"Run directory: {state.run_dir}")
        print("Artifacts:")
        for name, path in sorted(state.artifacts.items()):
            print(f"- {name}: {path}")
        return 0

    parser.print_help()
    return 0


def _build_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> WorkflowConfig:
    data: dict[str, Any] = {}
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            parser.error(f"config not found: {config_path}")
        data.update(json.loads(config_path.read_text(encoding="utf-8")))

    overrides = {
        "rna_h5ad_path": args.rna_h5ad_path,
        "tcr_path": args.tcr_path,
        "analysis_name": args.analysis_name,
        "output_root": args.output_root,
        "research_brief": args.research_brief,
        "research_brief_path": args.research_brief_path,
        "literature_cards_path": args.literature_cards_path,
        "rag_index_path": args.rag_index_path,
        "rag_top_k": args.rag_top_k,
        "model": args.model,
        "analysis_loops": args.analysis_loops,
        "repair_attempts": args.repair_attempts,
        "script_timeout_seconds": args.script_timeout_seconds,
    }
    for key, value in overrides.items():
        if value not in (None, ""):
            data[key] = value
    if args.execute:
        data["execute_script"] = True
    data["use_llm"] = True
    if args.deep_dive_enabled is False:
        data["deep_dive_enabled"] = False
    if args.mechanism_loop_enabled is False:
        data["mechanism_loop_enabled"] = False
    if args.downstream_analysis_enabled is False:
        data["downstream_analysis_enabled"] = False

    if not data.get("rna_h5ad_path"):
        parser.error("--rna is required unless provided in --config")
    if not data.get("tcr_path"):
        parser.error("--tcr is required unless provided in --config")
    return WorkflowConfig(**data)


if __name__ == "__main__":
    raise SystemExit(main())
