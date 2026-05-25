from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AgentSpec:
    name: str
    role: str
    capabilities: list[str] = field(default_factory=list)


@dataclass
class DatasetProfile:
    rna_path: str
    tcr_path: str
    rna_summary: list[str] = field(default_factory=list)
    tcr_summary: list[str] = field(default_factory=list)
    metadata_inventory: list[str] = field(default_factory=list)
    inferred_join_keys: list[str] = field(default_factory=list)
    guardrails: list[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        sections = [
            "# Dataset Profile",
            "## RNA",
            *[f"- {x}" for x in self.rna_summary],
            "## TCR",
            *[f"- {x}" for x in self.tcr_summary],
            "## Metadata Inventory",
            *[f"- {x}" for x in self.metadata_inventory],
            "## Inferred Join Keys",
            *[f"- {x}" for x in self.inferred_join_keys],
            "## Guardrails",
            *[f"- {x}" for x in self.guardrails],
        ]
        return "\n".join(sections)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LiteratureCard:
    title: str
    year: str = ""
    disease_or_condition: str = ""
    core_hypothesis: str = ""
    transferable_analysis_templates: str = ""
    source_url: str = ""
    relevance_score: float = 0.0

    def to_prompt_block(self) -> str:
        parts = [
            f"Title: {self.title}",
            f"Year: {self.year}" if self.year else "",
            f"Disease/context: {self.disease_or_condition}" if self.disease_or_condition else "",
            f"Core hypothesis: {self.core_hypothesis}" if self.core_hypothesis else "",
            (
                "Transferable analysis templates: "
                f"{self.transferable_analysis_templates}"
                if self.transferable_analysis_templates
                else ""
            ),
            f"Source: {self.source_url}" if self.source_url else "",
        ]
        return "\n".join(p for p in parts if p)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentResponse:
    agent_name: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisTask:
    task_id: str
    title: str
    rationale: str
    module: str
    outputs: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    status: str = "planned"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisPlan:
    title: str
    objective: str
    tasks: list[AnalysisTask] = field(default_factory=list)
    guardrails: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", "", f"Objective: {self.objective}", ""]
        lines.extend(["## Tasks", ""])
        for task in self.tasks:
            lines.extend(
                [
                    f"### {task.task_id}: {task.title}",
                    f"- Module: {task.module}",
                    f"- Status: {task.status}",
                    f"- Rationale: {task.rationale}",
                    f"- Outputs: {', '.join(task.outputs) if task.outputs else 'none'}",
                    f"- Dependencies: {', '.join(task.dependencies) if task.dependencies else 'none'}",
                    "",
                ]
            )
        if self.guardrails:
            lines.extend(["## Guardrails", ""])
            lines.extend(f"- {item}" for item in self.guardrails)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


@dataclass
class WorkflowConfig:
    rna_h5ad_path: str
    tcr_path: str
    analysis_name: str = "scrna_sctcr_case"
    output_root: str = "runs"
    research_brief_path: str | None = None
    research_brief: str = ""
    literature_cards_path: str | None = None
    rag_index_path: str | None = None
    rag_top_k: int = 10
    execute_script: bool = False
    use_llm: bool = True
    model: str = "gpt-5.4"
    analysis_loops: int = 6
    repair_attempts: int = 1
    script_timeout_seconds: int = 7200
    deep_dive_enabled: bool = True
    mechanism_loop_enabled: bool = True
    downstream_analysis_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def output_root_path(self) -> Path:
        return Path(self.output_root)


@dataclass
class WorkflowState:
    config: WorkflowConfig
    run_dir: Path
    profile: DatasetProfile | None = None
    literature_cards: list[LiteratureCard] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)

    def add_artifact(self, name: str, path: str | Path) -> None:
        self.artifacts[name] = str(path)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "run_dir": str(self.run_dir),
            "profile": self.profile.to_dict() if self.profile else None,
            "literature_cards": [card.to_dict() for card in self.literature_cards],
            "artifacts": self.artifacts,
        }
