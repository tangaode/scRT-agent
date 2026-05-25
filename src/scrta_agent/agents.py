from __future__ import annotations

from dataclasses import dataclass
import json
from importlib import resources
from typing import Callable

from .llm import LLMClient
from .schemas import AgentResponse, AgentSpec
from .utils import bullet_list, truncate_text


FallbackFn = Callable[[str, dict], str]


def _load_prompt(name: str) -> str:
    try:
        return resources.files("scrta_agent.prompts").joinpath(f"{name}.md").read_text(encoding="utf-8")
    except Exception:
        return f"You are the {name} agent in a scRNA-scTCR analysis team."


@dataclass
class ScRTAAgent:
    spec: AgentSpec
    llm: LLMClient
    fallback: FallbackFn

    @property
    def system_prompt(self) -> str:
        return _load_prompt(self.spec.name)

    def run(self, instruction: str, context: dict) -> AgentResponse:
        if self.llm.use_llm:
            prompt = self._build_prompt(instruction, context)
            content = self.llm.complete(self.system_prompt, prompt)
            return AgentResponse(
                agent_name=self.spec.name,
                content=content,
                metadata={"mode": "llm", "role": self.spec.role},
            )
        content = self.fallback(instruction, context)
        return AgentResponse(
            agent_name=self.spec.name,
            content=content,
            metadata={"mode": "deterministic", "role": self.spec.role},
        )

    def _build_prompt(self, instruction: str, context: dict) -> str:
        profile = truncate_text(context.get("dataset_profile", ""), 6000)
        literature = truncate_text(context.get("literature_context", ""), 9000)
        skills = truncate_text(context.get("skill_context", ""), 5000)
        environment = truncate_text(context.get("environment_context", ""), 4000)
        brief = truncate_text(context.get("research_brief", ""), 3000)
        prior = truncate_text(context.get("prior_outputs", ""), 5000)
        task_context = truncate_text(_format_task_context(context), 12000)
        return f"""# Instruction
{instruction}

# Dataset Profile
{profile}

# Research Brief
{brief}

# Environment Context
{environment}

# Skill Context
{skills}

# Literature Context
{literature}

# Task-Specific Context
{task_context}

# Requirement
Write all internal agent artifacts, scientific reasoning, plans, selections,
and machine-readable blocks in English. Workflow artifacts are manuscript
source material and must keep a consistent scientific vocabulary.

Use the RAG evidence above as prior scientific context, not as a target to
replicate. Separate (1) what prior papers already found, (2) what transferable
analysis pattern they teach, and (3) what new, dataset-testable hypothesis can
be proposed here. Prefer hypotheses that are meaningful, falsifiable, and not
just a restatement of a source paper. Keep TCR claims conservative unless the
retrieved evidence and current dataset support stronger language.

# Prior Team Outputs
{prior}
"""


def _format_task_context(context: dict) -> str:
    ignored = {
        "dataset_profile",
        "literature_context",
        "skill_context",
        "environment_context",
        "research_brief",
        "prior_outputs",
    }
    sections: list[str] = []
    for key in sorted(context):
        if key in ignored:
            continue
        value = context.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, str):
            rendered = value
        else:
            try:
                rendered = json.dumps(value, ensure_ascii=False, indent=2)
            except TypeError:
                rendered = str(value)
        sections.append(f"## {key}\n{rendered}")
    if not sections:
        return "No additional task-specific context was provided."
    return "\n\n".join(sections)


class ScRTATeam:
    """Fixed agent-as-tool team for paired scRNA/scTCR analysis."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.agents = {
            "leader": ScRTAAgent(
                AgentSpec(
                    name="leader",
                    role="workflow_orchestrator",
                    capabilities=["plan", "budget", "branch_selection"],
                ),
                llm,
                leader_fallback,
            ),
            "rna_analyst": ScRTAAgent(
                AgentSpec(
                    name="rna_analyst",
                    role="rna_state_program_designer",
                    capabilities=["qc", "annotation", "programs", "differential_state"],
                ),
                llm,
                rna_fallback,
            ),
            "methodologist": ScRTAAgent(
                AgentSpec(
                    name="methodologist",
                    role="statistical_design_and_controls",
                    capabilities=[
                        "patient_blocked_stats",
                        "clone_size_nulls",
                        "within_state_tests",
                    ],
                ),
                llm,
                methodologist_fallback,
            ),
            "tcr_analyst": ScRTAAgent(
                AgentSpec(
                    name="tcr_analyst",
                    role="clone_lineage_support",
                    capabilities=["clonotype_join", "clone_expansion", "lineage_controls"],
                ),
                llm,
                tcr_fallback,
            ),
            "t_cell_annotator": ScRTAAgent(
                AgentSpec(
                    name="t_cell_annotator",
                    role="llm_t_cell_subcluster_annotator",
                    capabilities=[
                        "t_cell_marker_interpretation",
                        "subcluster_labeling",
                        "annotation_confidence",
                    ],
                ),
                llm,
                t_cell_annotator_fallback,
            ),
            "integrator": ScRTAAgent(
                AgentSpec(
                    name="integrator",
                    role="cross_modal_hypothesis_builder",
                    capabilities=["rna_first_tcr_support", "branch_tests"],
                ),
                llm,
                integrator_fallback,
            ),
            "novelty_scout": ScRTAAgent(
                AgentSpec(
                    name="novelty_scout",
                    role="rag_to_novel_hypothesis_scout",
                    capabilities=["prior_work_mapping", "novelty_filtering", "hypothesis_gap_finding"],
                ),
                llm,
                novelty_scout_fallback,
            ),
            "hypothesis_selector": ScRTAAgent(
                AgentSpec(
                    name="hypothesis_selector",
                    role="candidate_hypothesis_selector",
                    capabilities=["hypothesis_ranking", "evidence_mapping", "branch_selection"],
                ),
                llm,
                hypothesis_selector_fallback,
            ),
            "hypothesis_generator": ScRTAAgent(
                AgentSpec(
                    name="hypothesis_generator",
                    role="rag_grounded_hypothesis_generator",
                    capabilities=[
                        "rag_literature_synthesis",
                        "dataset_reconnaissance_interpretation",
                        "novel_hypothesis_generation",
                    ],
                ),
                llm,
                hypothesis_generator_fallback,
            ),
            "deep_planner": ScRTAAgent(
                AgentSpec(
                    name="deep_planner",
                    role="selected_hypothesis_deep_dive_planner",
                    capabilities=["targeted_validation", "followup_tests", "stopping_rules"],
                ),
                llm,
                deep_planner_fallback,
            ),
            "skeptic": ScRTAAgent(
                AgentSpec(
                    name="skeptic",
                    role="confounder_and_overclaim_gate",
                    capabilities=["negative_controls", "overclaim_blocking", "ranking"],
                ),
                llm,
                skeptic_fallback,
            ),
            "code_writer": ScRTAAgent(
                AgentSpec(
                    name="code_writer",
                    role="plan_to_reproducible_code",
                    capabilities=["script_generation", "execution_repair", "artifact_contracts"],
                ),
                llm,
                code_writer_fallback,
            ),
            "visualizer": ScRTAAgent(
                AgentSpec(
                    name="visualizer",
                    role="publication_figure_designer",
                    capabilities=["python_plotting", "figure_panels", "visual_qc"],
                ),
                llm,
                visualizer_fallback,
            ),
            "result_interpreter": ScRTAAgent(
                AgentSpec(
                    name="result_interpreter",
                    role="deep_dive_result_interpreter",
                    capabilities=["support_assessment", "next_step_decision", "claim_calibration"],
                ),
                llm,
                result_interpreter_fallback,
            ),
            "biological_interpreter": ScRTAAgent(
                AgentSpec(
                    name="biological_interpreter",
                    role="biology_meaning_interpreter",
                    capabilities=["biological_meaning", "immune_context", "claim_calibration"],
                ),
                llm,
                biological_interpreter_fallback,
            ),
            "mechanism_mapper": ScRTAAgent(
                AgentSpec(
                    name="mechanism_mapper",
                    role="mechanism_evidence_mapper",
                    capabilities=["mechanism_mapping", "pathway_prioritization", "gene_program_annotation"],
                ),
                llm,
                mechanism_mapper_fallback,
            ),
            "next_test_planner": ScRTAAgent(
                AgentSpec(
                    name="next_test_planner",
                    role="downstream_validation_planner",
                    capabilities=["next_test_proposal", "external_validation", "orthogonal_validation"],
                ),
                llm,
                next_test_planner_fallback,
            ),
            "downstream_analyst": ScRTAAgent(
                AgentSpec(
                    name="downstream_analyst",
                    role="rag_grounded_downstream_analysis_agent",
                    capabilities=[
                        "rag_grounded_followup_plan",
                        "hypothesis_driven_scRNA_scTCR_analysis",
                        "scTCR_required_execution_contract",
                    ],
                ),
                llm,
                downstream_analyst_fallback,
            ),
            "system_manager": ScRTAAgent(
                AgentSpec(
                    name="system_manager",
                    role="environment_and_dependency_auditor",
                    capabilities=["environment_audit", "package_check", "runtime_notes"],
                ),
                llm,
                system_manager_fallback,
            ),
            "reporter": ScRTAAgent(
                AgentSpec(
                    name="reporter",
                    role="artifact_reporter",
                    capabilities=["summary", "figure_callouts", "next_steps"],
                ),
                llm,
                reporter_fallback,
            ),
        }

    def list_agents(self) -> list[dict]:
        return [
            {
                "name": agent.spec.name,
                "role": agent.spec.role,
                "capabilities": agent.spec.capabilities,
            }
            for agent in self.agents.values()
        ]

    def call_agent(self, agent_name: str, instruction: str, context: dict) -> AgentResponse:
        if agent_name not in self.agents:
            raise ValueError(f"Unknown agent '{agent_name}'. Available: {list(self.agents)}")
        return self.agents[agent_name].run(instruction, context)


def leader_fallback(instruction: str, context: dict) -> str:
    agents = context.get("agent_list", [])
    return f"""# Leader Plan

## Team Available
{bullet_list([f"{a['name']}: {a['role']}" for a in agents])}

## Operating Principle
- Use an RNA-first workflow: define cell states, disease programs, response contrasts, tissue/timepoint structure, and interpretable gene modules before making TCR claims.
- Use scTCR as lineage support: clone expansion, clone sharing, persistence, replacement, and clone-state coupling.
- Screen multiple branches shallowly before selecting one strong branch for deep validation.

## Execution Stages
1. Build dataset profile and verify RNA/TCR join assumptions.
2. Retrieve local scRNA-scTCR literature patterns relevant to the disease and study design.
3. Design RNA state/program analyses.
4. Design TCR clone-lineage support tests with confounder controls.
5. Integrate RNA and TCR evidence into 3-5 falsifiable hypotheses.
6. Run skeptic audit before execution.
7. Generate a standalone analysis script and final report scaffold.
"""


def rna_fallback(instruction: str, context: dict) -> str:
    return """# RNA Analyst Plan

## Primary RNA Outputs
- QC summary: cell/gene counts, mitochondrial/ribosomal flags when present, sample balance.
- Cell state map: use existing annotations when available; otherwise run neighbors/UMAP/Leiden and marker review.
- Program scores: cytotoxicity, exhaustion/dysfunction, interferon, tissue residency, memory/progenitor, proliferation.
- Contrast tests: response, disease group, tissue, timepoint, treatment arm, or user-defined covariates.

## RNA-First Branches
- State abundance branch: whether disease/response is explained by shifts in state composition.
- Within-state program branch: whether a program changes within the same annotated state.
- Transition branch: whether precursor-to-effector/exhausted gradients differ by condition.
- Patient/tissue branch: whether signals survive patient or tissue stratification.

## Minimum Artifacts
- RNA summary table.
- UMAP colored by state, condition, and core program scores.
- Program-by-state heatmap.
- Candidate contrasts with covariates and failure modes.
"""


def tcr_fallback(instruction: str, context: dict) -> str:
    return """# TCR Analyst Plan

## TCR Support Tests
- Join TCR cells to RNA cells using barcode-like columns; report matched fraction and unmatched causes.
- Define clone identity from patient-scoped CDR3 when present; otherwise use patient-scoped clonotype_id.
- Never merge cohort-level labels such as clonotype1 across patients without receptor sequence evidence.
- Compute clone size, clone expansion bins, clone sharing across samples/tissues/timepoints, and V/J/CDR3 summaries.
- Test whether RNA-defined states are clonally expanded after controlling for sample, patient, tissue, and clone size.

## Guardrails
- Do not infer antigen specificity from expansion alone.
- Do not claim receptor-sequence mechanism before RNA phenotype and clone-state coupling are established.
- Separate lineage persistence/replacement from simple abundance.

## Useful Outputs
- clone_state_summary.csv
- clone_expansion_by_group.csv
- clone_sharing_matrix.csv
- optional clone-size-matched permutation null.
"""


def methodologist_fallback(instruction: str, context: dict) -> str:
    return """# Methodologist Plan

## Required Statistical Modules
- Parse patient, sample, tissue, and timepoint labels before any cohort-level comparison.
- Use patient/sample-level summaries for group comparisons; cell-level tests are descriptive only.
- For longitudinal or tissue contrasts, compute within-patient paired summaries when both levels exist.
- Use clone-size-aware nulls by permuting labels within patient and clone-size strata.
- For CD8/Treg claims, compare groups within the same state rather than mixing state abundance with within-state biology.

## Minimum Outputs
- parsed_sample_metadata.csv
- sample_parse_summary.csv
- patient_blocked_summary.csv
- patient_blocked_tests.csv
- clone_size_null_tests.csv
- within_state_signature_differences.csv
- within_state_de_top_genes.csv
"""


def t_cell_annotator_fallback(instruction: str, context: dict) -> str:
    return """# T Cell Subcluster Annotation

LLM annotation was not available, so T-cell subclusters were left with their
numeric cluster IDs. Inspect `analysis_outputs/t_cell_cluster_marker_summary.csv`
and `analysis_outputs/t_cell_cluster_summary.csv` for marker-program evidence.

T_CELL_ANNOTATION_JSON
{
  "version": "scrta_t_cell_annotation_v1",
  "annotations": []
}
END_T_CELL_ANNOTATION_JSON
"""


def integrator_fallback(instruction: str, context: dict) -> str:
    return """# Integrator Unavailable

LLM mode is required for integrated hypothesis reasoning. No deterministic
hypothesis menu is provided.
"""


def novelty_scout_fallback(instruction: str, context: dict) -> str:
    return """# Novelty Scout Unavailable

LLM mode is required for novelty scouting. No deterministic novelty templates
are provided.
"""


def hypothesis_selector_fallback(instruction: str, context: dict) -> str:
    return """# Hypothesis Selector Unavailable

LLM mode is required for hypothesis selection. No deterministic selector
fallback is provided.
"""


def hypothesis_generator_fallback(instruction: str, context: dict) -> str:
    return """# RAG-Grounded Hypothesis Candidates

LLM mode is unavailable or failed, so no scientific hypothesis candidates were
generated. The workflow will not substitute an old hard-coded hypothesis menu.

HYPOTHESIS_CANDIDATES_JSON
{
  "language": "English",
  "candidates": []
}
END_HYPOTHESIS_CANDIDATES_JSON
"""


def deep_planner_fallback(instruction: str, context: dict) -> str:
    selected = context.get("selected_hypothesis", "No selected hypothesis was provided.")
    return f"""# Deep-Dive Plan

## Selected Hypothesis
{selected}

## Targeted Analyses
1. Translate the selected hypothesis into patient-level, tissue-level, state-level, and clone-support tests.
2. Identify which RNA programs, states, genes, and pathways are needed to test the hypothesis.
3. Add scTCR analyses only as support for lineage, persistence, expansion, sharing, state occupancy, repertoire diversity, or receptor follow-up priority.
4. Define accept, partial-support, and falsification criteria that map directly to the selected hypothesis.
5. Produce a calibrated support statement without adding a new biological claim.
"""


def skeptic_fallback(instruction: str, context: dict) -> str:
    return """# Skeptic Audit

## Required Controls
- Match RNA and TCR barcodes explicitly and report join rate.
- Block or stratify by patient/sample before group-level claims.
- Check whether clone expansion signals remain after conditioning on RNA state.
- Use clone-size-matched or sample-permuted nulls before claiming clone-state coupling.
- Treat low matched TCR fraction as a limitation, not a biological negative.

## Overclaim Blocks
- Expansion alone is not tumor reactivity.
- Shared clonotype alone is not migration.
- CDR3 similarity alone is not antigen identity.
- UMAP proximity alone is not lineage transition.

## Go/No-Go
Proceed with script generation if the first execution focuses on descriptive profiling, RNA programs, matched clone-state summaries, and conservative branch ranking.
"""


def code_writer_fallback(instruction: str, context: dict) -> str:
    return """# Code Writer Contract

## Implementation Strategy
- Generate one standalone Python script under the run directory.
- Keep all outputs under analysis_outputs/ with stable CSV/JSON/Markdown filenames.
- Prefer deterministic bounded modules over arbitrary ad hoc generated code.
- Make failures explicit in output files instead of silently skipping analyses.

## Required Modules
- Environment and input audit.
- Metadata parser for patient, sample, tissue, and timepoint.
- RNA/TCR barcode join and patient-scoped clone IDs.
- RNA signature scoring and UMAP outputs.
- Patient-blocked summaries/tests.
- Clone-size-aware permutation/null tests.
- CD8/Treg within-state signature and top-gene contrasts.
"""


def visualizer_fallback(instruction: str, context: dict) -> str:
    return """# Visualizer Unavailable

The visualizer must be an LLM-authored plotting-code agent. No deterministic
figure specification or fixed renderer fallback is provided.
"""


def result_interpreter_fallback(instruction: str, context: dict) -> str:
    return """# Deep-Dive Result Interpreter

## Interpretation Rules
- Supported: patient-aware and clone-level summaries agree with the selected hypothesis.
- Partially supported: reconnaissance trends remain but clone-level or shared-clone analyses are limited.
- Not supported: clone-level controls contradict the reconnaissance trend.

## Claim Calibration
- Preserve the selected hypothesis wording and calibrate support against it.
- Avoid "antigen specificity", "migration", or "TCR-driven mechanism" unless external evidence is added.
- Recommend another loop only when a decisive missing test can be run with available data.
"""


def biological_interpreter_fallback(instruction: str, context: dict) -> str:
    return """# Biological Interpreter

## Biological Meaning
- Interpret the selected hypothesis directly, using the deep-dive and downstream outputs as evidence.
- Explain RNA programs, cell states, tissue context, patient consistency, and tumor microenvironment mechanisms only when they are relevant to the selected hypothesis.
- Use scTCR as a supporting layer for lineage tracking, persistence, expansion, sharing, state occupancy, repertoire diversity, or receptor follow-up prioritization.
- Same-clone or expanded-clone evidence can support state occupancy, but does not prove antigen specificity.

## Claim Boundary
- Use "consistent with", "supports", and "nominates" for biological interpretation.
- Do not convert clone expansion into receptor mechanism without TCR motif, antigen database, or functional validation.
"""


def mechanism_mapper_fallback(instruction: str, context: dict) -> str:
    return """# Mechanism Mapper Unavailable

LLM mode is required for mechanism mapping. No deterministic mechanism-axis
template is provided.
"""


def next_test_planner_fallback(instruction: str, context: dict) -> str:
    return """# Next-Test Planner Unavailable

LLM mode is required for next-test planning. No deterministic next-test menu is
provided.
"""


def downstream_analyst_fallback(instruction: str, context: dict) -> str:
    return """# Downstream Analyst Unavailable

LLM mode is required for downstream analysis planning and code generation. No
deterministic downstream module menu is provided.
"""


def system_manager_fallback(instruction: str, context: dict) -> str:
    return """# System Manager Notes

## Runtime Checks
- Verify anndata before reading .h5ad files.
- Use scanpy when available for UMAP and signature scoring.
- Use scipy opportunistically for sparse matrices; fall back to simple summaries when unavailable.
- Preserve stdout/stderr logs for every execution attempt.

## Dependency Policy
- Do not install packages inside generated scripts.
- Report missing optional packages as limitations in analysis outputs.
"""


def reporter_fallback(instruction: str, context: dict) -> str:
    artifacts = context.get("artifact_list", [])
    return f"""# scRNA-scTCR Agent Run Report

## Summary
This run created a focused RNA-first paired scRNA/scTCR analysis plan, local literature context, hypothesis deep-dive outputs, biological interpretation, mechanism mapping, next-test proposals, mandatory downstream scTCR analyses, and publication-oriented figures when execution was enabled.

## Produced Artifacts
{bullet_list(artifacts)}

## Key Files To Inspect
- analysis_outputs/deep_dive/deep_dive_conclusion.md
- analysis_outputs/biology_mechanism/biological_interpretation.md
- analysis_outputs/biology_mechanism/mechanism_mapping.md
- analysis_outputs/biology_mechanism/next_test_proposals.md
- analysis_outputs/downstream/downstream_analysis_summary.md
- analysis_outputs/downstream/sctcr_repertoire_by_context.csv
- analysis_outputs/downstream/sctcr_clone_state_coupling.csv
- analysis_outputs/publication_figures/publication_figure_summary.md

## Recommended Next Action
Use downstream_analysis_summary.md and next_test_proposals.md to decide which mechanism should be strengthened first, then run an external cohort or receptor-level validation only after the within-cohort mechanism evidence is stable.
"""
