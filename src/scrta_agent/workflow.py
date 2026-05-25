from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import shutil

from .agents import ScRTATeam
from .artifacts import ArtifactStore
from .environment import collect_environment, render_environment_markdown
from .execution import execute_python_script
from .deep_dive import (
    DeepDiveSelection,
    extract_hypothesis_candidates,
    normalize_hypothesis_id,
    selection_from_agent_response,
    support_decision_from_result_interpreter,
)
from .literature import load_literature_cards, render_literature_context, retrieve_cards
from .llm import LLMClient
from .profiling import build_dataset_profile
from .rag import load_rag_chunks, render_rag_context, resolve_rag_index, retrieve_rag_chunks
from .result_inventory import build_result_inventory, render_result_inventory_markdown
from .schemas import AgentResponse, WorkflowConfig, WorkflowState
from .script_writer import (
    render_biology_mechanism_script,
    render_code_generation_note,
    render_deep_dive_script,
    render_downstream_analysis_script,
    render_joint_analysis_script,
    render_publication_figure_script,
)
from .skills import load_skill_context
from .utils import ensure_dir, read_text, slugify, truncate_text, utc_timestamp


class ScRTAWorkflow:
    """Pantheon-inspired fixed workflow for paired scRNA/scTCR analysis."""

    def __init__(self, config: WorkflowConfig, llm: LLMClient | None = None) -> None:
        self.config = config
        self.llm = llm or LLMClient(model=config.model, use_llm=config.use_llm)

    def run(self) -> WorkflowState:
        self.config.use_llm = True
        self.llm.use_llm = True
        self.llm.require_ready()
        run_id = f"{slugify(self.config.analysis_name, 'scrta_case')}_{utc_timestamp()}"
        run_dir = ensure_dir(self.config.output_root_path / run_id)
        store = ArtifactStore(run_dir)
        state = WorkflowState(config=self.config, run_dir=run_dir)
        team = ScRTATeam(self.llm)

        store.event("start", {"summary": f"started {self.config.analysis_name}"})
        self._add_artifact(state, "config", store.write_json("config", self.config.to_dict()))

        environment_info = collect_environment()
        environment_context = render_environment_markdown(environment_info)
        self._add_artifact(state, "environment", store.write_markdown("environment", environment_context))
        self._add_artifact(state, "environment_json", store.write_json("environment", environment_info))

        skill_context = load_skill_context()
        if skill_context:
            self._add_artifact(state, "skill_context", store.write_markdown("skill_context", skill_context))

        research_brief = self.config.research_brief.strip()
        if self.config.research_brief_path:
            research_brief = (research_brief + "\n\n" + read_text(self.config.research_brief_path)).strip()

        profile = build_dataset_profile(self.config.rna_h5ad_path, self.config.tcr_path)
        state.profile = profile
        self._add_artifact(state, "dataset_profile_md", store.write_markdown("dataset_profile", profile.to_prompt()))
        self._add_artifact(state, "dataset_profile_json", store.write_json("dataset_profile", profile.to_dict()))

        all_cards = load_literature_cards(self.config.literature_cards_path)
        retrieved_cards = retrieve_cards(all_cards, research_brief + "\n" + profile.to_prompt(), limit=8)
        state.literature_cards = retrieved_cards
        literature_context = render_literature_context(retrieved_cards)
        self._add_artifact(
            state,
            "literature_context",
            store.write_markdown("literature_context", literature_context),
        )
        self._add_artifact(
            state,
            "literature_cards",
            store.write_json("literature_cards", [card.to_dict() for card in retrieved_cards]),
        )
        rag_index = resolve_rag_index(self.config.rag_index_path)
        rag_chunks = load_rag_chunks(rag_index)
        if rag_index:
            self._add_artifact(
                state,
                "rag_index",
                store.write_markdown(
                    "rag_index",
                    f"# RAG Index\n\n- Path: {rag_index}\n- Loaded chunks: {len(rag_chunks)}\n",
                ),
            )
            mode_note = [
                "# RAG Usage Note",
                "",
                f"- RAG index: {rag_index}",
                f"- Loaded chunks: {len(rag_chunks)}",
                "- LLM mode: required",
                f"- LLM available: {self.llm.available}",
                f"- Model: {self.config.model}",
                "",
            ]
            mode_note.extend(
                [
                    "RAG contexts will be injected into LLM agent prompts.",
                    "Inspect `rag_context_*.md` files to verify the retrieved evidence for each agent.",
                    "If the LLM is unavailable, the workflow fails instead of silently using deterministic fallback.",
                ]
            )
            self._add_artifact(state, "rag_usage_note", store.write_markdown("rag_usage_note", "\n".join(mode_note)))

        base_context = {
            "dataset_profile": profile.to_prompt(),
            "research_brief": research_brief or "No user research brief provided.",
            "literature_context": literature_context,
            "skill_context": skill_context,
            "environment_context": environment_context,
            "agent_list": team.list_agents(),
        }
        prior_outputs: list[AgentResponse] = []

        system_manager = self._call_and_store(
            team,
            store,
            state,
            "system_manager",
            "Audit the runtime environment and dependency risks for this run.",
            base_context,
        )
        prior_outputs.append(system_manager)

        leader = self._call_and_store(
            team,
            store,
            state,
            "leader",
            "Create the dataset reconnaissance and initial multi-agent analysis plan for this paired scRNA/scTCR dataset.",
            self._context_with_rag(
                base_context,
                rag_chunks,
                "leader dataset reconnaissance multi-agent analysis plan",
                store=store,
                state=state,
                agent_name="leader",
            ),
        )
        prior_outputs.append(leader)

        rna = self._call_and_store(
            team,
            store,
            state,
            "rna_analyst",
            "Design the RNA-led state, program, and contrast analysis for this dataset.",
            self._context_with_rag(
                self._context_with_prior(base_context, prior_outputs),
                rag_chunks,
                "RNA state program contrast signature marker differential expression scRNA scTCR",
                store=store,
                state=state,
                agent_name="rna_analyst",
            ),
        )
        prior_outputs.append(rna)

        methodologist = self._call_and_store(
            team,
            store,
            state,
            "methodologist",
            "Design patient-blocked, clone-size-aware, and within-state statistical controls.",
            self._context_with_rag(
                self._context_with_prior(base_context, prior_outputs),
                rag_chunks,
                "patient-blocked statistics clone-size-aware null CD8 Treg within-state differential",
                store=store,
                state=state,
                agent_name="methodologist",
            ),
        )
        prior_outputs.append(methodologist)

        tcr = self._call_and_store(
            team,
            store,
            state,
            "tcr_analyst",
            "Design the TCR clone-lineage support analyses and barcode join checks.",
            self._context_with_rag(
                self._context_with_prior(base_context, prior_outputs),
                rag_chunks,
                "TCR clonotype clone expansion barcode join patient sample clone-state coupling controls",
                store=store,
                state=state,
                agent_name="tcr_analyst",
            ),
        )
        prior_outputs.append(tcr)

        novelty_scout = self._call_and_store(
            team,
            store,
            state,
            "novelty_scout",
            "Use RAG prior work to propose novel, dataset-testable scRNA-scTCR hypotheses without reproducing source-paper conclusions.",
            self._context_with_rag(
                self._context_with_prior(base_context, prior_outputs),
                rag_chunks,
                "novel hypothesis generation not reproduction prior literature gaps transferable scRNA scTCR mechanisms",
                store=store,
                state=state,
                agent_name="novelty_scout",
            ),
        )
        prior_outputs.append(novelty_scout)

        integrator = self._call_and_store(
            team,
            store,
            state,
            "integrator",
            "Integrate RNA and TCR plans into ranked, falsifiable hypotheses.",
            self._context_with_rag(
                self._context_with_prior(base_context, prior_outputs),
                rag_chunks,
                "paired scRNA scTCR integrated hypotheses clone-state coupling response tissue timepoint",
                store=store,
                state=state,
                agent_name="integrator",
            ),
        )
        prior_outputs.append(integrator)

        skeptic = self._call_and_store(
            team,
            store,
            state,
            "skeptic",
            "Audit the proposed analysis for confounding, overclaiming, and missing controls.",
            self._context_with_rag(
                self._context_with_prior(base_context, prior_outputs),
                rag_chunks,
                "scTCR confounder doublet clone size patient sample null controls overclaim antigen specificity",
                store=store,
                state=state,
                agent_name="skeptic",
            ),
        )
        prior_outputs.append(skeptic)

        code_writer = self._call_and_store(
            team,
            store,
            state,
            "code_writer",
            (
                "Review the dataset-reconnaissance execution contract. This first script "
                "must only produce reusable metadata, RNA-program, TCR-join, patient-blocked, "
                "clone-size-null, and within-state tables needed for later hypothesis "
                "generation. Do not create a broad overall analysis plan."
            ),
            self._context_with_prior(base_context, prior_outputs),
        )
        prior_outputs.append(code_writer)

        script_text = render_joint_analysis_script(self.config, run_dir, None)
        script_path = store.write_script("scrna_sctcr_joint_analysis.py", script_text)
        self._add_artifact(state, "joint_analysis_script", script_path)
        self._add_artifact(
            state,
            "code_generation",
            store.write_markdown("code_generation", render_code_generation_note(None, script_path)),
        )

        script_status_path = self._maybe_execute_script(store, state, script_path)
        self._add_artifact(state, "script_execution", script_status_path)
        t_cell_annotation = self._run_t_cell_annotation(
            team=team,
            store=store,
            state=state,
            base_context=base_context,
            prior_outputs=prior_outputs,
            rag_chunks=rag_chunks,
        )
        if t_cell_annotation:
            prior_outputs.append(t_cell_annotation)

        hypothesis_ready_for_followup = True
        if self.config.deep_dive_enabled:
            deep_dive_outputs = self._run_deep_dive_loop(
                team=team,
                store=store,
                state=state,
                base_context=base_context,
                prior_outputs=prior_outputs,
                rag_chunks=rag_chunks,
            )
            prior_outputs.extend(deep_dive_outputs)
            hypothesis_ready_for_followup = self._final_hypothesis_is_accepted(state.run_dir)

        if self.config.mechanism_loop_enabled and hypothesis_ready_for_followup:
            mechanism_outputs = self._run_biology_mechanism_loop(
                team=team,
                store=store,
                state=state,
                base_context=base_context,
                prior_outputs=prior_outputs,
                rag_chunks=rag_chunks,
            )
            prior_outputs.extend(mechanism_outputs)
        elif self.config.mechanism_loop_enabled:
            path = store.write_markdown(
                "biology_mechanism_execution",
                (
                    "# Biological Mechanism Loop Execution\n\n"
                    "Skipped because the hypothesis refinement loop did not produce a supported "
                    "or partially supported selected hypothesis. Inspect `hypothesis_refinement_summary.md`."
                ),
            )
            self._add_artifact(state, "biology_mechanism_execution", path)

        if self.config.downstream_analysis_enabled and hypothesis_ready_for_followup:
            downstream_outputs = self._run_downstream_analysis_loop(
                team=team,
                store=store,
                state=state,
                base_context=base_context,
                prior_outputs=prior_outputs,
                rag_chunks=rag_chunks,
            )
            prior_outputs.extend(downstream_outputs)
        elif self.config.downstream_analysis_enabled:
            path = store.write_markdown(
                "hypothesis_downstream_execution",
                (
                    "# Hypothesis Downstream Analysis Execution\n\n"
                    "Skipped because no supported or partially supported hypothesis was available after "
                    "the refinement loop. Inspect `hypothesis_refinement_summary.md`."
                ),
            )
            self._add_artifact(state, "hypothesis_downstream_execution", path)

        result_inventory = build_result_inventory(state.run_dir)
        result_inventory_json = store.write_json("available_results_inventory", result_inventory)
        self._add_artifact(state, "available_results_inventory_json", result_inventory_json)
        result_inventory_markdown = render_result_inventory_markdown(result_inventory)
        result_inventory_md = store.write_markdown("available_results_inventory", result_inventory_markdown)
        self._add_artifact(state, "available_results_inventory", result_inventory_md)

        if hypothesis_ready_for_followup:
            visualizer_context = self._context_with_prior(base_context, prior_outputs)
            visualizer_context["available_results_inventory"] = result_inventory_markdown
            visualizer_context["figure_design_order"] = "\n".join(
                [
                    "# Result-Driven Figure Design Order",
                    "",
                    "1. The downstream analyst has already generated and executed the selected-hypothesis analysis.",
                    "2. The available result inventory lists the actual CSV/JSON/PNG/PDF artifacts now present.",
                    "3. The visualizer must write a complete Python plotting script from those actual results only.",
                    "4. Do not specify a desired figure first and then assume a missing result table exists.",
                    "5. Omit any panel whose required table or image is absent from the inventory.",
                    "6. Do not rely on the old publication_figure_spec renderer or any fixed figure template.",
                ]
            )
            visualizer = self._call_and_store(
                team,
                store,
                state,
                "visualizer",
                (
                    "Design the publication figure set only after inspecting the executed "
                    "result inventory. Follow this order: selected hypothesis -> executed "
                    "downstream/deep-dive/mechanism results -> result-supported figure "
                    "claims -> a complete standalone Python plotting script. Use only tables "
                    "or images that are present in available_results_inventory. Do not create "
                    "a desired figure first and then point to a table that was not generated. "
                    "If a scientifically desirable panel lacks a real result table, omit it. "
                    "Do not use a fixed figure template, fixed renderer, or JSON figure spec. "
                    "Return the full executable script between PUBLICATION_FIGURE_PYTHON_SCRIPT "
                    "and END_PUBLICATION_FIGURE_PYTHON_SCRIPT."
                ),
                self._context_with_rag(
                    visualizer_context,
                    rag_chunks,
                    "publication figure design scRNA scTCR hypothesis-driven visualization patient-aware figures",
                    store=store,
                    state=state,
                    agent_name="visualizer",
                ),
            )
            prior_outputs.append(visualizer)
            figure_script_text = render_publication_figure_script(run_dir, visualizer.content)
            figure_script_path = store.write_script("publication_figures.py", figure_script_text)
            self._add_artifact(state, "publication_figure_script", figure_script_path)
            figure_status_path = self._execute_publication_figure_with_llm_repair(
                team=team,
                store=store,
                state=state,
                script_path=figure_script_path,
                visualizer_context=visualizer_context,
                rag_chunks=rag_chunks,
            )
            self._add_artifact(state, "publication_figure_execution", figure_status_path)
        else:
            path = store.write_markdown(
                "publication_figure_execution",
                (
                    "# Publication Figure Execution\n\n"
                    "Skipped because no supported or partially supported hypothesis was available. "
                    "The workflow will not generate publication figures for a rejected hypothesis."
                ),
            )
            self._add_artifact(state, "publication_figure_execution", path)

        reporter_context = self._context_with_prior(base_context, prior_outputs)
        reporter_context["artifact_list"] = [
            f"{name}: {path}" for name, path in sorted(state.artifacts.items())
        ]
        report_context_paths = {
            "analysis_summary": state.run_dir / "analysis_outputs" / "analysis_summary.md",
            "t_cell_subcluster_annotation": state.run_dir / "analysis_outputs" / "t_cell_subcluster_annotation.md",
            "rag_grounded_hypothesis_candidates": state.run_dir
            / "rag_grounded_hypothesis_candidates.md",
            "selected_hypothesis": state.run_dir / "selected_hypothesis.md",
            "hypothesis_refinement_summary": state.run_dir / "hypothesis_refinement_summary.md",
            "hypothesis_support_decision": state.run_dir / "hypothesis_support_decision.json",
            "deep_dive_interpretation": state.run_dir / "analysis_outputs" / "deep_dive" / "deep_dive_interpretation.md",
            "deep_dive_conclusion": state.run_dir / "analysis_outputs" / "deep_dive" / "deep_dive_conclusion.md",
            "biological_interpretation_output": state.run_dir
            / "analysis_outputs"
            / "biology_mechanism"
            / "biological_interpretation.md",
            "mechanism_mapping_output": state.run_dir
            / "analysis_outputs"
            / "biology_mechanism"
            / "mechanism_mapping.md",
            "next_test_proposals_output": state.run_dir
            / "analysis_outputs"
            / "biology_mechanism"
            / "next_test_proposals.md",
            "downstream_analysis_plan": state.run_dir
            / "analysis_outputs"
            / "downstream"
            / "downstream_analysis_plan.md",
            "downstream_analysis_summary": state.run_dir
            / "analysis_outputs"
            / "downstream"
            / "downstream_analysis_summary.md",
            "publication_figure_summary": state.run_dir
            / "analysis_outputs"
            / "publication_figures"
            / "publication_figure_summary.md",
        }
        for key, path in report_context_paths.items():
            if path.exists():
                reporter_context[key] = read_text(path)
        reporter = self._call_and_store(
            team,
            store,
            state,
            "reporter",
            "Write the final run report and tell the user what to inspect next.",
            self._context_with_rag(
                reporter_context,
                rag_chunks,
                "paired scRNA scTCR reporting conservative interpretation artifacts",
                store=store,
                state=state,
                agent_name="reporter",
            ),
            artifact_name="final_report",
        )
        prior_outputs.append(reporter)

        team_summary = self._render_team_summary(prior_outputs)
        self._add_artifact(state, "team_transcript", store.write_markdown("team_transcript", team_summary))
        self._add_artifact(state, "manifest", store.write_json("manifest", state.to_manifest()))
        store.event("finish", {"summary": f"finished {self.config.analysis_name}"})
        return state

    def _run_deep_dive_loop(
        self,
        team: ScRTATeam,
        store: ArtifactStore,
        state: WorkflowState,
        base_context: dict,
        prior_outputs: list[AgentResponse],
        rag_chunks: list,
    ) -> list[AgentResponse]:
        responses: list[AgentResponse] = []
        if not self.config.execute_script:
            path = store.write_markdown(
                "hypothesis_deep_dive_execution",
                "# Hypothesis Deep-Dive Execution\n\nSkipped. Run with --execute to produce dataset reconnaissance outputs before deep-dive.",
            )
            self._add_artifact(state, "hypothesis_deep_dive_execution", path)
            return responses

        dataset_reconnaissance_context = self._render_dataset_reconnaissance_context(state.run_dir)

        generator_context = dict(base_context)
        generator_context["dataset_reconnaissance_context"] = dataset_reconnaissance_context
        generator = self._call_and_store(
            team,
            store,
            state,
            "hypothesis_generator",
            (
                "After reading the RAG evidence and dataset reconnaissance tables, generate "
                "3-4 novel, biologically meaningful, falsifiable hypotheses for this "
                "dataset. Do not simply restate a source paper. Do not derive hypotheses "
                "from earlier team plans or from a fixed CD8/Treg/clone menu; use only the "
                "retrieved literature, disease context, dataset structure, and executed "
                "reconnaissance outputs."
            ),
            self._context_with_rag(
                generator_context,
                rag_chunks,
                (
                    "generate novel biology-first hypotheses from RAG literature and current dataset "
                    "disease context tissue context treatment response resistance immune state programs "
                    "tumor microenvironment mechanisms paired scRNA scTCR support when relevant"
                ),
                store=store,
                state=state,
                agent_name="hypothesis_generator",
            ),
            artifact_name="rag_grounded_hypothesis_candidates",
        )
        responses.append(generator)
        generated_candidates = extract_hypothesis_candidates(generator.content)
        if not generated_candidates:
            error_path = store.write_markdown(
                "hypothesis_generation_error",
                (
                    "# Hypothesis Generation Error\n\n"
                    "The LLM hypothesis_generator did not emit a parseable "
                    "`HYPOTHESIS_CANDIDATES_JSON` block. The workflow stops instead "
                    "of using deterministic fallback hypotheses.\n"
                ),
            )
            self._add_artifact(state, "hypothesis_generation_error", error_path)
            raise RuntimeError("hypothesis_generator did not emit parseable hypothesis candidates.")
        candidate_index_path = store.write_json(
            "hypothesis_candidate_index",
            {
                "source": "hypothesis_generator",
                "candidate_count": len(generated_candidates),
                "candidate_ids": sorted(generated_candidates),
                "audit_note": (
                    "This is only an audit manifest of LLM-generated candidate IDs. "
                    "It is not a scoring table, not a hard-coded menu, and not a "
                    "deterministic selection rule."
                ),
            },
        )
        self._add_artifact(state, "hypothesis_candidate_index", candidate_index_path)

        if self.config.interactive_hypothesis_selection:
            selection, selector_response = self._select_hypothesis_interactively(
                generated_candidates=generated_candidates,
                store=store,
                state=state,
            )
            responses.append(selector_response)
            selector_review_content = selector_response.content
        else:
            deep_context = dict(base_context)
            compact_candidates = {
                "source": "hypothesis_generator",
                "selection_instruction": (
                    "Select exactly one ID from this JSON. Preserve the selected "
                    "candidate's hypothesis_statement and plain_language_explanation."
                ),
                "candidates": [
                    {
                        "hypothesis_id": hyp_id,
                        "title": candidate.get("title", ""),
                        "hypothesis_statement": candidate.get("hypothesis_statement", ""),
                        "plain_language_explanation": candidate.get("plain_language_explanation", ""),
                        "prior_literature_pattern": (candidate.get("raw") or {}).get("prior_literature_pattern", ""),
                        "current_dataset_clue": (candidate.get("raw") or {}).get("current_dataset_clue", ""),
                        "innovative_claim": (candidate.get("raw") or {}).get("innovative_claim", ""),
                        "key_validation": (candidate.get("raw") or {}).get("key_validation", ""),
                        "falsification_criteria": (candidate.get("raw") or {}).get("falsification_criteria", ""),
                        "required_output_tables": (candidate.get("raw") or {}).get("required_output_tables", []),
                    }
                    for hyp_id, candidate in sorted(generated_candidates.items())
                ],
            }
            # Put the compact candidate block under an early-sorting key so the
            # selector sees it even when large reconnaissance context is truncated.
            deep_context["aa_hypothesis_candidates_for_selection"] = json.dumps(
                compact_candidates,
                ensure_ascii=False,
                indent=2,
            )
            deep_context["dataset_reconnaissance_context"] = dataset_reconnaissance_context
            deep_context["rag_grounded_hypothesis_candidates"] = generator.content
            deep_context["hypothesis_candidate_index"] = read_text(candidate_index_path)
            selector = self._call_and_store(
                team,
                store,
                state,
                "hypothesis_selector",
                (
                    "Select the hypothesis that should enter a targeted deep-dive loop. "
                    "Choose exactly one ID from the RAG-grounded hypothesis_generator "
                    "candidates. Use dataset reconnaissance outputs only as permissive context about "
                    "feasibility and immediate testability. Do not invent a new hypothesis, "
                    "do not use any hard-coded selection menu, and do not rewrite the selected "
                    "candidate into a different biological claim. "
                    "End with the required JSON block."
                ),
                self._context_with_rag(
                    deep_context,
                    rag_chunks,
                    "select hypothesis for deep validation after dataset reconnaissance scRNA scTCR results",
                    store=store,
                    state=state,
                    agent_name="hypothesis_selector",
                ),
            )
            responses.append(selector)
            selector_review_content = selector.content

            try:
                selection = selection_from_agent_response(selector.content, generated_candidates)
            except ValueError as exc:
                error_path = store.write_markdown(
                    "hypothesis_selection_error",
                    (
                        "# Hypothesis Selection Error\n\n"
                        "The LLM hypothesis_selector did not return a valid selection from "
                        "the LLM-generated candidate IDs. The workflow stops instead of "
                        "using a deterministic fallback selection.\n\n"
                        f"Error: {exc}\n"
                    ),
                )
                self._add_artifact(state, "hypothesis_selection_error", error_path)
                raise
            selection_payload = selection.to_dict()
            selection_payload["generator_candidate_count"] = len(generated_candidates)
            selection_payload["selection_provenance"] = (
                "The LLM hypothesis_generator first generated multiple candidates after RAG "
                "retrieval and dataset reconnaissance review. The LLM hypothesis_selector then "
                "selected exactly one candidate ID from that generator output. Reconnaissance "
                "tables were available only as context for feasibility and were not used to "
                "create a separate hard-coded hypothesis-selection table or built-in hypothesis menu. "
                "If the selector output is malformed, the workflow fails rather than selecting "
                "a deterministic fallback candidate."
            )
            selection_payload["llm_hypothesis_generator_artifact"] = "rag_grounded_hypothesis_candidates.md"
            selection_payload["llm_selector_artifact"] = "agent_hypothesis_selector.md"
            selection_payload["candidate_index_artifact"] = "hypothesis_candidate_index.json"
            selection_json = store.write_json("selected_hypothesis", selection_payload)
            selection_md_text = (
                selection.to_markdown()
                + "\n## Selection Provenance\n"
                + "This hypothesis was selected after `hypothesis_generator` generated "
                + "RAG-grounded candidates and `agent_hypothesis_selector` chose one "
                + "candidate ID with injected RAG evidence and optional dataset reconnaissance context. "
                + "No built-in hypothesis-selection table is generated. Inspect "
                + "`rag_context_hypothesis_generator.md`, "
                + "`rag_grounded_hypothesis_candidates.md`, "
                + "`rag_context_hypothesis_selector.md`, and "
                + "`agent_hypothesis_selector.md` to audit the sequence. Malformed selector "
                + "output causes workflow failure; no deterministic hypothesis fallback is used.\n"
            )
            selection_md = store.write_markdown("selected_hypothesis", selection_md_text)
            self._add_artifact(state, "selected_hypothesis_json", selection_json)
            self._add_artifact(state, "selected_hypothesis", selection_md)

        planner_context = dict(base_context)
        planner_context["selected_hypothesis"] = selection.to_markdown()
        planner_context["rag_grounded_hypothesis_candidates"] = generator.content
        planner_context["rag_grounded_selector_review"] = selector_review_content
        planner_context["dataset_reconnaissance_context"] = dataset_reconnaissance_context
        planner_context["deep_dive_runtime_contract"] = self._render_deep_dive_runtime_contract(state.run_dir)
        planner = self._call_and_store(
            team,
            store,
            state,
            "deep_planner",
            (
                "Create and implement the targeted second-stage validation plan for the selected "
                "hypothesis. Do not use a fixed CD8/Treg/clone validation program; choose only "
                "analyses that directly test this selected biological claim. Output a "
                "hypothesis-specific execution contract and a complete standalone Python script "
                "between DEEP_DIVE_PYTHON_SCRIPT and END_DEEP_DIVE_PYTHON_SCRIPT."
            ),
            self._context_with_rag(
                planner_context,
                rag_chunks,
                "deep-dive validation selected biological hypothesis RAG-guided dataset-specific tests",
                store=store,
                state=state,
                agent_name="deep_planner",
            ),
        )
        responses.append(planner)
        deep_plan_path = store.write_markdown("selected_hypothesis_deep_dive_plan", planner.content)
        self._add_artifact(state, "selected_hypothesis_deep_dive_plan", deep_plan_path)

        try:
            deep_script_text = render_deep_dive_script(state.run_dir, selection, planner.content)
        except ValueError as exc:
            path = store.write_markdown(
                "hypothesis_deep_dive_execution",
                (
                    "# Hypothesis Deep-Dive Execution\n\n"
                    "Failed before execution because the deep_planner did not emit a valid Python script block.\n\n"
                    f"Error: {exc}\n"
                ),
            )
            self._add_artifact(state, "hypothesis_deep_dive_execution", path)
            raise
        deep_script_path = store.write_script("hypothesis_deep_dive.py", deep_script_text)
        self._add_artifact(state, "hypothesis_deep_dive_script", deep_script_path)
        deep_status_path = self._execute_deep_dive_with_llm_repair(
            team=team,
            store=store,
            state=state,
            script_path=deep_script_path,
            deep_context=planner_context,
            selection=selection,
            rag_chunks=rag_chunks,
        )
        self._add_artifact(state, "hypothesis_deep_dive_execution", deep_status_path)
        self._ensure_deep_dive_selected_hypothesis(state.run_dir, selection)

        conclusion_path = state.run_dir / "analysis_outputs" / "deep_dive" / "deep_dive_conclusion.md"
        if conclusion_path.exists():
            self._add_artifact(state, "hypothesis_deep_dive_conclusion", conclusion_path)
            interpretation_context = self._context_with_prior(base_context, [*prior_outputs, *responses])
            interpretation_context["selected_hypothesis"] = selection.to_markdown()
            interpretation_context["deep_dive_conclusion"] = read_text(conclusion_path)
            self._add_deep_dive_result_context(interpretation_context, state.run_dir)
            interpreter = self._call_and_store(
                team,
                store,
                state,
                "result_interpreter",
                "Interpret the hypothesis deep-dive outputs and decide support level.",
                interpretation_context,
                artifact_name="agent_result_interpreter",
            )
            responses.append(interpreter)
            decision = self._record_hypothesis_support_decision(
                store=store,
                state=state,
                attempt=1,
                selection=selection,
                interpreter=interpreter,
            )
            rejected_hypotheses: list[dict] = []
            if not decision.accepted:
                rejected_hypotheses.append(
                    {
                        "attempt": 1,
                        "hypothesis_id": selection.hypothesis_id,
                        "title": selection.title,
                        "selected_hypothesis": selection.selected_hypothesis,
                        "status": decision.status,
                        "rejected_reason": decision.rejected_reason or decision.rationale,
                        "interpreter_artifact": "agent_result_interpreter.md",
                    }
                )
                self._archive_hypothesis_attempt(state.run_dir, "attempt_01_rejected")
                if max(1, int(self.config.analysis_loops or 1)) > 1:
                    responses.extend(
                        self._run_hypothesis_refinement_attempts(
                            team=team,
                            store=store,
                            state=state,
                            base_context=base_context,
                            rag_chunks=rag_chunks,
                            dataset_reconnaissance_context=dataset_reconnaissance_context,
                            rejected_hypotheses=rejected_hypotheses,
                            start_attempt=2,
                        )
                    )
                if not self._final_hypothesis_is_accepted(state.run_dir):
                    responses.extend(
                        self._write_evidence_grounded_final_conclusion(
                            team=team,
                            store=store,
                            state=state,
                            base_context=base_context,
                            rag_chunks=rag_chunks,
                            dataset_reconnaissance_context=dataset_reconnaissance_context,
                            rejected_hypotheses=rejected_hypotheses,
                        )
                    )
            self._write_hypothesis_refinement_summary(store, state, rejected_hypotheses)
        return responses

    def _select_hypothesis_interactively(
        self,
        generated_candidates: dict[str, dict[str, object]],
        store: ArtifactStore,
        state: WorkflowState,
    ) -> tuple[DeepDiveSelection, AgentResponse]:
        print("")
        print("Generated hypothesis candidates")
        print("--------------------------------")
        for hyp_id, candidate in sorted(generated_candidates.items()):
            print(f"{hyp_id}: {candidate.get('title', '')}")
            statement = str(candidate.get("hypothesis_statement") or "").strip()
            explanation = str(candidate.get("plain_language_explanation") or "").strip()
            if statement:
                print(f"  Hypothesis: {statement}")
            if explanation:
                print(f"  Explanation: {explanation}")
            print("")

        available = sorted(generated_candidates)
        selected_id = ""
        while selected_id not in generated_candidates:
            raw = input(f"Select hypothesis ID ({', '.join(available)}): ").strip()
            selected_id = normalize_hypothesis_id(raw)
            if selected_id not in generated_candidates:
                print("Please enter one of the displayed hypothesis IDs.")

        candidate = generated_candidates[selected_id]
        raw_candidate = candidate.get("raw") if isinstance(candidate.get("raw"), dict) else {}
        title = _interactive_edit("Title", str(candidate.get("title") or selected_id))
        statement = _interactive_edit("Hypothesis statement", str(candidate.get("hypothesis_statement") or ""))
        explanation = _interactive_edit(
            "Explanation",
            str(candidate.get("plain_language_explanation") or ""),
            multiline=True,
        )
        required_tests = _interactive_edit_list(
            "Required tests",
            raw_candidate.get("key_validation")
            if isinstance(raw_candidate.get("key_validation"), list)
            else [str(raw_candidate.get("key_validation") or "Run targeted validation for the selected hypothesis.")],
        )
        falsification_criteria = _interactive_edit_list(
            "Falsification criteria",
            raw_candidate.get("falsification_criteria")
            if isinstance(raw_candidate.get("falsification_criteria"), list)
            else [
                str(
                    raw_candidate.get("falsification_criteria")
                    or "The targeted validation analyses do not support the selected hypothesis."
                )
            ],
        )
        source_tables = _interactive_edit_list(
            "Source tables",
            raw_candidate.get("required_output_tables")
            if isinstance(raw_candidate.get("required_output_tables"), list)
            else ["rag_grounded_hypothesis_candidates.md"],
        )

        selection = DeepDiveSelection(
            hypothesis_id=selected_id,
            title=title,
            selected_hypothesis=statement,
            plain_language_explanation=explanation,
            rationale="Selected and optionally edited through the interactive workflow.",
            required_tests=required_tests,
            falsification_criteria=falsification_criteria,
            source_tables=source_tables,
            selected_candidate_source="interactive_hypothesis_selection",
            selected_candidate_text=str(candidate.get("source_text") or "").strip(),
            selection_mode="interactive_candidate_selection_for_deep_dive",
            data_support_level="not_assessed",
        )
        selection_payload = selection.to_dict()
        selection_payload["generator_candidate_count"] = len(generated_candidates)
        selection_payload["selection_provenance"] = (
            "The LLM hypothesis_generator produced candidates. The user then selected "
            "one candidate and could edit the title, hypothesis statement, explanation, "
            "required tests, falsification criteria, and source tables before deep-dive execution."
        )
        selection_payload["llm_hypothesis_generator_artifact"] = "rag_grounded_hypothesis_candidates.md"
        selection_payload["candidate_index_artifact"] = "hypothesis_candidate_index.json"
        selection_json = store.write_json("selected_hypothesis", selection_payload)
        selection_md = store.write_markdown(
            "selected_hypothesis",
            selection.to_markdown()
            + "\n## Selection Provenance\n"
            + selection_payload["selection_provenance"]
            + "\n",
        )
        self._add_artifact(state, "selected_hypothesis_json", selection_json)
        self._add_artifact(state, "selected_hypothesis", selection_md)
        response = AgentResponse(
            agent_name="interactive_hypothesis_selection",
            content=selection.to_markdown(),
            metadata={"mode": "interactive", "role": "human_candidate_selection"},
        )
        selector_path = store.write_markdown("agent_interactive_hypothesis_selection", response.content)
        self._add_artifact(state, "agent_interactive_hypothesis_selection", selector_path)
        selector_meta = store.write_json("agent_interactive_hypothesis_selection_metadata", asdict(response))
        self._add_artifact(state, "agent_interactive_hypothesis_selection_metadata", selector_meta)
        return selection, response

    def _run_hypothesis_refinement_attempts(
        self,
        team: ScRTATeam,
        store: ArtifactStore,
        state: WorkflowState,
        base_context: dict,
        rag_chunks: list,
        dataset_reconnaissance_context: str,
        rejected_hypotheses: list[dict],
        start_attempt: int,
    ) -> list[AgentResponse]:
        responses: list[AgentResponse] = []
        max_attempts = max(1, int(self.config.analysis_loops or 1))
        for attempt in range(start_attempt, max_attempts + 1):
            attempt_label = f"attempt_{attempt:02d}"
            store.event(
                "hypothesis_refinement",
                {"summary": f"starting {attempt_label} after rejected hypothesis"},
            )

            generator_context = dict(base_context)
            generator_context["dataset_reconnaissance_context"] = dataset_reconnaissance_context
            generator_context["rejected_hypotheses_from_prior_attempts"] = json.dumps(
                rejected_hypotheses,
                ensure_ascii=False,
                indent=2,
            )
            generator = self._call_and_store(
                team,
                store,
                state,
                "hypothesis_generator",
                (
                    "Generate 3-4 new or substantially revised biology-first hypotheses after reading "
                    "the RAG evidence, dataset reconnaissance tables, and rejected-hypothesis evidence. "
                    "Do not repeat any rejected hypothesis or its failed biological claim. Propose "
                    "hypotheses that the available data can falsify and that could still lead to a "
                    "biologically meaningful conclusion."
                ),
                self._context_with_rag(
                    generator_context,
                    rag_chunks,
                    (
                        "regenerate biological hypotheses after rejected deep-dive evidence disease context "
                        "scRNA scTCR RAG novel mechanisms"
                    ),
                    store=store,
                    state=state,
                    agent_name=f"hypothesis_generator_{attempt_label}",
                ),
                artifact_name=f"{attempt_label}_rag_grounded_hypothesis_candidates",
            )
            responses.append(generator)
            canonical_generator_path = store.write_markdown("rag_grounded_hypothesis_candidates", generator.content)
            self._add_artifact(state, "rag_grounded_hypothesis_candidates", canonical_generator_path)

            generated_candidates = extract_hypothesis_candidates(generator.content)
            if not generated_candidates:
                error_path = store.write_markdown(
                    f"{attempt_label}_hypothesis_generation_error",
                    (
                        "# Hypothesis Generation Error\n\n"
                        "The LLM hypothesis_generator did not emit a parseable "
                        "`HYPOTHESIS_CANDIDATES_JSON` block during hypothesis refinement.\n"
                    ),
                )
                self._add_artifact(state, f"{attempt_label}_hypothesis_generation_error", error_path)
                raise RuntimeError("hypothesis_generator did not emit parseable refinement candidates.")

            candidate_index = {
                "source": "hypothesis_generator",
                "attempt": attempt,
                "candidate_count": len(generated_candidates),
                "candidate_ids": sorted(generated_candidates),
                "rejected_hypotheses_considered": rejected_hypotheses,
                "audit_note": (
                    "This is an audit manifest of LLM-generated candidate IDs for a refinement attempt. "
                    "It is not a deterministic selection table."
                ),
            }
            candidate_index_path = store.write_json(f"{attempt_label}_hypothesis_candidate_index", candidate_index)
            canonical_candidate_index = store.write_json("hypothesis_candidate_index", candidate_index)
            self._add_artifact(state, "hypothesis_candidate_index", canonical_candidate_index)

            compact_candidates = {
                "source": "hypothesis_generator",
                "attempt": attempt,
                "selection_instruction": (
                    "Select exactly one ID from this JSON, excluding any rejected hypothesis or failed claim. "
                    "Preserve the selected candidate's hypothesis_statement and plain_language_explanation."
                ),
                "rejected_hypotheses_from_prior_attempts": rejected_hypotheses,
                "candidates": [
                    {
                        "hypothesis_id": hyp_id,
                        "title": candidate.get("title", ""),
                        "hypothesis_statement": candidate.get("hypothesis_statement", ""),
                        "plain_language_explanation": candidate.get("plain_language_explanation", ""),
                        "prior_literature_pattern": (candidate.get("raw") or {}).get("prior_literature_pattern", ""),
                        "current_dataset_clue": (candidate.get("raw") or {}).get("current_dataset_clue", ""),
                        "innovative_claim": (candidate.get("raw") or {}).get("innovative_claim", ""),
                        "key_validation": (candidate.get("raw") or {}).get("key_validation", ""),
                        "falsification_criteria": (candidate.get("raw") or {}).get("falsification_criteria", ""),
                        "required_output_tables": (candidate.get("raw") or {}).get("required_output_tables", []),
                    }
                    for hyp_id, candidate in sorted(generated_candidates.items())
                ],
            }
            deep_context = dict(base_context)
            deep_context["aa_hypothesis_candidates_for_selection"] = json.dumps(
                compact_candidates,
                ensure_ascii=False,
                indent=2,
            )
            deep_context["dataset_reconnaissance_context"] = dataset_reconnaissance_context
            deep_context["rag_grounded_hypothesis_candidates"] = generator.content
            deep_context["hypothesis_candidate_index"] = read_text(candidate_index_path)
            deep_context["rejected_hypotheses_from_prior_attempts"] = json.dumps(
                rejected_hypotheses,
                ensure_ascii=False,
                indent=2,
            )

            selector = self._call_and_store(
                team,
                store,
                state,
                "hypothesis_selector",
                (
                    "Select one new hypothesis for a targeted deep-dive. Exclude the rejected hypotheses "
                    "and avoid their failed biological claims. Choose exactly one ID from the current "
                    "hypothesis_generator candidates and end with the required JSON block."
                ),
                self._context_with_rag(
                    deep_context,
                    rag_chunks,
                    "select replacement hypothesis after rejected deep-dive result scRNA scTCR RAG",
                    store=store,
                    state=state,
                    agent_name=f"hypothesis_selector_{attempt_label}",
                ),
                artifact_name=f"{attempt_label}_agent_hypothesis_selector",
            )
            responses.append(selector)
            canonical_selector = store.write_markdown("agent_hypothesis_selector", selector.content)
            self._add_artifact(state, "agent_hypothesis_selector", canonical_selector)

            try:
                selection = selection_from_agent_response(selector.content, generated_candidates)
            except ValueError as exc:
                error_path = store.write_markdown(
                    f"{attempt_label}_hypothesis_selection_error",
                    (
                        "# Hypothesis Selection Error\n\n"
                        "The LLM hypothesis_selector did not return a valid selection from "
                        "the current LLM-generated candidate IDs.\n\n"
                        f"Error: {exc}\n"
                    ),
                )
                self._add_artifact(state, f"{attempt_label}_hypothesis_selection_error", error_path)
                raise

            selection_payload = selection.to_dict()
            selection_payload["attempt"] = attempt
            selection_payload["generator_candidate_count"] = len(generated_candidates)
            selection_payload["selection_provenance"] = (
                "Selected by the LLM hypothesis_selector during a hypothesis-refinement loop after "
                "prior selected hypotheses were rejected by deep-dive interpretation."
            )
            selection_payload["rejected_hypotheses_considered"] = rejected_hypotheses
            attempt_selection_json = store.write_json(f"{attempt_label}_selected_hypothesis", selection_payload)
            canonical_selection_json = store.write_json("selected_hypothesis", selection_payload)
            self._add_artifact(state, f"{attempt_label}_selected_hypothesis_json", attempt_selection_json)
            self._add_artifact(state, "selected_hypothesis_json", canonical_selection_json)
            selection_md_text = (
                selection.to_markdown()
                + "\n## Selection Provenance\n"
                + "This hypothesis was selected in a refinement loop after prior hypothesis failure. "
                + "Rejected hypotheses were supplied as context and must not be repeated.\n"
            )
            attempt_selection_md = store.write_markdown(f"{attempt_label}_selected_hypothesis", selection_md_text)
            canonical_selection_md = store.write_markdown("selected_hypothesis", selection_md_text)
            self._add_artifact(state, f"{attempt_label}_selected_hypothesis", attempt_selection_md)
            self._add_artifact(state, "selected_hypothesis", canonical_selection_md)

            planner_context = dict(base_context)
            planner_context["selected_hypothesis"] = selection.to_markdown()
            planner_context["rag_grounded_hypothesis_candidates"] = generator.content
            planner_context["rag_grounded_selector_review"] = selector.content
            planner_context["dataset_reconnaissance_context"] = dataset_reconnaissance_context
            planner_context["deep_dive_runtime_contract"] = self._render_deep_dive_runtime_contract(state.run_dir)
            planner_context["rejected_hypotheses_from_prior_attempts"] = json.dumps(
                rejected_hypotheses,
                ensure_ascii=False,
                indent=2,
            )
            planner = self._call_and_store(
                team,
                store,
                state,
                "deep_planner",
                (
                    "Create and implement a targeted second-stage validation plan for this replacement "
                    "hypothesis. Do not repeat tests whose failure already rejected prior hypotheses "
                    "unless they are needed as controls. Choose analyses that directly test this "
                    "selected biological claim. Output a hypothesis-specific execution contract and a "
                    "complete standalone Python script between DEEP_DIVE_PYTHON_SCRIPT and "
                    "END_DEEP_DIVE_PYTHON_SCRIPT."
                ),
                self._context_with_rag(
                    planner_context,
                    rag_chunks,
                    "replacement hypothesis deep-dive validation RAG dataset-specific tests",
                    store=store,
                    state=state,
                    agent_name=f"deep_planner_{attempt_label}",
                ),
                artifact_name=f"{attempt_label}_agent_deep_planner",
            )
            responses.append(planner)
            canonical_planner = store.write_markdown("agent_deep_planner", planner.content)
            self._add_artifact(state, "agent_deep_planner", canonical_planner)
            attempt_deep_plan_path = store.write_markdown(f"{attempt_label}_selected_hypothesis_deep_dive_plan", planner.content)
            canonical_deep_plan_path = store.write_markdown("selected_hypothesis_deep_dive_plan", planner.content)
            self._add_artifact(state, f"{attempt_label}_selected_hypothesis_deep_dive_plan", attempt_deep_plan_path)
            self._add_artifact(state, "selected_hypothesis_deep_dive_plan", canonical_deep_plan_path)

            shutil.rmtree(state.run_dir / "analysis_outputs" / "deep_dive", ignore_errors=True)
            try:
                deep_script_text = render_deep_dive_script(state.run_dir, selection, planner.content)
            except ValueError as exc:
                path = store.write_markdown(
                    "hypothesis_deep_dive_execution",
                    (
                        "# Hypothesis Deep-Dive Execution\n\n"
                        "Failed before execution because the deep_planner did not emit a valid Python script block.\n\n"
                        f"Error: {exc}\n"
                    ),
                )
                self._add_artifact(state, "hypothesis_deep_dive_execution", path)
                raise
            deep_script_path = store.write_script("hypothesis_deep_dive.py", deep_script_text)
            self._add_artifact(state, "hypothesis_deep_dive_script", deep_script_path)
            deep_status_path = self._execute_deep_dive_with_llm_repair(
                team=team,
                store=store,
                state=state,
                script_path=deep_script_path,
                deep_context=planner_context,
                selection=selection,
                rag_chunks=rag_chunks,
            )
            self._add_artifact(state, "hypothesis_deep_dive_execution", deep_status_path)
            self._ensure_deep_dive_selected_hypothesis(state.run_dir, selection)

            conclusion_path = state.run_dir / "analysis_outputs" / "deep_dive" / "deep_dive_conclusion.md"
            if conclusion_path.exists():
                self._add_artifact(state, "hypothesis_deep_dive_conclusion", conclusion_path)
            interpretation_context = self._context_with_prior(base_context, responses)
            interpretation_context["selected_hypothesis"] = selection.to_markdown()
            if conclusion_path.exists():
                interpretation_context["deep_dive_conclusion"] = read_text(conclusion_path)
            self._add_deep_dive_result_context(interpretation_context, state.run_dir)
            interpreter = self._call_and_store(
                team,
                store,
                state,
                "result_interpreter",
                "Interpret the replacement hypothesis deep-dive outputs and decide support level.",
                interpretation_context,
                artifact_name=f"{attempt_label}_agent_result_interpreter",
            )
            responses.append(interpreter)
            canonical_interpreter = store.write_markdown("agent_result_interpreter", interpreter.content)
            self._add_artifact(state, "agent_result_interpreter", canonical_interpreter)
            decision = self._record_hypothesis_support_decision(
                store=store,
                state=state,
                attempt=attempt,
                selection=selection,
                interpreter=interpreter,
            )
            if decision.accepted:
                return responses
            rejected_hypotheses.append(
                {
                    "attempt": attempt,
                    "hypothesis_id": selection.hypothesis_id,
                    "title": selection.title,
                    "selected_hypothesis": selection.selected_hypothesis,
                    "status": decision.status,
                    "rejected_reason": decision.rejected_reason or decision.rationale,
                    "interpreter_artifact": f"{attempt_label}_agent_result_interpreter.md",
                }
            )
            self._archive_hypothesis_attempt(state.run_dir, f"{attempt_label}_rejected")
        return responses

    def _write_evidence_grounded_final_conclusion(
        self,
        team: ScRTATeam,
        store: ArtifactStore,
        state: WorkflowState,
        base_context: dict,
        rag_chunks: list,
        dataset_reconnaissance_context: str,
        rejected_hypotheses: list[dict],
    ) -> list[AgentResponse]:
        """Ask the LLM to narrow failed hypotheses into one data-supported conclusion.

        This is deliberately not a deterministic hypothesis fallback. It is a
        final LLM synthesis step used after all proposed hypotheses have been
        rejected or judged inconclusive. The generated conclusion must be
        narrower than the failed claims and anchored in executed outputs.
        """
        responses: list[AgentResponse] = []
        store.event(
            "evidence_grounded_final_conclusion",
            {
                "summary": (
                    "all selected hypotheses were rejected; asking LLM to synthesize "
                    "a conservative data-supported conclusion"
                )
            },
        )

        rescue_context = dict(base_context)
        rescue_context["dataset_reconnaissance_context"] = dataset_reconnaissance_context
        rescue_context["rejected_hypotheses_from_prior_attempts"] = json.dumps(
            rejected_hypotheses,
            ensure_ascii=False,
            indent=2,
        )
        rescue_context["available_results_inventory"] = render_result_inventory_markdown(
            build_result_inventory(state.run_dir)
        )
        self._add_deep_dive_result_context(rescue_context, state.run_dir)

        final_candidate = self._call_and_store(
            team,
            store,
            state,
            "hypothesis_generator",
            (
                "All previously selected hypotheses were rejected or inconclusive after "
                "targeted deep-dive testing. Do not repeat, rescue, or soften those failed "
                "claims. Generate exactly one conservative, evidence-grounded final "
                "conclusion candidate from the executed outputs and RAG context. This "
                "candidate should be biologically useful, but it must be narrower than the "
                "rejected hypotheses and directly supported by actual tables or summaries. "
                "It may be a negative or boundary-setting conclusion if that is the honest "
                "result. Use HYP-1 only. End with the required HYPOTHESIS_CANDIDATES_JSON "
                "block containing exactly one candidate."
            ),
            self._context_with_rag(
                rescue_context,
                rag_chunks,
                (
                    "evidence-grounded final conclusion after rejected hypotheses scRNA scTCR "
                    "dataset-supported conservative biological conclusion"
                ),
                store=store,
                state=state,
                agent_name="evidence_grounded_final_conclusion",
            ),
            artifact_name="final_evidence_grounded_conclusion_candidate",
        )
        responses.append(final_candidate)

        candidates = extract_hypothesis_candidates(final_candidate.content)
        if candidates:
            hyp_id, candidate = next(iter(sorted(candidates.items())))
            raw = candidate.get("raw") if isinstance(candidate.get("raw"), dict) else {}
            source_tables = raw.get("required_output_tables") or []
            if isinstance(source_tables, str):
                source_tables = [item.strip() for item in source_tables.split(",") if item.strip()]
            if not isinstance(source_tables, list):
                source_tables = []
            title = str(candidate.get("title") or "Evidence-grounded final conclusion").strip()
            statement = str(candidate.get("hypothesis_statement") or "").strip()
            explanation = str(candidate.get("plain_language_explanation") or "").strip()
        else:
            hyp_id = "HYP-1"
            title = "Evidence-grounded final conclusion"
            fallback_lines = []
            for line in final_candidate.content.splitlines():
                line = line.strip().strip("- ")
                if not line or line.startswith("#") or line.startswith("HYPOTHESIS_"):
                    continue
                if line in {"{", "}", "[", "]"} or line.startswith('"'):
                    continue
                fallback_lines.append(line)
                if len(" ".join(fallback_lines)) > 500:
                    break
            statement = " ".join(fallback_lines).strip() or (
                "The executed analyses did not support the prior generated mechanistic hypotheses; "
                "the final conclusion must be restricted to the conservative evidence summarized "
                "by the LLM in final_evidence_grounded_conclusion_candidate.md."
            )
            explanation = truncate_text(final_candidate.content, 2000)
            source_tables = ["final_evidence_grounded_conclusion_candidate.md"]

        selection = DeepDiveSelection(
            hypothesis_id=hyp_id,
            title=title,
            selected_hypothesis=statement,
            rationale=(
                "Final evidence-grounded conclusion synthesized by the LLM after all earlier "
                "selected hypotheses were rejected or inconclusive. It is intended to replace "
                "the failed broad claims with a narrower conclusion that can be carried into "
                "mechanism interpretation and downstream analysis without fabricating support."
            ),
            required_tests=[
                "Use only executed result tables, deep-dive outputs, and RAG-supported biological context.",
                "Do not reintroduce any rejected biological claim unless new executed evidence supports it.",
            ],
            falsification_criteria=[
                "If downstream analyses contradict this narrowed conclusion, mark it as not supported."
            ],
            source_tables=source_tables or ["available_results_inventory", "hypothesis_attempts"],
            plain_language_explanation=explanation,
            selected_candidate_source="hypothesis_generator_final_evidence_grounded_conclusion",
            selected_candidate_text=truncate_text(final_candidate.content, 12000),
            selection_mode="evidence_grounded_final_conclusion_after_rejected_hypotheses",
            data_support_level="narrow_evidence_grounded",
        )

        deep_dir = ensure_dir(state.run_dir / "analysis_outputs" / "deep_dive")
        selection_md_text = (
            selection.to_markdown()
            + "\n## Final Evidence-Grounded Conclusion Mode\n"
            + "All earlier selected hypotheses were rejected or inconclusive. This LLM-generated "
            + "conclusion is deliberately narrower and must be interpreted only within the "
            + "evidence boundary described above.\n"
        )
        selection_payload = selection.to_dict()
        selection_payload.update(
            {
                "attempt": "final_evidence_grounded_conclusion",
                "selection_provenance": (
                    "Generated by the LLM hypothesis_generator after all configured "
                    "hypothesis-refinement attempts failed. This is not a hard-coded fallback."
                ),
                "rejected_hypotheses_considered": rejected_hypotheses,
            }
        )
        selected_md = store.write_markdown("selected_hypothesis", selection_md_text)
        selected_json = store.write_json("selected_hypothesis", selection_payload)
        self._add_artifact(state, "selected_hypothesis", selected_md)
        self._add_artifact(state, "selected_hypothesis_json", selected_json)
        (deep_dir / "selected_hypothesis.md").write_text(selection_md_text, encoding="utf-8")
        (deep_dir / "selected_hypothesis.json").write_text(
            json.dumps(selection_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        conclusion_lines = [
            "# Evidence-Grounded Final Conclusion",
            "",
            "## Conclusion",
            statement,
            "",
        ]
        if explanation:
            conclusion_lines.extend(["## Explanation", explanation, ""])
        conclusion_lines.extend(
            [
                "## Why This Replaced The Failed Hypotheses",
                "The configured hypothesis-refinement attempts did not yield a supported selected hypothesis. "
                "The workflow therefore asked the LLM to synthesize a narrower conclusion from the executed "
                "data products instead of stopping or building a story around a rejected claim.",
                "",
                "## Rejected Hypotheses Considered",
            ]
        )
        if rejected_hypotheses:
            for item in rejected_hypotheses:
                conclusion_lines.extend(
                    [
                        f"- Attempt {item.get('attempt')}: {item.get('title')} "
                        f"({item.get('status')}) - {item.get('rejected_reason')}",
                    ]
                )
        else:
            conclusion_lines.append("- No rejected hypothesis records were available.")
        conclusion_path = deep_dir / "deep_dive_conclusion.md"
        conclusion_path.write_text("\n".join(conclusion_lines).rstrip() + "\n", encoding="utf-8")
        self._add_artifact(state, "hypothesis_deep_dive_conclusion", conclusion_path)

        plan_path = deep_dir / "deep_dive_analysis_plan.md"
        plan_path.write_text(
            "\n".join(
                [
                    "# Final Evidence-Grounded Conclusion Plan",
                    "",
                    "No additional fixed deep-dive script was run for this final step. The conclusion was "
                    "generated by the LLM from the executed reconnaissance/deep-dive outputs and the "
                    "failure reasons of prior hypotheses.",
                    "",
                    "Downstream and mechanism agents must treat this as a narrowed evidence boundary, not "
                    "as permission to revive rejected claims.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        manifest = {
            "mode": "evidence_grounded_final_conclusion_after_rejected_hypotheses",
            "candidate_artifact": "final_evidence_grounded_conclusion_candidate.md",
            "selected_hypothesis": selection_payload,
            "rejected_hypotheses_considered": rejected_hypotheses,
            "source_tables": source_tables,
        }
        (deep_dir / "deep_dive_result_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (deep_dir / "deep_dive_summary.json").write_text(
            json.dumps(
                {
                    "status": "final_evidence_grounded_conclusion",
                    "accepted_for_followup": True,
                    "conclusion": statement,
                    "source": "LLM hypothesis_generator after rejected hypotheses",
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        decision_payload = {
            "status": "partially_supported",
            "rationale": (
                "Accepted for follow-up as a narrowed evidence-grounded conclusion after "
                "the broader generated hypotheses were rejected. This does not rescue any "
                "failed claim; it constrains downstream work to the supported conclusion."
            ),
            "next_action": "continue",
            "rejected_reason": "",
            "accepted": True,
            "attempt": "final_evidence_grounded_conclusion",
            "hypothesis_id": selection.hypothesis_id,
            "title": selection.title,
            "selected_hypothesis": selection.selected_hypothesis,
            "interpreter_agent": "hypothesis_generator_final_evidence_grounded_conclusion",
            "decision_mode": "evidence_grounded_final_conclusion_after_rejected_hypotheses",
        }
        final_decision_path = store.write_json("final_evidence_grounded_hypothesis_support_decision", decision_payload)
        canonical_decision_path = store.write_json("hypothesis_support_decision", decision_payload)
        self._add_artifact(state, "final_evidence_grounded_hypothesis_support_decision", final_decision_path)
        self._add_artifact(state, "hypothesis_support_decision", canonical_decision_path)
        (deep_dir / "hypothesis_support_decision.json").write_text(
            json.dumps(decision_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        interpretation_path = deep_dir / "deep_dive_interpretation.md"
        interpretation_path.write_text(
            "# Final Evidence-Grounded Interpretation\n\n"
            + decision_payload["rationale"]
            + "\n\n"
            + "Mechanism, downstream, and figure agents may continue only by using this narrowed "
            + "conclusion and its source evidence.\n",
            encoding="utf-8",
        )
        self._add_artifact(state, "deep_dive_interpretation", interpretation_path)
        return responses

    def _record_hypothesis_support_decision(
        self,
        store: ArtifactStore,
        state: WorkflowState,
        attempt: int,
        selection,
        interpreter: AgentResponse,
    ):
        decision = support_decision_from_result_interpreter(interpreter.content)
        payload = decision.to_dict()
        payload.update(
            {
                "attempt": attempt,
                "hypothesis_id": selection.hypothesis_id,
                "title": selection.title,
                "selected_hypothesis": selection.selected_hypothesis,
                "interpreter_agent": interpreter.agent_name,
            }
        )
        attempt_label = f"attempt_{attempt:02d}"
        attempt_path = store.write_json(f"{attempt_label}_hypothesis_support_decision", payload)
        canonical_path = store.write_json("hypothesis_support_decision", payload)
        self._add_artifact(state, f"{attempt_label}_hypothesis_support_decision", attempt_path)
        self._add_artifact(state, "hypothesis_support_decision", canonical_path)
        deep_dir = ensure_dir(state.run_dir / "analysis_outputs" / "deep_dive")
        (deep_dir / "deep_dive_interpretation.md").write_text(interpreter.content.strip() + "\n", encoding="utf-8")
        (deep_dir / "hypothesis_support_decision.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self._add_artifact(state, "deep_dive_interpretation", deep_dir / "deep_dive_interpretation.md")
        return decision

    def _archive_hypothesis_attempt(self, run_dir: Path, attempt_name: str) -> None:
        archive_dir = ensure_dir(run_dir / "analysis_outputs" / "hypothesis_attempts" / attempt_name)
        deep_dir = run_dir / "analysis_outputs" / "deep_dive"
        if deep_dir.exists():
            shutil.copytree(deep_dir, archive_dir / "deep_dive", dirs_exist_ok=True)
        for filename in [
            "rag_grounded_hypothesis_candidates.md",
            "hypothesis_candidate_index.json",
            "agent_hypothesis_selector.md",
            "selected_hypothesis.md",
            "selected_hypothesis.json",
            "selected_hypothesis_deep_dive_plan.md",
            "agent_result_interpreter.md",
            "hypothesis_support_decision.json",
        ]:
            src = run_dir / filename
            if src.exists():
                shutil.copy2(src, archive_dir / filename)

    def _write_hypothesis_refinement_summary(
        self,
        store: ArtifactStore,
        state: WorkflowState,
        rejected_hypotheses: list[dict],
    ) -> None:
        run_dir = state.run_dir
        decision_path = run_dir / "hypothesis_support_decision.json"
        decision = json.loads(read_text(decision_path)) if decision_path.exists() else {}
        accepted = bool(decision.get("accepted"))
        lines = [
            "# Hypothesis Refinement Summary",
            "",
            f"- Maximum hypothesis attempts: {max(1, int(self.config.analysis_loops or 1))}",
            f"- Final status: {decision.get('status', 'not_assessed')}",
            f"- Final hypothesis accepted for follow-up: {accepted}",
            "",
        ]
        if decision:
            lines.extend(
                [
                    "## Final Hypothesis",
                    f"- Attempt: {decision.get('attempt')}",
                    f"- ID: {decision.get('hypothesis_id')}",
                    f"- Title: {decision.get('title')}",
                    f"- Statement: {decision.get('selected_hypothesis')}",
                    f"- Rationale: {decision.get('rationale') or decision.get('rejected_reason', '')}",
                    "",
                ]
            )
        if rejected_hypotheses:
            lines.extend(["## Rejected Hypotheses", ""])
            for item in rejected_hypotheses:
                lines.extend(
                    [
                        f"### Attempt {item.get('attempt')}: {item.get('title')}",
                        f"- Hypothesis: {item.get('selected_hypothesis')}",
                        f"- Status: {item.get('status')}",
                        f"- Rejected reason: {item.get('rejected_reason')}",
                        "",
                    ]
                )
        if not accepted:
            lines.extend(
                [
                    "## Workflow Decision",
                    "No supported or partially supported hypothesis was found within the configured loop limit. "
                    "The workflow could not synthesize an evidence-grounded final conclusion, so mechanism mapping, "
                    "downstream analysis, and publication figures are skipped to avoid building a story around a "
                    "rejected hypothesis.",
                    "",
                ]
            )
        elif decision.get("decision_mode") == "evidence_grounded_final_conclusion_after_rejected_hypotheses":
            lines.extend(
                [
                    "## Workflow Decision",
                    "All earlier selected hypotheses were rejected or inconclusive. Instead of stopping, the workflow "
                    "asked the LLM to synthesize a narrower evidence-grounded final conclusion from the executed "
                    "outputs and the rejected-hypothesis evidence. Follow-up modules may continue, but only within "
                    "this narrowed conclusion boundary.",
                    "",
                ]
            )
        summary_path = store.write_markdown("hypothesis_refinement_summary", "\n".join(lines))
        self._add_artifact(state, "hypothesis_refinement_summary", summary_path)
        summary_json_path = store.write_json(
            "hypothesis_refinement_summary",
            {
                "max_attempts": max(1, int(self.config.analysis_loops or 1)),
                "final_decision": decision,
                "rejected_hypotheses": rejected_hypotheses,
            },
        )
        self._add_artifact(state, "hypothesis_refinement_summary_json", summary_json_path)

    @staticmethod
    def _final_hypothesis_is_accepted(run_dir: Path) -> bool:
        decision_path = run_dir / "hypothesis_support_decision.json"
        if not decision_path.exists():
            return False
        try:
            data = json.loads(read_text(decision_path))
        except Exception:
            return False
        return bool(data.get("accepted"))

    def _run_t_cell_annotation(
        self,
        team: ScRTATeam,
        store: ArtifactStore,
        state: WorkflowState,
        base_context: dict,
        prior_outputs: list[AgentResponse],
        rag_chunks: list,
    ) -> AgentResponse | None:
        output_dir = state.run_dir / "analysis_outputs"
        marker_path = output_dir / "t_cell_cluster_marker_summary.csv"
        cluster_path = output_dir / "t_cell_cluster_summary.csv"
        occupied_path = output_dir / "t_cell_occupied_clonotypes_by_group.csv"
        if not marker_path.exists() and not cluster_path.exists():
            return None
        annotation_context = self._context_with_prior(base_context, prior_outputs)
        annotation_context["t_cell_cluster_marker_summary"] = truncate_text(read_text(marker_path), 12000)
        annotation_context["t_cell_cluster_summary"] = truncate_text(read_text(cluster_path), 8000)
        if occupied_path.exists():
            annotation_context["t_cell_occupied_clonotypes_by_group"] = truncate_text(read_text(occupied_path), 6000)
        annotation_context["annotation_task"] = "\n".join(
            [
                "Annotate T-cell subclusters from the fixed baseline T-cell reclustering module.",
                "Use marker-program evidence, existing annotations, and scTCR support conservatively.",
                "Return a JSON mapping suitable for later plotting/reporting.",
            ]
        )
        response = self._call_and_store(
            team,
            store,
            state,
            "t_cell_annotator",
            "Define each T-cell subcluster from marker-program and existing-annotation evidence.",
            self._context_with_rag(
                annotation_context,
                rag_chunks,
                "T cell subcluster annotation CD4 CD8 Treg exhausted cytotoxic proliferating MAIT gamma delta scTCR",
                store=store,
                state=state,
                agent_name="t_cell_annotator",
            ),
            artifact_name="t_cell_subcluster_annotation",
        )
        annotation_copy = output_dir / "t_cell_subcluster_annotation.md"
        annotation_copy.write_text(response.content.strip() + "\n", encoding="utf-8")
        self._add_artifact(state, "t_cell_subcluster_annotation_output", annotation_copy)
        return response

    def _run_biology_mechanism_loop(
        self,
        team: ScRTATeam,
        store: ArtifactStore,
        state: WorkflowState,
        base_context: dict,
        prior_outputs: list[AgentResponse],
        rag_chunks: list,
    ) -> list[AgentResponse]:
        responses: list[AgentResponse] = []
        if not self.config.execute_script:
            path = store.write_markdown(
                "biology_mechanism_execution",
                (
                    "# Biological Mechanism Loop Execution\n\n"
                    "Skipped. Run with --execute to produce analysis outputs before biological interpretation, "
                    "mechanism mapping, and next-test proposal."
                ),
            )
            self._add_artifact(state, "biology_mechanism_execution", path)
            return responses

        selected_hypothesis_path = state.run_dir / "analysis_outputs" / "deep_dive" / "selected_hypothesis.md"
        deep_conclusion_path = state.run_dir / "analysis_outputs" / "deep_dive" / "deep_dive_conclusion.md"
        deep_interpretation_path = state.run_dir / "analysis_outputs" / "deep_dive" / "deep_dive_interpretation.md"
        mechanism_context = dict(base_context)
        mechanism_context["selected_hypothesis"] = read_text(selected_hypothesis_path)
        mechanism_context["deep_dive_conclusion"] = read_text(deep_conclusion_path)
        if deep_interpretation_path.exists():
            mechanism_context["deep_dive_interpretation"] = read_text(deep_interpretation_path)

        biological_interpreter = self._call_and_store(
            team,
            store,
            state,
            "biological_interpreter",
            "Interpret the selected deep-dive hypothesis as conservative immunobiology.",
            self._context_with_rag(
                mechanism_context,
                rag_chunks,
                "biological interpretation selected hypothesis dataset-specific mechanisms scRNA scTCR support when relevant",
                store=store,
                state=state,
                agent_name="biological_interpreter",
            ),
        )
        responses.append(biological_interpreter)

        mapper_context = dict(base_context)
        mapper_context["selected_hypothesis"] = mechanism_context["selected_hypothesis"]
        mapper_context["deep_dive_conclusion"] = mechanism_context["deep_dive_conclusion"]
        if "deep_dive_interpretation" in mechanism_context:
            mapper_context["deep_dive_interpretation"] = mechanism_context["deep_dive_interpretation"]
        mapper_context["biological_interpretation"] = biological_interpreter.content
        mechanism_mapper = self._call_and_store(
            team,
            store,
            state,
            "mechanism_mapper",
            (
                "Map the validated hypothesis to mechanism axes and downstream analysis targets. "
                "Do not force a fixed CD8/Treg/clone axis list; choose axes that follow from "
                "the selected hypothesis, RAG evidence, and executed results."
            ),
            self._context_with_rag(
                mapper_context,
                rag_chunks,
                "mechanism mapping selected hypothesis dataset-specific immune biology scRNA scTCR support when relevant",
                store=store,
                state=state,
                agent_name="mechanism_mapper",
            ),
        )
        responses.append(mechanism_mapper)

        script_text = render_biology_mechanism_script(state.run_dir)
        script_path = store.write_script("biology_mechanism.py", script_text)
        self._add_artifact(state, "biology_mechanism_script", script_path)
        status_path = self._maybe_execute_biology_mechanism_script(store, state, script_path)
        self._add_artifact(state, "biology_mechanism_execution", status_path)

        bio_dir = state.run_dir / "analysis_outputs" / "biology_mechanism"
        artifact_paths = {
            "biological_interpretation_output": bio_dir / "biological_interpretation.md",
            "mechanism_mapping_output": bio_dir / "mechanism_mapping.md",
            "next_test_proposals_output": bio_dir / "next_test_proposals.md",
            "mechanism_evidence_map": bio_dir / "mechanism_evidence_map.csv",
            "biological_signal_table": bio_dir / "biological_signal_table.csv",
            "biology_mechanism_summary": bio_dir / "biological_mechanism_summary.json",
        }
        for name, path in artifact_paths.items():
            if path.exists():
                self._add_artifact(state, name, path)

        next_context = dict(base_context)
        next_context["biological_interpretation"] = read_text(artifact_paths["biological_interpretation_output"])
        next_context["mechanism_mapping"] = read_text(artifact_paths["mechanism_mapping_output"])
        next_context["next_test_table"] = read_text(artifact_paths["next_test_proposals_output"])
        next_test_planner = self._call_and_store(
            team,
            store,
            state,
            "next_test_planner",
            "Rank the next tests needed to strengthen the mechanism-level biological story.",
            self._context_with_rag(
                next_context,
                rag_chunks,
                "next tests selected hypothesis validation scRNA scTCR external cohort functional validation when relevant",
                store=store,
                state=state,
                agent_name="next_test_planner",
            ),
        )
        responses.append(next_test_planner)
        return responses

    def _run_downstream_analysis_loop(
        self,
        team: ScRTATeam,
        store: ArtifactStore,
        state: WorkflowState,
        base_context: dict,
        prior_outputs: list[AgentResponse],
        rag_chunks: list,
    ) -> list[AgentResponse]:
        responses: list[AgentResponse] = []
        if not self.config.execute_script:
            path = store.write_markdown(
                "hypothesis_downstream_execution",
                (
                    "# Hypothesis Downstream Analysis Execution\n\n"
                    "Skipped. Run with --execute to produce dataset reconnaissance and deep-dive outputs before downstream analysis."
                ),
            )
            self._add_artifact(state, "hypothesis_downstream_execution", path)
            return responses

        selected_hypothesis_path = state.run_dir / "analysis_outputs" / "deep_dive" / "selected_hypothesis.md"
        if not selected_hypothesis_path.exists():
            path = store.write_markdown(
                "hypothesis_downstream_execution",
                (
                    "# Hypothesis Downstream Analysis Execution\n\n"
                    "Skipped because no selected hypothesis artifact was found. "
                    "Enable and execute the deep-dive loop first."
                ),
            )
            self._add_artifact(state, "hypothesis_downstream_execution", path)
            return responses

        downstream_context = dict(base_context)
        downstream_context["selected_hypothesis"] = read_text(selected_hypothesis_path)
        downstream_context["dataset_reconnaissance_context"] = self._render_dataset_reconnaissance_context(state.run_dir)
        downstream_context["downstream_runtime_contract"] = "\n".join(
            [
                "# Downstream Runtime Contract",
                "",
                f"- Run directory: {state.run_dir}",
                f"- Analysis outputs directory: {state.run_dir / 'analysis_outputs'}",
                f"- Downstream output directory: {state.run_dir / 'analysis_outputs' / 'downstream'}",
                f"- Script path that will be executed: {state.run_dir / 'scripts' / 'hypothesis_downstream_analysis.py'}",
                "",
                "The downstream_analyst must write a complete standalone Python script.",
                "The workflow will extract only the code between DOWNSTREAM_PYTHON_SCRIPT and END_DOWNSTREAM_PYTHON_SCRIPT.",
                "There is no fixed downstream execution template after this agent response.",
                "The script must choose analyses based on the selected hypothesis and available output tables.",
            ]
        )
        optional_context_paths = {
            "deep_dive_conclusion": state.run_dir / "analysis_outputs" / "deep_dive" / "deep_dive_conclusion.md",
            "deep_dive_interpretation": state.run_dir
            / "analysis_outputs"
            / "deep_dive"
            / "deep_dive_interpretation.md",
            "hypothesis_support_decision": state.run_dir
            / "analysis_outputs"
            / "deep_dive"
            / "hypothesis_support_decision.json",
            "biological_interpretation": state.run_dir
            / "analysis_outputs"
            / "biology_mechanism"
            / "biological_interpretation.md",
            "mechanism_mapping": state.run_dir
            / "analysis_outputs"
            / "biology_mechanism"
            / "mechanism_mapping.md",
            "next_test_proposals": state.run_dir
            / "analysis_outputs"
            / "biology_mechanism"
            / "next_test_proposals.md",
        }
        for key, path in optional_context_paths.items():
            if path.exists():
                downstream_context[key] = read_text(path)

        output_dir = state.run_dir / "analysis_outputs"
        if output_dir.exists():
            downstream_context["available_analysis_outputs"] = "\n".join(
                sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file())[:120]
            )

        downstream_analyst = self._call_and_store(
            team,
            store,
            state,
            "downstream_analyst",
            (
                "Design and implement the selected-hypothesis downstream analysis after reading RAG, "
                "dataset reconnaissance outputs, deep-dive results, biological interpretation, and mechanism mapping. "
                "Do not rely on a fixed downstream template. Output a hypothesis-specific execution contract and a "
                "complete standalone Python script between DOWNSTREAM_PYTHON_SCRIPT and END_DOWNSTREAM_PYTHON_SCRIPT. "
                "Do not force pseudobulk, pathway, repertoire, clone-state, same-clone, or receptor modules. "
                "Consider scTCR because this is a paired dataset, but include only scTCR analyses that are "
                "scientifically relevant to the selected hypothesis and feasible from available outputs."
            ),
            self._context_with_rag(
                downstream_context,
                rag_chunks,
                (
                    "downstream analysis selected biological hypothesis RAG-guided dataset-specific scRNA scTCR "
                    "mechanistic validation support layers when relevant"
                ),
                store=store,
                state=state,
                agent_name="downstream_analyst",
            ),
        )
        responses.append(downstream_analyst)

        try:
            script_text = render_downstream_analysis_script(state.run_dir, downstream_analyst.content)
        except ValueError as exc:
            path = store.write_markdown(
                "hypothesis_downstream_execution",
                (
                    "# Hypothesis Downstream Analysis Execution\n\n"
                    "Failed before execution because the downstream_analyst did not emit a valid Python script block.\n\n"
                    f"Error: {exc}\n"
                ),
            )
            self._add_artifact(state, "hypothesis_downstream_execution", path)
            raise
        script_path = store.write_script("hypothesis_downstream_analysis.py", script_text)
        self._add_artifact(state, "hypothesis_downstream_script", script_path)
        status_path = self._execute_downstream_analysis_with_llm_repair(
            team=team,
            store=store,
            state=state,
            script_path=script_path,
            downstream_context=downstream_context,
            rag_chunks=rag_chunks,
        )
        self._add_artifact(state, "hypothesis_downstream_execution", status_path)

        downstream_dir = state.run_dir / "analysis_outputs" / "downstream"
        artifact_paths = {
            "downstream_analysis_plan": downstream_dir / "downstream_analysis_plan.md",
            "downstream_analysis_summary": downstream_dir / "downstream_analysis_summary.md",
            "downstream_analysis_summary_json": downstream_dir / "downstream_analysis_summary.json",
            "sctcr_repertoire_by_context": downstream_dir / "sctcr_repertoire_by_context.csv",
            "sctcr_clone_state_coupling": downstream_dir / "sctcr_clone_state_coupling.csv",
            "sctcr_clone_state_score_contrasts": downstream_dir / "sctcr_clone_state_score_contrasts.csv",
            "sctcr_same_clone_program_shifts": downstream_dir / "sctcr_same_clone_program_shifts.csv",
            "sctcr_receptor_feature_summary": downstream_dir / "sctcr_receptor_feature_summary.csv",
            "mechanism_priority_table": downstream_dir / "mechanism_priority_table.csv",
            "focus_state_program_scores_by_context": downstream_dir / "focus_state_program_scores_by_context.csv",
            "focus_state_pseudobulk_de": downstream_dir / "focus_state_pseudobulk_de.csv",
        }
        for name, path in artifact_paths.items():
            if path.exists():
                self._add_artifact(state, name, path)

        summary_path = artifact_paths["downstream_analysis_summary"]
        if summary_path.exists():
            responses.append(
                AgentResponse(
                    agent_name="downstream_analysis_executor",
                    content=read_text(summary_path),
                    metadata={"mode": "script", "role": "downstream_analysis_execution"},
                )
            )
        return responses

    def _maybe_execute_deep_dive_script(
        self, store: ArtifactStore, state: WorkflowState, script_path: Path
    ) -> Path:
        store.event("hypothesis_deep_dive_execution", {"summary": "running hypothesis deep-dive script"})
        result = execute_python_script(
            script_path=script_path,
            run_dir=state.run_dir,
            timeout_seconds=min(self.config.script_timeout_seconds, 2400),
            repair_attempts=0,
            log_prefix="hypothesis_deep_dive",
        )
        self._add_artifact(state, "hypothesis_deep_dive_stdout", result.stdout_path)
        self._add_artifact(state, "hypothesis_deep_dive_stderr", result.stderr_path)
        self._add_artifact(
            state,
            "hypothesis_deep_dive_execution_json",
            store.write_json("hypothesis_deep_dive_execution", result.to_dict()),
        )
        return store.write_markdown("hypothesis_deep_dive_execution", result.to_markdown())

    def _execute_deep_dive_with_llm_repair(
        self,
        team: ScRTATeam,
        store: ArtifactStore,
        state: WorkflowState,
        script_path: Path,
        deep_context: dict,
        selection,
        rag_chunks: list,
    ) -> Path:
        """Run the LLM-authored deep-dive script and ask the LLM for repairs on failure."""
        max_attempts = max(1, int(self.config.repair_attempts) + 1)
        current_script = script_path
        last_result = None
        for attempt in range(1, max_attempts + 1):
            log_prefix = "hypothesis_deep_dive" if attempt == 1 else f"hypothesis_deep_dive_repair{attempt - 1}"
            store.event(
                "hypothesis_deep_dive_execution",
                {"summary": f"running LLM-authored deep-dive script attempt {attempt}/{max_attempts}"},
            )
            result = execute_python_script(
                script_path=current_script,
                run_dir=state.run_dir,
                timeout_seconds=min(self.config.script_timeout_seconds, 2400),
                repair_attempts=0,
                log_prefix=log_prefix,
            )
            last_result = result
            self._add_artifact(state, f"{log_prefix}_stdout", result.stdout_path)
            self._add_artifact(state, f"{log_prefix}_stderr", result.stderr_path)
            self._add_artifact(
                state,
                f"{log_prefix}_execution_json",
                store.write_json(f"{log_prefix}_execution", result.to_dict()),
            )
            if result.returncode == 0:
                self._add_artifact(state, "hypothesis_deep_dive_stdout", result.stdout_path)
                self._add_artifact(state, "hypothesis_deep_dive_stderr", result.stderr_path)
                self._add_artifact(
                    state,
                    "hypothesis_deep_dive_execution_json",
                    store.write_json("hypothesis_deep_dive_execution", result.to_dict()),
                )
                return store.write_markdown("hypothesis_deep_dive_execution", result.to_markdown())

            if attempt >= max_attempts:
                break

            repair_context = dict(deep_context)
            repair_context["selected_hypothesis"] = selection.to_markdown()
            repair_context["failed_deep_dive_script"] = truncate_text(read_text(current_script), 24000)
            repair_context["failed_stdout"] = truncate_text(read_text(Path(result.stdout_path)), 8000)
            repair_context["failed_stderr"] = truncate_text(read_text(Path(result.stderr_path)), 12000)
            repair_context["repair_requirement"] = "\n".join(
                [
                    "The previous LLM-authored deep-dive script failed.",
                    "Generate a repaired complete standalone Python script.",
                    "Do not switch to a generic fixed deep-dive workflow.",
                    "Keep the selected hypothesis and original execution contract logic unless the error requires a minimal change.",
                    "Use only local outputs under analysis_outputs/ and write only under analysis_outputs/deep_dive/.",
                    "Always write selected_hypothesis.md, deep_dive_analysis_plan.md, deep_dive_execution_contract.json, deep_dive_result_manifest.json, deep_dive_conclusion.md, and deep_dive_summary.json.",
                    "Output the full repaired script between DEEP_DIVE_PYTHON_SCRIPT and END_DEEP_DIVE_PYTHON_SCRIPT.",
                ]
            )
            repair_response = self._call_and_store(
                team,
                store,
                state,
                "deep_planner",
                (
                    "Repair the failed selected-hypothesis deep-dive Python script using the stderr/stdout "
                    "and current run artifacts. Return a full corrected script between the required markers."
                ),
                self._context_with_rag(
                    repair_context,
                    rag_chunks,
                    "repair selected hypothesis deep-dive Python script scRNA scTCR stderr",
                    store=store,
                    state=state,
                    agent_name=f"deep_planner_repair_{attempt}",
                ),
                artifact_name=f"agent_deep_planner_repair_{attempt}",
            )
            try:
                repaired_script = render_deep_dive_script(state.run_dir, selection, repair_response.content)
            except ValueError as exc:
                store.write_markdown(
                    f"hypothesis_deep_dive_repair_{attempt}_extraction_error",
                    (
                        "# Deep-Dive Repair Extraction Error\n\n"
                        f"The repair agent did not emit a valid Python script block.\n\nError: {exc}\n"
                    ),
                )
                break
            current_script = store.write_script(f"hypothesis_deep_dive_repair_{attempt}.py", repaired_script)
            self._add_artifact(state, f"hypothesis_deep_dive_repair_{attempt}_script", current_script)

        assert last_result is not None
        self._add_artifact(state, "hypothesis_deep_dive_stdout", last_result.stdout_path)
        self._add_artifact(state, "hypothesis_deep_dive_stderr", last_result.stderr_path)
        self._add_artifact(
            state,
            "hypothesis_deep_dive_execution_json",
            store.write_json("hypothesis_deep_dive_execution", last_result.to_dict()),
        )
        return store.write_markdown("hypothesis_deep_dive_execution", last_result.to_markdown())

    def _maybe_execute_biology_mechanism_script(
        self, store: ArtifactStore, state: WorkflowState, script_path: Path
    ) -> Path:
        store.event(
            "biology_mechanism_execution",
            {"summary": "running biological interpretation and mechanism mapping script"},
        )
        result = execute_python_script(
            script_path=script_path,
            run_dir=state.run_dir,
            timeout_seconds=min(self.config.script_timeout_seconds, 1800),
            repair_attempts=0,
            log_prefix="biology_mechanism",
        )
        self._add_artifact(state, "biology_mechanism_stdout", result.stdout_path)
        self._add_artifact(state, "biology_mechanism_stderr", result.stderr_path)
        self._add_artifact(
            state,
            "biology_mechanism_execution_json",
            store.write_json("biology_mechanism_execution", result.to_dict()),
        )
        return store.write_markdown("biology_mechanism_execution", result.to_markdown())

    def _maybe_execute_downstream_analysis_script(
        self, store: ArtifactStore, state: WorkflowState, script_path: Path
    ) -> Path:
        store.event(
            "hypothesis_downstream_execution",
            {"summary": "running selected-hypothesis downstream scRNA/scTCR analysis script"},
        )
        result = execute_python_script(
            script_path=script_path,
            run_dir=state.run_dir,
            timeout_seconds=min(self.config.script_timeout_seconds, 1800),
            repair_attempts=0,
            log_prefix="hypothesis_downstream",
        )
        self._add_artifact(state, "hypothesis_downstream_stdout", result.stdout_path)
        self._add_artifact(state, "hypothesis_downstream_stderr", result.stderr_path)
        self._add_artifact(
            state,
            "hypothesis_downstream_execution_json",
            store.write_json("hypothesis_downstream_execution", result.to_dict()),
        )
        return store.write_markdown("hypothesis_downstream_execution", result.to_markdown())

    def _execute_downstream_analysis_with_llm_repair(
        self,
        team: ScRTATeam,
        store: ArtifactStore,
        state: WorkflowState,
        script_path: Path,
        downstream_context: dict,
        rag_chunks: list,
    ) -> Path:
        """Run the LLM-authored downstream script and ask the LLM for repairs on failure."""
        max_attempts = max(1, int(self.config.repair_attempts) + 1)
        current_script = script_path
        last_result = None
        for attempt in range(1, max_attempts + 1):
            log_prefix = "hypothesis_downstream" if attempt == 1 else f"hypothesis_downstream_repair{attempt - 1}"
            store.event(
                "hypothesis_downstream_execution",
                {"summary": f"running LLM-authored downstream script attempt {attempt}/{max_attempts}"},
            )
            result = execute_python_script(
                script_path=current_script,
                run_dir=state.run_dir,
                timeout_seconds=min(self.config.script_timeout_seconds, 1800),
                repair_attempts=0,
                log_prefix=log_prefix,
            )
            last_result = result
            self._add_artifact(state, f"{log_prefix}_stdout", result.stdout_path)
            self._add_artifact(state, f"{log_prefix}_stderr", result.stderr_path)
            self._add_artifact(
                state,
                f"{log_prefix}_execution_json",
                store.write_json(f"{log_prefix}_execution", result.to_dict()),
            )
            if result.returncode == 0:
                self._add_artifact(state, "hypothesis_downstream_stdout", result.stdout_path)
                self._add_artifact(state, "hypothesis_downstream_stderr", result.stderr_path)
                self._add_artifact(
                    state,
                    "hypothesis_downstream_execution_json",
                    store.write_json("hypothesis_downstream_execution", result.to_dict()),
                )
                return store.write_markdown("hypothesis_downstream_execution", result.to_markdown())

            if attempt >= max_attempts:
                break

            repair_context = dict(downstream_context)
            repair_context["failed_downstream_script"] = truncate_text(read_text(current_script), 24000)
            repair_context["failed_stdout"] = truncate_text(read_text(Path(result.stdout_path)), 8000)
            repair_context["failed_stderr"] = truncate_text(read_text(Path(result.stderr_path)), 12000)
            repair_context["repair_requirement"] = "\n".join(
                [
                    "The previous LLM-authored downstream script failed.",
                    "Generate a repaired complete standalone Python script.",
                    "Do not switch to a generic fixed downstream workflow.",
                    "Keep the selected hypothesis and the original execution contract logic unless the error requires a minimal change.",
                    "Output the full repaired script between DOWNSTREAM_PYTHON_SCRIPT and END_DOWNSTREAM_PYTHON_SCRIPT.",
                ]
            )
            repair_response = self._call_and_store(
                team,
                store,
                state,
                "downstream_analyst",
                (
                    "Repair the failed selected-hypothesis downstream Python script using the stderr/stdout "
                    "and current run artifacts. Return a full corrected script between the required markers."
                ),
                self._context_with_rag(
                    repair_context,
                    rag_chunks,
                    "repair downstream hypothesis-specific Python script scRNA scTCR selected hypothesis stderr",
                    store=store,
                    state=state,
                    agent_name=f"downstream_analyst_repair_{attempt}",
                ),
                artifact_name=f"agent_downstream_analyst_repair_{attempt}",
            )
            try:
                repaired_script = render_downstream_analysis_script(state.run_dir, repair_response.content)
            except ValueError as exc:
                store.write_markdown(
                    f"hypothesis_downstream_repair_{attempt}_extraction_error",
                    (
                        "# Downstream Repair Extraction Error\n\n"
                        f"The repair agent did not emit a valid Python script block.\n\nError: {exc}\n"
                    ),
                )
                break
            current_script = store.write_script(f"hypothesis_downstream_analysis_repair_{attempt}.py", repaired_script)
            self._add_artifact(state, f"hypothesis_downstream_repair_{attempt}_script", current_script)

        assert last_result is not None
        self._add_artifact(state, "hypothesis_downstream_stdout", last_result.stdout_path)
        self._add_artifact(state, "hypothesis_downstream_stderr", last_result.stderr_path)
        self._add_artifact(
            state,
            "hypothesis_downstream_execution_json",
            store.write_json("hypothesis_downstream_execution", last_result.to_dict()),
        )
        return store.write_markdown("hypothesis_downstream_execution", last_result.to_markdown())

    def _execute_publication_figure_with_llm_repair(
        self,
        team: ScRTATeam,
        store: ArtifactStore,
        state: WorkflowState,
        script_path: Path,
        visualizer_context: dict,
        rag_chunks: list,
    ) -> Path:
        """Run the LLM-authored figure script and ask the LLM for repairs on failure."""
        if not self.config.execute_script:
            return store.write_markdown(
                "publication_figure_execution",
                "# Publication Figure Execution\n\nSkipped. Run with --execute after analysis outputs exist.",
            )
        max_attempts = max(1, int(self.config.repair_attempts) + 1)
        current_script = script_path
        last_result = None
        for attempt in range(1, max_attempts + 1):
            log_prefix = "publication_figure" if attempt == 1 else f"publication_figure_repair{attempt - 1}"
            store.event(
                "publication_figure_execution",
                {"summary": f"running LLM-authored publication figure script attempt {attempt}/{max_attempts}"},
            )
            result = execute_python_script(
                script_path=current_script,
                run_dir=state.run_dir,
                timeout_seconds=min(self.config.script_timeout_seconds, 1800),
                repair_attempts=0,
                log_prefix=log_prefix,
            )
            last_result = result
            self._add_artifact(state, f"{log_prefix}_stdout", result.stdout_path)
            self._add_artifact(state, f"{log_prefix}_stderr", result.stderr_path)
            self._add_artifact(
                state,
                f"{log_prefix}_execution_json",
                store.write_json(f"{log_prefix}_execution", result.to_dict()),
            )
            if result.returncode == 0:
                self._add_artifact(state, "publication_figure_stdout", result.stdout_path)
                self._add_artifact(state, "publication_figure_stderr", result.stderr_path)
                self._add_artifact(
                    state,
                    "publication_figure_execution_json",
                    store.write_json("publication_figure_execution", result.to_dict()),
                )
                return store.write_markdown("publication_figure_execution", result.to_markdown())

            if attempt >= max_attempts:
                break

            repair_context = dict(visualizer_context)
            repair_context["failed_publication_figure_script"] = truncate_text(read_text(current_script), 24000)
            repair_context["failed_stdout"] = truncate_text(read_text(Path(result.stdout_path)), 8000)
            repair_context["failed_stderr"] = truncate_text(read_text(Path(result.stderr_path)), 12000)
            repair_context["repair_requirement"] = "\n".join(
                [
                    "The previous LLM-authored publication figure script failed.",
                    "Generate a repaired complete standalone Python script.",
                    "Do not switch to a generic fixed figure workflow.",
                    "Do not use publication_figure_spec JSON or a fixed renderer template.",
                    "Keep the selected hypothesis and actual available_results_inventory as the figure source of truth.",
                    "Do not reference file names or columns from previous runs.",
                    "Every panel must be guarded by file-existence and column-existence checks.",
                    "If an optional panel table or column is missing, skip that panel and continue; do not raise.",
                    "Only raise SystemExit if no real rendered figure can be produced at all.",
                    "For TCR/same-clone tables, dynamically detect available numeric score or delta columns instead of requiring fixed legacy names.",
                    "Output the full repaired script between PUBLICATION_FIGURE_PYTHON_SCRIPT and END_PUBLICATION_FIGURE_PYTHON_SCRIPT.",
                ]
            )
            repair_response = self._call_and_store(
                team,
                store,
                state,
                "visualizer",
                (
                    "Repair the failed publication figure Python script using the stderr/stdout "
                    "and current run artifacts. Return a full corrected script between the required markers."
                ),
                self._context_with_rag(
                    repair_context,
                    rag_chunks,
                    "repair publication figure Python script scRNA scTCR selected hypothesis stderr",
                    store=store,
                    state=state,
                    agent_name=f"visualizer_repair_{attempt}",
                ),
                artifact_name=f"agent_visualizer_repair_{attempt}",
            )
            try:
                repaired_script = render_publication_figure_script(state.run_dir, repair_response.content)
            except ValueError as exc:
                store.write_markdown(
                    f"publication_figure_repair_{attempt}_extraction_error",
                    (
                        "# Publication Figure Repair Extraction Error\n\n"
                        f"The repair agent did not emit a valid Python script block.\n\nError: {exc}\n"
                    ),
                )
                break
            current_script = store.write_script(f"publication_figures_repair_{attempt}.py", repaired_script)
            self._add_artifact(state, f"publication_figure_repair_{attempt}_script", current_script)

        assert last_result is not None
        self._add_artifact(state, "publication_figure_stdout", last_result.stdout_path)
        self._add_artifact(state, "publication_figure_stderr", last_result.stderr_path)
        self._add_artifact(
            state,
            "publication_figure_execution_json",
            store.write_json("publication_figure_execution", last_result.to_dict()),
        )
        return store.write_markdown("publication_figure_execution", last_result.to_markdown())

    def _maybe_execute_figure_script(
        self, store: ArtifactStore, state: WorkflowState, script_path: Path
    ) -> Path:
        if not self.config.execute_script:
            return store.write_markdown(
                "publication_figure_execution",
                "# Publication Figure Execution\n\nSkipped. Run with --execute after analysis outputs exist.",
            )
        store.event("publication_figure_execution", {"summary": "running publication figure script"})
        result = execute_python_script(
            script_path=script_path,
            run_dir=state.run_dir,
            timeout_seconds=min(self.config.script_timeout_seconds, 1800),
            repair_attempts=0,
            log_prefix="publication_figure",
        )
        self._add_artifact(state, "publication_figure_stdout", result.stdout_path)
        self._add_artifact(state, "publication_figure_stderr", result.stderr_path)
        self._add_artifact(
            state,
            "publication_figure_execution_json",
            store.write_json("publication_figure_execution", result.to_dict()),
        )
        return store.write_markdown("publication_figure_execution", result.to_markdown())

    def _maybe_execute_script(
        self, store: ArtifactStore, state: WorkflowState, script_path: Path
    ) -> Path:
        if not self.config.execute_script:
            return store.write_markdown(
                "script_execution",
                "# Script Execution\n\nSkipped. Run with --execute to execute the generated analysis script.",
            )

        store.event("script_execution", {"summary": "running generated analysis script"})
        result = execute_python_script(
            script_path=script_path,
            run_dir=state.run_dir,
            timeout_seconds=self.config.script_timeout_seconds,
            repair_attempts=self.config.repair_attempts,
            log_prefix="script",
        )
        self._add_artifact(state, "script_stdout", result.stdout_path)
        self._add_artifact(state, "script_stderr", result.stderr_path)
        self._add_artifact(state, "script_execution_json", store.write_json("script_execution", result.to_dict()))
        return store.write_markdown("script_execution", result.to_markdown())

    def _call_and_store(
        self,
        team: ScRTATeam,
        store: ArtifactStore,
        state: WorkflowState,
        agent_name: str,
        instruction: str,
        context: dict,
        artifact_name: str | None = None,
    ) -> AgentResponse:
        response = team.call_agent(agent_name, instruction, context)
        name = artifact_name or f"agent_{agent_name}"
        path = store.write_markdown(name, response.content)
        self._add_artifact(state, name, path)
        meta_path = store.write_json(f"{name}_metadata", asdict(response))
        self._add_artifact(state, f"{name}_metadata", meta_path)
        return response

    def _context_with_rag(
        self,
        context: dict,
        rag_chunks: list,
        agent_query: str,
        store: ArtifactStore | None = None,
        state: WorkflowState | None = None,
        agent_name: str | None = None,
    ) -> dict:
        if not rag_chunks:
            return context
        query = "\n".join(
            [
                agent_query,
                context.get("research_brief", ""),
                context.get("dataset_profile", ""),
                context.get("prior_outputs", ""),
            ]
        )
        retrieved = retrieve_rag_chunks(
            rag_chunks,
            query,
            limit=max(1, int(self.config.rag_top_k)),
        )
        rag_context = render_rag_context(retrieved)
        if store and state and agent_name:
            path = store.write_markdown(f"rag_context_{agent_name}", rag_context)
            self._add_artifact(state, f"rag_context_{agent_name}", path)
        new_context = dict(context)
        new_context["literature_context"] = (
            context.get("literature_context", "").rstrip()
            + "\n\n"
            + rag_context
        ).strip()
        return new_context

    @staticmethod
    def _context_with_prior(base_context: dict, responses: list[AgentResponse]) -> dict:
        context = dict(base_context)
        # Keep the newest specialist outputs first because agent prompts are
        # length-limited and late-stage reporting needs deep-dive/mechanism
        # results more than early planning prose.
        context["prior_outputs"] = "\n\n".join(
            f"# {response.agent_name}\n{response.content}" for response in reversed(responses)
        )
        return context

    @staticmethod
    def _render_deep_dive_runtime_contract(run_dir: Path) -> str:
        output_dir = run_dir / "analysis_outputs"
        available = []
        if output_dir.exists():
            available = sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file())
        lines = [
            "# Deep-Dive Runtime Contract",
            "",
            f"- Run directory: {run_dir}",
            f"- Analysis outputs directory: {output_dir}",
            f"- Deep-dive output directory: {output_dir / 'deep_dive'}",
            f"- Script path that will be executed: {run_dir / 'scripts' / 'hypothesis_deep_dive.py'}",
            "",
            "The deep_planner must write a complete standalone Python script.",
            "The workflow will extract only the code between DEEP_DIVE_PYTHON_SCRIPT and END_DEEP_DIVE_PYTHON_SCRIPT.",
            "There is no fixed deep-dive execution template after this agent response.",
            "The script must choose analyses based on the selected hypothesis and available output tables.",
            "",
            "## Available analysis_outputs files",
        ]
        if available:
            lines.extend(f"- {item}" for item in available[:160])
            if len(available) > 160:
                lines.append(f"- ... {len(available) - 160} additional files omitted")
        else:
            lines.append("- No analysis_outputs files are currently available.")
        return "\n".join(lines)

    @staticmethod
    def _add_deep_dive_result_context(context: dict, run_dir: Path) -> None:
        deep_dir = run_dir / "analysis_outputs" / "deep_dive"
        if not deep_dir.exists():
            context["deep_dive_result_inventory"] = "No analysis_outputs/deep_dive directory exists."
            return
        files = sorted(path for path in deep_dir.rglob("*") if path.is_file())
        context["deep_dive_result_inventory"] = "\n".join(
            f"- {path.relative_to(deep_dir)} ({path.stat().st_size} bytes)" for path in files
        )
        priority_names = {
            "deep_dive_conclusion.md",
            "deep_dive_result_manifest.json",
            "deep_dive_summary.json",
            "deep_dive_execution_contract.json",
            "selected_hypothesis.md",
        }
        selected: list[Path] = []
        selected.extend([path for path in files if path.name in priority_names])
        selected.extend([path for path in files if path.suffix.lower() in {".csv", ".json", ".md"} and path not in selected])
        sections: list[str] = []
        for path in selected[:12]:
            try:
                rel = path.relative_to(deep_dir)
                sections.extend(
                    [
                        f"## {rel}",
                        "```text",
                        truncate_text(read_text(path), 4500),
                        "```",
                        "",
                    ]
                )
            except Exception:
                continue
        context["deep_dive_result_files"] = "\n".join(sections).strip()

    @staticmethod
    def _ensure_deep_dive_selected_hypothesis(run_dir: Path, selection) -> None:
        deep_dir = ensure_dir(run_dir / "analysis_outputs" / "deep_dive")
        selected_md = deep_dir / "selected_hypothesis.md"
        if not selected_md.exists():
            selected_md.write_text(selection.to_markdown(), encoding="utf-8")
        selected_json = deep_dir / "selected_hypothesis.json"
        if not selected_json.exists():
            selected_json.write_text(
                json.dumps(selection.to_dict(), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    @staticmethod
    def _render_dataset_reconnaissance_context(run_dir: Path) -> str:
        output_dir = run_dir / "analysis_outputs"
        files = [
            ("analysis_summary", output_dir / "analysis_summary.md", 4000),
            ("clone_state_summary", output_dir / "clone_state_summary.csv", 5000),
            ("patient_blocked_tests", output_dir / "patient_blocked_tests.csv", 5000),
            ("clone_size_null_tests", output_dir / "clone_size_null_tests.csv", 5000),
            (
                "within_state_signature_differences",
                output_dir / "within_state_signature_differences.csv",
                5000,
            ),
            ("within_state_de_top_genes", output_dir / "within_state_de_top_genes.csv", 5000),
            ("t_cell_cluster_summary", output_dir / "t_cell_cluster_summary.csv", 5000),
            ("t_cell_cluster_marker_summary", output_dir / "t_cell_cluster_marker_summary.csv", 7000),
            ("t_cell_occupied_clonotypes_by_group", output_dir / "t_cell_occupied_clonotypes_by_group.csv", 5000),
            ("t_cell_subcluster_annotation", output_dir / "t_cell_subcluster_annotation.md", 5000),
            ("top_clones", output_dir / "top_clones.csv", 3000),
        ]
        sections: list[str] = [
            "# Dataset Reconnaissance And Feasibility Context",
            "",
            "These outputs describe available metadata, RNA programs, TCR join structure,",
            "patient/tissue/timepoint contrasts, clone-size controls, and tables that can",
            "make a generated hypothesis executable. They are not a hypothesis screen,",
            "validation result, or rejection rule.",
        ]
        for label, path, limit in files:
            if not path.exists():
                continue
            sections.extend(
                [
                    "",
                    f"## {label}",
                    "```text",
                    truncate_text(read_text(path), limit),
                    "```",
                ]
            )
        if len(sections) == 1:
            sections.append("\nNo dataset reconnaissance output files were available.")
        return "\n".join(sections)

    @staticmethod
    def _render_team_summary(responses: list[AgentResponse]) -> str:
        parts = ["# Team Transcript", ""]
        for response in responses:
            parts.extend([f"## {response.agent_name}", "", response.content.strip(), ""])
        return "\n".join(parts).strip() + "\n"

    @staticmethod
    def _add_artifact(state: WorkflowState, name: str, path: str | Path) -> None:
        state.add_artifact(name, path)


def _interactive_edit(label: str, current: str, multiline: bool = False) -> str:
    print("")
    print(f"{label}:")
    if current:
        print(current)
    if multiline:
        print("Enter replacement text. Submit an empty line immediately to keep the current text.")
        print("End replacement text with a line containing only a single period.")
        first = input("> ")
        if not first.strip():
            return current
        lines = [first]
        while True:
            line = input("> ")
            if line.strip() == ".":
                break
            lines.append(line)
        replacement = "\n".join(lines).strip()
        return replacement or current
    replacement = input("Replacement (blank keeps current): ").strip()
    return replacement or current


def _interactive_edit_list(label: str, current: object) -> list[str]:
    values: list[str]
    if isinstance(current, list):
        values = [str(item).strip() for item in current if str(item).strip()]
    elif current:
        values = [str(current).strip()]
    else:
        values = []
    print("")
    print(f"{label}:")
    for item in values:
        print(f"- {item}")
    print("Enter replacement items separated by semicolons, or leave blank to keep current.")
    replacement = input("Replacement list: ").strip()
    if not replacement:
        return values
    edited = [item.strip() for item in replacement.split(";") if item.strip()]
    return edited or values
