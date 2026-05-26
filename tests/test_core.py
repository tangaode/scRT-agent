from __future__ import annotations

from pathlib import Path
import gzip
import zipfile

import pytest

from scrta_agent.agents import ScRTATeam
from scrta_agent.data_import import (
    choose_rna_sources,
    choose_tcr_sources,
    clone_size_category,
    materialize_input_paths,
    prepare_inputs,
    split_user_paths,
)
from scrta_agent.deep_dive import DeepDiveSelection
from scrta_agent.llm import LLMClient
import scrta_agent.llm as llm_module
from scrta_agent.rag import RagChunk, retrieve_rag_chunks
from scrta_agent.schemas import WorkflowConfig
from scrta_agent.script_writer import (
    render_biology_mechanism_script,
    render_deep_dive_script,
    render_downstream_analysis_script,
    render_joint_analysis_script,
    render_publication_figure_script,
)
from scrta_agent.workflow import ScRTAWorkflow


def test_fixed_team_has_expected_roles() -> None:
    team = ScRTATeam(LLMClient(use_llm=False))
    names = {agent["name"] for agent in team.list_agents()}
    assert names == {
        "leader",
        "rna_analyst",
        "methodologist",
        "tcr_analyst",
        "t_cell_annotator",
        "integrator",
        "novelty_scout",
        "hypothesis_generator",
        "hypothesis_selector",
        "deep_planner",
        "skeptic",
        "code_writer",
        "visualizer",
        "result_interpreter",
        "biological_interpreter",
        "mechanism_mapper",
        "next_test_planner",
        "downstream_analyst",
        "system_manager",
        "reporter",
    }


def test_agent_fallback_runs_without_llm() -> None:
    team = ScRTATeam(LLMClient(use_llm=False))
    response = team.call_agent(
        "skeptic",
        "Audit this plan.",
        {
            "dataset_profile": "",
            "research_brief": "",
            "literature_context": "",
            "prior_outputs": "",
        },
    )
    assert response.metadata["mode"] == "deterministic"
    assert "Expansion alone is not" in response.content


def test_rendered_analysis_script_compiles(tmp_path: Path) -> None:
    config = WorkflowConfig(
        rna_h5ad_path=str(tmp_path / "rna.h5ad"),
        tcr_path=str(tmp_path / "tcr.csv"),
        analysis_name="compile_test",
        output_root=str(tmp_path),
    )
    script = render_joint_analysis_script(config, tmp_path)
    compile(script, "scrna_sctcr_joint_analysis.py", "exec")
    assert "RNA_PATH" in script
    assert "SIGNATURES" in script
    assert "patient_blocked_stats" in script
    assert "clone_size_aware_null_tests" in script
    assert "run_fixed_t_cell_baseline" in script
    assert "t_cell_occupied_clonotypes_by_group.csv" in script
    assert "plot_occupied_clonotypes_by_group" in script


def test_rendered_deep_dive_script_compiles(tmp_path: Path) -> None:
    selection = DeepDiveSelection(
        hypothesis_id="HYP-1",
        title="Compile-test selected hypothesis",
        selected_hypothesis="A selected hypothesis can be passed into the deep-dive script.",
        rationale="Compile-time placeholder.",
        required_tests=["Compile rendered script."],
        falsification_criteria=["The script does not compile."],
        source_tables=["rag_grounded_hypothesis_candidates.md"],
    )
    llm_output = """
DEEP_DIVE_PYTHON_SCRIPT
```python
import json

DEEP_DIVE_DIR.mkdir(parents=True, exist_ok=True)
(DEEP_DIVE_DIR / "selected_hypothesis.md").write_text(SELECTED_HYPOTHESIS["selected_hypothesis"], encoding="utf-8")
(DEEP_DIVE_DIR / "deep_dive_analysis_plan.md").write_text("# Plan\\n", encoding="utf-8")
(DEEP_DIVE_DIR / "deep_dive_execution_contract.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
(DEEP_DIVE_DIR / "deep_dive_result_manifest.json").write_text(json.dumps([]), encoding="utf-8")
(DEEP_DIVE_DIR / "deep_dive_conclusion.md").write_text("# Conclusion\\n", encoding="utf-8")
(DEEP_DIVE_DIR / "deep_dive_summary.json").write_text(json.dumps({"status": "tested"}), encoding="utf-8")
```
END_DEEP_DIVE_PYTHON_SCRIPT
"""
    script = render_deep_dive_script(tmp_path, selection, llm_output)
    compile(script, "hypothesis_deep_dive.py", "exec")
    assert "Auto-extracted from deep_planner" in script
    assert "DEEP_DIVE_DIR" in script
    assert "deep_dive_conclusion.md" in script


def test_deep_dive_script_requires_llm_code_block(tmp_path: Path) -> None:
    selection = DeepDiveSelection(
        hypothesis_id="HYP-1",
        title="Compile-test selected hypothesis",
        selected_hypothesis="A selected hypothesis can be passed into the deep-dive script.",
        rationale="Compile-time placeholder.",
        required_tests=["Compile rendered script."],
        falsification_criteria=["The script does not compile."],
        source_tables=["rag_grounded_hypothesis_candidates.md"],
    )
    with pytest.raises(ValueError):
        render_deep_dive_script(tmp_path, selection, "# plan only")


def test_rendered_biology_mechanism_script_compiles(tmp_path: Path) -> None:
    script = render_biology_mechanism_script(tmp_path)
    compile(script, "biology_mechanism.py", "exec")
    assert "mechanism_evidence_map.csv" in script
    assert "next_test_proposals.csv" in script


def test_rendered_downstream_analysis_script_compiles(tmp_path: Path) -> None:
    llm_output = """
DOWNSTREAM_PYTHON_SCRIPT
```python
from pathlib import Path

out = Path("analysis_outputs") / "downstream"
out.mkdir(parents=True, exist_ok=True)
(out / "downstream_analysis_summary.md").write_text("# Summary\\n", encoding="utf-8")
```
END_DOWNSTREAM_PYTHON_SCRIPT
"""
    script = render_downstream_analysis_script(tmp_path, llm_output)
    compile(script, "hypothesis_downstream_analysis.py", "exec")
    assert "Auto-extracted from downstream_analyst" in script
    assert "downstream_analysis_summary.md" in script


def test_downstream_script_requires_llm_code_block(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        render_downstream_analysis_script(tmp_path, "# plan only")


def test_rendered_publication_figure_script_compiles(tmp_path: Path) -> None:
    llm_output = """
PUBLICATION_FIGURE_PYTHON_SCRIPT
```python
import json

summary = FIG_DIR / "publication_figure_summary.md"
summary.write_text("# Figures\\n", encoding="utf-8")
(FIG_DIR / "publication_figure_qc.json").write_text(json.dumps([]), encoding="utf-8")
```
END_PUBLICATION_FIGURE_PYTHON_SCRIPT
"""
    script = render_publication_figure_script(tmp_path, llm_output)
    compile(script, "publication_figures.py", "exec")
    assert "Auto-extracted from visualizer" in script
    assert "publication_figure_summary.md" in script


def test_marked_publication_script_allows_notes_after_fence(tmp_path: Path) -> None:
    llm_output = """
PUBLICATION_FIGURE_PYTHON_SCRIPT
```python
summary = FIG_DIR / "publication_figure_summary.md"
summary.write_text("# Figures\\n", encoding="utf-8")
```
These notes should not be treated as executable Python.
END_PUBLICATION_FIGURE_PYTHON_SCRIPT
"""
    script = render_publication_figure_script(tmp_path, llm_output)
    compile(script, "publication_figures.py", "exec")
    assert "These notes" not in script


def test_marked_publication_script_allows_unclosed_fence(tmp_path: Path) -> None:
    llm_output = """
PUBLICATION_FIGURE_PYTHON_SCRIPT
```python
summary = FIG_DIR / "publication_figure_summary.md"
summary.write_text("# Figures\\n", encoding="utf-8")
END_PUBLICATION_FIGURE_PYTHON_SCRIPT
"""
    script = render_publication_figure_script(tmp_path, llm_output)
    compile(script, "publication_figures.py", "exec")
    assert "```python" not in script


def test_publication_figure_script_requires_llm_code_block(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        render_publication_figure_script(tmp_path, "# figure plan only")


def test_rag_query_expands_known_geo_accession() -> None:
    chunks = [
        RagChunk(
            chunk_id="paper",
            doc_id="39889705",
            title="Contrasting cytotoxic and regulatory T cell responses underlying distinct clinical outcomes to anti-PD-1 plus lenvatinib therapy in cancer.",
            text="HBV-specific CD8 cTem Tpex KIR FOXP3 Treg dCODE dextramer scTCR clonotype.",
            year="2025",
        ),
        RagChunk(
            chunk_id="generic",
            doc_id="generic",
            title="Generic paired single-cell TCR analysis",
            text="paired scRNA scTCR clonotype analysis",
        ),
    ]
    hits = retrieve_rag_chunks(chunks, "GSE235863 paired scRNA scTCR analysis", limit=1)
    assert hits[0].doc_id == "39889705"


def test_interactive_path_split_and_clone_bins() -> None:
    paths = split_user_paths("a; b\nc")
    assert [path.name for path in paths] == ["a", "b", "c"]
    assert clone_size_category(1) == "Single"
    assert clone_size_category(5) == "Small"
    assert clone_size_category(20) == "Medium"
    assert clone_size_category(100) == "Large"
    assert clone_size_category(500) == "Hyperexpanded"


def test_project_folder_archive_materialization_discovers_tcr_tables(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    archive = project / "sample_a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            "sample_a/filtered_contig_annotations.csv",
            "barcode,raw_clonotype_id,cdr3\nAAAC-1,clonotype1,CASSA\n",
        )

    materialized, notes = materialize_input_paths([project], tmp_path / "extracted")
    sources = choose_tcr_sources(materialized, {})

    assert notes
    assert len(sources) == 1
    assert sources[0].name == "filtered_contig_annotations.csv"


def test_h5ad_under_tcr_named_parent_is_valid_rna_input(tmp_path: Path) -> None:
    project = tmp_path / "scRNA-TCR-project"
    project.mkdir()
    h5ad = project / "sample_rna.h5ad"
    h5ad.write_bytes(b"placeholder")

    sources = choose_rna_sources([h5ad], {})
    folder_sources = choose_rna_sources([project], {})

    assert sources == [h5ad.resolve()]
    assert folder_sources == [h5ad.resolve()]


def test_plan_review_display_omits_truncated_execution_prose() -> None:
    plan_text = """
Short deep-dive plan: I will implement a hypothesis-specific validator that first reads the selected
hypothesis text and then searches the existing analysis outputs for the exact metadata, state labels,
and signatures needed to test the claim in a generated Python script.

DEEP_DIVE_PYTHON_SCRIPT
print("placeholder")
END_DEEP_DIVE_PYTHON_SCRIPT
"""

    display = ScRTAWorkflow._render_plan_review_display("Deep-Dive", plan_text)

    assert "[truncated]" not in display
    assert "Short deep-dive plan" not in display
    assert "I will implement" not in display
    assert "Python script" not in display
    assert "Review the selected hypothesis" in display


def test_llm_client_loads_root_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".scrta_agent.env"
    env_file.write_text(
        "\ufeffSCRTA_AGENT_API_KEY=dummy_key\nSCRTA_AGENT_API_BASE=https://example.invalid/v1\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("SCRTA_AGENT_API_KEY", raising=False)
    monkeypatch.delenv("SCRTA_AGENT_API_BASE", raising=False)
    monkeypatch.setenv("SCRTA_AGENT_ENV_FILE", str(env_file))
    monkeypatch.setattr(llm_module, "_LOCAL_ENV_LOADED", False)

    client = LLMClient(use_llm=True)

    assert client.api_key == "dummy_key"
    assert client.base_url == "https://example.invalid/v1"


def test_prepare_inputs_handles_prefixed_geo_10x_triplets_and_tcr_gz(tmp_path: Path) -> None:
    pytest.importorskip("anndata")
    pytest.importorskip("scipy")
    pd = pytest.importorskip("pandas")

    raw = tmp_path / "GSE_RAW"
    raw.mkdir()
    _write_gzip_text(
        raw / "GSM1_sampleA.matrix.mtx.gz",
        "%%MatrixMarket matrix coordinate integer general\n%\n2 2 2\n1 1 3\n2 2 4\n",
    )
    _write_gzip_text(raw / "GSM1_sampleA.barcodes.tsv.gz", "AAAC-1\nTTTG-1\n")
    _write_gzip_text(raw / "GSM1_sampleA.features.tsv.gz", "ENSG1\tGZMB\tGene Expression\nENSG2\tTCF7\tGene Expression\n")
    _write_gzip_text(
        raw / "GSM2_sampleA-TCR_filtered_contig_annotations.csv.gz",
        "barcode,raw_clonotype_id,cdr3\nAAAC-1,clonotype1,CASSA\nTTTG-1,clonotype2,CASSB\n",
    )
    _write_gzip_text(
        raw / "GSM3_sampleA-BCR_filtered_contig_annotations.csv.gz",
        "barcode,raw_clonotype_id,cdr3\nAAAC-1,bcr1,CARDR\n",
    )

    result = prepare_inputs(
        rna_inputs=str(raw),
        tcr_inputs=str(raw),
        output_dir=tmp_path / "prepared",
        require_llm_plan=False,
    )

    tcr = pd.read_csv(result.tcr_path)
    assert Path(result.rna_h5ad_path).exists()
    assert set(tcr["clonotype_id"]) == {"clonotype1", "clonotype2"}
    assert tcr["sample_id"].str.contains("TCR").all()


def _write_gzip_text(path: Path, text: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(text)
