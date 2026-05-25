# scRT-agent

scRT-agent is a focused workflow for paired single-cell RNA sequencing
(scRNA-seq) and single-cell T cell receptor sequencing (scTCR-seq) analysis.
It combines a fixed single-cell analysis layer with role-specific LLM agents for
literature-grounded hypothesis generation, clone-aware validation, downstream
analysis planning, mechanism interpretation, and report generation.

The workflow is RNA-first: transcriptional states, programs, tissues,
conditions and patient-level contrasts define the biological question. TCR
clonotypes are used as supporting evidence for lineage, persistence, clone
expansion, sharing, receptor follow-up priority and state occupancy. Clone
expansion or sharing is not interpreted as antigen specificity without
orthogonal evidence.

## Features

- Dataset profiling for `.h5ad` scRNA-seq objects and tabular scTCR files.
- Standard paired scRNA/scTCR analysis script generation.
- T-cell subclustering and marker-based state annotation support.
- Clone-size bins compatible with common repertoire summaries.
- Patient-aware, tissue-aware and clone-size-aware summary tables.
- RAG context injection from a local JSONL literature index.
- Biology-first hypothesis generation and hypothesis selection.
- LLM-written deep-dive and downstream analysis scripts.
- Biological interpretation, mechanism mapping and next-test proposals.
- Exported run artifacts, scripts, logs, tables and figures.

## Installation

```bash
git clone https://github.com/tangaode/scRT-agent.git
cd scRT-agent
pip install -e ".[analysis,llm]"
```

For development:

```bash
pip install -e ".[analysis,llm,dev]"
pytest
```

## Required Inputs

The main workflow expects:

- `rna_h5ad_path`: an `.h5ad` file containing the scRNA-seq matrix and cell
  metadata.
- `tcr_path`: a tabular scTCR file such as a 10x
  `filtered_contig_annotations.csv` file or a table containing barcode,
  clonotype, chain, CDR3 and V/J gene columns.

Useful RNA metadata columns include patient, sample, tissue, condition,
timepoint, response group, cluster and cell type. The workflow attempts to
profile available metadata and infer join keys before analysis.

## LLM Configuration

Set one of the following API keys before running the workflow:

```bash
export OPENAI_API_KEY="your_api_key"
# or
export SCRTA_AGENT_API_KEY="your_api_key"
```

For OpenAI-compatible endpoints:

```bash
export SCRTA_AGENT_API_BASE="https://your-compatible-endpoint/v1"
```

The default model can be overridden with `--model`.

## Quick Start

```bash
scrta-agent run \
  --rna /path/to/sample.h5ad \
  --tcr /path/to/filtered_contig_annotations.csv \
  --analysis-name example_scrna_sctcr \
  --out ./runs \
  --brief "Identify RNA-defined T-cell states with conservative TCR lineage support." \
  --execute
```

With a local RAG index:

```bash
scrta-agent run \
  --rna /path/to/sample.h5ad \
  --tcr /path/to/tcr.tsv.gz \
  --analysis-name example_rag_run \
  --out ./runs \
  --rag-index /path/to/rag_chunks.jsonl \
  --rag-top-k 10 \
  --brief "Propose and test biology-first hypotheses for this paired scRNA/scTCR cohort." \
  --execute
```

Disable optional loops if needed:

```bash
scrta-agent run \
  --rna /path/to/sample.h5ad \
  --tcr /path/to/tcr.tsv \
  --out ./runs \
  --no-deep-dive \
  --no-mechanism-loop \
  --no-downstream-analysis
```

## Build a Local RAG Index

The repository includes helper scripts for legal/open full-text retrieval and
structured card generation. A typical local build is:

```bash
python scripts/build_scrna_sctcr_rag.py \
  --out ./rag_kb/scrna_sctcr \
  --seed-csv /path/to/literature_cards.csv
```

The resulting JSONL chunks can be passed with `--rag-index`.

## Output Structure

Each run writes a timestamped directory under the selected output root. Common
artifacts include:

- `dataset_profile.md` and `dataset_profile.json`
- `environment.md` and `environment.json`
- `rag_context_*.md`
- `agent_*.md`
- `rag_grounded_hypothesis_candidates.md`
- `selected_hypothesis.md` and `selected_hypothesis.json`
- `scripts/scrna_sctcr_joint_analysis.py`
- `scripts/hypothesis_deep_dive.py`
- `scripts/hypothesis_downstream_analysis.py`
- `scripts/biology_mechanism.py`
- `scripts/publication_figures.py`
- `analysis_outputs/*.csv`
- `analysis_outputs/figures/*.png`
- `analysis_outputs/publication_figures/*.pdf`
- `final_report.md`

## Command Reference

List agent roles:

```bash
scrta-agent agents
scrta-agent agents --json
```

Run from a JSON config:

```bash
scrta-agent run --config examples/config.example.json
```

Important options:

- `--execute`: run the generated analysis script.
- `--repair-attempts N`: retry script execution after transient failures.
- `--script-timeout SECONDS`: set script execution timeout.
- `--rag-index PATH`: inject local RAG chunks into agent prompts.
- `--rag-top-k N`: number of retrieved chunks per agent call.
- `--model MODEL`: LLM model name.

## TCR Interpretation Guardrails

scRT-agent treats TCR evidence conservatively:

- Clone expansion supports clonal enrichment, not antigen specificity.
- Shared clonotypes support lineage relatedness or state occupancy, not
  migration by themselves.
- CDR3 similarity or V/J usage can prioritize receptor follow-up, but does not
  establish antigen identity without experimental validation.
- Patient structure, sample composition and clone-size effects should be
  controlled before drawing cohort-level conclusions.

## Repository Contents

- `src/scrta_agent/`: package source code.
- `src/scrta_agent/prompts/`: role prompts.
- `src/scrta_agent/templates/`: generated Python script templates.
- `scripts/`: optional RAG and literature preparation utilities.
- `skills/`: domain workflow rules loaded by the package.
- `examples/`: minimal configuration example.
- `tests/`: lightweight package tests.

## License

This repository is provided for research use. Add a project-specific license
before redistribution if required by your institution or journal.
