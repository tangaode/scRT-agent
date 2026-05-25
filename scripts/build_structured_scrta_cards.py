from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

from build_scrna_sctcr_rag import PaperRecord, now_stamp, slugify, write_json


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scrta_agent.llm import LLMClient  # noqa: E402


CARD_FIELDS = [
    "doc_id",
    "title",
    "year",
    "journal",
    "pmid",
    "pmcid",
    "doi",
    "source_url",
    "relevance_class",
    "topic_tags",
    "study_object",
    "disease_or_physiological_condition",
    "species",
    "sample_size",
    "sample_types",
    "cohort_or_group_design",
    "core_hypothesis",
    "scRNA_methods",
    "scTCR_methods",
    "joint_analysis_methods",
    "scRNA_results",
    "scRNA_biological_question_answered",
    "scTCR_results",
    "scTCR_biological_question_answered",
    "joint_analysis_results",
    "article_conclusion",
    "limitations_or_caveats",
    "hypothesis_generation_logic",
    "new_hypotheses_suggested_by_this_study",
    "reusable_analysis_patterns_for_agent",
    "code_generation_guidance",
    "evidence_confidence",
    "extraction_method",
    "extraction_notes",
]


SYSTEM_PROMPT = """You are building a structured scientific RAG knowledge base for an autonomous scRNA-seq + scTCR-seq analysis agent.

Extract only information supported by the supplied paper text. Write in precise English. If a field is not stated, write "not stated".

The most important goal is to teach the agent how papers move from a study question to a testable hypothesis, then to scRNA/scTCR/joint analyses, then to biologically bounded conclusions. Do not overclaim antigen specificity, migration, lineage transition, or treatment causality unless the text explicitly supports it.

Return exactly one JSON object. Do not use Markdown.
"""


USER_TEMPLATE = """Paper metadata:
Title: {title}
Year: {year}
PMID: {pmid}
PMCID: {pmcid}
DOI: {doi}
Topic tags: {topic_tags}

Paper text snippets:
{evidence}

Return a JSON object with exactly these keys:
- study_object
- disease_or_physiological_condition
- species
- sample_size
- sample_types
- cohort_or_group_design
- core_hypothesis
- scRNA_methods
- scTCR_methods
- joint_analysis_methods
- scRNA_results
- scRNA_biological_question_answered
- scTCR_results
- scTCR_biological_question_answered
- joint_analysis_results
- article_conclusion
- limitations_or_caveats
- hypothesis_generation_logic
- new_hypotheses_suggested_by_this_study
- reusable_analysis_patterns_for_agent
- code_generation_guidance
- evidence_confidence
- extraction_notes

Field guidance:
- sample_size should include numbers of patients, donors, samples, tissues, cells, clones, or datasets when stated.
- cohort_or_group_design should describe controls, diseases, treatment arms, time points, tissues, responders/nonresponders, or other groups.
- hypothesis_generation_logic should explain how the paper turns prior knowledge or observations into a testable hypothesis.
- new_hypotheses_suggested_by_this_study should propose cautious follow-up hypotheses an analysis agent could test in new paired scRNA-scTCR data.
- code_generation_guidance should say what analysis code/modules the agent should generate because of this paper.
"""


def bool_from_csv(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def record_from_csv(row: dict[str, str]) -> PaperRecord:
    kwargs: dict[str, Any] = {}
    bool_fields = {"open_access", "full_text_downloaded"}
    float_fields = {"relevance_score"}
    field_names = {f.name for f in fields(PaperRecord)}
    for key in field_names:
        value = row.get(key, "")
        if key in bool_fields:
            kwargs[key] = bool_from_csv(value)
        elif key in float_fields:
            try:
                kwargs[key] = float(value or 0)
            except ValueError:
                kwargs[key] = 0.0
        else:
            kwargs[key] = value
    return PaperRecord(**kwargs)


def read_records(path: Path) -> list[PaperRecord]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        return [record_from_csv(row) for row in csv.DictReader(fh)]


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) >= 80]


KEYWORD_WEIGHTS = {
    "patient": 5,
    "patients": 5,
    "donor": 5,
    "donors": 5,
    "sample": 4,
    "cohort": 5,
    "group": 3,
    "control": 3,
    "responder": 4,
    "nonresponder": 4,
    "single-cell": 5,
    "scrna": 6,
    "rna-seq": 4,
    "tcr": 8,
    "sctcr": 8,
    "vdj": 6,
    "clonotype": 8,
    "clonal": 7,
    "expansion": 5,
    "trajectory": 4,
    "pseudotime": 4,
    "umap": 2,
    "cluster": 3,
    "differential": 3,
    "exhaust": 4,
    "cytotoxic": 4,
    "conclusion": 4,
    "suggest": 3,
    "indicate": 3,
    "hypothesis": 5,
    "we hypothesized": 8,
    "we found": 6,
    "we show": 6,
    "our results": 5,
}


def paragraph_score(paragraph: str) -> int:
    lower = paragraph.lower()
    score = 0
    for keyword, weight in KEYWORD_WEIGHTS.items():
        if keyword in lower:
            score += weight
    if re.search(r"\b\d+\s+(patients|donors|samples|cells|clonotypes|mice|subjects)\b", lower):
        score += 8
    return score


def make_evidence_pack(record: PaperRecord, max_chars: int = 16000) -> str:
    pieces = [
        f"TITLE: {record.title}",
        f"ABSTRACT_OR_CARD: {record.abstract}",
    ]
    text = ""
    if record.text_path and Path(record.text_path).exists():
        text = Path(record.text_path).read_text(encoding="utf-8", errors="replace")
    text = clean_text(text)
    if not text:
        return "\n\n".join(pieces)[:max_chars]

    paragraphs = split_paragraphs(text)
    intro = "\n\n".join(paragraphs[:4])
    scored = sorted(
        ((paragraph_score(p), idx, p) for idx, p in enumerate(paragraphs)),
        key=lambda x: (x[0], -x[1]),
        reverse=True,
    )
    selected: list[str] = []
    seen: set[int] = set()
    for score, idx, paragraph in scored:
        if score <= 0:
            break
        if idx in seen:
            continue
        seen.add(idx)
        selected.append(paragraph)
        if sum(len(p) for p in selected) >= max_chars - 5000:
            break

    pieces.append("EARLY_CONTEXT:\n" + intro)
    pieces.append("HIGH_VALUE_SNIPPETS:\n" + "\n\n".join(selected))
    return "\n\n".join(pieces)[:max_chars]


def record_full_text(record: PaperRecord, max_chars: int = 50000) -> str:
    pieces = [record.title, record.abstract]
    if record.text_path and Path(record.text_path).exists():
        pieces.append(Path(record.text_path).read_text(encoding="utf-8", errors="replace")[:max_chars])
    return clean_text("\n\n".join(p for p in pieces if p))


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if 50 <= len(p.strip()) <= 450]


def extract_sentences(text: str, required_any: tuple[str, ...], bonus_any: tuple[str, ...] = (), limit: int = 3) -> str:
    scored: list[tuple[int, int, str]] = []
    for idx, sentence in enumerate(split_sentences(text)):
        lower = sentence.lower()
        if required_any and not any(term in lower for term in required_any):
            continue
        score = sum(3 for term in required_any if term in lower)
        score += sum(1 for term in bonus_any if term in lower)
        if re.search(r"\b\d[\d,]*\b", sentence):
            score += 1
        scored.append((score, -idx, sentence))
    scored.sort(reverse=True)
    selected: list[str] = []
    seen: set[str] = set()
    for _, _, sentence in scored:
        norm = sentence.lower()
        if norm in seen:
            continue
        selected.append(sentence)
        seen.add(norm)
        if len(selected) >= limit:
            break
    return " ".join(selected) if selected else "not stated"


def empty_card(record: PaperRecord, method: str, note: str) -> dict[str, Any]:
    card = {
        "doc_id": record.doc_id,
        "title": record.title,
        "year": record.year,
        "journal": record.journal,
        "pmid": record.pmid,
        "pmcid": record.pmcid,
        "doi": record.doi,
        "source_url": record.source_url,
        "relevance_class": record.relevance_class,
        "topic_tags": record.topic_tags,
        "extraction_method": method,
        "extraction_notes": note,
    }
    for field in CARD_FIELDS:
        card.setdefault(field, "not stated")
    return card


def heuristic_card(record: PaperRecord, note: str = "") -> dict[str, Any]:
    text = record_full_text(record)
    card = empty_card(record, "heuristic", note or "Rule-based fallback extraction.")
    lower = text.lower()

    condition_terms = []
    for term in (
        "cancer",
        "tumor",
        "melanoma",
        "lung cancer",
        "colorectal cancer",
        "covid-19",
        "autoimmune",
        "infection",
        "diabetes",
        "arthritis",
        "healthy",
    ):
        if term in lower:
            condition_terms.append(term)
    sample_hits = re.findall(
        r"\b\d[\d,]*\s+(?:patients|donors|samples|cells|clonotypes|mice|subjects|tissues)\b",
        text,
        flags=re.IGNORECASE,
    )
    design = extract_sentences(
        text,
        ("patients", "donors", "samples", "cohort", "group", "control", "treated", "responder", "nonresponder", "tissue"),
        ("single-cell", "scrna", "tcr", "clonotype", "tumor", "disease"),
        limit=4,
    )
    scrna_methods = extract_sentences(
        text,
        ("scrna", "single-cell rna", "single cell rna", "transcriptom", "seurat", "scanpy", "cluster"),
        ("method", "analysis", "umap", "marker", "differential", "signature"),
        limit=4,
    )
    sctcr_methods = extract_sentences(
        text,
        ("tcr", "vdj", "cdr3", "clonotype", "repertoire"),
        ("method", "analysis", "clone", "diversity", "trav", "trbv"),
        limit=4,
    )
    joint_methods = extract_sentences(
        text,
        ("scrna", "single-cell", "tcr", "clonotype", "clonal"),
        ("integrat", "combined", "paired", "joint", "transcriptom", "repertoire", "state"),
        limit=4,
    )
    scrna_results = extract_sentences(
        text,
        ("scrna", "single-cell rna", "transcriptom", "cluster", "state", "signature"),
        ("identified", "revealed", "showed", "found", "enriched", "increased", "decreased"),
        limit=4,
    )
    sctcr_results = extract_sentences(
        text,
        ("tcr", "clonotype", "clonal", "repertoire", "cdr3"),
        ("identified", "revealed", "showed", "expanded", "shared", "diversity", "enriched"),
        limit=4,
    )
    conclusion = extract_sentences(
        text[-12000:] if len(text) > 12000 else text,
        ("conclude", "conclusion", "demonstrate", "suggest", "indicate", "show", "reveal"),
        ("single-cell", "tcr", "clonal", "transcriptom", "tumor", "immune"),
        limit=4,
    )

    card.update(
        {
            "study_object": "T cells profiled by single-cell transcriptomics and TCR/clonotype information.",
            "disease_or_physiological_condition": "; ".join(dict.fromkeys(condition_terms)) or "not stated",
            "species": "human" if "human" in lower or "patients" in lower else ("mouse" if "mice" in lower or "murine" in lower else "not stated"),
            "sample_size": "; ".join(dict.fromkeys(sample_hits[:12])) or "not stated",
            "sample_types": "not stated",
            "cohort_or_group_design": design,
            "core_hypothesis": "The study likely tests whether T cell transcriptional states are linked to clonotype structure, expansion, sharing, or antigen-receptor features.",
            "scRNA_methods": scrna_methods if scrna_methods != "not stated" else "Single-cell RNA-seq analysis, clustering, marker/signature interpretation, and cell-state annotation when described.",
            "scTCR_methods": sctcr_methods if sctcr_methods != "not stated" else "TCR repertoire or clonotype analysis, including CDR3/V(D)J/clonal expansion metrics when described.",
            "joint_analysis_methods": joint_methods if joint_methods != "not stated" else "Link clonotype/TCR features to RNA-defined cell states, tissues, groups, or outcomes.",
            "scRNA_results": scrna_results,
            "scRNA_biological_question_answered": "Which T cell states or transcriptional programs are present in the studied condition.",
            "scTCR_results": sctcr_results,
            "scTCR_biological_question_answered": "Whether T cell responses show clonal expansion, restriction, sharing, or repertoire shifts.",
            "joint_analysis_results": extract_sentences(
                text,
                ("clonotype", "clonal", "tcr", "single-cell", "scrna"),
                ("state", "integrat", "paired", "link", "associate", "correlat", "expanded"),
                limit=4,
            ),
            "article_conclusion": conclusion,
            "limitations_or_caveats": "Do not infer antigen specificity, migration, or causal differentiation from clonotype/RNA association alone unless explicitly validated.",
            "hypothesis_generation_logic": "Use prior disease context or observed T cell-state variation to propose a testable link between clonotype expansion/sharing and RNA-defined functional states.",
            "new_hypotheses_suggested_by_this_study": "In a new paired scRNA-scTCR dataset, test whether expanded clonotypes are enriched in specific RNA states, tissues, treatment-response groups, or cytotoxic/exhaustion/residency programs.",
            "reusable_analysis_patterns_for_agent": "Profile RNA-defined T cell states; define patient/sample-scoped clonotypes; quantify expansion; test clone-state enrichment; compare repertoire metrics across groups; report caveats.",
            "code_generation_guidance": "Generate code for sample-aware RNA-TCR joins, clonotype definition, expansion bins, repertoire summaries, state-by-clone enrichment, group comparisons, and evidence-linked figures.",
            "evidence_confidence": "low" if not record.full_text_downloaded else "medium",
        }
    )
    return card


def extract_json(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def llm_card(record: PaperRecord, client: LLMClient) -> dict[str, Any]:
    evidence = make_evidence_pack(record)
    user_prompt = USER_TEMPLATE.format(
        title=record.title,
        year=record.year,
        pmid=record.pmid,
        pmcid=record.pmcid,
        doi=record.doi,
        topic_tags=record.topic_tags,
        evidence=evidence,
    )
    response = client.complete(SYSTEM_PROMPT, user_prompt, temperature=0)
    parsed = extract_json(response)
    card = empty_card(record, "llm_gpt", "")
    for key in parsed:
        if key in CARD_FIELDS:
            card[key] = parsed[key]
    card["extraction_method"] = f"llm:{client.model}"
    card["extraction_notes"] = str(parsed.get("extraction_notes", "") or "")
    return card


def card_to_markdown(card: dict[str, Any]) -> str:
    lines = [f"# {card['title']}", ""]
    for key in CARD_FIELDS:
        if key in {"title"}:
            continue
        value = card.get(key, "not stated")
        if isinstance(value, list):
            value = "; ".join(str(x) for x in value)
        lines.append(f"## {key}")
        lines.append(str(value).strip() or "not stated")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def card_to_rag_text(card: dict[str, Any]) -> str:
    ordered = [
        "study_object",
        "disease_or_physiological_condition",
        "species",
        "sample_size",
        "cohort_or_group_design",
        "core_hypothesis",
        "scRNA_methods",
        "scTCR_methods",
        "joint_analysis_methods",
        "scRNA_results",
        "scTCR_results",
        "joint_analysis_results",
        "article_conclusion",
        "limitations_or_caveats",
        "hypothesis_generation_logic",
        "new_hypotheses_suggested_by_this_study",
        "reusable_analysis_patterns_for_agent",
        "code_generation_guidance",
    ]
    lines = [f"Structured scRNA-scTCR RAG card for: {card['title']}"]
    for key in ordered:
        value = card.get(key, "not stated")
        if isinstance(value, list):
            value = "; ".join(str(x) for x in value)
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def write_outputs(kb_dir: Path, cards: list[dict[str, Any]]) -> None:
    out_dir = kb_dir / "structured_cards"
    md_dir = out_dir / "markdown"
    md_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "scrna_sctcr_structured_cards.jsonl"
    csv_path = out_dir / "scrna_sctcr_structured_cards.csv"
    chunks_path = out_dir / "structured_rag_chunks.jsonl"
    combined_path = out_dir / "rag_structured_plus_fulltext_chunks.jsonl"

    with jsonl_path.open("w", encoding="utf-8") as fh:
        for card in cards:
            fh.write(json.dumps(card, ensure_ascii=False) + "\n")
            filename = f"{slugify(str(card.get('doc_id') or card.get('doi') or card.get('title')), 80)}.md"
            (md_dir / filename).write_text(card_to_markdown(card), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CARD_FIELDS)
        writer.writeheader()
        for card in cards:
            row = {}
            for key in CARD_FIELDS:
                value = card.get(key, "not stated")
                if isinstance(value, list):
                    value = "; ".join(str(x) for x in value)
                elif isinstance(value, dict):
                    value = json.dumps(value, ensure_ascii=False)
                row[key] = value
            writer.writerow(row)

    with chunks_path.open("w", encoding="utf-8") as fh:
        for card in cards:
            item = {
                "chunk_id": f"structured_{card.get('doc_id')}",
                "doc_id": card.get("doc_id", ""),
                "title": card.get("title", ""),
                "text": card_to_rag_text(card),
                "source_url": card.get("source_url", ""),
                "pmid": card.get("pmid", ""),
                "pmcid": card.get("pmcid", ""),
                "doi": card.get("doi", ""),
                "year": card.get("year", ""),
                "journal": card.get("journal", ""),
                "section": "structured_scientific_card",
                "is_full_text": True,
                "relevance_score": 100.0 if card.get("relevance_class") == "core" else 80.0,
            }
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    source_chunks = kb_dir / "rag_chunks.jsonl"
    with combined_path.open("w", encoding="utf-8") as out:
        out.write(chunks_path.read_text(encoding="utf-8", errors="replace"))
        if source_chunks.exists():
            out.write(source_chunks.read_text(encoding="utf-8", errors="replace"))

    latest = Path("E:/scRTA/rag_kb/scrna_sctcr_rag_current")
    latest.mkdir(parents=True, exist_ok=True)
    (latest / "structured_cards.jsonl").write_text(jsonl_path.read_text(encoding="utf-8"), encoding="utf-8")
    (latest / "rag_chunks.jsonl").write_text(combined_path.read_text(encoding="utf-8"), encoding="utf-8")

    index_path = kb_dir / "index.json"
    index: dict[str, Any] = {}
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8", errors="replace"))
    paths = index.setdefault("paths", {})
    paths.update(
        {
            "structured_cards_jsonl": str(jsonl_path),
            "structured_cards_csv": str(csv_path),
            "structured_cards_markdown": str(md_dir),
            "structured_rag_chunks_jsonl": str(chunks_path),
            "rag_structured_plus_fulltext_chunks": str(combined_path),
        }
    )
    index["structured_cards"] = len(cards)
    index["structured_rag_chunks"] = len(cards)
    index["updated_at"] = now_stamp()
    notes = index.setdefault("notes", [])
    note = "Structured English scRNA-scTCR evidence cards were generated to teach study design, hypotheses, methods, results, caveats, and reusable analysis/code patterns."
    if note not in notes:
        notes.append(note)
    write_json(index_path, index)
    (latest / "index.json").write_text(index_path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")


def existing_cards(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cards: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                card = json.loads(line)
            except json.JSONDecodeError:
                continue
            cards[str(card.get("doc_id") or card.get("doi") or card.get("title"))] = card
    return cards


def main() -> int:
    parser = argparse.ArgumentParser(description="Build English structured scientific cards for scRNA-scTCR RAG.")
    parser.add_argument("--kb", default="E:/scRTA/rag_kb/scrna_sctcr_focused_20260509")
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--model", default=os.environ.get("SCRTA_STRUCTURED_RAG_MODEL", "gpt-5.4"))
    parser.add_argument("--max-records", type=int, default=0, help="0 means all records.")
    parser.add_argument("--llm-max-records", type=int, default=0, help="0 means no extra cap beyond max-records.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    kb_dir = Path(args.kb)
    records = read_records(kb_dir / "papers.csv")
    records.sort(
        key=lambda r: (
            {"core": 0, "high": 1, "supporting": 2}.get(r.relevance_class, 3),
            not r.full_text_downloaded,
            -float(r.relevance_score or 0),
        )
    )
    if args.max_records:
        records = records[: args.max_records]

    out_dir = kb_dir / "structured_cards"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "scrna_sctcr_structured_cards.jsonl"
    cards_by_key = existing_cards(checkpoint_path) if args.resume else {}
    client = LLMClient(use_llm=args.use_llm, model=args.model) if args.use_llm else None

    llm_done = 0
    with checkpoint_path.open("a" if args.resume else "w", encoding="utf-8") as checkpoint:
        for idx, record in enumerate(records, start=1):
            key = str(record.doc_id or record.doi or record.title)
            if key in cards_by_key:
                print(f"[{idx}/{len(records)}] SKIP {record.title[:90]}")
                continue
            card: dict[str, Any]
            try:
                if client and client.available and (
                    args.llm_max_records == 0 or llm_done < args.llm_max_records
                ):
                    card = llm_card(record, client)
                    llm_done += 1
                else:
                    card = heuristic_card(record)
            except Exception as exc:
                card = heuristic_card(record, note=f"LLM extraction failed: {type(exc).__name__}: {exc}")
            checkpoint.write(json.dumps(card, ensure_ascii=False) + "\n")
            checkpoint.flush()
            print(f"[{idx}/{len(records)}] {card['extraction_method']} {record.title[:90]}")
            time.sleep(0.05)

    cards = list(existing_cards(checkpoint_path).values())
    # Preserve the record ordering in the final output.
    order = {str(r.doc_id or r.doi or r.title): i for i, r in enumerate(records)}
    cards.sort(key=lambda c: order.get(str(c.get("doc_id") or c.get("doi") or c.get("title")), 10**9))
    write_outputs(kb_dir, cards)
    print(json.dumps({"structured_cards": len(cards), "llm_cards_this_run": llm_done}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
