from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import truncate_text


TOKEN_RE = re.compile(r"[a-zA-Z0-9_+-]{3,}")
ACCESSION_RE = re.compile(r"\b(?:GSE|GSM|SRP|ERP|PRJNA)\d+\b", re.IGNORECASE)


KNOWN_DATASET_ALIASES = {
    "GSE235863": (
        "Contrasting cytotoxic and regulatory T cell responses underlying distinct clinical outcomes "
        "to anti-PD-1 plus lenvatinib therapy in cancer PMID 39889705 DOI 10.1016/j.ccell.2025.01.001 "
        "hepatocellular carcinoma HCC HBV-specific CD8 T cells dCODE dextramer deCODE-Dextramer-seq "
        "GZMK CD8 Teff Tem cTem circulating Tem Tpex CXCL13 KIR FOXP3 Treg responders nonresponders "
        "combination therapy anti-PD-1 monotherapy scRNA-seq scTCR clonotype"
    ),
}


@dataclass
class RagChunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    source_url: str = ""
    pmid: str = ""
    pmcid: str = ""
    doi: str = ""
    year: str = ""
    section: str = ""
    score: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RagChunk":
        return cls(
            chunk_id=str(data.get("chunk_id", "")),
            doc_id=str(data.get("doc_id", "")),
            title=str(data.get("title", "")),
            text=str(data.get("text", "")),
            source_url=str(data.get("source_url", "")),
            pmid=str(data.get("pmid", "")),
            pmcid=str(data.get("pmcid", "")),
            doi=str(data.get("doi", "")),
            year=str(data.get("year", "")),
            section=str(data.get("section", "")),
            score=float(data.get("score") or 0.0),
        )

    def citation(self) -> str:
        bits = []
        if self.year:
            bits.append(self.year)
        if self.pmid:
            bits.append(f"PMID:{self.pmid}")
        if self.pmcid:
            bits.append(self.pmcid)
        if self.doi:
            bits.append(f"DOI:{self.doi}")
        return "; ".join(bits)


def token_set(text: str) -> set[str]:
    return {x.lower() for x in TOKEN_RE.findall(text or "")}


def expand_rag_query(query: str) -> str:
    """Expand query text with known dataset aliases and paper-specific terms.

    This keeps a dataset accession such as GSE235863 from being treated as an
    isolated token when the corresponding RAG chunks are indexed by paper title,
    PMID, DOI, or biological terms rather than by GEO accession.
    """
    if not query:
        return query
    additions = []
    for accession in sorted({m.group(0).upper() for m in ACCESSION_RE.finditer(query)}):
        alias = KNOWN_DATASET_ALIASES.get(accession)
        if alias:
            additions.append(f"{accession}: {alias}")
    if not additions:
        return query
    return query.rstrip() + "\n\n# Dataset-specific RAG aliases\n" + "\n".join(additions)


def load_rag_chunks(path: str | Path | None) -> list[RagChunk]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    chunks: list[RagChunk] = []
    with p.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(RagChunk.from_dict(json.loads(line)))
            except json.JSONDecodeError:
                continue
    return chunks


def retrieve_rag_chunks(chunks: list[RagChunk], query: str, limit: int = 10) -> list[RagChunk]:
    if not chunks:
        return []
    query = expand_rag_query(query)
    q_terms = token_set(query)
    if not q_terms:
        return []
    scored: list[RagChunk] = []
    for chunk in chunks:
        title_terms = token_set(chunk.title)
        text_terms = token_set(chunk.text)
        overlap = len(q_terms & text_terms)
        title_overlap = len(q_terms & title_terms)
        domain_bonus = 0
        lower = f"{chunk.title} {chunk.text}".lower()
        for term in (
            "single-cell",
            "single cell",
            "scrna",
            "tcr",
            "clonotype",
            "clonal",
            "t cell receptor",
            "paired",
        ):
            if term in lower:
                domain_bonus += 1
        score = overlap + title_overlap * 3 + domain_bonus * 0.5
        if chunk.section == "structured_scientific_card":
            score += 3.0
        if chunk.section in {
            "paper_card",
            "mechanism_atlas",
            "gene_program",
            "analysis_template",
            "hypothesis_archetype",
            "guardrail",
        }:
            score += 4.0
        if chunk.section.startswith("evidence_unit"):
            score += 2.5
        query_lower = query.lower()
        if chunk.section == "hypothesis_archetype" and any(
            term in query_lower for term in ("hypothesis", "candidate", "novel")
        ):
            score += 8.0
        if chunk.section == "analysis_template" and any(
            term in query_lower for term in ("analysis", "downstream", "template", "test")
        ):
            score += 8.0
        if chunk.section in {"mechanism_atlas", "gene_program"} and any(
            term in query_lower for term in ("mechanism", "pathway", "program", "molecular")
        ):
            score += 8.0
        if chunk.section == "guardrail" and any(
            term in query_lower for term in ("specificity", "claim", "conservative", "overclaim", "migration")
        ):
            score += 6.0
        if score > 0:
            scored_chunk = RagChunk(**{**chunk.__dict__, "score": float(score)})
            scored.append(scored_chunk)
    scored.sort(key=lambda x: x.score, reverse=True)

    # Keep document diversity so one long paper does not crowd out everything.
    selected: list[RagChunk] = []
    seen_docs: dict[str, int] = {}
    for chunk in scored:
        if seen_docs.get(chunk.doc_id, 0) >= 2:
            continue
        selected.append(chunk)
        seen_docs[chunk.doc_id] = seen_docs.get(chunk.doc_id, 0) + 1
        if len(selected) >= limit:
            break
    return selected


def render_rag_context(chunks: list[RagChunk], max_chars_per_chunk: int = 1600) -> str:
    if not chunks:
        return "# RAG Evidence\n\nNo RAG chunks were retrieved.\n"
    lines = ["# RAG Evidence", ""]
    for idx, chunk in enumerate(chunks, start=1):
        lines.append(f"## {idx}. {chunk.title}")
        cite = chunk.citation()
        if cite:
            lines.append(f"Source: {cite}")
        if chunk.source_url:
            lines.append(f"URL: {chunk.source_url}")
        if chunk.section:
            lines.append(f"Section: {chunk.section}")
        lines.append(f"Retrieval score: {chunk.score:.2f}")
        lines.append("")
        lines.append(truncate_text(chunk.text.strip(), max_chars_per_chunk))
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def default_rag_index_candidates() -> list[Path]:
    return [
        Path("E:/scRTA/rag_kb/scrna_sctcr_rag_v2/rag_chunks.jsonl"),
        Path("E:/scRTA/rag_kb/scrna_sctcr_rag_current/rag_chunks.jsonl"),
        Path("E:/scRTA/rag_kb/scrna_sctcr_open_access_20260509/rag_chunks.jsonl"),
        Path("G:/scRNA_scTCR/scRTA/knowledge_base/scrna_sctcr_literature_20260508_v2/scrna_sctcr_literature_cards.jsonl"),
    ]


def resolve_rag_index(path: str | None) -> Path | None:
    if path:
        p = Path(path)
        return p if p.exists() else None
    for candidate in default_rag_index_candidates():
        if candidate.exists():
            return candidate
    return None
