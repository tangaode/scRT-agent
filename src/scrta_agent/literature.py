from __future__ import annotations

import csv
import re
from pathlib import Path

from .schemas import LiteratureCard


TEXT_FIELDS = (
    "title",
    "disease_or_condition",
    "core_hypothesis",
    "scrna_analyses",
    "sctcr_analyses",
    "article_results",
    "conclusion_scientific_problem",
    "transferable_analysis_templates",
)


def _terms(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-zA-Z0-9_+-]{3,}", text.lower())}


def load_literature_cards(path: str | Path | None) -> list[LiteratureCard]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    cards: list[LiteratureCard] = []
    with p.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cards.append(
                LiteratureCard(
                    title=row.get("title", ""),
                    year=row.get("year", ""),
                    disease_or_condition=row.get("disease_or_condition", ""),
                    core_hypothesis=row.get("core_hypothesis", ""),
                    transferable_analysis_templates=row.get(
                        "transferable_analysis_templates", ""
                    ),
                    source_url=row.get("source_url", ""),
                    relevance_score=float(row.get("relevance_score") or 0.0),
                )
            )
    return cards


def retrieve_cards(cards: list[LiteratureCard], query: str, limit: int = 8) -> list[LiteratureCard]:
    if not cards:
        return []
    query_terms = _terms(query)
    scored = []
    for card in cards:
        blob = " ".join(str(getattr(card, field, "")) for field in card.__dataclass_fields__)
        overlap = len(query_terms & _terms(blob))
        score = overlap * 10.0 + card.relevance_score
        scored.append((score, card))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [card for score, card in scored[:limit] if score > 0]


def render_literature_context(cards: list[LiteratureCard]) -> str:
    if not cards:
        return "# Literature Context\n\nNo local literature cards were provided or matched.\n"
    blocks = ["# Literature Context", ""]
    for idx, card in enumerate(cards, start=1):
        blocks.append(f"## {idx}. {card.title}")
        blocks.append(card.to_prompt_block())
        blocks.append("")
    return "\n".join(blocks).strip()
