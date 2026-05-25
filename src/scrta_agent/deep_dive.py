from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class DeepDiveSelection:
    hypothesis_id: str
    title: str
    selected_hypothesis: str
    rationale: str
    required_tests: list[str]
    falsification_criteria: list[str]
    source_tables: list[str]
    plain_language_explanation: str = ""
    selected_candidate_source: str = "hypothesis_generator"
    selected_candidate_text: str = ""
    selection_mode: str = "candidate_selection_for_deep_dive"
    data_support_level: str = "not_assessed"
    not_selected_reasons: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["not_selected_reasons"] = self.not_selected_reasons or {}
        return data

    def to_markdown(self) -> str:
        lines = [
            "# Hypothesis Deep-Dive Selection",
            "",
            f"- Hypothesis ID: {self.hypothesis_id}",
            f"- Title: {self.title}",
            "",
            "## Selected Hypothesis",
            self.selected_hypothesis,
            "",
        ]
        if self.plain_language_explanation:
            lines.extend(["## Plain-Language Explanation", self.plain_language_explanation, ""])
        if self.selected_candidate_source:
            lines.extend(["## Selected Candidate Source", self.selected_candidate_source, ""])
        if self.selected_candidate_text:
            lines.extend(["## Original Candidate Text", self.selected_candidate_text.strip(), ""])
        lines.extend(
            [
                "## Rationale",
                self.rationale,
                "",
                "## Selection Mode",
                self.selection_mode,
                "",
                "## Initial Data Support Level",
                self.data_support_level,
                "",
                "## Required Tests",
                *[f"- {item}" for item in self.required_tests],
                "",
                "## Falsification Criteria",
                *[f"- {item}" for item in self.falsification_criteria],
                "",
                "## Source Tables",
                *[f"- {item}" for item in self.source_tables],
            ]
        )
        if self.not_selected_reasons:
            lines.extend(["", "## Deferred Candidate Reasons"])
            lines.extend([f"- {key}: {value}" for key, value in self.not_selected_reasons.items()])
        return "\n".join(lines).rstrip() + "\n"


@dataclass
class HypothesisSupportDecision:
    status: str
    rationale: str = ""
    next_action: str = ""
    rejected_reason: str = ""

    @property
    def accepted(self) -> bool:
        if self.status == "supported":
            return True
        if self.status != "partially_supported":
            return False
        next_action = self.next_action.strip().lower().replace("-", "_").replace(" ", "_")
        return next_action in {"", "continue", "proceed", "accept", "accepted"} and not self.rejected_reason.strip()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["accepted"] = self.accepted
        return data


def extract_hypothesis_candidates(content: str) -> dict[str, dict[str, Any]]:
    """Extract generator candidates from JSON when possible, markdown otherwise."""
    candidates: dict[str, dict[str, Any]] = {}
    if not content:
        return candidates

    match = re.search(
        r"HYPOTHESIS_CANDIDATES_JSON\s*(\{.*?\})\s*END_HYPOTHESIS_CANDIDATES_JSON",
        content,
        flags=re.DOTALL,
    )
    if match:
        try:
            data = json.loads(match.group(1))
            for raw in data.get("candidates", []):
                if not isinstance(raw, dict):
                    continue
                hyp_id = normalize_hypothesis_id(str(raw.get("hypothesis_id") or ""))
                if not hyp_id:
                    continue
                candidates[hyp_id] = {
                    "hypothesis_id": hyp_id,
                    "title": str(raw.get("title") or "").strip(),
                    "hypothesis_statement": str(raw.get("hypothesis_statement") or "").strip(),
                    "plain_language_explanation": str(raw.get("plain_language_explanation") or "").strip(),
                    "source_text": "",
                    "raw": raw,
                }
        except Exception:
            candidates = {}

    section_pattern = re.compile(
        r"(?ms)^##\s+(HYP[- ]?\d+|H\d+)[：:]\s*(.*?)\n(.*?)(?=^##\s+(?:HYP[- ]?\d+|H\d+)[：:]|\Z)"
    )
    for section in section_pattern.finditer(content):
        hyp_id = normalize_hypothesis_id(section.group(1))
        if not hyp_id:
            continue
        title = section.group(2).strip()
        body = section.group(3).strip()
        statement = _extract_bullet_value(body, ["Hypothesis statement"])
        explanation = _extract_bullet_value(body, ["In plain language"])
        existing = candidates.setdefault(
            hyp_id,
            {
                "hypothesis_id": hyp_id,
                "title": title,
                "hypothesis_statement": statement,
                "plain_language_explanation": explanation,
                "source_text": "",
                "raw": {},
            },
        )
        if not existing.get("title"):
            existing["title"] = title
        if not existing.get("hypothesis_statement"):
            existing["hypothesis_statement"] = statement
        if not existing.get("plain_language_explanation"):
            existing["plain_language_explanation"] = explanation
        existing["source_text"] = f"## {hyp_id}: {title}\n{body}".strip()
    return candidates


def support_decision_from_result_interpreter(content: str) -> HypothesisSupportDecision:
    """Parse the result_interpreter decision.

    New interpreter prompts require a structured JSON block. A conservative
    text heuristic is kept so older runs can still be audited and so malformed
    interpreter responses do not accidentally pass as supported.
    """
    match = re.search(
        r"HYPOTHESIS_SUPPORT_DECISION_JSON\s*(\{.*?\})\s*END_HYPOTHESIS_SUPPORT_DECISION_JSON",
        content or "",
        flags=re.DOTALL,
    )
    if match:
        try:
            data = json.loads(match.group(1))
            status = _normalize_support_status(str(data.get("status") or "inconclusive"))
            return HypothesisSupportDecision(
                status=status,
                rationale=str(data.get("rationale") or "").strip(),
                next_action=str(data.get("next_action") or "").strip(),
                rejected_reason=str(data.get("rejected_reason") or "").strip(),
            )
        except Exception:
            pass

    text = (content or "").strip()
    lower = text.lower()
    if re.search(r"\bnot[- ]supported\b|\bunsupported\b|decision:\s*not supported|status:\s*not supported", lower):
        status = "not_supported"
    elif re.search(r"\bpartially[- ]supported\b|partial support", lower):
        status = "partially_supported"
    elif re.search(r"\binconclusive\b|insufficient evidence|not enough evidence", lower):
        status = "inconclusive"
    elif re.search(r"\bsupported\b|validated", lower):
        status = "supported"
    else:
        status = "inconclusive"
    return HypothesisSupportDecision(
        status=status,
        rationale=_first_nonempty_line(text),
        next_action="regenerate_hypothesis" if status in {"not_supported", "inconclusive"} else "continue",
        rejected_reason=text[:1200] if status in {"not_supported", "inconclusive"} else "",
    )


def _normalize_support_status(value: str) -> str:
    value = value.strip().lower().replace("-", "_").replace(" ", "_")
    if value in {"support", "supported", "validated", "accepted"}:
        return "supported"
    if value in {"partial", "partially_supported", "partly_supported", "partial_support"}:
        return "partially_supported"
    if value in {"not_supported", "unsupported", "rejected", "not_validated"}:
        return "not_supported"
    return "inconclusive"


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip().strip("*")
        if line:
            return line[:500]
    return ""


def normalize_hypothesis_id(value: str) -> str:
    value = value.strip().upper().replace(" ", "-")
    match = re.search(r"(?:HYP-|H)(\d+)", value)
    if not match:
        return ""
    return f"HYP-{int(match.group(1))}"


def selection_from_agent_response(
    content: str,
    hypothesis_candidates: dict[str, dict[str, Any]] | None = None,
) -> DeepDiveSelection:
    """Parse selector JSON and preserve the selected generator candidate text.

    This parser is intentionally strict. If the LLM selector does not return a
    valid candidate ID from the LLM-generated candidate set, the workflow fails
    instead of silently using a deterministic selection fallback.
    """
    candidates = hypothesis_candidates or {}
    if not candidates:
        raise ValueError("No parseable hypothesis candidates were produced by hypothesis_generator.")
    match = re.search(
        r"FINAL_SELECTED_HYPOTHESIS_JSON\s*(\{.*?\})\s*END_FINAL_SELECTED_HYPOTHESIS_JSON",
        content or "",
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError("hypothesis_selector did not emit FINAL_SELECTED_HYPOTHESIS_JSON.")
    try:
        data = json.loads(match.group(1))
    except Exception as exc:
        raise ValueError(f"hypothesis_selector emitted invalid selection JSON: {exc}") from exc

    hyp_id = normalize_hypothesis_id(str(data.get("hypothesis_id") or ""))
    if hyp_id and hyp_id in candidates:
        return _selection_from_candidate(
            hyp_id,
            candidates[hyp_id],
            rationale=str(data.get("rationale") or "Selected by the LLM hypothesis_selector."),
            required_tests=_list_field(data, "required_tests", []),
            falsification_criteria=_list_field(data, "falsification_criteria", []),
            source_tables=_list_field(data, "source_tables", []),
            selection_mode=str(data.get("selection_mode") or "candidate_selection_for_deep_dive"),
            data_support_level=str(data.get("data_support_level") or "not_assessed"),
            not_selected_reasons=(
                data.get("not_selected_reasons") if isinstance(data.get("not_selected_reasons"), dict) else {}
            ),
        )

    raise ValueError(
        "hypothesis_selector selected an unknown hypothesis ID. "
        f"Selected={data.get('hypothesis_id')!r}; available={sorted(candidates)}"
    )


def _selection_from_candidate(
    hyp_id: str,
    candidate: dict[str, Any],
    rationale: str,
    required_tests: list[str] | None = None,
    falsification_criteria: list[str] | None = None,
    source_tables: list[str] | None = None,
    selection_mode: str = "candidate_selection_for_deep_dive",
    data_support_level: str = "not_assessed",
    not_selected_reasons: dict[str, str] | None = None,
) -> DeepDiveSelection:
    raw = candidate.get("raw") if isinstance(candidate.get("raw"), dict) else {}
    return DeepDiveSelection(
        hypothesis_id=hyp_id,
        title=str(candidate.get("title") or raw.get("title") or hyp_id),
        selected_hypothesis=str(
            candidate.get("hypothesis_statement") or raw.get("hypothesis_statement") or ""
        ).strip(),
        plain_language_explanation=str(
            candidate.get("plain_language_explanation") or raw.get("plain_language_explanation") or ""
        ).strip(),
        rationale=rationale,
        required_tests=required_tests or _candidate_text_list(raw, "key_validation", "Run the selected hypothesis deep-dive plan."),
        falsification_criteria=falsification_criteria
        or _candidate_text_list(raw, "falsification_criteria", "The required validation analyses do not support the selected hypothesis."),
        source_tables=source_tables or list(raw.get("required_output_tables") or ["rag_grounded_hypothesis_candidates.md"]),
        selected_candidate_source="hypothesis_generator",
        selected_candidate_text=str(candidate.get("source_text") or "").strip(),
        selection_mode=selection_mode,
        data_support_level=data_support_level,
        not_selected_reasons=not_selected_reasons or {},
    )


def _candidate_text_list(raw: dict[str, Any], key: str, default: str) -> list[str]:
    value = raw.get(key)
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return [default]


def _list_field(data: dict[str, Any], name: str, default: list[str]) -> list[str]:
    value = data.get(name, default)
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return default


def _extract_bullet_value(section_body: str, labels: list[str]) -> str:
    for label in labels:
        pattern = re.compile(
            rf"(?ms)^-\s*{re.escape(label)}\s*[：:]\s*(.*?)(?=^\-\s*[\w\u4e00-\u9fff /-]+\s*[：:]|\Z)"
        )
        match = pattern.search(section_body)
        if match:
            return " ".join(match.group(1).strip().split())
    return ""
