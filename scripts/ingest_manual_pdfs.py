from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from dataclasses import fields
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from build_scrna_sctcr_rag import (
    PaperRecord,
    build_chunks,
    now_stamp,
    slugify,
    write_json,
    write_manual_download_list,
    write_metadata_audit,
    write_records,
)
from fetch_legal_open_fulltext_pdfs import extract_pdf_text


DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)


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


def normalize(value: str) -> str:
    value = (value or "").lower()
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def title_similarity(a: str, b: str) -> float:
    na = normalize(a)
    nb = normalize(b)
    if not na or not nb:
        return 0.0
    if na in nb or nb in na:
        return 0.92
    return SequenceMatcher(None, na, nb).ratio()


def clean_doi(value: str) -> str:
    value = value.strip().rstrip(".,);]")
    value = re.sub(r"</?[^>]+>", "", value)
    return value.lower()


def extract_dois(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for match in DOI_RE.findall(text or ""):
        doi = clean_doi(match)
        if doi and doi not in seen:
            seen.add(doi)
            out.append(doi)
    return out


def pdf_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def collect_pdfs(paths: list[Path]) -> list[Path]:
    pdfs: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix.lower() == ".pdf":
            pdfs.append(path)
        elif path.is_dir():
            pdfs.extend(
                p
                for p in path.rglob("*.pdf")
                if "__MACOSX" not in str(p) and not p.name.startswith("._")
            )
    unique: dict[str, Path] = {}
    for pdf in pdfs:
        if pdf.stat().st_size < 2048:
            continue
        try:
            digest = pdf_hash(pdf)
        except Exception:
            continue
        unique.setdefault(digest, pdf)
    return sorted(unique.values(), key=lambda p: p.name.lower())


def first_page_hint(text: str) -> str:
    lines = []
    for raw in (text or "").splitlines()[:80]:
        line = re.sub(r"\s+", " ", raw).strip()
        if 15 <= len(line) <= 220:
            lines.append(line)
    return " ".join(lines[:10])


def match_record(pdf: Path, text: str, records: list[PaperRecord]) -> tuple[PaperRecord | None, float, str]:
    dois = extract_dois(text[:25000])
    doi_map = {clean_doi(rec.doi): rec for rec in records if rec.doi}
    for doi in dois:
        if doi in doi_map:
            return doi_map[doi], 1.0, f"doi:{doi}"

    haystacks = [
        pdf.stem,
        first_page_hint(text),
        text[:5000],
    ]
    best: tuple[PaperRecord | None, float, str] = (None, 0.0, "")
    for rec in records:
        for label, haystack in zip(("filename", "first_page", "body_start"), haystacks, strict=False):
            score = title_similarity(rec.title, haystack)
            if score > best[1]:
                best = (rec, score, label)
    if best[1] >= 0.68:
        return best
    return None, best[1], best[2]


def refresh_index(kb_dir: Path, records: list[PaperRecord], manual_ingested: int) -> dict[str, Any]:
    chunks_dir = kb_dir / "rag_chunks"
    if chunks_dir.exists() and str(chunks_dir.resolve()).lower().startswith(str(kb_dir.resolve()).lower()):
        for old in chunks_dir.glob("*.md"):
            old.unlink()
    chunk_count = build_chunks(records, kb_dir)
    write_records(records, kb_dir)
    manual_count = write_manual_download_list(records, kb_dir)
    audit_count = write_metadata_audit(records, kb_dir)

    index_path = kb_dir / "index.json"
    index: dict[str, Any] = {}
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8", errors="replace"))
    index.update(
        {
            "schema_version": "scrta_agent_rag.v2",
            "updated_at": now_stamp(),
            "records": len(records),
            "full_text_downloaded": sum(1 for r in records if r.full_text_downloaded),
            "manual_pdf_ingested_this_run": manual_ingested,
            "rag_chunks": chunk_count,
            "manual_download_needed": manual_count,
            "metadata_audit_records": audit_count,
        }
    )
    paths = index.setdefault("paths", {})
    paths.update(
        {
            "manual_pdf_text": str(kb_dir / "manual_ingest" / "text"),
            "manual_pdf_archive": str(kb_dir / "manual_ingest" / "pdf"),
            "manual_pdf_manifest": str(kb_dir / "manual_ingest" / "manual_pdf_manifest.csv"),
        }
    )
    notes = index.setdefault("notes", [])
    note = "Local PDFs were extracted and matched to records where DOI/title evidence was sufficient."
    if note not in notes:
        notes.append(note)
    write_json(index_path, index)

    latest = Path("E:/scRTA/rag_kb/scrna_sctcr_rag_current")
    latest.mkdir(parents=True, exist_ok=True)
    for name in ("index.json", "papers.csv", "papers.jsonl", "rag_chunks.jsonl"):
        src = kb_dir / name
        if src.exists():
            (latest / name).write_text(src.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest local legal PDFs into the scRTA RAG KB.")
    parser.add_argument("--kb", default="E:/scRTA/rag_kb/scrna_sctcr_focused_20260509")
    parser.add_argument(
        "--pdf-dir",
        action="append",
        default=[],
        help="Directory or PDF file. Can be specified multiple times.",
    )
    args = parser.parse_args()

    kb_dir = Path(args.kb)
    records = read_records(kb_dir / "papers.csv")
    sources = [Path(p) for p in args.pdf_dir] or [kb_dir / "manual_ingest" / "raw"]
    pdfs = collect_pdfs(sources)

    archive_dir = kb_dir / "manual_ingest" / "pdf"
    text_dir = kb_dir / "manual_ingest" / "text"
    archive_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = kb_dir / "manual_ingest" / "manual_pdf_manifest.csv"
    ingested = 0
    with manifest_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "pdf_path",
                "matched",
                "match_score",
                "match_reason",
                "doc_id",
                "title",
                "doi",
                "text_path",
                "note",
            ],
        )
        writer.writeheader()
        for idx, pdf in enumerate(pdfs, start=1):
            text = extract_pdf_text(pdf)
            note = ""
            rec: PaperRecord | None = None
            score = 0.0
            reason = ""
            text_path = ""
            if len(text) < 500:
                note = "text_extraction_failed_or_too_short"
            else:
                rec, score, reason = match_record(pdf, text, records)
                if rec:
                    base = f"{slugify(rec.doc_id or rec.doi or rec.title, 60)}_{slugify(rec.title, 70)}"
                    target_pdf = archive_dir / f"{base}.pdf"
                    target_text = text_dir / f"{base}.txt"
                    if not target_pdf.exists():
                        shutil.copy2(pdf, target_pdf)
                    target_text.write_text(text, encoding="utf-8")
                    rec.full_text_downloaded = True
                    rec.text_path = str(target_text)
                    rec.download_status = "user_pdf_ingested"
                    rec.manual_download_priority = ""
                    text_path = str(target_text)
                    ingested += 1
                else:
                    note = "no_confident_record_match"
                    target_text = text_dir / f"unmatched_{idx:03d}_{slugify(pdf.stem, 80)}.txt"
                    target_text.write_text(text, encoding="utf-8")
                    text_path = str(target_text)
            writer.writerow(
                {
                    "pdf_path": str(pdf),
                    "matched": bool(rec),
                    "match_score": f"{score:.3f}",
                    "match_reason": reason,
                    "doc_id": rec.doc_id if rec else "",
                    "title": rec.title if rec else "",
                    "doi": rec.doi if rec else "",
                    "text_path": text_path,
                    "note": note,
                }
            )
            print(f"[{idx}/{len(pdfs)}] {'MATCH' if rec else 'MISS'} {pdf.name}")

    index = refresh_index(kb_dir, records, ingested)
    print(json.dumps(index, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
