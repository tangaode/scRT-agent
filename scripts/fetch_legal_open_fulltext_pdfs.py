from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, fields
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


NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
USER_AGENT = "scrta-agent-open-fulltext-fetcher/0.1"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


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


def request_json(url: str, timeout: int = 45) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with NO_PROXY_OPENER.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def request_bytes(url: str, timeout: int = 75) -> tuple[bytes, str, str]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*"})
    with NO_PROXY_OPENER.open(req, timeout=timeout) as resp:
        data = resp.read()
        content_type = resp.headers.get("content-type", "")
        final_url = resp.geturl()
    return data, content_type, final_url


def parse_full_text_urls(value: str) -> list[str]:
    urls: list[str] = []
    for part in (value or "").split(";"):
        part = part.strip()
        if not part:
            continue
        match = re.search(r"https?://.+$", part)
        if match:
            urls.append(match.group(0).strip())
    return urls


def is_probable_pdf_url(url: str) -> bool:
    lower = url.lower()
    return (
        ".pdf" in lower
        or "type=printable" in lower
        or "/article/file" in lower
        or "pdf=render" in lower
        or lower.endswith("/pdf")
    )


def unpaywall_urls(doi: str, email: str) -> list[str]:
    if not doi or not email:
        return []
    encoded = urllib.parse.quote(doi.strip(), safe="")
    url = f"https://api.unpaywall.org/v2/{encoded}?email={urllib.parse.quote(email)}"
    try:
        data = request_json(url)
    except Exception:
        return []
    urls: list[str] = []
    locations = []
    best = data.get("best_oa_location")
    if isinstance(best, dict):
        locations.append(best)
    locations.extend(x for x in data.get("oa_locations") or [] if isinstance(x, dict))
    for loc in locations:
        for key in ("url_for_pdf", "url"):
            value = str(loc.get(key) or "").strip()
            if value:
                urls.append(value)
    return urls


def semantic_scholar_urls(doi: str) -> list[str]:
    if not doi:
        return []
    paper_id = "DOI:" + doi.strip()
    url = "https://api.semanticscholar.org/graph/v1/paper/" + urllib.parse.quote(
        paper_id, safe=":"
    ) + "?fields=openAccessPdf,title"
    try:
        data = request_json(url)
    except Exception:
        return []
    pdf = data.get("openAccessPdf") or {}
    value = str(pdf.get("url") or "").strip() if isinstance(pdf, dict) else ""
    return [value] if value else []


def openalex_urls(doi: str) -> list[str]:
    if not doi:
        return []
    url = "https://api.openalex.org/works/https://doi.org/" + urllib.parse.quote(doi.strip(), safe="/.")
    try:
        data = request_json(url)
    except Exception:
        return []
    urls: list[str] = []
    open_access = data.get("open_access") or {}
    if isinstance(open_access, dict) and open_access.get("oa_url"):
        urls.append(str(open_access["oa_url"]))
    for location_key in ("primary_location", "best_oa_location"):
        loc = data.get(location_key) or {}
        if isinstance(loc, dict):
            for key in ("pdf_url", "landing_page_url"):
                value = str(loc.get(key) or "").strip()
                if value:
                    urls.append(value)
    return urls


def biorxiv_pdf_urls(doi: str) -> list[str]:
    if not doi or not doi.startswith("10.1101/"):
        return []
    suffix = doi.split("/", 1)[1]
    # These are public preprint endpoints. The exact Europe PMC URL is used
    # first when available; these are fallback forms for older records.
    return [
        f"https://www.biorxiv.org/content/10.1101/{suffix}.full.pdf",
        f"https://www.medrxiv.org/content/10.1101/{suffix}.full.pdf",
    ]


def unique_urls(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for url in urls:
        url = url.strip()
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(url)
    return out


def candidate_urls(record: PaperRecord, email: str) -> list[str]:
    urls: list[str] = []
    urls.extend(parse_full_text_urls(record.full_text_urls))
    urls.extend(unpaywall_urls(record.doi, email))
    urls.extend(semantic_scholar_urls(record.doi))
    urls.extend(openalex_urls(record.doi))
    urls.extend(biorxiv_pdf_urls(record.doi))
    urls.append(record.source_url)
    pdf_first = sorted(unique_urls(urls), key=lambda u: 0 if is_probable_pdf_url(u) else 1)
    return pdf_first


def pdf_text_with_fitz(path: Path) -> str:
    try:
        import fitz  # type: ignore
    except Exception:
        return ""
    try:
        doc = fitz.open(str(path))
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return "\n\n".join(x.strip() for x in pages if x.strip())
    except Exception:
        return ""


def pdf_text_with_pypdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        return ""
    try:
        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "").strip() for page in reader.pages)
    except Exception:
        return ""


def extract_pdf_text(path: Path) -> str:
    text = pdf_text_with_fitz(path)
    if len(text) >= 500:
        return text
    text = pdf_text_with_pypdf(path)
    return text if len(text) >= 500 else ""


def safe_pdf_basename(record: PaperRecord) -> str:
    key = record.pmcid or record.pmid or record.doi or hashlib.sha1(record.title.encode()).hexdigest()[:12]
    return f"{slugify(key, 40)}_{slugify(record.title, 70)}"


def try_download_pdf(record: PaperRecord, out_dir: Path, email: str) -> tuple[bool, str]:
    pdf_dir = out_dir / "articles_pdf"
    text_dir = out_dir / "articles_pdf_text"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    basename = safe_pdf_basename(record)
    pdf_path = pdf_dir / f"{basename}.pdf"
    text_path = text_dir / f"{basename}.txt"
    errors: list[str] = []

    for url in candidate_urls(record, email):
        if not url:
            continue
        try:
            data, content_type, final_url = request_bytes(url)
        except Exception as exc:
            errors.append(f"{url} -> {type(exc).__name__}")
            time.sleep(0.3)
            continue
        if not (data.startswith(b"%PDF") or "pdf" in content_type.lower() or final_url.lower().endswith(".pdf")):
            errors.append(f"{url} -> non_pdf:{content_type}")
            time.sleep(0.2)
            continue
        pdf_path.write_bytes(data)
        text = extract_pdf_text(pdf_path)
        if not text:
            errors.append(f"{url} -> pdf_text_extract_failed")
            continue
        text_path.write_text(text, encoding="utf-8")
        record.full_text_downloaded = True
        record.text_path = str(text_path)
        record.download_status = "downloaded_open_pdf"
        record.manual_download_priority = ""
        record.full_text_urls = add_url_note(record.full_text_urls, f"downloaded_pdf:{final_url}")
        return True, final_url
    return False, "; ".join(errors[-5:])


def add_url_note(existing: str, value: str) -> str:
    values = [x for x in (existing or "").split(";") if x]
    if value and value not in values:
        values.append(value)
    return ";".join(values)


def refresh_index(out_dir: Path, records: list[PaperRecord], downloaded_now: int, attempted: int) -> dict[str, Any]:
    chunks_dir = out_dir / "rag_chunks"
    if chunks_dir.exists() and str(chunks_dir.resolve()).lower().startswith(str(out_dir.resolve()).lower()):
        for old in chunks_dir.glob("*.md"):
            old.unlink()
    chunk_count = build_chunks(records, out_dir)
    write_records(records, out_dir)
    manual_count = write_manual_download_list(records, out_dir)
    audit_count = write_metadata_audit(records, out_dir)

    index_path = out_dir / "index.json"
    index: dict[str, Any] = {}
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8", errors="replace"))
    paths = index.setdefault("paths", {})
    paths.update(
        {
            "papers_csv": str(out_dir / "papers.csv"),
            "papers_jsonl": str(out_dir / "papers.jsonl"),
            "rag_chunks_jsonl": str(out_dir / "rag_chunks.jsonl"),
            "rag_chunks_dir": str(out_dir / "rag_chunks"),
            "articles_pdf": str(out_dir / "articles_pdf"),
            "articles_pdf_text": str(out_dir / "articles_pdf_text"),
            "manual_download_needed_csv": str(out_dir / "manual_download_needed.csv"),
            "metadata_audit_csv": str(out_dir / "metadata_audit.csv"),
        }
    )
    index.update(
        {
            "schema_version": "scrta_agent_rag.v2",
            "updated_at": now_stamp(),
            "records": len(records),
            "full_text_downloaded": sum(1 for r in records if r.full_text_downloaded),
            "open_pdf_downloaded_this_run": downloaded_now,
            "open_pdf_attempted_this_run": attempted,
            "rag_chunks": chunk_count,
            "manual_download_needed": manual_count,
            "metadata_audit_records": audit_count,
        }
    )
    notes = index.setdefault("notes", [])
    note = "Additional public PDFs were fetched only from legal OA/public endpoints: Europe PMC URLs, Unpaywall, Semantic Scholar, OpenAlex, bioRxiv/medRxiv, and publisher OA PDF links."
    if note not in notes:
        notes.append(note)
    write_json(index_path, index)

    latest = Path("E:/scRTA/rag_kb/scrna_sctcr_rag_current")
    latest.mkdir(parents=True, exist_ok=True)
    for name in ("index.json", "papers.csv", "papers.jsonl", "rag_chunks.jsonl"):
        src = out_dir / name
        if src.exists():
            (latest / name).write_text(src.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch legal open PDFs and rebuild scRTA RAG chunks.")
    parser.add_argument("--kb", default="E:/scRTA/rag_kb/scrna_sctcr_focused_20260509")
    parser.add_argument("--email", default=os.environ.get("UNPAYWALL_EMAIL", "scrta@example.com"))
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit.")
    args = parser.parse_args()

    kb_dir = Path(args.kb)
    records_path = kb_dir / "papers.csv"
    records = read_records(records_path)
    targets = [rec for rec in records if not rec.full_text_downloaded]
    if args.limit:
        targets = targets[: args.limit]

    log_path = kb_dir / "legal_pdf_fetch_log.csv"
    downloaded = 0
    attempted = 0
    with log_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["title", "doi", "pmid", "pmcid", "success", "source_or_error"],
        )
        writer.writeheader()
        for rec in targets:
            attempted += 1
            ok, detail = try_download_pdf(rec, kb_dir, args.email)
            if ok:
                downloaded += 1
            writer.writerow(
                {
                    "title": rec.title,
                    "doi": rec.doi,
                    "pmid": rec.pmid,
                    "pmcid": rec.pmcid,
                    "success": ok,
                    "source_or_error": detail,
                }
            )
            print(f"[{attempted}/{len(targets)}] {'OK' if ok else 'MISS'} {rec.title[:100]}")
            time.sleep(0.3)

    index = refresh_index(kb_dir, records, downloaded, attempted)
    print(json.dumps(index, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
