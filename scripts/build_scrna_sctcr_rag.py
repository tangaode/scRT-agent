from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


EUROPE_PMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EUROPE_PMC_FULLTEXT = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
NCBI_BIOC_JSON = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"
NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# The queries are intentionally narrow around joint transcriptome + TCR analysis.
# Broader immunology papers can still enter through seed cards, but Europe PMC
# discovery is biased toward papers that can teach the agent analysis practice.
QUERY_SET = [
    {
        "label": "exact_scrna_sctcr",
        "query": '("scRNA-seq" OR "single-cell RNA sequencing" OR "single cell RNA sequencing") AND ("scTCR-seq" OR "single-cell TCR" OR "single cell TCR" OR "TCR-seq")',
    },
    {
        "label": "paired_transcriptome_tcr",
        "query": '("paired single-cell" OR "paired single cell" OR "paired transcriptome") AND ("T cell receptor" OR TCR OR clonotype OR "V(D)J")',
    },
    {
        "label": "immune_profiling_vdj",
        "query": '("single-cell immune profiling" OR "single cell immune profiling" OR "5\' immune profiling") AND (TCR OR "V(D)J" OR clonotype)',
    },
    {
        "label": "clonotype_cell_state",
        "query": '("single-cell" OR "single cell") AND (transcriptome OR transcriptomic OR RNA) AND (clonotype OR clonotypes OR "clonal expansion") AND ("T cell" OR "T cells")',
    },
    {
        "label": "tcr_repertoire_transcriptome",
        "query": '("TCR repertoire" OR "T cell receptor repertoire") AND ("single-cell" OR "single cell") AND (transcriptome OR transcriptomic OR RNA)',
    },
    {
        "label": "tumor_til_joint",
        "query": '("tumor infiltrating" OR TIL OR cancer OR tumour OR tumor) AND ("single-cell RNA" OR scRNA-seq) AND (TCR OR clonotype OR "T cell receptor")',
    },
    {
        "label": "immunotherapy_joint",
        "query": '("immune checkpoint" OR immunotherapy OR "PD-1" OR "PD-L1" OR "CTLA-4") AND ("single-cell RNA" OR scRNA-seq) AND (TCR OR clonotype)',
    },
    {
        "label": "tissue_clone_sharing",
        "query": '("single-cell RNA" OR scRNA-seq) AND (TCR OR clonotype) AND ("clonal sharing" OR "shared clonotypes" OR tissue OR migration)',
    },
]

WEIGHTED_TERMS = (
    ("scrna-seq and sctcr-seq", 12),
    ("scrna-seq and single-cell tcr", 12),
    ("single-cell rna sequencing and tcr", 11),
    ("single cell rna sequencing and tcr", 11),
    ("single-cell tcr sequencing", 10),
    ("single cell tcr sequencing", 10),
    ("sctcr-seq", 10),
    ("paired single-cell", 9),
    ("paired single cell", 9),
    ("single-cell immune profiling", 8),
    ("single cell immune profiling", 8),
    ("v(d)j", 7),
    ("tcr-seq", 7),
    ("t cell receptor", 6),
    ("tcr repertoire", 6),
    ("clonotype", 5),
    ("clonotypes", 5),
    ("clonotypic", 5),
    ("clonal expansion", 5),
    ("shared clonotypes", 5),
    ("single-cell rna", 5),
    ("single cell rna", 5),
    ("scrna-seq", 5),
    ("transcriptomic state", 4),
    ("tumor infiltrating", 3),
    ("exhaust", 3),
    ("cytotoxic", 3),
    ("tissue resident", 3),
    ("immunotherapy", 3),
)

TOPIC_TAG_TERMS = {
    "paired_scRNA_scTCR": (
        "scrna-seq and sctcr-seq",
        "sctcr-seq",
        "single-cell tcr sequencing",
        "single cell tcr sequencing",
        "paired single-cell",
        "paired single cell",
        "v(d)j",
        "single-cell immune profiling",
    ),
    "clone_expansion": ("clonal expansion", "expanded clonotype", "expanded clonotypes", "clonally expanded"),
    "clone_sharing_tissue": ("shared clonotype", "shared clonotypes", "clonal sharing", "migration", "tissue"),
    "tumor_immunology": ("tumor", "tumour", "cancer", "til", "tumor infiltrating"),
    "checkpoint_immunotherapy": ("immunotherapy", "checkpoint", "pd-1", "pd-l1", "ctla-4", "anti-pd"),
    "exhaustion_cytotoxicity": ("exhaust", "pdcd1", "lag3", "havcr2", "cytotoxic", "gzmb", "prf1"),
    "tissue_residency": ("tissue resident", "trm", "cxcr6", "znf683", "cd69"),
    "repertoire_metrics": ("tcr repertoire", "cdr3", "trav", "trbv", "diversity", "vdj"),
    "method_qc": ("doublet", "quality control", "benchmark", "pipeline", "method"),
}


@dataclass
class PaperRecord:
    doc_id: str
    title: str
    year: str = ""
    journal: str = ""
    pmid: str = ""
    pmcid: str = ""
    doi: str = ""
    abstract: str = ""
    source_url: str = ""
    open_access: bool = False
    full_text_downloaded: bool = False
    xml_path: str = ""
    text_path: str = ""
    full_text_urls: str = ""
    relevance_score: float = 0.0
    relevance_class: str = ""
    topic_tags: str = ""
    hit_queries: str = ""
    source_kind: str = ""
    download_status: str = ""
    manual_download_priority: str = ""
    seed_title: str = ""
    metadata_note: str = ""


def now_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def slugify(value: str, limit: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return (slug[:limit].strip("_") or "untitled")


def norm_pmcid(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text if text.upper().startswith("PMC") else f"PMC{text}"


def request_json(url: str, params: dict[str, str] | None = None, timeout: int = 60) -> Any:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "scrta-agent-rag/0.1"})
    last_exc: Exception | None = None
    for _ in range(3):
        try:
            with NO_PROXY_OPENER.open(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8", errors="replace"))
        except Exception as exc:
            last_exc = exc
            time.sleep(1.0)
    raise last_exc or RuntimeError("request_json failed")


def request_text(url: str, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "scrta-agent-rag/0.1"})
    last_exc: Exception | None = None
    for _ in range(3):
        try:
            with NO_PROXY_OPENER.open(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            last_exc = exc
            time.sleep(1.0)
    raise last_exc or RuntimeError("request_text failed")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_seed_cards(path: Path | None) -> list[dict[str, Any]]:
    if not path or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(dict(row))
    return rows


def normalize_title(value: str) -> str:
    value = html.unescape(value or "")
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9]+", " ", value.lower())).strip()


def title_similarity(a: str, b: str) -> float:
    na = normalize_title(a)
    nb = normalize_title(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def add_unique_csv(existing: str, value: str) -> str:
    values = [x.strip() for x in (existing or "").split(";") if x.strip()]
    if value and value not in values:
        values.append(value)
    return ";".join(values)


def escape_query_text(value: str) -> str:
    return value.replace('"', " ").strip()


def key_for_record(row: dict[str, Any]) -> str:
    for key in ("pmcid", "pmid", "doi"):
        value = str(row.get(key) or row.get(key.upper()) or "").strip()
        if value:
            if key == "pmcid":
                value = norm_pmcid(value)
            return f"{key}:{value.lower()}"
    return "title:" + hashlib.sha1(str(row.get("title", "")).lower().encode()).hexdigest()


def key_for_paper(record: PaperRecord) -> str:
    if record.pmid:
        return f"pmid:{record.pmid.lower()}"
    if record.pmcid:
        return f"pmcid:{norm_pmcid(record.pmcid).lower()}"
    if record.doi:
        return f"doi:{record.doi.lower()}"
    return "title:" + hashlib.sha1(record.title.lower().encode()).hexdigest()


def record_from_seed(row: dict[str, Any]) -> PaperRecord:
    pmid = str(row.get("pmid", "")).strip()
    pmcid = norm_pmcid(row.get("pmcid", ""))
    doi = str(row.get("doi", "")).strip()
    title = str(row.get("title", "")).strip()
    source_url = str(row.get("source_url", "")).strip()
    if not source_url and pmid:
        source_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    doc_id = pmcid or pmid or doi or hashlib.sha1(title.lower().encode()).hexdigest()[:12]
    abstract = "\n".join(
        str(row.get(k, "")).strip()
        for k in (
            "core_hypothesis",
            "scrna_analyses",
            "sctcr_analyses",
            "article_results",
            "conclusion_scientific_problem",
            "transferable_analysis_templates",
        )
        if str(row.get(k, "")).strip()
    )
    return PaperRecord(
        doc_id=doc_id,
        title=title,
        year=str(row.get("year", "")).strip(),
        journal=str(row.get("journal", "")).strip(),
        pmid=pmid,
        pmcid=pmcid,
        doi=doi,
        abstract=abstract,
        source_url=source_url,
        open_access=bool(pmcid),
        relevance_score=float(row.get("relevance_score") or 0.0),
        source_kind="seed_card",
        seed_title=title,
    )


def europe_pmc_search(query: str, page_size: int = 100, cursor: str = "*") -> dict[str, Any]:
    return request_json(
        EUROPE_PMC_SEARCH,
        {
            "query": query,
            "format": "json",
            "pageSize": str(page_size),
            "cursorMark": cursor,
            "resultType": "core",
            "synonym": "true",
        },
    )


def discover_europe_pmc(max_results_per_query: int = 250) -> list[dict[str, Any]]:
    all_results: list[dict[str, Any]] = []
    seen = set()
    for query_spec in QUERY_SET:
        query = query_spec["query"]
        label = query_spec["label"]
        cursor = "*"
        fetched = 0
        while fetched < max_results_per_query:
            data = europe_pmc_search(query, page_size=100, cursor=cursor)
            results = data.get("resultList", {}).get("result", [])
            if not results:
                break
            for row in results:
                key = key_for_record(row)
                if key not in seen:
                    row["_hit_queries"] = label
                    all_results.append(row)
                    seen.add(key)
                else:
                    for existing in all_results:
                        if key_for_record(existing) == key:
                            existing["_hit_queries"] = add_unique_csv(existing.get("_hit_queries", ""), label)
                            break
            fetched += len(results)
            next_cursor = data.get("nextCursorMark")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor
            time.sleep(0.2)
    return all_results


def lookup_europe_pmc_row(seed: PaperRecord) -> dict[str, Any] | None:
    queries: list[str] = []
    if seed.pmcid:
        queries.append(f"PMCID:{norm_pmcid(seed.pmcid)}")
    if seed.pmid:
        queries.append(f"EXT_ID:{seed.pmid} AND SRC:MED")
    if seed.doi:
        queries.append(f'DOI:"{escape_query_text(seed.doi)}"')
    if seed.title:
        queries.append(f'TITLE:"{escape_query_text(seed.title)}"')
    for query in queries:
        try:
            data = europe_pmc_search(query, page_size=3)
        except Exception:
            continue
        results = data.get("resultList", {}).get("result", [])
        if not results:
            continue
        if query.startswith("TITLE:"):
            ranked = sorted(results, key=lambda x: title_similarity(seed.title, str(x.get("title", ""))), reverse=True)
            best = ranked[0]
            if title_similarity(seed.title, str(best.get("title", ""))) < 0.72:
                continue
            best["_hit_queries"] = "seed_title_lookup"
            return best
        results[0]["_hit_queries"] = "seed_identifier_lookup"
        return results[0]
    return None


def extract_full_text_urls(row: dict[str, Any]) -> str:
    urls: list[str] = []
    url_list = row.get("fullTextUrlList") or {}
    if isinstance(url_list, dict):
        entries = url_list.get("fullTextUrl") or []
        if isinstance(entries, dict):
            entries = [entries]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            url = str(entry.get("url", "")).strip()
            availability = str(entry.get("availability", "")).strip()
            style = str(entry.get("documentStyle", "")).strip()
            if url:
                label = ",".join(x for x in (availability, style) if x)
                urls.append(f"{label}:{url}" if label else url)
    return ";".join(urls)


def record_from_europe_pmc(row: dict[str, Any]) -> PaperRecord:
    pmid = str(row.get("pmid", "")).strip()
    pmcid = norm_pmcid(row.get("pmcid", ""))
    doi = str(row.get("doi", "")).strip()
    title = str(row.get("title", "")).strip()
    source_url = f"https://europepmc.org/article/MED/{pmid}" if pmid else ""
    if pmcid:
        source_url = f"https://europepmc.org/article/PMC/{pmcid.replace('PMC', '')}"
    elif doi:
        source_url = f"https://doi.org/{doi}"
    doc_id = pmcid or pmid or doi or hashlib.sha1(title.lower().encode()).hexdigest()[:12]
    full_text_urls = extract_full_text_urls(row)
    return PaperRecord(
        doc_id=doc_id,
        title=title,
        year=str(row.get("pubYear", "")).strip(),
        journal=str(row.get("journalTitle", "")).strip(),
        pmid=pmid,
        pmcid=pmcid,
        doi=doi,
        abstract=str(row.get("abstractText", "")).strip(),
        source_url=source_url,
        open_access=str(row.get("isOpenAccess", "")).upper() == "Y"
        or bool(pmcid)
        or "Open access" in full_text_urls
        or "Free" in full_text_urls,
        full_text_urls=full_text_urls,
        relevance_score=score_relevance(title + "\n" + str(row.get("abstractText", ""))),
        hit_queries=str(row.get("_hit_queries", "")).strip(),
        source_kind="europe_pmc_discovery",
    )


def score_relevance(text: str) -> float:
    lower = text.lower()
    score = 0.0
    for term, weight in WEIGHTED_TERMS:
        if term in lower:
            score += weight
    if ("single-cell" in lower or "single cell" in lower or "scrna" in lower) and (
        "tcr" in lower or "t cell receptor" in lower or "clonotype" in lower
    ):
        score += 8
    if ("transcriptome" in lower or "transcriptomic" in lower or "rna" in lower) and (
        "repertoire" in lower or "clonal" in lower
    ):
        score += 4
    if "b cell receptor" in lower and "t cell receptor" not in lower and " tcr" not in lower:
        score -= 8
    if "single-cell atac" in lower and "rna" not in lower and "transcript" not in lower:
        score -= 5
    return max(score, 0.0)


def assign_topic_tags(text: str) -> str:
    lower = text.lower()
    tags: list[str] = []
    for tag, terms in TOPIC_TAG_TERMS.items():
        if any(term in lower for term in terms):
            tags.append(tag)
    return ";".join(tags)


def relevance_class(score: float) -> str:
    if score >= 34:
        return "core"
    if score >= 22:
        return "high"
    if score >= 14:
        return "supporting"
    return "low"


def is_relevant(record: PaperRecord) -> bool:
    blob = f"{record.title}\n{record.abstract}".lower()
    has_single_cell = any(x in blob for x in ("single-cell", "single cell", "scrna", "single cell"))
    has_tcr = any(
        x in blob
        for x in (
            " tcr",
            "tcr-",
            "sctcr",
            "t cell receptor",
            "clonotype",
            "clonotypic",
            "vdj",
            "v(d)j",
            "repertoire",
        )
    )
    has_rna_context = any(
        x in blob
        for x in (
            "rna",
            "transcriptome",
            "transcriptomic",
            "gene expression",
            "immune profiling",
            "multiomic",
            "multi-omic",
        )
    )
    is_bcr_only = "b cell receptor" in blob and "t cell receptor" not in blob and " tcr" not in blob
    return has_single_cell and has_tcr and has_rna_context and not is_bcr_only and score_relevance(blob) >= 14


def finalize_record_annotations(record: PaperRecord) -> None:
    blob = f"{record.title}\n{record.abstract}"
    record.relevance_score = max(record.relevance_score, score_relevance(blob))
    record.topic_tags = assign_topic_tags(blob)
    record.relevance_class = relevance_class(record.relevance_score)


def combine_text(primary: str, secondary: str) -> str:
    primary = (primary or "").strip()
    secondary = (secondary or "").strip()
    if not primary:
        return secondary
    if not secondary or secondary in primary:
        return primary
    return f"{primary}\n\nSeed literature-card notes:\n{secondary}"


def record_from_validated_seed(row: dict[str, Any]) -> PaperRecord:
    seed = record_from_seed(row)
    lookup = lookup_europe_pmc_row(seed)
    if not lookup:
        seed.metadata_note = "seed_not_found_in_europe_pmc"
        finalize_record_annotations(seed)
        return seed

    rec = record_from_europe_pmc(lookup)
    rec.source_kind = "seed_validated_europe_pmc"
    rec.seed_title = seed.title
    rec.hit_queries = add_unique_csv(rec.hit_queries, "seed")
    sim = title_similarity(seed.title, rec.title)
    if seed.title and rec.title and sim < 0.55:
        rec.metadata_note = f"seed_title_mismatch_corrected:{sim:.2f}"
        rec.relevance_score = score_relevance(f"{rec.title}\n{rec.abstract}")
    else:
        rec.abstract = combine_text(rec.abstract, seed.abstract)
        rec.relevance_score = max(rec.relevance_score, seed.relevance_score)
    finalize_record_annotations(rec)
    return rec


def merge_records(old: PaperRecord | None, new: PaperRecord) -> PaperRecord:
    if old is None:
        return new
    # Prefer validated/discovered Europe PMC metadata for bibliographic fields,
    # but keep any richer abstract/card notes and provenance from both sides.
    chosen = new if new.source_kind in {"seed_validated_europe_pmc", "europe_pmc_discovery"} else old
    other = old if chosen is new else new
    chosen.abstract = combine_text(chosen.abstract, other.abstract)
    chosen.hit_queries = add_unique_csv(chosen.hit_queries, other.hit_queries)
    chosen.source_kind = add_unique_csv(chosen.source_kind, other.source_kind)
    chosen.full_text_urls = add_unique_csv(chosen.full_text_urls, other.full_text_urls)
    chosen.relevance_score = max(chosen.relevance_score, other.relevance_score)
    if other.metadata_note:
        chosen.metadata_note = add_unique_csv(chosen.metadata_note, other.metadata_note)
    if other.seed_title and not chosen.seed_title:
        chosen.seed_title = other.seed_title
    finalize_record_annotations(chosen)
    return chosen


def deduplicate_by_title(records: list[PaperRecord]) -> list[PaperRecord]:
    merged: dict[str, PaperRecord] = {}
    for rec in records:
        title_key = normalize_title(rec.title)
        if not title_key:
            title_key = key_for_paper(rec)
        old = merged.get(title_key)
        if old is None:
            merged[title_key] = rec
            continue
        # Merge into the record with richer identifiers/source URL.
        old_richness = sum(bool(x) for x in (old.pmid, old.pmcid, old.doi, old.source_url))
        rec_richness = sum(bool(x) for x in (rec.pmid, rec.pmcid, rec.doi, rec.source_url))
        if rec_richness >= old_richness:
            merged[title_key] = merge_records(rec, old)
        else:
            merged[title_key] = merge_records(old, rec)
    return list(merged.values())


def xml_to_text(xml_text: str) -> str:
    try:
        root = ET.fromstring(xml_text.encode("utf-8"))
    except Exception:
        return ""
    chunks: list[str] = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag in {"article-title", "title", "abstract", "p", "caption"}:
            text = " ".join(" ".join(elem.itertext()).split())
            if text and len(text) > 20:
                chunks.append(text)
    return "\n\n".join(chunks)


def bioc_json_to_text(json_text: str) -> str:
    try:
        data = json.loads(json_text)
    except Exception:
        return ""
    passages: list[str] = []
    if isinstance(data, list):
        docs = data
    elif isinstance(data, dict):
        docs = data.get("documents") or []
        if not docs and "passages" in data:
            docs = [data]
    else:
        docs = []
    for doc in docs:
        for passage in doc.get("passages", []):
            text = " ".join(str(passage.get("text", "")).split())
            if text and len(text) > 20:
                passages.append(text)
    return "\n\n".join(passages)


def download_full_text(record: PaperRecord, out_dir: Path) -> PaperRecord:
    if not record.pmcid:
        record.download_status = "no_pmcid_for_xml_download"
        return record
    pmcid = record.pmcid if record.pmcid.startswith("PMC") else f"PMC{record.pmcid}"
    basename = f"{pmcid}_{slugify(record.title, 60)}"
    xml_path = out_dir / "articles_xml" / f"{basename}.xml"
    text_path = out_dir / "articles_text" / f"{basename}.txt"
    if not text_path.exists():
        existing_text = next((out_dir / "articles_text").glob(f"{pmcid}_*.txt"), None)
        existing_xml = next((out_dir / "articles_xml").glob(f"{pmcid}_*.xml"), None)
        if existing_text:
            text_path = existing_text
        if existing_xml:
            xml_path = existing_xml
    if text_path.exists() and text_path.stat().st_size > 100:
        record.full_text_downloaded = True
        record.xml_path = str(xml_path)
        record.text_path = str(text_path)
        record.download_status = "downloaded_open_full_text"
        return record
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    xml_text = ""
    text = ""
    try:
        xml_text = request_text(EUROPE_PMC_FULLTEXT.format(pmcid=pmcid), timeout=60)
        if xml_text and "<" in xml_text[:200]:
            xml_path.write_text(xml_text, encoding="utf-8")
            text = xml_to_text(xml_text)
    except Exception:
        pass
    if not text:
        try:
            bioc = request_text(NCBI_BIOC_JSON.format(pmcid=pmcid), timeout=60)
            text = bioc_json_to_text(bioc)
            if text and not xml_text:
                (out_dir / "articles_bioc_json" / f"{basename}.json").parent.mkdir(parents=True, exist_ok=True)
                (out_dir / "articles_bioc_json" / f"{basename}.json").write_text(bioc, encoding="utf-8")
        except Exception:
            pass
    if text:
        text_path.write_text(text, encoding="utf-8")
        record.full_text_downloaded = True
        record.xml_path = str(xml_path) if xml_path.exists() else ""
        record.text_path = str(text_path)
        record.download_status = "downloaded_open_full_text"
    elif record.pmcid:
        record.download_status = "pmc_full_text_download_failed"
    return record


def finalize_download_status(record: PaperRecord) -> None:
    if record.full_text_downloaded:
        record.download_status = "downloaded_open_full_text"
        record.manual_download_priority = ""
        return
    if not record.download_status:
        if record.pmcid:
            record.download_status = "pmc_full_text_download_failed"
        elif record.open_access or "Free" in record.full_text_urls or "Open access" in record.full_text_urls:
            record.download_status = "open_or_free_no_pmc_xml_manual_check"
        else:
            record.download_status = "no_open_full_text_found"
    if record.relevance_class in {"core", "high"}:
        record.manual_download_priority = "high"
    elif record.relevance_class == "supporting":
        record.manual_download_priority = "medium"
    else:
        record.manual_download_priority = "low"


def chunk_text(text: str, chunk_words: int = 450, overlap_words: int = 80) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_words)
        chunk = " ".join(words[start:end]).strip()
        if len(chunk) > 200:
            chunks.append(chunk)
        if end == len(words):
            break
        start = max(end - overlap_words, start + 1)
    return chunks


def build_chunks(records: list[PaperRecord], out_dir: Path) -> int:
    chunks_path = out_dir / "rag_chunks.jsonl"
    chunks_dir = out_dir / "rag_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with chunks_path.open("w", encoding="utf-8") as out:
        for rec in records:
            text = ""
            if rec.text_path and Path(rec.text_path).exists():
                text = Path(rec.text_path).read_text(encoding="utf-8", errors="replace")
            if not text:
                text = f"{rec.title}\n\n{rec.abstract}".strip()
            if not text:
                continue
            for idx, chunk in enumerate(chunk_text(text), start=1):
                count += 1
                chunk_id = f"{rec.doc_id}_{idx:04d}"
                item = {
                    "chunk_id": chunk_id,
                    "doc_id": rec.doc_id,
                    "title": rec.title,
                    "text": chunk,
                    "source_url": rec.source_url,
                    "pmid": rec.pmid,
                    "pmcid": rec.pmcid,
                    "doi": rec.doi,
                    "year": rec.year,
                    "journal": rec.journal,
                    "section": "full_text" if rec.full_text_downloaded else "abstract_or_card",
                    "is_full_text": rec.full_text_downloaded,
                    "relevance_score": rec.relevance_score,
                }
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
                md_name = f"{count:05d}_{slugify(rec.doc_id)}_{idx:04d}.md"
                (chunks_dir / md_name).write_text(
                    f"# {rec.title}\n\nSource: {rec.source_url}\n\n{chunk}\n",
                    encoding="utf-8",
                )
    return count


def write_records(records: list[PaperRecord], out_dir: Path) -> None:
    jsonl = out_dir / "papers.jsonl"
    csv_path = out_dir / "papers.csv"
    with jsonl.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(records[0]).keys()) if records else [])
        if records:
            writer.writeheader()
            for rec in records:
                writer.writerow(asdict(rec))


def write_manual_download_list(records: list[PaperRecord], out_dir: Path) -> int:
    missing = [
        rec
        for rec in records
        if not rec.full_text_downloaded and rec.manual_download_priority in {"high", "medium"}
    ]
    missing.sort(
        key=lambda x: (
            {"high": 0, "medium": 1, "low": 2}.get(x.manual_download_priority, 3),
            -x.relevance_score,
            x.year,
        )
    )
    fieldnames = [
        "manual_download_priority",
        "relevance_class",
        "relevance_score",
        "title",
        "year",
        "journal",
        "pmid",
        "pmcid",
        "doi",
        "source_url",
        "full_text_urls",
        "topic_tags",
        "download_status",
        "metadata_note",
    ]
    with (out_dir / "manual_download_needed.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in missing:
            writer.writerow({field: getattr(rec, field) for field in fieldnames})
    return len(missing)


def write_metadata_audit(records: list[PaperRecord], out_dir: Path) -> int:
    audited = [rec for rec in records if rec.metadata_note or (rec.seed_title and rec.seed_title != rec.title)]
    fieldnames = [
        "title",
        "seed_title",
        "metadata_note",
        "pmid",
        "pmcid",
        "doi",
        "source_url",
        "relevance_score",
        "source_kind",
    ]
    with (out_dir / "metadata_audit.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in audited:
            writer.writerow({field: getattr(rec, field) for field in fieldnames})
    return len(audited)


def write_search_strategy(out_dir: Path, index: dict[str, Any]) -> None:
    lines = [
        "# scRNA-scTCR RAG Knowledge Base Build Report",
        "",
        "## Method",
        "",
        "This knowledge base follows the common literature-RAG pattern: targeted corpus discovery, identifier-normalized metadata, open full-text acquisition where legally exposed, provenance-preserving chunking, and a manual-download queue for high-value papers without retrievable full text.",
        "",
        "## Targeted Europe PMC Queries",
        "",
    ]
    for spec in QUERY_SET:
        lines.append(f"- `{spec['label']}`: {spec['query']}")
    lines.extend(
        [
            "",
            "## Build Summary",
            "",
            f"- Records retained: {index['records']}",
            f"- Open full texts downloaded: {index['full_text_downloaded']}",
            f"- RAG chunks: {index['rag_chunks']}",
            f"- Manual-download candidates: {index['manual_download_needed']}",
            "",
            "## Inclusion Rule",
            "",
            "A paper was retained only if title/abstract/card text contained single-cell context, TCR/clonotype/repertoire context, and RNA/transcriptome/immune-profiling context with a minimum relevance score.",
            "",
            "## Files",
            "",
            "- `papers.csv` / `papers.jsonl`: normalized metadata and provenance.",
            "- `rag_chunks.jsonl`: runtime RAG chunks used by the scRTA agent.",
            "- `manual_download_needed.csv`: high/medium priority papers that the script could not fetch as full text.",
            "- `metadata_audit.csv`: seed-card identifier/title corrections and validation notes.",
            "- `articles_text/` and `articles_xml/`: downloaded open full text.",
            "",
            "## Copyright Boundary",
            "",
            "The builder downloads only open text exposed through Europe PMC / PMC APIs. It does not bypass subscriptions or paywalls.",
            "",
        ]
    )
    (out_dir / "BUILD_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an open scRNA-scTCR RAG knowledge base.")
    parser.add_argument("--out", default="E:/scRTA/rag_kb/scrna_sctcr_focused_20260509")
    parser.add_argument("--seed-csv", default="G:/scRNA_scTCR/scRTA/knowledge_base/scrna_sctcr_literature_20260508_v2/scrna_sctcr_literature_cards.csv")
    parser.add_argument("--max-results-per-query", type=int, default=450)
    parser.add_argument("--max-fulltext-downloads", type=int, default=420)
    parser.add_argument("--no-discover", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    seed_rows = read_seed_cards(Path(args.seed_csv) if args.seed_csv else None)
    discovered_rows: list[dict[str, Any]] = []
    if not args.no_discover:
        discovered_rows = discover_europe_pmc(args.max_results_per_query)
        write_json(raw_dir / "europe_pmc_discovery.json", discovered_rows)

    records_by_key: dict[str, PaperRecord] = {}
    for row in seed_rows:
        rec = record_from_validated_seed(row)
        if is_relevant(rec):
            key = key_for_paper(rec)
            records_by_key[key] = merge_records(records_by_key.get(key), rec)
        time.sleep(0.05)
    for row in discovered_rows:
        rec = record_from_europe_pmc(row)
        finalize_record_annotations(rec)
        if is_relevant(rec):
            key = key_for_paper(rec)
            records_by_key[key] = merge_records(records_by_key.get(key), rec)
    records = sorted(
        deduplicate_by_title(list(records_by_key.values())),
        key=lambda x: (
            {"core": 0, "high": 1, "supporting": 2, "low": 3}.get(x.relevance_class, 4),
            -x.relevance_score,
            x.title.lower(),
        ),
    )

    downloaded = 0
    for rec in records:
        if downloaded >= args.max_fulltext_downloads:
            break
        if rec.pmcid:
            before = rec.full_text_downloaded
            download_full_text(rec, out_dir)
            if rec.full_text_downloaded and not before:
                downloaded += 1
            time.sleep(0.15)
    for rec in records:
        finalize_download_status(rec)

    chunk_count = build_chunks(records, out_dir)
    write_records(records, out_dir)
    manual_count = write_manual_download_list(records, out_dir)
    audit_count = write_metadata_audit(records, out_dir)
    class_counts: dict[str, int] = {}
    topic_counts: dict[str, int] = {}
    for rec in records:
        class_counts[rec.relevance_class] = class_counts.get(rec.relevance_class, 0) + 1
        for tag in rec.topic_tags.split(";"):
            if tag:
                topic_counts[tag] = topic_counts.get(tag, 0) + 1
    index = {
        "schema_version": "scrta_agent_rag.v2",
        "created_at": now_stamp(),
        "records": len(records),
        "full_text_downloaded": sum(1 for r in records if r.full_text_downloaded),
        "rag_chunks": chunk_count,
        "manual_download_needed": manual_count,
        "metadata_audit_records": audit_count,
        "relevance_class_counts": class_counts,
        "topic_tag_counts": dict(sorted(topic_counts.items(), key=lambda x: x[0])),
        "paths": {
            "papers_csv": str(out_dir / "papers.csv"),
            "papers_jsonl": str(out_dir / "papers.jsonl"),
            "rag_chunks_jsonl": str(out_dir / "rag_chunks.jsonl"),
            "rag_chunks_dir": str(out_dir / "rag_chunks"),
            "articles_xml": str(out_dir / "articles_xml"),
            "articles_text": str(out_dir / "articles_text"),
            "manual_download_needed_csv": str(out_dir / "manual_download_needed.csv"),
            "metadata_audit_csv": str(out_dir / "metadata_audit.csv"),
            "build_report": str(out_dir / "BUILD_REPORT.md"),
        },
        "sources": {
            "seed_csv": args.seed_csv,
            "europe_pmc_search_api": EUROPE_PMC_SEARCH,
            "europe_pmc_full_text_api": EUROPE_PMC_FULLTEXT,
            "ncbi_pmc_bioc_api": NCBI_BIOC_JSON,
            "query_labels": [spec["label"] for spec in QUERY_SET],
        },
        "notes": [
            "Open access full text was downloaded where Europe PMC/PMC exposed XML/BioC text.",
            "Non-open or unavailable full text records are represented by abstracts or existing literature cards.",
            "Seed-card PMID/PMCID/DOI values were validated against Europe PMC; title mismatches are recorded in metadata_audit.csv.",
            "No paywalled content was bypassed.",
        ],
    }
    write_json(out_dir / "index.json", index)
    write_search_strategy(out_dir, index)
    latest = Path("E:/scRTA/rag_kb/scrna_sctcr_rag_current")
    latest.mkdir(parents=True, exist_ok=True)
    for name in ("index.json", "papers.csv", "papers.jsonl", "rag_chunks.jsonl"):
        src = out_dir / name
        if src.exists():
            (latest / name).write_text(src.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    print(json.dumps(index, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
