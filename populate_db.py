from __future__ import annotations

import json
import re
import time
import sqlite3
import urllib.request
import urllib.parse
from datetime import datetime, date
from pathlib import Path
from typing import Optional

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()
llm = ChatGroq(model="llama-3.3-70b-versatile")


# ─────────────────────────────────────────────
# 0) Config
# ─────────────────────────────────────────────

_BASE_DIR = Path(__file__).parent
DB_FILE   = _BASE_DIR / "microlearn.db"

VALID_SUBJECTS = [
    "human behavior", "psychology", "persuasion", "anatomy", "physiology",
    "sales psychology", "archeology", "medicine", "endocrinology",
    "behavioral economics", "neuroscience", "evolutionary biology",
    "anthropology", "cognitive science", "nutrition science",
    "sleep science", "geopolitics", "health & fitness", "social sciences",
]

MIN_CITATIONS    = 2    # minimum citation count (Semantic Scholar)
WINDOW_SIZE      = 4    # number of sources active per populate call

# ─────────────────────────────────────────────
# 1) DB setup
# ─────────────────────────────────────────────

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS papers (
            doi             TEXT PRIMARY KEY,
            title           TEXT NOT NULL,
            abstract        TEXT,
            full_text       TEXT,
            subject         TEXT,
            source          TEXT,
            published_at    TEXT,
            is_preprint     INTEGER DEFAULT 0,
            citation_count  INTEGER DEFAULT 0,
            is_open_access  INTEGER DEFAULT 1,
            quality_ok      INTEGER DEFAULT 0,
            full_text_ok    INTEGER DEFAULT 0,
            served          INTEGER DEFAULT 0,
            in_reading_list INTEGER DEFAULT 0,
            saved           INTEGER DEFAULT 0,
            date_added      TEXT,
            date_served     TEXT,
            card_title      TEXT,
            card_intro      TEXT,
            filename        TEXT
        );

        CREATE TABLE IF NOT EXISTS offsets (
            subject          TEXT    NOT NULL,
            source           TEXT    NOT NULL,
            offset           INTEGER DEFAULT 0,
            last_reset       TEXT,
            exhausted        INTEGER DEFAULT 0,
            PRIMARY KEY (subject, source)
        );

        CREATE TABLE IF NOT EXISTS subject_state (
            subject          TEXT PRIMARY KEY,
            window_start     INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_subject_served
            ON papers(subject, served, quality_ok, full_text_ok);
    """)
    # Migrate existing DBs — add columns if missing
    for col, definition in [("card_title", "TEXT"), ("card_intro", "TEXT"), ("filename", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE papers ADD COLUMN {col} {definition}")
        except sqlite3.OperationalError:
            pass
    # Remove fetchable_count column if present (legacy migration)
    try:
        conn.execute("ALTER TABLE offsets DROP COLUMN fetchable_count")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()


def _upsert_paper(conn: sqlite3.Connection, paper: dict) -> bool:
    """Insert or ignore (DOI is primary key). Returns True if inserted."""
    try:
        conn.execute("""
            INSERT OR IGNORE INTO papers
            (doi, title, abstract, full_text, subject, source, published_at,
             is_preprint, citation_count, is_open_access, quality_ok, full_text_ok,
             card_title, card_intro, date_added)
            VALUES (:doi, :title, :abstract, :full_text, :subject, :source, :published_at,
                    :is_preprint, :citation_count, :is_open_access, :quality_ok, :full_text_ok,
                    :card_title, :card_intro, :date_added)
        """, {
            **paper,
            "full_text":    paper.get("full_text") or None,
            "quality_ok":   int(paper.get("quality_ok", 0)),
            "full_text_ok": int(paper.get("full_text_ok", 0)),
            "card_title":   paper.get("card_title") or None,
            "card_intro":   paper.get("card_intro") or None,
            "date_added":   datetime.now().isoformat(timespec="seconds"),
        })
        return conn.total_changes > 0
    except Exception as exc:
        print(f"[db] upsert failed for {paper.get('doi')}: {exc}")
        return False


# Sources that sort by recency — use date cursor strategy
RECENCY_SOURCES = {"arxiv", "biorxiv"}
# Sources that sort by relevance — reset offset periodically
RELEVANCE_SOURCES = {"pubmed", "semantic_scholar", "openalex", "crossref",
                     "europe_pmc", "ssrn", "base"}
RESET_STALE_DAYS = 7    # reset relevance-sorted source offsets after N days
TOP_REFETCH_N   = 3     # re-fetch top 3 from offset=0 per source (recency refresh)
ADVANCE_N       = 10    # fetch 10 from current offset per source


def _get_source_state(conn: sqlite3.Connection, subject: str, source: str) -> dict:
    row = conn.execute(
        "SELECT offset, last_reset, exhausted FROM offsets WHERE subject=? AND source=?",
        (subject, source)
    ).fetchone()
    return dict(row) if row else {"offset": 0, "last_reset": None, "exhausted": 0}


def _set_source_state(conn: sqlite3.Connection, subject: str, source: str,
                      offset: int, exhausted: bool = False, reset: bool = False) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        """INSERT INTO offsets (subject, source, offset, last_reset, exhausted)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(subject, source) DO UPDATE SET
               offset     = excluded.offset,
               last_reset = CASE WHEN ? THEN ? ELSE last_reset END,
               exhausted  = excluded.exhausted""",
        (subject, source, offset, now, int(exhausted),
         int(reset), now)
    )
    conn.commit()


def _is_stale(last_reset: str | None) -> bool:
    """True if last_reset is older than RESET_STALE_DAYS."""
    if not last_reset:
        return True
    try:
        reset_date = datetime.fromisoformat(last_reset).date()
        return (date.today() - reset_date).days >= RESET_STALE_DAYS
    except Exception:
        return True


def _get_window_start(conn: sqlite3.Connection, subject: str) -> int:
    row = conn.execute(
        "SELECT window_start FROM subject_state WHERE subject=?", (subject,)
    ).fetchone()
    return row["window_start"] if row else 0


def _advance_window(conn: sqlite3.Connection, subject: str, current: int) -> None:
    next_start = (current + 1) % len(FETCHERS)
    conn.execute(
        """INSERT INTO subject_state (subject, window_start) VALUES (?, ?)
           ON CONFLICT(subject) DO UPDATE SET window_start = excluded.window_start""",
        (subject, next_start)
    )
    conn.commit()


# ─────────────────────────────────────────────
# 2) HTTP helper
# ─────────────────────────────────────────────

def _get(url: str, headers: Optional[dict] = None, timeout: int = 12) -> Optional[dict]:
    """GET JSON from url. Returns None on failure."""
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "MicroLearnBot/1.0 (research aggregator)",
            **(headers or {}),
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="ignore"))
    except Exception as exc:
        print(f"[http] GET failed {url[:80]}: {exc}")
        return None


def _normalize_doi(doi: str) -> str:
    """Strip URL prefix if present."""
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", doi).strip()


# ─────────────────────────────────────────────
# 3) Source fetchers
#    Each returns List[dict] with keys:
#    doi, title, abstract, published_at,
#    is_preprint, citation_count, is_open_access, source
# ─────────────────────────────────────────────

# ── 3a) PubMed ────────────────────────────────

def _pubmed_fetch(subject: str, offset: int, n: int) -> list[dict]:
    query = urllib.parse.quote(
        f"{subject} AND free full text[filter] AND (\"2020\"[PDAT]:\"3000\"[PDAT])"
    )
    search_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        f"?db=pubmed&term={query}&retstart={offset}&retmax={n}&retmode=json&sort=relevance"
    )
    data = _get(search_url)
    if not data:
        return []

    ids = data.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []

    # Fetch abstracts in batch
    fetch_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=pubmed&id={','.join(ids)}&retmode=xml"
    )
    try:
        req = urllib.request.Request(fetch_url, headers={"User-Agent": "MicroLearnBot/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            xml = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return []

    results = []
    articles = re.findall(r"<PubmedArticle>(.*?)</PubmedArticle>", xml, re.DOTALL)
    for art in articles:
        title_m    = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", art, re.DOTALL)
        abstract_m = re.search(r"<AbstractText[^>]*>(.*?)</AbstractText>", art, re.DOTALL)
        doi_m      = re.search(r'<ArticleId IdType="doi">(.*?)</ArticleId>', art)
        year_m     = re.search(r"<PubDate>.*?<Year>(\d{4})", art, re.DOTALL)
        pmcid_m    = re.search(r'<ArticleId IdType="pmc">(.*?)</ArticleId>', art)

        doi = _normalize_doi(doi_m.group(1).strip()) if doi_m else None
        if not doi:
            continue

        results.append({
            "doi":            doi,
            "title":          re.sub(r"<[^>]+>", "", title_m.group(1)).strip() if title_m else "",
            "abstract":       re.sub(r"<[^>]+>", "", abstract_m.group(1)).strip() if abstract_m else "",
            "published_at":   f"{year_m.group(1)}-01-01" if year_m else None,
            "is_preprint":    0,
            "citation_count": 0,   # PubMed doesn't return citations
            "is_open_access": 1 if pmcid_m else 0,
            "source":         "pubmed",
        })
        time.sleep(0.1)   # NCBI rate limit

    return results


# ── 3b) Semantic Scholar ──────────────────────

def _semantic_scholar_fetch(subject: str, offset: int, n: int) -> list[dict]:
    query = urllib.parse.quote(f"{subject} controversial finding OR unexpected result OR landmark study")
    url   = (
        f"https://api.semanticscholar.org/graph/v1/paper/search"
        f"?query={query}&offset={offset}&limit={n}"
        f"&fields=title,abstract,year,externalIds,isOpenAccess,citationCount,publicationTypes"
        f"&openAccessPdf=true"
    )
    data = _get(url)
    if not data:
        return []

    results = []
    for p in data.get("data", []):
        if not p.get("isOpenAccess"):
            continue
        doi = p.get("externalIds", {}).get("DOI")
        if not doi:
            continue
        doi = _normalize_doi(doi)

        pub_types  = p.get("publicationTypes") or []
        is_preprint = 1 if any(t in ["Preprint"] for t in pub_types) else 0

        results.append({
            "doi":            doi,
            "title":          p.get("title") or "",
            "abstract":       p.get("abstract") or "",
            "published_at":   f"{p['year']}-01-01" if p.get("year") else None,
            "is_preprint":    is_preprint,
            "citation_count": p.get("citationCount") or 0,
            "is_open_access": 1,
            "source":         "semantic_scholar",
        })

    return results


# ── 3c) arXiv ─────────────────────────────────

_ARXIV_SUBJECT_MAP = {
    "psychology":           "q-bio.NC",
    "neuroscience":         "q-bio.NC",
    "cognitive science":    "cs.AI",
    "behavioral economics": "econ.GN",
    "geopolitics":          "econ.GN",
    "social sciences":      "econ.GN",
    "health & fitness":     "q-bio.TO",
    "medicine":             "q-bio.TO",
}

def _arxiv_fetch(subject: str, offset: int, n: int) -> list[dict]:
    cat   = _ARXIV_SUBJECT_MAP.get(subject, "")
    query = urllib.parse.quote(f"all:{subject} AND ti:study")
    cat_q = f"+AND+cat:{cat}" if cat else ""
    url   = (
        f"http://export.arxiv.org/api/query"
        f"?search_query={query}{cat_q}&start={offset}&max_results={n}&sortBy=relevance"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MicroLearnBot/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            xml = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return []

    results = []
    entries = re.findall(r"<entry>(.*?)</entry>", xml, re.DOTALL)
    for e in entries:
        title_m    = re.search(r"<title>(.*?)</title>", e, re.DOTALL)
        summary_m  = re.search(r"<summary>(.*?)</summary>", e, re.DOTALL)
        doi_m      = re.search(r'<arxiv:doi[^>]*>(.*?)</arxiv:doi>', e)
        id_m       = re.search(r"<id>(.*?)</id>", e)
        date_m     = re.search(r"<published>([\d\-]+)", e)

        # Use arXiv ID as DOI fallback
        doi = _normalize_doi(doi_m.group(1).strip()) if doi_m else None
        if not doi and id_m:
            arxiv_id = id_m.group(1).split("/abs/")[-1].strip()
            doi = f"arxiv:{arxiv_id}"
        if not doi:
            continue

        results.append({
            "doi":            doi,
            "title":          re.sub(r"\s+", " ", title_m.group(1)).strip() if title_m else "",
            "abstract":       re.sub(r"\s+", " ", summary_m.group(1)).strip() if summary_m else "",
            "published_at":   date_m.group(1)[:10] if date_m else None,
            "is_preprint":    1,
            "citation_count": 0,
            "is_open_access": 1,
            "source":         "arxiv",
        })

    return results


# ── 3d) OpenAlex ──────────────────────────────

def _openalex_fetch(subject: str, offset: int, n: int) -> list[dict]:
    query = urllib.parse.quote(subject)
    url   = (
        f"https://api.openalex.org/works"
        f"?search={query}&filter=is_oa:true,has_doi:true"
        f"&per_page={n}&page={offset // n + 1}"
        f"&select=doi,title,abstract_inverted_index,publication_date,cited_by_count,type"
        f"&sort=cited_by_count:desc"
    )
    data = _get(url, headers={"mailto": "microlearn@localhost"})
    if not data:
        return []

    results = []
    for w in data.get("results", []):
        doi = w.get("doi")
        if not doi:
            continue
        doi = _normalize_doi(doi)

        # Reconstruct abstract from inverted index
        inv = w.get("abstract_inverted_index") or {}
        abstract = ""
        if inv:
            word_positions = [(pos, word) for word, positions in inv.items() for pos in positions]
            abstract = " ".join(w for _, w in sorted(word_positions))

        results.append({
            "doi":            doi,
            "title":          w.get("title") or "",
            "abstract":       abstract,
            "published_at":   (w.get("publication_date") or "")[:10] or None,
            "is_preprint":    1 if w.get("type") == "preprint" else 0,
            "citation_count": w.get("cited_by_count") or 0,
            "is_open_access": 1,
            "source":         "openalex",
        })

    return results


# ── 3e) Europe PMC ────────────────────────────

def _europe_pmc_fetch(subject: str, offset: int, n: int) -> list[dict]:
    query = urllib.parse.quote(
        f"{subject} AND OPEN_ACCESS:Y AND (HAS_ABSTRACT:Y)"
    )
    page = offset // n + 1
    url = (
        f"https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        f"?query={query}&resultType=core&format=json"
        f"&pageSize={n}&page={page}&sort=CITED+desc"
    )
    data = _get(url)
    if not data:
        return []

    results = []
    for r in data.get("resultList", {}).get("result", []):
        doi = r.get("doi")
        if not doi:
            continue
        abstract = r.get("abstractText") or ""
        results.append({
            "doi":            _normalize_doi(doi),
            "title":          r.get("title") or "",
            "abstract":       abstract,
            "published_at":   str(r["pubYear"]) + "-01-01" if r.get("pubYear") else None,
            "is_preprint":    1 if r.get("source") in ["PPR", "BIORXIV", "MEDRXIV"] else 0,
            "citation_count": r.get("citedByCount") or 0,
            "is_open_access": 1,
            "source":         "europe_pmc",
        })

    return results


# ── 3f) bioRxiv / medRxiv ─────────────────────

def _biorxiv_fetch(subject: str, offset: int, n: int) -> list[dict]:
    # bioRxiv/medRxiv API: date-based cursor, no subject search — use keyword in title
    servers = ["biorxiv", "medrxiv"]
    n_each  = max(1, n // len(servers))
    results = []
    for server in servers:
        url = (
            f"https://api.biorxiv.org/details/{server}/2020-01-01/3000-01-01"
            f"/{offset}/{n_each}/json"
        )
        data = _get(url)
        if not data:
            continue
        for p in data.get("collection", []):
            # Filter by subject keyword in title or category
            title_lower = (p.get("title") or "").lower()
            cat_lower   = (p.get("category") or "").lower()
            if subject.lower() not in title_lower and subject.lower() not in cat_lower:
                continue
            doi = p.get("doi")
            if not doi:
                continue
            results.append({
                "doi":            _normalize_doi(doi),
                "title":          p.get("title") or "",
                "abstract":       p.get("abstract") or "",
                "published_at":   p.get("date") or None,
                "is_preprint":    1,
                "citation_count": 0,
                "is_open_access": 1,
                "source":         server,
            })

    return results[:n]


def _parse_crossref_date(pub: dict) -> str | None:
    dp = (pub.get("date-parts") or [[]])[0]
    if len(dp) >= 2:
        return f"{dp[0]}-{dp[1]:02d}-01"
    if len(dp) == 1:
        return f"{dp[0]}-01-01"
    return None


# ── 3g) CrossRef ──────────────────────────────

def _crossref_fetch(subject: str, offset: int, n: int) -> list[dict]:
    query = urllib.parse.quote(f"{subject} study findings methodology")
    url   = (
        f"https://api.crossref.org/works"
        f"?query={query}&rows={n}&offset={offset}"
        f"&filter=has-abstract:true,has-license:true"
        f"&select=DOI,title,abstract,published,is-referenced-by-count,license,type"
        f"&mailto=microlearn@localhost"
    )
    data = _get(url)
    if not data:
        return []

    results = []
    for w in (data.get("message") or {}).get("items") or []:
        doi = w.get("DOI")
        if not doi:
            continue

        # Check for open license
        licenses = w.get("license") or []
        is_oa = int(any(
            "creativecommons" in (lic.get("URL") or "").lower()
            for lic in licenses
        ))
        if not is_oa:
            continue

        pub  = w.get("published") or {}
        date_str = _parse_crossref_date(pub)

        title_list = w.get("title") or []
        abstract   = re.sub(r"<[^>]+>", "", w.get("abstract") or "")

        results.append({
            "doi":            _normalize_doi(doi),
            "title":          title_list[0] if title_list else "",
            "abstract":       abstract,
            "published_at":   date_str,
            "is_preprint":    1 if w.get("type") == "posted-content" else 0,
            "citation_count": w.get("is-referenced-by-count") or 0,
            "is_open_access": is_oa,
            "source":         "crossref",
        })

    return results


# ── 3h) SSRN ──────────────────────────────────

def _ssrn_fetch(subject: str, offset: int, n: int) -> list[dict]:
    # SSRN has no public API — use CrossRef filter for SSRN preprints
    query = urllib.parse.quote(f"{subject}")
    url   = (
        f"https://api.crossref.org/works"
        f"?query={query}&rows={n}&offset={offset}"
        f"&filter=type:posted-content,member:246"   # member 246 = SSRN/Elsevier
        f"&select=DOI,title,abstract,published,type"
        f"&mailto=microlearn@localhost"
    )
    data = _get(url)
    if not data:
        return []

    results = []
    for w in (data.get("message") or {}).get("items") or []:
        doi  = w.get("DOI")
        if not doi:
            continue
        pub  = w.get("published") or {}
        date_str = _parse_crossref_date(pub)
        title_list = w.get("title") or []
        results.append({
            "doi":            _normalize_doi(doi),
            "title":          title_list[0] if title_list else "",
            "abstract":       re.sub(r"<[^>]+>", "", w.get("abstract") or ""),
            "published_at":   date_str,
            "is_preprint":    1,
            "citation_count": 0,
            "is_open_access": 1,
            "source":         "ssrn",
        })

    return results


# ── 3i) BASE ──────────────────────────────────

def _base_fetch(subject: str, offset: int, n: int) -> list[dict]:
    # BASE offers OAI-PMH and a search API (free, no key needed)
    query = urllib.parse.quote(f"{subject} study")
    url   = (
        f"https://api.base-search.net/cgi-bin/BaseHttpSearchInterface.fcgi"
        f"?func=PerformSearch&query=subj:{query}&hits={n}&offset={offset}"
        f"&fields=dcidentifier,dctitle,dcabstract,dcdate,dcdoi"
        f"&format=json&boost=oa"
    )
    data = _get(url)
    if not data:
        return []

    results = []
    docs = (data.get("response") or {}).get("docs") or []
    for d in docs:
        doi = (d.get("dcdoi") or [""])[0]
        if not doi:
            continue
        results.append({
            "doi":            _normalize_doi(doi),
            "title":          (d.get("dctitle") or [""])[0],
            "abstract":       (d.get("dcabstract") or [""])[0],
            "published_at":   (d.get("dcdate") or [""])[0][:10] or None,
            "is_preprint":    0,
            "citation_count": 0,
            "is_open_access": 1,
            "source":         "base",
        })

    return results


# ─────────────────────────────────────────────
# 4) Inline validation
# ─────────────────────────────────────────────

def _validate(paper: dict) -> bool:
    """
    Validate a paper inline during fetch. Returns True if paper passes.
    Checks:
      - Has DOI
      - Has non-empty title and abstract (min 100 chars)
      - Is open access
      - Citation count >= MIN_CITATIONS (skip for preprints — they're new)
      - Not a retracted paper (basic title check)
    """
    if not paper.get("doi"):
        return False
    if not paper.get("title") or len(paper.get("title", "")) < 5:
        return False
    abstract = paper.get("abstract") or ""
    if len(abstract) < 100:
        return False
    if not paper.get("is_open_access"):
        return False
    # Skip citation floor for preprints — they haven't had time to accumulate
    if not paper.get("is_preprint") and paper.get("citation_count", 0) < MIN_CITATIONS:
        return False
    # Basic retraction check
    title_lower = paper.get("title", "").lower()
    if "retraction" in title_lower or "retracted" in title_lower:
        return False
    return True



# ─────────────────────────────────────────────
# 4b) Full text fetchers (called after metadata validated)
# ─────────────────────────────────────────────

def _fetch_full_text_arxiv(doi: str) -> str:
    """Fetch full text from arXiv HTML (preferred) or PDF fallback."""
    arxiv_id = doi.replace("arxiv:", "").split("/")[-1]
    # Try HTML version first (cleaner text)
    for url in [
        f"https://ar5iv.org/html/{arxiv_id}",
        f"https://arxiv.org/html/{arxiv_id}",
    ]:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "MicroLearnBot/1.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                raw = r.read().decode("utf-8", errors="ignore")
            text = re.sub(r"<[^>]+>", " ", raw)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > 500:
                return text[:60000]
        except Exception:
            continue
    # PDF fallback via pdfplumber/fitz
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    try:
        req = urllib.request.Request(pdf_url, headers={"User-Agent": "MicroLearnBot/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            pdf_bytes = r.read()
        import io
        try:
            import fitz
            doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
            return " ".join(p.get_text() for p in doc)[:60000]
        except ImportError:
            pass
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                return " ".join(p.extract_text() or "" for p in pdf.pages)[:60000]
        except ImportError:
            pass
    except Exception:
        pass
    return ""


def _fetch_full_text_pmc(doi: str) -> str:
    """Fetch full text from PubMed Central XML."""
    # Convert DOI to PMCID via ID converter
    try:
        conv_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={urllib.parse.quote(doi)}&format=json"
        req = urllib.request.Request(conv_url, headers={"User-Agent": "MicroLearnBot/1.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        records = data.get("records", [])
        pmcid = records[0].get("pmcid") if records else None
        if not pmcid:
            return ""
        fetch_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            f"?db=pmc&id={pmcid}&rettype=full&retmode=xml"
        )
        req2 = urllib.request.Request(fetch_url, headers={"User-Agent": "MicroLearnBot/1.0"})
        with urllib.request.urlopen(req2, timeout=15) as r:
            xml = r.read().decode("utf-8", errors="ignore")
        text = re.sub(r"<[^>]+>", " ", xml)
        return re.sub(r"\s+", " ", text).strip()[:60000]
    except Exception as e:
        print(f"[pmc] full text fetch failed for {doi}: {e}")
        return ""


def _fetch_full_text_doi(doi: str) -> str:
    """Generic DOI landing page fetch — last resort. Rejects short/paywall pages."""
    try:
        url = f"https://doi.org/{doi}"
        req = urllib.request.Request(url, headers={"User-Agent": "MicroLearnBot/1.0"})
        with urllib.request.urlopen(req, timeout=12) as r:
            raw = r.read().decode("utf-8", errors="ignore")
        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"\s+", " ", text).strip()
        # Reject paywall/abstract pages — require substantial text (>1500 words)
        if len(text.split()) < 1500:
            return ""
        return text[:60000]
    except Exception:
        return ""


def fetch_full_text(doi: str, source: str) -> str:
    """
    Dispatch full text fetch by source.
    Returns empty string if unavailable — caller handles gracefully.
    """
    doi_lower = doi.lower()
    src_lower = (source or "").lower()

    if "arxiv" in doi_lower or "arxiv" in src_lower:
        return _fetch_full_text_arxiv(doi)
    if src_lower in ("pubmed", "europe_pmc", "biorxiv", "medrxiv"):
        text = _fetch_full_text_pmc(doi)
        return text if text else _fetch_full_text_doi(doi)
    return _fetch_full_text_doi(doi)


# ─────────────────────────────────────────────
# 4c) Combined LLM quality filter + card generation
# ─────────────────────────────────────────────

QUALITY_AND_CARD_SYSTEM = """You are an editorial filter and card writer for a micro-learning platform.

For each paper, do TWO things in one pass:

STEP 1 — Quality check. A paper is suitable if it has ALL of the following:
  - Real methodology (not just a stated conclusion)
  - Measurable outcomes or empirical data
  - A non-obvious mechanism worth explaining to a curious non-expert
  - Not an editorial, opinion piece, letter, commentary, or literature review

STEP 2 — If suitable, write a card (title + intro). If not suitable, set card_title and card_intro to null.

Card rules:
  title:
    - State WHAT was found, not just the topic.
    - Bad:  "The surprising truth about sleep"
    - Good: "Seven hours of sleep reshapes which genes are active in immune cells"
    - Do NOT start with "New study shows…" or "Researchers find…"

  intro (2–3 sentences):
    - Write a catchy hook that makes the reader want to know more.
    - Do NOT reveal the conclusion.
    - No unexplained jargon.
    - Based only on what is present in the abstract — do not invent.

Return a JSON object with a single key "decisions" containing an array. Each element:
  {"doi": "<doi>", "suitable": true/false, "card_title": "<title or null>", "card_intro": "<intro or null>"}
"""

class CardDecision(BaseModel):
    doi:        str
    suitable:   bool
    card_title: Optional[str] = None
    card_intro: Optional[str] = None

class CardBatch(BaseModel):
    decisions: list[CardDecision]

def _llm_quality_and_cards(papers: list[dict]) -> dict[str, CardDecision]:
    """
    Combined quality filter + card generation in one LLM call per batch.
    Returns dict of doi -> CardDecision for approved papers only.
    Returns empty dict on error (fail-closed).
    """
    if not papers:
        return {}

    batch_text = "\n".join(
        f"[{i}] doi={p['doi']} | title={p.get('title','')[:120]} | abstract={p.get('abstract','')[:400]}"
        for i, p in enumerate(papers)
    )

    try:
        processor = llm.with_structured_output(CardBatch)
        result = processor.invoke([
            SystemMessage(content=QUALITY_AND_CARD_SYSTEM),
            HumanMessage(content=f"Papers to evaluate:\n{batch_text}"),
        ])
        return {d.doi: d for d in result.decisions if d.suitable}
    except Exception as exc:
        print(f"[quality_and_cards] LLM call failed: {exc} — rejecting batch (fail-closed)")
        return {}

# ─────────────────────────────────────────────
# 5) Population orchestrator
# ─────────────────────────────────────────────

FETCHERS = [
    ("pubmed",           _pubmed_fetch),
    ("arxiv",            _arxiv_fetch),
    ("openalex",         _openalex_fetch),
    ("europe_pmc",       _europe_pmc_fetch),
    ("biorxiv",          _biorxiv_fetch),
    ("semantic_scholar", _semantic_scholar_fetch),
    ("crossref",         _crossref_fetch),
    ("ssrn",             _ssrn_fetch),
    ("base",             _base_fetch),
]

def _fetch_from_source(source_name: str, fetcher, subject: str,
                       state: dict, conn: sqlite3.Connection) -> tuple[list[dict], bool]:
    """
    Fetch papers from one source using 10+3 split strategy:
      - 3 from offset=0 (top refresh — catches new papers)
      - 10 from current offset (advances into unseen)
    Returns (papers, exhausted).
    Handles recency vs relevance source strategies.
    """
    is_recency = source_name in RECENCY_SOURCES
    is_relevance = source_name in RELEVANCE_SOURCES
    current_offset = state["offset"]
    last_reset = state.get("last_reset")

    # For relevance-sorted sources: reset offset if stale
    if is_relevance and _is_stale(last_reset):
        print(f"  [{source_name}] stale — resetting offset to 0")
        current_offset = 0
        _set_source_state(conn, subject, source_name,
                          offset=0, reset=True)

    papers: list[dict] = []

    # Top-3 refresh (skip for recency sources — their offset 0 is always latest)
    if not is_recency and current_offset > 0:
        try:
            top = fetcher(subject, 0, TOP_REFETCH_N)
            papers.extend(top)
        except Exception as exc:
            print(f"  [{source_name}] top-refresh error: {exc}")

    # Main fetch from current offset
    try:
        main = fetcher(subject, current_offset, ADVANCE_N)
        papers.extend(main)
    except Exception as exc:
        print(f"  [{source_name}] main fetch error: {exc}")
        main = []

    exhausted = len(main) == 0
    return papers, exhausted

def _process_papers(papers: list[dict], subject: str,
                    conn: sqlite3.Connection) -> tuple[int, int]:
    """
    Validate → LLM quality filter → fetch full text → upsert.
    Returns (inserted, tombstoned).
    """
    validated = []
    for p in papers:
        p["subject"] = subject
        if _validate(p):
            validated.append(p)

    if not validated:
        return 0, 0

    # Dedup by DOI before LLM call
    seen: dict[str, dict] = {}
    for p in validated:
        seen.setdefault(p["doi"], p)
    validated = list(seen.values())

    # Combined LLM quality filter + card generation in batches of 10
    approved: dict[str, CardDecision] = {}
    for i in range(0, len(validated), 10):
        approved.update(_llm_quality_and_cards(validated[i:i + 10]))

    inserted = tombstoned = 0

    for p in validated:
        decision = approved.get(p["doi"])
        if decision is None:
            p.update({"full_text": None, "quality_ok": 0, "full_text_ok": 0,
                      "card_title": None, "card_intro": None})
            if _upsert_paper(conn, p):
                tombstoned += 1
            continue
        full_text = fetch_full_text(p["doi"], p.get("source", ""))
        p.update({
            "full_text":    full_text,
            "quality_ok":   1,
            "full_text_ok": 1 if full_text else 0,
            "card_title":   decision.card_title,
            "card_intro":   decision.card_intro,
        })
        if _upsert_paper(conn, p):
            inserted += 1
        time.sleep(0.2)

    return inserted, tombstoned

def populate(subjects: Optional[list[str]] = None) -> None:
    """
    Single-pass fetch for each subject using a rolling window of WINDOW_SIZE sources.
    Each source is called once per populate() call: 3 top-refresh + 10 main fetch.
    Window advances by 1 after each call (persisted in subject_state table).
    Call populate() again across sessions to gradually build the pool.
    """
    init_db()
    conn = get_conn()
    targets = subjects or VALID_SUBJECTS

    for subject in targets:
        print(f"\n[populate] subject: {subject}")
        total_inserted = 0

        window_start = _get_window_start(conn, subject)
        source_count = len(FETCHERS)
        window_indices = [(window_start + i) % source_count for i in range(WINDOW_SIZE)]
        active_sources = [FETCHERS[i] for i in window_indices]
        print(f"  window sources: {[s for s, _ in active_sources]}")

        for source_name, fetcher in active_sources:
            state = _get_source_state(conn, subject, source_name)
            if state["exhausted"]:
                print(f"  [{source_name}] exhausted — skipping")
                continue

            papers, exhausted = _fetch_from_source(
                source_name, fetcher, subject, state, conn
            )
            ins, tomb = _process_papers(papers, subject, conn)
            total_inserted += ins

            new_offset = state["offset"] + ADVANCE_N
            _set_source_state(
                conn, subject, source_name,
                offset=new_offset,
                exhausted=exhausted,
            )
            print(f"  [{source_name}] +{ins} inserted +{tomb} tombstoned | exhausted={exhausted}")
            time.sleep(0.3)

        _advance_window(conn, subject, window_start)
        print(f"  total: {total_inserted} inserted for '{subject}'")

    conn.close()
    print("\n[populate] done.")

def unserved_count(subject: str) -> int:
    """Return count of unserved quality-approved papers with full text for a subject."""
    conn = get_conn()
    row  = conn.execute(
        """SELECT COUNT(*) as n FROM papers
           WHERE subject=? AND served=0 AND quality_ok=1 AND full_text_ok=1""",
        (subject,)
    ).fetchone()
    conn.close()
    return row["n"] if row else 0

# ─────────────────────────────────────────────
# 6) Reading list / saved DB operations
# ─────────────────────────────────────────────

def mark_served(doi: str) -> None:
    """Mark paper as served (shown as a card)."""
    conn = get_conn()
    conn.execute(
        "UPDATE papers SET served=1, date_served=? WHERE doi=?",
        (datetime.now().isoformat(timespec="seconds"), doi),
    )
    conn.commit()
    conn.close()

def get_reading_list() -> list[dict]:
    """Articles in reading list, sorted by generation time (newest first)."""
    conn = get_conn()
    rows = conn.execute(
        """SELECT doi, title, card_intro, subject, source, published_at, filename
           FROM papers WHERE in_reading_list=1 ORDER BY date_served DESC"""
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_saved() -> list[dict]:
    """Articles marked saved/liked, sorted by generation time (newest first)."""
    conn = get_conn()
    rows = conn.execute(
        """SELECT doi, title, card_intro, subject, source, published_at, filename
           FROM papers WHERE saved=1 ORDER BY date_served DESC"""
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def move_to_saved(dois: list[str]) -> None:
    """Move articles from reading list to saved/liked atomically."""
    if not dois:
        return
    conn = get_conn()
    conn.executemany(
        "UPDATE papers SET saved=1, in_reading_list=0 WHERE doi=?",
        [(doi,) for doi in dois],
    )
    conn.commit()
    conn.close()

def delete_articles(dois: list[str]) -> None:
    """Remove articles from reading list or saved. served stays 1 — paper never resurfaces as a card."""
    if not dois:
        return
    conn = get_conn()
    conn.executemany(
        "UPDATE papers SET in_reading_list=0, saved=0 WHERE doi=?",
        [(doi,) for doi in dois],
    )
    conn.commit()
    conn.close()

def mark_in_reading_list(doi: str, filename: str = "") -> None:
    """Mark paper as in reading list with filename. card_intro preserved from populate time."""
    conn = get_conn()
    conn.execute(
        """UPDATE papers
           SET in_reading_list=1, filename=?, date_served=?
           WHERE doi=?""",
        (filename, datetime.now().isoformat(timespec="seconds"), doi),
    )
    conn.commit()
    conn.close()

# ─────────────────────────────────────────────
# 7) Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    subjects = sys.argv[1:] or None
    populate(subjects)