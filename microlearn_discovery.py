from __future__ import annotations

import re
from pathlib import Path
from datetime import date, datetime
from typing import TypedDict, List, Optional, Annotated

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from populate_db import (
    get_conn as _get_db,
    mark_served as _mark_served,
    mark_in_reading_list as _mark_in_reading_list,
    DB_FILE,
    VALID_SUBJECTS,
)


# ─────────────────────────────────────────────
# 0) LLM
# ─────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()
llm = ChatGroq(model="llama-3.3-70b-versatile")


# ─────────────────────────────────────────────
# 1) Constants
# ─────────────────────────────────────────────

TARGET_CARDS  = 10
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 80
TOP_K         = 4

READING_LIST_DIR = DB_FILE.parent / "reading_list"



def _query_db_for_subject(subject: str, n: int) -> list[dict]:
    """
    Fetch n unserved quality-approved papers for subject from DB.
    Uses random seed for shuffled selection — same seed gives same order,
    different seeds surface different papers from the available pool.
    """
    conn = _get_db()
    rows = conn.execute(
        """SELECT doi, title, card_title, card_intro, subject, source,
                  published_at
           FROM papers
           WHERE subject=? AND served=0 AND quality_ok=1 AND full_text_ok=1
           ORDER BY RANDOM()
           LIMIT ?""",
        (subject, n)
    ).fetchall()
    conn.close()

    if not rows:
        return []

    return [dict(r) for r in rows]




# ─────────────────────────────────────────────
# 2) Schemas
# ─────────────────────────────────────────────

class ArticleCard(BaseModel):
    id:                 int = 0
    subject:            str
    title:              str
    intro:              str
    doi:                str
    source:             Optional[str] = None
    published_at:       Optional[str] = None


class Task(BaseModel):
    task_id:  str
    title:    str
    goal:     str
    bullets:  List[str] = Field(..., min_length=2, max_length=8)


class Plan(BaseModel):
    article_title: str
    doi:           str
    subject:       str
    tasks:         List[Task] = Field(..., min_length=4, max_length=8)


class Section(BaseModel):
    order:   int
    content: str


# ─────────────────────────────────────────────
# 3) RAG store
# ─────────────────────────────────────────────

class RAGStore:
    """
    Hybrid RAG: BM25 (keyword) + semantic (embedding) retrieval.
    Both passes return top_k chunks; results are unioned, deduplicated,
    then re-ranked by combined normalised score.
    Falls back to BM25-only when sentence-transformers is unavailable.
    """

    def __init__(self):
        self._chunks: dict[str, List[str]]    = {}
        self._embeddings: dict[str, list]     = {}
        self._bm25: dict[str, object]         = {}
        self._encoder                         = None
        self._use_embeddings                  = False
        self._try_load_encoder()

    def _try_load_encoder(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._encoder       = SentenceTransformer("all-MiniLM-L6-v2")
            self._use_embeddings = True
        except Exception:
            self._use_embeddings = False

    @staticmethod
    def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        words = text.split()
        if not words:
            return []
        chunks, i = [], 0
        while i < len(words):
            chunks.append(" ".join(words[i: i + size]))
            i += size - overlap
        return chunks

    def _build_bm25(self, key: str, chunks: List[str]):
        try:
            from rank_bm25 import BM25Okapi
            tokenised      = [re.findall(r"\w+", c.lower()) for c in chunks]
            self._bm25[key] = BM25Okapi(tokenised)
        except ImportError:
            self._bm25[key] = None   # fall back to keyword overlap scoring

    def index(self, key: str, full_text: str) -> int:
        chunks = self._chunk_text(full_text)
        if not chunks:
            return 0
        self._chunks[key] = chunks
        self._build_bm25(key, chunks)
        if self._use_embeddings:
            self._embeddings[key] = self._encoder.encode(
                chunks, show_progress_bar=False
            ).tolist()
        return len(chunks)

    def is_indexed(self, key: str) -> bool:
        return key in self._chunks

    def chunk_count(self, key: str) -> int:
        return len(self._chunks.get(key, []))

    def retrieve(self, key: str, query: str, top_k: int = TOP_K) -> List[str]:
        chunks = self._chunks.get(key, [])
        if not chunks:
            return []
        return self._retrieve_hybrid(key, query, top_k, chunks)

    # ── BM25 pass ────────────────────────────────────────────────────────────

    def _bm25_scores(self, key: str, query: str, chunks: List[str]):
        """Return normalised BM25 scores array (length == len(chunks))."""
        import numpy as np
        bm25 = self._bm25.get(key)
        if bm25 is not None:
            tokens = re.findall(r"\w+", query.lower())
            raw    = np.array(bm25.get_scores(tokens), dtype=float)
        else:
            # Fallback: simple term-overlap count
            terms = set(re.findall(r"\w+", query.lower()))
            raw   = np.array(
                [len(terms & set(re.findall(r"\w+", c.lower()))) for c in chunks],
                dtype=float,
            )
        mx = raw.max()
        return raw / mx if mx > 0 else raw

    # ── Semantic pass ─────────────────────────────────────────────────────────

    def _semantic_scores(self, key: str, query: str, chunks: List[str]):
        """Return normalised cosine-similarity scores array."""
        import numpy as np
        stored = np.array(self._embeddings[key])
        q_vec  = np.array(self._encoder.encode([query], show_progress_bar=False)[0])
        norms  = np.linalg.norm(stored, axis=1) * np.linalg.norm(q_vec)
        sims   = stored.dot(q_vec) / (norms + 1e-9)
        mn, mx = sims.min(), sims.max()
        return (sims - mn) / (mx - mn + 1e-9)

    # ── Hybrid retrieval ──────────────────────────────────────────────────────

    def _retrieve_hybrid(self, key: str, query: str, top_k: int, chunks: List[str]) -> List[str]:
        import numpy as np

        bm25_scores = self._bm25_scores(key, query, chunks)

        if self._use_embeddings and key in self._embeddings:
            sem_scores   = self._semantic_scores(key, query, chunks)
            combined     = 0.5 * bm25_scores + 0.5 * sem_scores
        else:
            combined = bm25_scores

        # top_k from each pass separately → union → re-rank by combined score
        bm25_top  = set(bm25_scores.argsort()[::-1][:top_k].tolist())
        if self._use_embeddings and key in self._embeddings:
            sem_top = set(sem_scores.argsort()[::-1][:top_k].tolist())
        else:
            sem_top = set()

        candidate_idx = sorted(bm25_top | sem_top, key=lambda i: combined[i], reverse=True)
        return [chunks[i] for i in candidate_idx[:top_k]]


rag_store: RAGStore = RAGStore()


# ─────────────────────────────────────────────
# 4) State
# ─────────────────────────────────────────────

def _sections_reducer(current: List, update: List) -> List:
    """Fan-out appends sections; None resets for next article."""
    if update is None:
        return []
    return current + update

def _articles_reducer(current: List, update: List) -> List:
    """Appends articles on normal updates; None resets (refresh or new selection)."""
    if update is None:
        return []
    return current + update


class State(TypedDict):
    as_of:                  str
    selected_cards:         List[ArticleCard]
    current_article_index:  int
    current_plan:           Optional[Plan]
    sections:               Annotated[List[Section], _sections_reducer]
    completed_articles:     Annotated[List[dict], _articles_reducer]
    full_text:              str


# ─────────────────────────────────────────────
# 5) Full-text fetch on demand (deep dive only)
# ─────────────────────────────────────────────

def fetch_and_index_for_deepdive(doi: str) -> str:
    """
    Fetches full_text from DB, indexes into RAG store, returns full_text for orchestrator planning.
    No-op index if already indexed.
    """
    conn = _get_db()
    row = conn.execute("SELECT full_text FROM papers WHERE doi=?", (doi,)).fetchone()
    conn.close()
    if not row:
        return ""
    full_text = row["full_text"] or ""
    if not rag_store.is_indexed(doi):
        if full_text:
            rag_store.index(doi, full_text)
        else:
            print(f"[deepdive] no full text for {doi} — skipping index")
    return full_text



# ─────────────────────────────────────────────
# 6) discovery_node
# ─────────────────────────────────────────────

def discovery_node(subjects: List[str]) -> List[ArticleCard]:
    n_subjects = len(subjects)

    base      = TARGET_CARDS // n_subjects
    remainder = TARGET_CARDS % n_subjects
    alloc     = [base + (1 if i < remainder else 0) for i in range(n_subjects)]

    new_cards: list[ArticleCard] = []
    next_id = 0

    for subject, n in zip(subjects, alloc):
        rows = _query_db_for_subject(subject, n)
        for row in rows:
            doi = row.get("doi")
            if not doi:
                continue
            card = ArticleCard(
                id           = next_id,
                subject      = subject,
                title        = row.get("card_title") or row.get("title") or "",
                intro        = row.get("card_intro") or "",
                doi          = doi,
                source       = row.get("source"),
                published_at = str(row["published_at"])[:10] if row.get("published_at") else None,
            )
            new_cards.append(card)
            next_id += 1
            _mark_served(doi)

    return new_cards







# ─────────────────────────────────────────────
# 7) fetcher_node
# ─────────────────────────────────────────────

def fetcher_node(state: State) -> dict:
    idx   = state.get("current_article_index", 0)
    cards = state.get("selected_cards", [])
    if idx >= len(cards):
        return {}

    full_text = fetch_and_index_for_deepdive(cards[idx].doi)
    return {"full_text": full_text}


# ─────────────────────────────────────────────
# 8) orchestrator_node
# ─────────────────────────────────────────────

ORCHESTRATOR_SYSTEM = """You are a senior editor planning a micro-learning article from a research paper.

Your reader is a curious non-expert — intelligent and motivated, but without domain training.

STRUCTURE:
  FIRST (mandatory): intro
    - task_id: "intro"
    - Combines: hook that makes the reader want to read on + why this finding matters to a non-expert

  MIDDLE (2-5 sections, mandatory coverage):
    - Decide count, names, and content freely based on what the paper actually contains
    - task_id: a short descriptive slug you choose (e.g. "study_design", "key_finding", "mechanism")
    - Must collectively cover all important aspects of the paper — methodology, findings, mechanism, context, implications
    - Only omit content that is deeply expert-level and has no bearing on understanding the core idea
    - Do NOT map sections to a fixed template — let the paper's structure dictate the sections

  LAST (mandatory): takeaways
    - task_id: "takeaways"
    - Key lessons the reader should walk away with

SECTION PLANNING PRINCIPLES:
  - Think of sections as logical steps that build on each other — each section should earn the next
  - Before assigning a section's bullets, ask: what does the reader need to already understand for this section to land?
    If the answer is "something from a previous section", make sure that prior section covers it
  - Order sections so prerequisite knowledge always comes before the section that needs it

FOR EACH SECTION PRODUCE:
  task_id   — short descriptive slug ('intro' and 'takeaways' fixed; middle sections free-form)
  title     — specific, engaging display heading (LLM decides freely)
  goal      — one sentence starting with a verb describing what this section achieves for the reader
  bullets   — 3–8 RAG queries (see RAG BULLET RULES below)

RAG BULLET RULES (critical — workers use each bullet directly as a retrieval query):
  Each bullet must contain BOTH:
    1. A keyword anchor — a specific term, measurement, concept, or named element from the paper
    2. A semantic question — the precise aspect the section needs to explain or establish
  Format: "<keyword anchor> — <specific question it answers>"

  The goal is to surface the exact chunks workers need — be specific enough that a semantic search
  returns the right passage, not a generic paragraph.

  Examples:
    BAD : "explain the methodology"
    GOOD: "cortisol assay timing — how and when did researchers collect samples and what controls were used"
    BAD : "discuss findings"
    GOOD: "hippocampal volume reduction — what magnitude of change was observed and in which participant subgroups"
    BAD : "background context"
    GOOD: "dopamine reward pathway — what is its normal function and why is its disruption relevant to this finding"

  For intro: first bullet should retrieve the core finding; remaining bullets should retrieve
  context needed to explain why it matters.
  For takeaways: bullets should retrieve the strongest evidence and implications to ground the lessons.
"""


def orchestrator_node(state: State) -> dict:
    idx   = state.get("current_article_index", 0)
    cards = state.get("selected_cards", [])
    if idx >= len(cards):
        print(f"[orchestrator] idx={idx} out of range (len={len(cards)}) — skipping")
        return {"current_plan": None}

    card = cards[idx]
    doi  = card.doi

    full_text = state.get("full_text", "")
    ft_words  = rag_store.chunk_count(doi)
    source_block = (
        f"Full text indexed for RAG retrieval ({ft_words} chunks). "
        f"Workers retrieve precise chunks — write bullets as specific RAG queries.\n\n"
        f"Full text (for planning context):\n{' '.join(full_text.split()[:12000])}"
    ) if ft_words else f"Full text:\n{' '.join(full_text.split()[:12000])}" if full_text else "No full text available."

    planner = llm.with_structured_output(Plan)
    plan    = planner.invoke([
        SystemMessage(content=ORCHESTRATOR_SYSTEM),
        HumanMessage(content=(
            f"as_of: {state['as_of']}\n"
            f"subject: {card.subject}\n"
            f"title: {card.title}\n"
            f"doi: {doi}\n"
            f"source: {card.source or 'unknown'}\n"
            f"published_at: {card.published_at or 'unknown'}\n\n"
            f"{source_block}\n\n"
            f"Decide how many sections this paper warrants and produce the plan now."
        )),
    ])

    plan.doi     = doi
    plan.subject = card.subject

    return {"current_plan": plan}


def fanout(state: State) -> list[Send]:
    plan = state.get("current_plan")
    if not plan:
        return []
    return [
        Send("worker", {
            "task":  task,
            "plan":  plan,
            "order": i,
            "as_of": state["as_of"],
        })
        for i, task in enumerate(plan.tasks)
    ]


# ─────────────────────────────────────────────
# 9) worker_node — abstract as primary source
# ─────────────────────────────────────────────

WORKER_SYSTEM = """You are a science writer for a micro-learning platform.
Your reader is intelligent but has no domain background. Your job is to build
understanding from the ground up — in logical order, never leaving an unexplained leap.

Hard constraints:
- Start with '## <Section Title>' using the exact title provided.
- Use the bullets as retrieval-guided content targets — each bullet identifies a concept or finding
  the section must cover. Do not treat them as subheadings or list them explicitly.
  Weave all bullet topics into cohesive prose.
- Output ONLY the section Markdown — no preamble, no commentary.

Source rules:
- RAG chunks are your ONLY factual source. Do not invent data, sample sizes, p-values,
  or outcomes not present in the retrieved chunks. If no chunks are provided, write only
  what the plan bullets state — do not fabricate specifics.

Logical chain rules (apply to every sentence):
- Before making any claim that depends on a concept, establish that concept inline
  in one sentence — just enough for the next step to click. Do not over-explain.
- Every sentence must either: introduce a necessary concept, advance the argument,
  or deliver a finding. Cut anything that does neither.
- Chain sentences so each one earns the next — no orphaned facts, no sudden jumps.
- Primary goal: preserve every piece of information that is part of the logical chain
  leading to the finding's conclusion, including prerequisite concepts needed to follow
  that chain. Do not cut important details or explanations.
- Secondary goal: cut repetition, hedging language, over-qualification, and tangential
  details that have no bearing on the core idea.

Length:
- Write exactly as long as the content demands — no padding, no repetition.
- If a concept needs one sentence, use one.
- Target the minimum word count that preserves the full logical chain and all evidence.
- Cut any sentence that doesn't introduce a concept, advance the argument, or deliver a finding.

Jargon policy:
- Rewrite technical terms into plain language wherever possible.
- Field-critical terms: keep, define inline on first use.
  Format: "term (plain-language explanation)"
- Never leave an undefined acronym or domain term.

Writing style:
- Short paragraphs (2–4 sentences). Confident, direct tone.
- No filler phrases: "it is worth noting", "importantly", "interestingly",
  "it goes without saying", "this suggests that".
- Concrete over abstract: prefer mechanisms, numbers, and examples over vague claims.
"""


def worker_node(payload: dict) -> dict:
    task  = payload["task"]
    plan  = payload["plan"]
    order = payload.get("order", 0)
    as_of = payload.get("as_of", "")

    context_block = ""
    if rag_store.is_indexed(plan.doi):
        retrieved: list[str] = []
        seen: set[str] = set()
        for bullet in task.bullets:
            for c in rag_store.retrieve(key=plan.doi, query=bullet, top_k=TOP_K):
                if c not in seen:
                    seen.add(c)
                    retrieved.append(c)
        if retrieved:
            context_block = (
                "\n\nRetrieved context (PRIMARY — hybrid BM25 + semantic):\n"
                + "\n\n---\n".join(f"[Chunk {i+1}]:\n{c}" for i, c in enumerate(retrieved))
            )
    else:
        print(f"[worker] no RAG index for {plan.doi} — section will lack source context")

    bullets_text = "\n".join(f"- {b}" for b in task.bullets)

    section_md: str = llm.invoke([
        SystemMessage(content=WORKER_SYSTEM),
        HumanMessage(content=(
            f"Article title  : {plan.article_title}\n"
            f"Subject        : {plan.subject}\n"
            f"DOI            : {plan.doi}\n"
            f"as_of          : {as_of}\n\n"
            f"Section task_id    : {task.task_id}\n"
            f"Section title      : {task.title}\n"
            f"Goal               : {task.goal}\n\n"
            f"Bullets to cover:\n{bullets_text}"
            f"{context_block}"
        )),
    ]).content

    return {"sections": [Section(order=order, content=section_md)]}


# ─────────────────────────────────────────────
# 10) reducer_node
# ─────────────────────────────────────────────

def reducer_node(state: State) -> dict:
    sorted_sections = sorted(state.get("sections", []), key=lambda s: s.order)

    idx  = state.get("current_article_index", 0)
    card = state["selected_cards"][idx]

    article_title = card.title
    subject       = card.subject
    doi           = card.doi
    source        = card.source       or ""
    published_at  = card.published_at or ""

    header = (
        f"# {article_title}\n\n"
        f"*Subject: {subject}*  \n"
        f"*DOI: [{doi}](https://doi.org/{doi})*  \n"
        f"*Source: {source}*  \n"
        + (f"*Published: {published_at}*\n\n" if published_at else "\n")
    )

    body     = "\n\n".join(s.content for s in sorted_sections)
    markdown = header + body

    article = {
        "title":    article_title,
        "subject":  subject,
        "doi":      doi,
        "source":   source,
        "intro":    card.intro,
        "markdown": markdown,
    }

    saved_path          = save_article(article)
    article["filename"] = saved_path.name

    return {
        "completed_articles":    [article],   # operator.add appends
        "current_article_index": idx + 1,
        "sections":              None,         # sentinel resets for next article
    }


# ─────────────────────────────────────────────
# 11) loop_condition
# ─────────────────────────────────────────────

def loop_condition(state: State) -> str:
    idx   = state["current_article_index"]
    total = len(state.get("selected_cards", []))
    return "fetcher" if idx < total else END


# ─────────────────────────────────────────────
# 12) Build graph
# ─────────────────────────────────────────────

g = StateGraph(State)

g.add_node("fetcher",      fetcher_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker",       worker_node)
g.add_node("reducer",      reducer_node)

g.add_edge(START,          "fetcher")
g.add_edge("fetcher",      "orchestrator")
g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker",       "reducer")
g.add_conditional_edges("reducer", loop_condition, {
    "fetcher": "fetcher",
    END:       END,
})

import sqlite3 as _sqlite3
_checkpointer = SqliteSaver(_sqlite3.connect(str(DB_FILE), check_same_thread=False))
app = g.compile(checkpointer=_checkpointer)


# ─────────────────────────────────────────────
# 13) Reading list persistence — DB-backed
# ─────────────────────────────────────────────

def _slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text.lower())
    text = re.sub(r"[\s_-]+", "-", text).strip("-")
    return text[:80] or "article"


def save_article(article: dict) -> Path:
    """Save markdown to reading_list/<slug>.md; idempotent on replay."""
    READING_LIST_DIR.mkdir(parents=True, exist_ok=True)

    slug    = _slugify(article["title"])
    md_path = READING_LIST_DIR / f"{slug}.md"
    counter = 2
    while md_path.exists():
        # Check if existing file is for the same DOI (replay) — skip write
        existing = md_path.read_text(encoding="utf-8")
        if article.get("doi", "") in existing:
            print(f"[reading_list] already exists (replay): {md_path.name}")
            return md_path
        md_path = READING_LIST_DIR / f"{slug}-{counter}.md"
        counter += 1

    md_path.write_text(article["markdown"], encoding="utf-8")

    if doi := article.get("doi"):
        _mark_in_reading_list(doi, md_path.name)

    print(f"[reading_list] saved: {md_path.name}")
    return md_path


# ─────────────────────────────────────────────
# 14) Public API
# ─────────────────────────────────────────────

def discovery(subjects: List[str]) -> List[ArticleCard]:
    """
    Calls discovery_node directly (no graph) and returns a list of ArticleCards.
    Selection happens in the UI before app.invoke is called per card.
    """
    subjects = [s for s in subjects if s in VALID_SUBJECTS]
    if not subjects:
        raise ValueError(f"No valid subjects. Choose from:\n{VALID_SUBJECTS}")

    cards = discovery_node(subjects)

    print(f"\n{'='*80}")
    print(f"SUBJECTS        : {subjects}")
    print(f"CARDS GENERATED : {len(cards)}")
    print(f"{'='*80}\n")
    for c in cards:
        print(f"[{c.id}] ({c.subject}) {c.title}")
        print(f"  INTRO  : {c.intro[:120]}...")
        print(f"  DOI    : {c.doi}")
        print(f"  SOURCE : {c.source}\n")

    return cards


def generate_article(card: ArticleCard, as_of: Optional[str] = None) -> dict:
    """
    Invoke the graph for a single card. Each call gets a fresh thread_id.
    Returns the completed article dict written to the reading list.
    """
    if not as_of:
        as_of = date.today().isoformat()

    thread_id = f"gen-{card.doi.replace('/', '_')}-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    config    = {"configurable": {"thread_id": thread_id}}

    initial_state: State = {
        "as_of":                 as_of,
        "selected_cards":        [card],
        "current_article_index": 0,
        "current_plan":          None,
        "sections":              [],
        "completed_articles":    [],
        "full_text":             "",
    }

    final_state = app.invoke(initial_state, config)
    completed   = final_state.get("completed_articles", [])
    return completed[0] if completed else {}