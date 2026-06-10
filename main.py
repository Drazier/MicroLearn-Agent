"""
main.py — Streamlit entry point for MicroLearn.
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from microlearn_discovery import (
    ArticleCard,
    READING_LIST_DIR,
    discovery,
    generate_article,
)
from populate_db import (
    init_db,
    get_reading_list,
    get_saved,
    move_to_saved,
    delete_articles,
    populate,
    unserved_count,
    VALID_SUBJECTS,
)


# ─────────────────────────────────────────────
# Subject → color mapping (19 subjects)
# ─────────────────────────────────────────────

SUBJECT_COLORS: dict[str, str] = {
    "human behavior":       "#4f86c6",
    "psychology":           "#7b5ea7",
    "persuasion":           "#c2666d",
    "anatomy":              "#4caf82",
    "physiology":           "#3d9e6e",
    "sales psychology":     "#e07b39",
    "archeology":           "#a0845c",
    "medicine":             "#3b8dbf",
    "endocrinology":        "#5bb8a0",
    "behavioral economics": "#c08d3a",
    "neuroscience":         "#9b59b6",
    "evolutionary biology": "#27ae8f",
    "anthropology":         "#b07d48",
    "cognitive science":    "#5472d3",
    "nutrition science":    "#6aaa3a",
    "sleep science":        "#4a7fbd",
    "geopolitics":          "#c0392b",
    "health & fitness":     "#2ecc71",
    "social sciences":      "#8e44ad",
}

def subject_tag_html(subject: str) -> str:
    color = SUBJECT_COLORS.get(subject.lower(), "#888888")
    return (
        f'<span style="background:{color};color:#fff;padding:2px 8px;'
        f'border-radius:12px;font-size:0.72rem;font-weight:600;'
        f'letter-spacing:0.03em;white-space:nowrap;">'
        f'{subject}</span>'
    )


# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────

def _inject_css() -> None:
    st.markdown("""
<style>
/* ── Tab nav ── */
div[data-testid="stHorizontalBlock"] > div > div[data-testid="stButton"] > button {
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 0.45rem 1.4rem;
    border: 1.5px solid #dee2e6;
    background: transparent;
    color: #555;
    transition: all .15s;
}
div[data-testid="stHorizontalBlock"] > div > div[data-testid="stButton"] > button:hover {
    background: #f0f4ff;
    border-color: #5472d3;
    color: #5472d3;
}
button[data-active="true"] {
    background: #5472d3 !important;
    color: #fff !important;
    border-color: #5472d3 !important;
}

/* ── Cards ── */
.ml-card {
    border: 1px solid #e2e6ea;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 4px;
    background: #fff;
    transition: box-shadow .15s;
}
.ml-card:hover { box-shadow: 0 2px 12px rgba(0,0,0,.08); }
.ml-card-title { font-weight: 700; font-size: 0.97rem; margin: 6px 0 4px; line-height: 1.35; }
.ml-card-intro { font-size: 0.85rem; color: #555; line-height: 1.5; margin-top: 6px; }
.ml-card-meta  { font-size: 0.75rem; color: #888; margin-top: 6px; }

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Disk helpers
# ─────────────────────────────────────────────

def get_article_markdown(filename: str) -> str:
    path = READING_LIST_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


# ─────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────

def _init_session() -> None:
    defaults = {
        "active_tab":         "reading_list",
        "subjects":           [],
        "cards":              [],
        "selected_card_ids":  set(),
        "generating":         False,
        "pending_cards":      [],
        "generation_as_of":   "",
        "reading_selected":   set(),
        "saved_selected":     set(),
        "rl_expanded_doi":    None,
        "saved_expanded_doi": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────
# Discovery actions
# ─────────────────────────────────────────────

UNSERVED_THRESHOLD = 5

def _ensure_db_stocked(subjects: list[str]) -> None:
    low = [s for s in subjects if unserved_count(s) < UNSERVED_THRESHOLD]
    if low:
        populate(low)

def action_start_discovery(subjects: list[str]) -> None:
    _ensure_db_stocked(subjects)
    cards = discovery(subjects)
    st.session_state.cards             = cards
    st.session_state.selected_card_ids = set()
    st.session_state.subjects          = subjects

def action_generate_articles() -> None:
    if not st.session_state.selected_card_ids:
        return
    selected_cards = [c for c in st.session_state.cards if c.id in st.session_state.selected_card_ids]
    st.session_state.cards             = [c for c in st.session_state.cards
                                           if c.id not in st.session_state.selected_card_ids]
    st.session_state.selected_card_ids = set()
    st.session_state.generating        = True
    st.session_state.pending_cards     = selected_cards
    st.session_state.generation_as_of  = datetime.now().strftime("%Y-%m-%d")
    st.session_state.active_tab        = "reading_list"

def action_select_all_cards() -> None:
    st.session_state.selected_card_ids = {c.id for c in st.session_state.cards}

# ─────────────────────────────────────────────
# Reading list actions
# ─────────────────────────────────────────────

def action_select_all_reading() -> None:
    st.session_state.reading_selected = {r["doi"] for r in get_reading_list()}

def action_delete_reading() -> None:
    delete_articles(list(st.session_state.reading_selected))
    st.session_state.reading_selected = set()
    st.session_state.rl_expanded_doi  = None

def action_move_to_saved() -> None:
    move_to_saved(list(st.session_state.reading_selected))
    st.session_state.reading_selected = set()
    st.session_state.rl_expanded_doi  = None

# ─────────────────────────────────────────────
# Saved actions
# ─────────────────────────────────────────────

def action_select_all_saved() -> None:
    st.session_state.saved_selected = {r["doi"] for r in get_saved()}

def action_delete_saved() -> None:
    delete_articles(list(st.session_state.saved_selected))
    st.session_state.saved_selected     = set()
    st.session_state.saved_expanded_doi = None


# ─────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────

def _tab_nav() -> None:
    """Render top 3 tab buttons."""
    tabs       = [("discover", "🔍 Discover"), ("reading_list", "📖 Reading List"), ("saved", "⭐ Saved")]
    active     = st.session_state.active_tab
    generating = st.session_state.generating
    cols       = st.columns([1, 1, 1, 4])
    for i, (key, label) in enumerate(tabs):
        with cols[i]:
            frozen = generating and key != "reading_list"
            if st.button(label, key=f"nav_{key}", use_container_width=True,
                         type="primary" if active == key else "secondary",
                         disabled=frozen):
                st.session_state.active_tab = key
                st.rerun()


def _render_discover_card(card: ArticleCard, idx: int) -> None:
    """Render a single discover card with checkbox."""
    selected = st.session_state.selected_card_ids
    checked  = card.id in selected

    with st.container():
        st.markdown(f'<div class="ml-card">', unsafe_allow_html=True)
        col_chk, col_body = st.columns([0.06, 0.94])

        with col_chk:
            new_val = st.checkbox("", value=checked, key=f"card_chk_{card.id}_{idx}",
                                  label_visibility="collapsed")
            if new_val != checked:
                if new_val:
                    st.session_state.selected_card_ids.add(card.id)
                else:
                    st.session_state.selected_card_ids.discard(card.id)

        with col_body:
            st.markdown(subject_tag_html(card.subject), unsafe_allow_html=True)
            st.markdown(f'<div class="ml-card-title">{card.title}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ml-card-intro">{card.intro}</div>', unsafe_allow_html=True)
            meta_parts = []
            if card.source:
                meta_parts.append(card.source)
            if card.published_at:
                meta_parts.append(card.published_at[:4])
            if meta_parts:
                st.markdown(f'<div class="ml-card-meta">{" · ".join(meta_parts)}</div>',
                            unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def _render_article_card(article: dict, selected_key: str, expanded_key: str) -> None:
    """Render a reading-list / saved card with checkbox + expandable markdown."""
    doi      = article["doi"]
    selected = st.session_state[selected_key]
    checked  = doi in selected
    expanded = st.session_state[expanded_key] == doi

    with st.container():
        st.markdown('<div class="ml-card">', unsafe_allow_html=True)
        col_chk, col_body = st.columns([0.06, 0.94])

        with col_chk:
            safe_doi = doi.replace("/", "_").replace(".", "_")
            new_val  = st.checkbox("", value=checked,
                                   key=f"{selected_key}_chk_{safe_doi}",
                                   label_visibility="collapsed")
            if new_val != checked:
                if new_val:
                    st.session_state[selected_key].add(doi)
                else:
                    st.session_state[selected_key].discard(doi)

        with col_body:
            subject = article.get("subject", "")
            if subject:
                st.markdown(subject_tag_html(subject), unsafe_allow_html=True)
            st.markdown(f'<div class="ml-card-title">{article["title"]}</div>',
                        unsafe_allow_html=True)
            intro = article.get("card_intro", "")
            if intro:
                st.markdown(f'<div class="ml-card-intro">{intro}</div>',
                            unsafe_allow_html=True)
            meta_parts = []
            if article.get("source"):
                meta_parts.append(article["source"])
            if article.get("published_at"):
                meta_parts.append(str(article["published_at"])[:4])
            if meta_parts:
                st.markdown(f'<div class="ml-card-meta">{" · ".join(meta_parts)}</div>',
                            unsafe_allow_html=True)

            toggle_label = "▲ Hide article" if expanded else "▼ Read article"
            if st.button(toggle_label, key=f"{selected_key}_expand_{safe_doi}",
                         use_container_width=False):
                st.session_state[expanded_key] = None if expanded else doi
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    if expanded:
        filename = article.get("filename", "")
        if filename:
            md = get_article_markdown(filename)
            if md:
                with st.expander("", expanded=True):
                    st.markdown(md)
            else:
                st.info("Article content not found on disk.")
        else:
            st.info("No article file linked yet.")


# ─────────────────────────────────────────────
# Tab renderers
# ─────────────────────────────────────────────

def _render_discover() -> None:
    st.markdown("### 🔍 Discover")

    # Subject multiselect
    chosen = st.multiselect(
        "Choose subjects to explore",
        options=VALID_SUBJECTS,
        default=st.session_state.subjects,
        placeholder="Select one or more subjects…",
    )

    if not chosen:
        st.session_state.subjects = []
        st.info("Select at least one subject above to start discovering articles.")
        return

    if st.button("🃏 Show New Cards", key="disc_show", type="primary"):
        with st.spinner("Loading cards…"):
            action_start_discovery(chosen)
        st.rerun()

    cards: list[ArticleCard] = st.session_state.cards
    selected: set            = st.session_state.selected_card_ids

    if not cards:
        st.info("Press **Show New Cards** to load articles for your selected subjects.")
        return

    # Action buttons — all disabled while generating
    generating = st.session_state.generating
    col1, col2, _ = st.columns([1, 1.3, 5])
    with col1:
        if st.button("☑ Select All", key="disc_select_all", use_container_width=True, disabled=generating):
            action_select_all_cards()
            st.rerun()
    with col2:
        disabled = len(selected) == 0 or generating
        if st.button("✨ Generate Articles", key="disc_generate",
                     use_container_width=True, disabled=disabled):
            action_generate_articles()
            st.rerun()

    if selected:
        st.caption(f"{len(selected)} card(s) selected")

    # Card grid
    if cards:
        cols = st.columns(2)
        for i, card in enumerate(cards):
            with cols[i % 2]:
                _render_discover_card(card, i)


def _render_reading_list() -> None:
    st.markdown("### 📖 Reading List")

    # ── One-at-a-time streaming generation ──
    if st.session_state.generating and st.session_state.pending_cards:
        card    = st.session_state.pending_cards[0]
        remaining = len(st.session_state.pending_cards)
        with st.spinner(f"Generating article {card.title[:60]}… ({remaining} left)"):
            generate_article(card, st.session_state.generation_as_of)
            st.session_state.pending_cards = st.session_state.pending_cards[1:]
        if not st.session_state.pending_cards:
            st.session_state.generating = False
        st.rerun()

    articles = get_reading_list()
    selected: set = st.session_state.reading_selected
    generating    = st.session_state.generating

    if not articles and not generating:
        st.info("Your reading list is empty. Generate articles in the Discover tab.")
        return

    if generating:
        remaining = len(st.session_state.pending_cards)
        st.info(f"⏳ Generating… {remaining} article(s) remaining. Already-generated articles are readable below.")

    # Action buttons — frozen while generating; expand toggles are NOT buttons so always work
    col1, col2, col3, _ = st.columns([1, 1, 1, 4])
    with col1:
        if st.button("☑ Select All", key="rl_select_all", use_container_width=True, disabled=generating):
            action_select_all_reading()
            st.rerun()
    with col2:
        disabled = len(selected) == 0 or generating
        if st.button("🗑 Delete", key="rl_delete",
                     use_container_width=True, disabled=disabled):
            action_delete_reading()
            st.rerun()
    with col3:
        disabled = len(selected) == 0 or generating
        if st.button("⭐ Save", key="rl_save",
                     use_container_width=True, disabled=disabled):
            action_move_to_saved()
            st.rerun()

    if selected:
        st.caption(f"{len(selected)} article(s) selected")

    for article in articles:
        _render_article_card(article,
                             selected_key="reading_selected",
                             expanded_key="rl_expanded_doi")


def _render_saved() -> None:
    st.markdown("### ⭐ Saved")
    articles = get_saved()
    selected: set = st.session_state.saved_selected

    if not articles:
        st.info("No saved articles yet. Save articles from your Reading List.")
        return

    # Action buttons
    col1, col2, _ = st.columns([1, 1, 5])
    with col1:
        if st.button("☑ Select All", key="saved_select_all", use_container_width=True):
            action_select_all_saved()
            st.rerun()
    with col2:
        disabled = len(selected) == 0
        if st.button("🗑 Delete", key="saved_delete",
                     use_container_width=True, disabled=disabled):
            action_delete_saved()
            st.rerun()

    if selected:
        st.caption(f"{len(selected)} article(s) selected")

    for article in articles:
        _render_article_card(article,
                             selected_key="saved_selected",
                             expanded_key="saved_expanded_doi")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main() -> None:
    init_db()
    _init_session()

    st.set_page_config(page_title="MicroLearn", layout="wide")
    _inject_css()

    _tab_nav()
    st.divider()

    tab = st.session_state.active_tab
    if tab == "discover":
        _render_discover()
    elif tab == "reading_list":
        _render_reading_list()
    elif tab == "saved":
        _render_saved()


if __name__ == "__main__":
    main()