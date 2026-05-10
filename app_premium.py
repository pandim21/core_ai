"""app_premium.py — C.O.R.E. AI │ Premium Analytical Dashboard.

Completely re-designed UI with:
  • Animated agent-pipeline sidebar
  • ESG risk radar chart (Plotly)
  • Financial overview bar chart (Plotly)
  • Colour-coded stage-gate decision banner
  • COO key-rationale cards
  • Per-agent tabbed report sections
  • Downloadable markdown report

Run with:
    streamlit run app_premium.py
"""

from __future__ import annotations

import queue
import re
import threading
import time
from datetime import datetime

import plotly.graph_objects as go
import streamlit as st

from coreai_ import run_evaluation, _strip_inline_markdown, list_prospects

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="C.O.R.E. AI — Mining Intelligence Platform",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ──────────────────────────────────────────────────────────── */
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
[data-testid="stAppViewContainer"] { background: #080c18; }
[data-testid="stSidebar"]          { background: #05080f; border-right: 1px solid #1a2540; }
[data-testid="stSidebar"] *        { color: #c5d0e6 !important; }

/* ── Typography ──────────────────────────────────────────────────────── */
h1, h2, h3 { color: #e8edf8 !important; }

/* ── AMD badge ───────────────────────────────────────────────────────── */
.amd-badge {
    background: linear-gradient(90deg,#ED1C24,#c01018);
    color:#fff; font-size:.65rem; font-weight:800;
    letter-spacing:.12em; padding:3px 10px; border-radius:4px;
    display:inline-block; margin-bottom:4px;
}

/* ── Sidebar agent rows ──────────────────────────────────────────────── */
.ag-row {
    display:flex; align-items:center; gap:10px;
    padding:9px 0; border-bottom:1px solid #1a2540;
}
.ag-row:last-child { border-bottom:none; }
.ag-icon  { font-size:1.25rem; }
.ag-role  { font-size:.82rem; font-weight:600; color:#e2e8f0; }
.ag-model { font-size:.7rem;  font-family:'Courier New',monospace; color:#f0c040; }

/* ── Decision banner ─────────────────────────────────────────────────── */
.banner {
    border-radius:12px; padding:26px 32px; margin-bottom:20px;
    display:flex; align-items:center; gap:22px;
}
.banner-emoji  { font-size:2.8rem; }
.banner-verdict{ font-size:1.9rem; font-weight:800; letter-spacing:.05em; }
.banner-sub    { font-size:.9rem;  opacity:.85; margin-top:4px; }

.proceed-pea { background:linear-gradient(135deg,#0a2e16,#0f3d20); color:#b7f5c8;
               border:1px solid #1a7a35; }
.proceed-pfs { background:linear-gradient(135deg,#08204a,#0d2d60); color:#c3d9ff;
               border:1px solid #2a5abb; }
.proceed-dfs { background:linear-gradient(135deg,#1e0a50,#280d6a); color:#ddd0ff;
               border:1px solid #6035cc; }
.reject      { background:linear-gradient(135deg,#3a0a0a,#520d0d); color:#ffd0d0;
               border:1px solid #cc2020; }
.unknown     { background:linear-gradient(135deg,#111827,#1f2937); color:#d1d5db;
               border:1px solid #374151; }

/* ── Metric cards ────────────────────────────────────────────────────── */
.metric-card {
    background:#0e1425; border:1px solid #1a2540;
    border-radius:10px; padding:16px 18px; text-align:center;
}
.metric-label { font-size:.68rem; text-transform:uppercase;
                letter-spacing:.1em; color:#5a7aaa; margin-bottom:6px; }
.metric-value { font-size:1.05rem; font-weight:700; color:#e8edf8; }

/* ── Rationale cards ─────────────────────────────────────────────────── */
.rationale-card {
    background:#0e1425; border-left:4px solid #ED1C24;
    border-radius:0 8px 8px 0; padding:14px 18px; margin-bottom:12px;
}
.rationale-num { font-size:.75rem; color:#ED1C24; font-weight:800;
                 text-transform:uppercase; letter-spacing:.1em; margin-bottom:4px; }
.rationale-text{ font-size:.88rem; color:#c5d0e6; line-height:1.6; }

/* ── Section container (tab content) ────────────────────────────────── */
.section-box {
    background:#0e1425; border:1px solid #1a2540; border-radius:8px;
    padding:22px 26px; font-size:.875rem; line-height:1.8;
    color:#c5d0e6; white-space:pre-wrap;
}

/* ── Landing cards (how it works) ────────────────────────────────────── */
.lcard {
    background:#0e1425; border:1px solid #1a2540; border-radius:12px;
    padding:22px; text-align:center; height:100%;
}
.lcard-icon  { font-size:2.2rem; margin-bottom:10px; }
.lcard-title { font-size:.95rem; font-weight:700; color:#e8edf8; margin-bottom:6px; }
.lcard-model { font-size:.72rem; font-family:'Courier New',monospace;
               color:#f0c040; margin-bottom:8px; }
.lcard-desc  { font-size:.78rem; color:#64748b; line-height:1.6; }
.arrow       { font-size:1.5rem; color:#1a2540;
               display:flex; align-items:center; justify-content:center; }

/* ── Tabs ────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap:4px; background:#0e1425;
    border-radius:8px 8px 0 0; padding:4px 4px 0;
    border-bottom:1px solid #1a2540;
}
.stTabs [data-baseweb="tab"] {
    border-radius:6px 6px 0 0; padding:8px 18px;
    font-size:.82rem; color:#64748b !important;
}
.stTabs [aria-selected="true"] {
    background:#080c18; color:#e8edf8 !important;
    border-bottom:2px solid #ED1C24;
}

/* ── Sidebar history button ──────────────────────────────────────────── */
.stButton>button {
    background:#0e1425 !important; border:1px solid #1a2540 !important;
    color:#c5d0e6 !important; border-radius:6px !important;
    font-size:.8rem !important;
}
.stButton>button:hover {
    border-color:#ED1C24 !important; color:#fff !important;
}

/* ── Live evaluation checklist ───────────────────────────────────────── */
.ck-card {
    background: linear-gradient(180deg,#0c1322 0%,#0a1020 100%);
    border: 1px solid #1a2540;
    border-radius: 14px;
    padding: 20px 24px 16px;
    margin: 6px 0 22px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.35);
}
.ck-head {
    display: flex; align-items: baseline; justify-content: space-between;
    margin-bottom: 14px;
}
.ck-title {
    font-size: .82rem; font-weight: 700; color: #e8edf8;
    letter-spacing: .04em;
}
.ck-elapsed {
    font-size: .72rem; color: #5a7aaa;
    font-family: 'Courier New', monospace;
}
.ck-list { display: flex; flex-direction: column; gap: 0; }
.ck-row {
    display: flex; align-items: center; gap: 14px;
    padding: 12px 4px;
    border-bottom: 1px solid #14223a;
    transition: opacity .25s ease, color .25s ease;
}
.ck-row:last-child { border-bottom: none; }
.ck-row.pending  { opacity: .42; }
.ck-row.running  { opacity: 1; }
.ck-row.done     { opacity: 1; }
.ck-status {
    width: 22px; height: 22px;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 1rem; font-weight: 800;
    border-radius: 50%;
    flex-shrink: 0;
}
.ck-row.pending .ck-status { color: #5a7aaa; border: 1px solid #1a2540; }
.ck-row.running .ck-status { color: #f0c040; border: 1px solid #f0c040; background: rgba(240,192,64,0.08); }
.ck-row.done    .ck-status { color: #fff; background: #16a34a; border: 1px solid #16a34a; }
.ck-icon { font-size: 1.35rem; line-height: 1; }
.ck-text { flex: 1; min-width: 0; }
.ck-label { font-size: .92rem; font-weight: 600; color: #e8edf8; }
.ck-row.pending .ck-label { color: #94a3b8; }
.ck-model {
    font-size: .7rem; font-family: 'Courier New', monospace;
    color: #f0c040; margin-top: 2px;
}
.ck-row.pending .ck-model { color: #475569; }
.ck-state {
    font-size: .7rem; text-transform: uppercase;
    letter-spacing: .12em; font-weight: 700;
}
.ck-row.pending .ck-state { color: #475569; }
.ck-row.running .ck-state { color: #f0c040; }
.ck-row.done    .ck-state { color: #22c55e; }
.ck-spinner {
    display: inline-block;
    width: 12px; height: 12px;
    border: 2px solid #f0c040;
    border-right-color: transparent;
    border-radius: 50%;
    animation: ck-spin .75s linear infinite;
}
@keyframes ck-spin { to { transform: rotate(360deg); } }
.ck-footer {
    margin-top: 12px; padding-top: 12px;
    border-top: 1px solid #14223a;
    font-size: .72rem; color: #64748b;
    display: flex; justify-content: space-between;
}
.ck-footer .ck-done-msg { color: #22c55e; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
for key, default in [("reports", []), ("active_index", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────────────────────────
# Agent roster
# ─────────────────────────────────────────────────────────────────────────────
# (role_key, icon, role_label, model_label, summary)
# Model labels mirror the assignments in .env exactly so the sidebar and
# the live progress checklist never drift from the actual deployed stack.
# role_key matches the value coreai_.run_evaluation passes to its
# progress_callback ("geology" / "env" / "economy" / "coo"); see
# _render_progress_checklist for how it's consumed.
AGENTS = [
    ("geology", "🔬", "Senior Exploration Geologist", "Mixtral-8x22B",
     "JORC 2012-compliant resource classification and drill interpretation."),
    ("env",     "🌿", "Mining & Environmental Engineer", "Mistral-Small-24B",
     "IFC PS1–PS8, GISTM, ICMM, FPIC under ILO 169."),
    ("economy", "📊", "Senior Mining Economist", "Qwen-2.5-72B",
     "Scoping CAPEX/OPEX, IRR/NPV, jurisdiction risk premium."),
    ("coo",     "🏢", "Chief Operating Officer", "Llama-3.3-70B",
     "Stage-gate PROCEED / REJECT with eight-point fatal-flaw test."),
]
AGENT_ROLE_ORDER = tuple(a[0] for a in AGENTS)
AGENT_BY_ROLE    = {a[0]: a for a in AGENTS}

# ─────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_sections(report: str) -> dict[str, str]:
    pattern = re.compile(r"^## (.+?)$", re.MULTILINE)
    matches = list(pattern.finditer(report))
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        name  = m.group(1).strip()
        start = m.end()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(report)
        sections[name] = report[start:end].strip()
    return sections


def _parse_briefing_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            fields[k.strip()] = v.strip()
    return fields


def _parse_recommendation(coo: str) -> tuple[str, str, str]:
    m = re.search(r"Recommendation:\s*(PROCEED\s+to\s+\w+|REJECT)", coo, re.I)
    if not m:
        return "EVALUATING…", "unknown", "🔄"
    rec = m.group(1).strip().upper()
    if "REJECT" in rec: return "REJECT",        "reject",      "🚫"
    if "DFS"    in rec: return "PROCEED to DFS", "proceed-dfs", "✅"
    if "PFS"    in rec: return "PROCEED to PFS", "proceed-pfs", "✅"
    return "PROCEED to PEA", "proceed-pea", "✅"


def _parse_stage_gate(coo: str) -> str:
    m = re.search(r"Stage-Gate Position:\s*(.+)", coo)
    return m.group(1).strip() if m else ""


def _parse_rationale(coo: str) -> list[str]:
    """Return up to 3 numbered rationale points from the COO output."""
    block_m = re.search(r"Key\s+Rationale\s*:\s*(.*?)$", coo, re.DOTALL | re.I)
    if not block_m:
        return []
    block = block_m.group(1).strip()
    points = re.findall(r"\d+[.)]\s+(.+?)(?=\n\s*\d+[.)]|\Z)", block, re.DOTALL)
    return [p.strip().replace("\n", " ") for p in points[:3]]


def _compute_esg_scores(env_text: str, coo_text: str) -> dict[str, float]:
    """Keyword-weighted ESG risk scores (1 = low, 5 = high)."""
    corpus = (env_text + " " + coo_text).lower()

    def _score(base_words: list[str], high_words: list[str]) -> float:
        base  = sum(1.0 for w in base_words if w in corpus)
        high  = sum(2.0 for w in high_words if w in corpus)
        return min(5.0, max(1.0, round((base + high) / max(len(base_words), 1) * 2.5, 1)))

    return {
        "Ecological":   _score(
            ["biodiversity","habitat","ps6","protected","species","endemic"],
            ["critical habitat","no-go","gistm","unresolvable","severely"]),
        "Water & Tailings": _score(
            ["tailings","water","drainage","balance","gistm","tailing"],
            ["catastrophic","acid","contamination","scarcity","toxic"]),
        "ARD / AMD":    _score(
            ["ard","amd","acid rock","acid mine","sulphide","pyrite"],
            ["catastrophic","severe","high potential","unmanageable"]),
        "Social Licence": _score(
            ["indigenous","fpic","community","ilo","undrip","traditional"],
            ["opposition","conflict","denied","refused","cannot proceed"]),
        "Permitting":   _score(
            ["permit","eia","esia","regulatory","approval","timeline"],
            ["blocked","refused","no pathway","10 years","unresolvable"]),
        "ESG Finance":  _score(
            ["equator","ifc","sasb","ep4","esg","financing","barrier"],
            ["disqualify","fail","stranded","unable to finance"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────────────────────────────────────
_PAPER  = "rgba(0,0,0,0)"
_PLOT   = "#0e1425"
_GRID   = "#1a2540"
_TEXT   = "#c5d0e6"


def _esg_radar(scores: dict[str, float]) -> go.Figure:
    cats   = list(scores.keys()) + [list(scores.keys())[0]]   # close the polygon
    vals   = list(scores.values()) + [list(scores.values())[0]]

    fig = go.Figure(go.Scatterpolar(
        r      = vals,
        theta  = cats,
        fill   = "toself",
        fillcolor = "rgba(237,28,36,0.20)",
        line   = dict(color="#ED1C24", width=2),
        marker = dict(size=6, color="#ED1C24"),
        name   = "Risk Level",
    ))
    fig.update_layout(
        paper_bgcolor = _PAPER,
        plot_bgcolor  = _PAPER,
        font          = dict(color=_TEXT, size=11),
        margin        = dict(l=40, r=40, t=30, b=30),
        polar = dict(
            bgcolor      = _PLOT,
            radialaxis   = dict(
                visible   = True, range=[0, 5],
                gridcolor = _GRID, linecolor=_GRID,
                tickfont  = dict(color="#5a7aaa", size=9),
                tickvals  = [1, 2, 3, 4, 5],
                ticktext  = ["Min", "Low", "Med", "High", "Max"],
            ),
            angularaxis  = dict(
                gridcolor = _GRID, linecolor=_GRID,
                tickfont  = dict(color=_TEXT, size=10),
            ),
        ),
        showlegend = False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Live progress checklist
# ─────────────────────────────────────────────────────────────────────────────

def _render_progress_checklist(
    completed: set[str],
    elapsed_s: int,
    finished: bool = False,
) -> str:
    """Render the per-agent progress card as a single HTML string.

    ``completed`` is the set of role keys ('geology', 'env', 'economy', 'coo')
    whose tasks have already returned. Within the iteration we mark every
    completed role as ``done``, the first uncompleted role as ``running``,
    and the rest as ``pending``. When ``finished`` is True nothing is marked
    running — used for the final frame after kickoff returns successfully.
    """
    rows = []
    found_running = False
    for role_key in AGENT_ROLE_ORDER:
        _, icon, role_label, model_label, _summary = AGENT_BY_ROLE[role_key]
        if role_key in completed:
            state_cls, state_text = "done", "Done"
            status_html = "✓"
        elif not finished and not found_running:
            state_cls, state_text = "running", "Running"
            status_html = '<span class="ck-spinner"></span>'
            found_running = True
        else:
            state_cls, state_text = "pending", "Queued"
            status_html = "○"
        rows.append(
            f'<div class="ck-row {state_cls}">'
            f'  <span class="ck-status">{status_html}</span>'
            f'  <span class="ck-icon">{icon}</span>'
            f'  <div class="ck-text">'
            f'    <div class="ck-label">{role_label}</div>'
            f'    <div class="ck-model">{model_label}</div>'
            f'  </div>'
            f'  <span class="ck-state">{state_text}</span>'
            f'</div>'
        )

    n_done = len(completed & set(AGENT_ROLE_ORDER))
    n_total = len(AGENT_ROLE_ORDER)
    if finished:
        footer = (
            f'<span>Pipeline complete</span>'
            f'<span class="ck-done-msg">✓ {n_done}/{n_total} agents · '
            f'{elapsed_s}s total</span>'
        )
    else:
        footer = (
            f'<span>Sequential agent pipeline · AMD Instinct MI300X</span>'
            f'<span>{n_done}/{n_total} done · {elapsed_s}s elapsed</span>'
        )

    return (
        '<div class="ck-card">'
        '  <div class="ck-head">'
        '    <span class="ck-title">Multi-Agent Evaluation</span>'
        f'    <span class="ck-elapsed">{elapsed_s:>3}s</span>'
        '  </div>'
        f'  <div class="ck-list">{"".join(rows)}</div>'
        f'  <div class="ck-footer">{footer}</div>'
        '</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Landing page (no reports yet)
# ─────────────────────────────────────────────────────────────────────────────

def _render_landing() -> None:
    st.markdown("""
    <div style='text-align:center;padding:40px 0 20px;'>
      <div style='font-size:3.5rem;'>⛏️</div>
      <h1 style='font-size:2.2rem;font-weight:900;letter-spacing:.03em;
                 margin:8px 0 4px;color:#e8edf8;'>C.O.R.E. AI</h1>
      <p style='color:#5a7aaa;font-size:.95rem;margin:0;'>
        Collaborative Operational Risk Evaluator &nbsp;·&nbsp;
        AMD MI300X &nbsp;·&nbsp; CrewAI Multi-Agent Pipeline
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    st.markdown(
        "<p style='text-align:center;color:#64748b;font-size:.8rem;"
        "text-transform:uppercase;letter-spacing:.12em;margin-bottom:12px;'>"
        "How it works</p>",
        unsafe_allow_html=True,
    )

    cols = st.columns([3, 1, 3, 1, 3, 1, 3])
    for i, (_role_key, icon, role, model, desc) in enumerate(AGENTS):
        with cols[i * 2]:
            st.markdown(f"""
            <div class="lcard">
              <div class="lcard-icon">{icon}</div>
              <div class="lcard-title">{role}</div>
              <div class="lcard-model">{model}</div>
              <div class="lcard-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
        if i < len(AGENTS) - 1:
            with cols[i * 2 + 1]:
                st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, title, body in [
        (c1, "🛡️  Guardrail Validation",
         "Every agent output passes a deterministic Python validator "
         "before continuing. JORC violations, LaTeX leakage, and PEA/PFS "
         "conflation are caught and force a retry."),
        (c2, "🚫  Genuine REJECT Capability",
         "The COO applies a six-point fatal-flaw test and issues REJECT "
         "when economics, environment, jurisdiction, or social licence "
         "cannot support advancement — not just PROCEED."),
        (c3, "⚡  AMD MI300X — 192 GB HBM3",
         "All four open-weight models are loaded simultaneously in the "
         "MI300X's unified memory pool, enabling true heterogeneous "
         "multi-agent inference without GPU swapping."),
    ]:
        with col:
            st.markdown(f"""
            <div class="lcard" style='text-align:left;'>
              <div class="lcard-title" style='font-size:1rem;margin-bottom:8px;'>
                {title}</div>
              <div class="lcard-desc" style='font-size:.82rem;color:#94a3b8;
                line-height:1.65;'>{body}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:48px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;color:#374151;font-size:.8rem;'>"
        "Select a prospect in the sidebar and click ▶ Run Evaluation to begin.</p>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Report renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_report(record: dict) -> None:
    content  = record["content"]
    sections = _parse_sections(content)
    bf       = _parse_briefing_fields(sections.get("Prospect Briefing", ""))
    coo_text = sections.get("COO Executive Summary", "")
    env_text = sections.get("Environmental Impact",  "")
    fin_text = sections.get("Financial Briefing",    "")

    verdict, css_cls, icon = _parse_recommendation(coo_text)
    stage_gate             = _parse_stage_gate(coo_text)

    # ── Decision banner ──────────────────────────────────────────────────
    st.markdown(f"""
    <div class="banner {css_cls}">
      <div class="banner-emoji">{icon}</div>
      <div>
        <div class="banner-verdict">{verdict}</div>
        <div class="banner-sub">{stage_gate or record['prospect']}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Prospect info metrics ────────────────────────────────────────────
    meta = [
        ("Prospect",     bf.get("Prospect Name",           "—")),
        ("Country",      bf.get("Country",                 "—")),
        ("Commodity",    bf.get("Target Commodity",        "—")),
        ("Depth (m)",    bf.get("Est. Depth to Target (m)","—")),
        ("Logistics km", bf.get("Logistics Distance (km)", "—")),
    ]
    cols = st.columns(len(meta))
    for col, (label, val) in zip(cols, meta):
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{val or "—"}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── ESG risk chart ───────────────────────────────────────────────────
    st.markdown(
        "<p style='font-size:.72rem;text-transform:uppercase;"
        "letter-spacing:.1em;color:#5a7aaa;margin-bottom:4px;'>"
        "ESG Risk Profile</p>",
        unsafe_allow_html=True,
    )
    scores = _compute_esg_scores(env_text, coo_text)
    st.plotly_chart(
        _esg_radar(scores),
        width="stretch",
        config={"displayModeBar": False},
    )

    # ── Key rationale cards ──────────────────────────────────────────────
    rationale = _parse_rationale(_strip_inline_markdown(coo_text))
    if rationale:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:.72rem;text-transform:uppercase;"
            "letter-spacing:.1em;color:#5a7aaa;margin-bottom:8px;'>"
            "Key Rationale</p>",
            unsafe_allow_html=True,
        )
        for i, point in enumerate(rationale, 1):
            st.markdown(f"""
            <div class="rationale-card">
              <div class="rationale-num">Rationale {i}</div>
              <div class="rationale-text">{point}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Tabbed report sections ───────────────────────────────────────────
    tab_g, tab_e, tab_f, tab_c = st.tabs([
        "🔬  Geological",
        "🌿  Environmental",
        "📊  Financial",
        "🏢  COO Decision",
    ])

    def _tab_content(section_name: str) -> str:
        raw = sections.get(section_name, f"No {section_name.lower()} output found.")
        return _strip_inline_markdown(raw)

    with tab_g:
        st.markdown(
            f'<div class="section-box">{_tab_content("Geological Assessment")}</div>',
            unsafe_allow_html=True,
        )
    with tab_e:
        st.markdown(
            f'<div class="section-box">{_tab_content("Environmental Impact")}</div>',
            unsafe_allow_html=True,
        )
    with tab_f:
        st.markdown(
            f'<div class="section-box">{_tab_content("Financial Briefing")}</div>',
            unsafe_allow_html=True,
        )
    with tab_c:
        st.markdown(
            f'<div class="section-box">{_tab_content("COO Executive Summary")}</div>',
            unsafe_allow_html=True,
        )

    # ── Download ─────────────────────────────────────────────────────────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    safe_name = record["prospect"].replace(" ", "_")
    ts        = record["ts"].replace(" ", "").replace("·", "").replace(":", "")
    st.download_button(
        label     = "⬇  Download Full Report (.md)",
        data      = content,
        file_name = f"CORE_{safe_name}_{ts}.md",
        mime      = "text/markdown",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<span class="amd-badge">AMD DEVELOPER CLOUD · MI300X</span>',
                unsafe_allow_html=True)
    st.markdown(
        "<h2 style='margin:4px 0 2px;font-size:1.3rem;font-weight:900;'>"
        "⛏️ C.O.R.E. AI</h2>"
        "<p style='margin:0;font-size:.73rem;color:#374151;'>"
        "Collaborative Operational Risk Evaluator</p>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Agent pipeline
    st.markdown(
        "<p style='font-size:.68rem;text-transform:uppercase;"
        "letter-spacing:.12em;color:#374151;margin-bottom:6px;'>"
        "Agent Pipeline</p>",
        unsafe_allow_html=True,
    )
    for _role_key, icon, role, model, _summary in AGENTS:
        st.markdown(f"""
        <div class="ag-row">
          <span class="ag-icon">{icon}</span>
          <div>
            <div class="ag-role">{role}</div>
            <div class="ag-model">{model}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Prospect selector
    st.markdown(
        "<p style='font-size:.68rem;text-transform:uppercase;"
        "letter-spacing:.12em;color:#374151;margin-bottom:4px;'>"
        "Select Prospect</p>",
        unsafe_allow_html=True,
    )

    run_clicked       = False
    selected_prospect = None
    selected_display  = ""

    try:
        prospects     = list_prospects()
        display_names = [p["display"] for p in prospects]
        name_map      = {p["display"]: p["name"] for p in prospects}

        selected_display  = st.selectbox("Prospect", display_names,
                                         label_visibility="collapsed")
        selected_prospect = name_map[selected_display]
        run_clicked       = st.button("▶  Run Evaluation",
                                      type="primary",
                                      use_container_width=True)
    except Exception as exc:
        st.error(f"Could not load prospects: {exc}")

    # History
    if st.session_state.reports:
        st.markdown("---")
        st.markdown(
            "<p style='font-size:.68rem;text-transform:uppercase;"
            "letter-spacing:.12em;color:#374151;margin-bottom:6px;'>"
            "Report History</p>",
            unsafe_allow_html=True,
        )
        for i, rpt in enumerate(reversed(st.session_state.reports)):
            actual_i = len(st.session_state.reports) - 1 - i
            v, _, ico = _parse_recommendation(rpt["content"])
            if st.button(f"{ico}  {rpt['prospect']}", key=f"h_{actual_i}",
                         use_container_width=True):
                st.session_state.active_index = actual_i

    st.markdown("---")
    st.markdown(
        "<p style='font-size:.65rem;color:#1f2937;'>"
        "Powered by CrewAI · AMD ROCm</p>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main area — header
# ─────────────────────────────────────────────────────────────────────────────
h_left, h_right = st.columns([6, 2])
with h_left:
    st.markdown(
        "<h1 style='margin:0;padding:0;font-size:1.9rem;font-weight:900;"
        "color:#e8edf8;'>C.O.R.E. AI"
        "<span style='color:#ED1C24;'> ·</span> Mining Intelligence Platform</h1>"
        "<p style='margin:2px 0 0;color:#374151;font-size:.82rem;'>"
        "Multi-Agent ESG Due Diligence &nbsp;·&nbsp; "
        "Cognitive-Demand Routing on AMD MI300X &nbsp;·&nbsp; "
        "CrewAI + Open-Weight LLMs</p>",
        unsafe_allow_html=True,
    )
with h_right:
    st.markdown(
        "<div style='text-align:right;padding-top:4px;'>"
        "<span class='amd-badge'>AMD INSTINCT MI300X</span></div>",
        unsafe_allow_html=True,
    )

st.markdown(
    "<hr style='border:none;border-top:1px solid #1a2540;margin:12px 0 20px;'>",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Main area — evaluation + display
# ─────────────────────────────────────────────────────────────────────────────
if run_clicked and selected_prospect:
    # ── Live progress: run evaluation in a worker thread so we can tick
    # the per-agent checklist as each task completes. The worker pushes
    # role-strings into ``progress_q`` from inside CrewAI's task callback;
    # the main Streamlit thread polls the queue and redraws the placeholder
    # so judges see the pipeline advance instead of a static spinner.
    st.markdown(
        f"<p style='font-size:.78rem;color:#5a7aaa;margin:6px 0 0;'>"
        f"Evaluating <strong style='color:#e8edf8;'>{selected_display}</strong> "
        f"on AMD Instinct MI300X …</p>",
        unsafe_allow_html=True,
    )

    progress_q: queue.Queue = queue.Queue()
    result_q:   queue.Queue = queue.Queue()

    def _on_agent_done(role: str, _q: queue.Queue = progress_q) -> None:
        _q.put(role)

    def _worker() -> None:
        try:
            content = run_evaluation(
                selected_prospect,
                progress_callback=_on_agent_done,
            )
            result_q.put(("ok", content))
        except Exception as exc:    # noqa: BLE001
            result_q.put(("error", exc))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    progress_box = st.empty()
    completed: set[str] = set()
    started = time.monotonic()

    # Initial paint so the card appears immediately (before the first task
    # callback fires, which happens ~30-40s into the geology run).
    progress_box.markdown(
        _render_progress_checklist(completed, 0),
        unsafe_allow_html=True,
    )

    while worker.is_alive() or not progress_q.empty():
        # Drain anything the worker pushed since the last poll.
        try:
            while True:
                completed.add(progress_q.get_nowait())
        except queue.Empty:
            pass
        elapsed = int(time.monotonic() - started)
        progress_box.markdown(
            _render_progress_checklist(completed, elapsed),
            unsafe_allow_html=True,
        )
        time.sleep(0.4)

    # Final drain after the worker exits, in case a callback fired between
    # the last poll and worker termination.
    try:
        while True:
            completed.add(progress_q.get_nowait())
    except queue.Empty:
        pass

    status, payload = result_q.get()
    elapsed = int(time.monotonic() - started)

    if status == "ok":
        progress_box.markdown(
            _render_progress_checklist(completed, elapsed, finished=True),
            unsafe_allow_html=True,
        )
        st.session_state.reports.append({
            "prospect": selected_display,
            "content":  payload,
            "ts":       datetime.now().strftime("%H:%M · %d %b %Y"),
        })
        st.session_state.active_index = len(st.session_state.reports) - 1
        # Brief pause so the "complete" frame is visible, then rerun to
        # render the report below the (now hidden) checklist.
        time.sleep(0.6)
        st.rerun()
    else:
        # Show the partial checklist as-is so the user sees how far the
        # pipeline got before failing.
        progress_box.markdown(
            _render_progress_checklist(completed, elapsed, finished=True),
            unsafe_allow_html=True,
        )
        st.error(f"Evaluation failed: {payload}")

# Display active report
if st.session_state.active_index is not None and st.session_state.reports:
    idx = st.session_state.active_index
    if 0 <= idx < len(st.session_state.reports):
        _render_report(st.session_state.reports[idx])
    else:
        st.session_state.active_index = None

# Landing page when nothing has run yet
if not st.session_state.reports:
    _render_landing()
