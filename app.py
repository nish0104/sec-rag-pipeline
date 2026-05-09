"""
app.py - SEC Filing Q&A Engine
Bloomberg-style dark professional UI
"""

import streamlit as st
import chromadb
from pipeline import run_pipeline
from src.rag.chain import SECQueryChain
from src.embedding.embedder import run_embedding

st.set_page_config(page_title="SEC Filing Q&A", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0a0e1a; color: #e0e4ef; }
    
    /* ── Sidebar ── */
    [data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #1e2535; }
    [data-testid="stSidebar"] .stButton button {
        background: #1e2535; color: #a0aec0; border: 1px solid #2d3748;
        border-radius: 4px; font-size: 0.8rem; transition: all 0.2s;
    }
    [data-testid="stSidebar"] .stButton button:hover { background: #2d3748; color: #fff; border-color: #00a8ff; }
    
    /* ── Primary button ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00a8ff, #0080cc) !important;
        color: #0a0e1a !important; font-weight: 700 !important;
        border: none !important; border-radius: 4px !important;
        letter-spacing: 0.05em; text-transform: uppercase; font-size: 0.75rem !important;
    }
    .stButton > button[kind="primary"]:hover { opacity: 0.9; transform: translateY(-1px); }
    
    /* ── Inputs ── */
    .stTextInput input, .stSelectbox select {
        background: #1e2535 !important; color: #e0e4ef !important;
        border: 1px solid #2d3748 !important; border-radius: 4px !important;
    }
    .stTextInput input:focus { border-color: #00a8ff !important; box-shadow: 0 0 0 2px rgba(245,158,11,0.2) !important; }
    
    /* ── Chat messages ── */
    [data-testid="stChatMessage"] { background: #111827; border: 1px solid #1e2535; border-radius: 6px; margin: 0.5rem 0; }
    
    /* ── Metric cards ── */
    .metric-card {
        background: #111827; border: 1px solid #1e2535; border-radius: 6px;
        padding: 1rem 1.2rem; margin-bottom: 0.5rem;
    }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #00a8ff; }
    .metric-label { font-size: 0.72rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.2rem; }

    /* ── Header ── */
    .bb-header {
        background: linear-gradient(90deg, #0d1117, #111827);
        border-bottom: 2px solid #00a8ff;
        padding: 1rem 1.5rem; margin-bottom: 1.5rem;
        display: flex; align-items: center; gap: 1rem;
    }
    .bb-logo { font-size: 1.4rem; font-weight: 800; color: #00a8ff; letter-spacing: -0.02em; }
    .bb-subtitle { font-size: 0.78rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.1em; }
    .bb-ticker-badge {
        background: #00a8ff; color: #0a0e1a; font-weight: 800;
        padding: 0.25rem 0.75rem; border-radius: 3px; font-size: 0.9rem;
        letter-spacing: 0.05em;
    }

    /* ── Status badges ── */
    .badge-ready { background: #064e3b; color: #34d399; border: 1px solid #065f46; padding: 0.2rem 0.7rem; border-radius: 20px; font-size: 0.72rem; font-weight: 600; }
    .badge-not-ready { background: #450a0a; color: #f87171; border: 1px solid #7f1d1d; padding: 0.2rem 0.7rem; border-radius: 20px; font-size: 0.72rem; font-weight: 600; }

    /* ── Section headers ── */
    .section-header {
        font-size: 0.7rem; font-weight: 600; color: #6b7280;
        text-transform: uppercase; letter-spacing: 0.12em;
        border-bottom: 1px solid #1e2535; padding-bottom: 0.4rem; margin-bottom: 0.8rem;
    }

    /* ── Answer box ── */
    .answer-container {
        background: #111827; border: 1px solid #1e2535;
        border-left: 3px solid #00a8ff;
        border-radius: 0 6px 6px 0; padding: 1.2rem; margin: 0.5rem 0;
    }

    /* ── Source chunk ── */
    .source-chip {
        display: inline-block; background: #1e2535; color: #a0aec0;
        border: 1px solid #2d3748; border-radius: 3px;
        padding: 0.15rem 0.5rem; font-size: 0.7rem;
        margin-right: 0.3rem; margin-bottom: 0.3rem;
    }
    .source-text { font-size: 0.8rem; color: #9ca3af; line-height: 1.5; }

    /* ── Quick question buttons ── */
    .stButton > button {
        background: #111827 !important; color: #a0aec0 !important;
        border: 1px solid #2d3748 !important; border-radius: 4px !important;
        font-size: 0.78rem !important; transition: all 0.15s !important;
    }
    .stButton > button:hover { border-color: #00a8ff !important; color: #00a8ff !important; background: #1a2035 !important; }

    /* ── Chat input ── */
    [data-testid="stChatInput"] textarea {
        background: #111827 !important; border: 1px solid #2d3748 !important;
        color: #e0e4ef !important; border-radius: 6px !important;
    }
    [data-testid="stChatInput"] textarea:focus { border-color: #00a8ff !important; }

    /* ── Divider ── */
    hr { border-color: #1e2535 !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: #0a0e1a; }
    ::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_ticker_indexed(ticker):
    try:
        client = chromadb.PersistentClient(path="data/embeddings")
        col = client.get_collection("sec_filings")
        return len(col.get(where={"ticker": ticker.upper()}, limit=1)["ids"]) > 0
    except:
        return False

def get_indexed_tickers():
    try:
        client = chromadb.PersistentClient(path="data/embeddings")
        col = client.get_collection("sec_filings")
        return sorted(set(m["ticker"] for m in col.get()["metadatas"]))
    except:
        return []

def get_chunk_count(ticker):
    try:
        client = chromadb.PersistentClient(path="data/embeddings")
        col = client.get_collection("sec_filings")
        return len(col.get(where={"ticker": ticker.upper()})["ids"])
    except:
        return 0

# ── Session State ─────────────────────────────────────────────────────────────

for key, default in [("messages", []), ("current_ticker", None), ("chain", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="bb-logo">◈ SECQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="bb-subtitle">SEC Filing Intelligence</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Data Source</div>', unsafe_allow_html=True)
    ticker = st.text_input("", value="AAPL", placeholder="Ticker e.g. AAPL", label_visibility="collapsed").upper().strip()
    form_type = st.selectbox("", ["10-K", "10-Q"], label_visibility="collapsed")
    num_filings = st.slider("Filings to fetch", 1, 5, 1)

    st.markdown("<br>", unsafe_allow_html=True)
    already_indexed = is_ticker_indexed(ticker)
    if already_indexed:
        chunk_count = get_chunk_count(ticker)
        st.markdown(f'<span class="badge-ready">● INDEXED — {chunk_count} chunks</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="badge-not-ready">○ NOT INDEXED</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⟳  FETCH & INDEX", use_container_width=True, type="primary"):
        with st.status(f"Processing {ticker}...", expanded=True) as status:
            try:
                st.write("📥 Fetching from SEC EDGAR...")
                run_pipeline(ticker=ticker, form_type=form_type, limit=num_filings)
                st.write("🧠 Embedding into ChromaDB...")
                run_embedding(ticker=ticker, reset=True)
                status.update(label=f"✅ {ticker} ready!", state="complete")
                st.session_state.current_ticker = ticker
                st.session_state.chain = None
                st.session_state.messages = []
                st.rerun()
            except Exception as e:
                status.update(label="❌ Error", state="error")
                st.error(str(e))

    st.divider()
    indexed = get_indexed_tickers()
    if indexed:
        st.markdown('<div class="section-header">Indexed Securities</div>', unsafe_allow_html=True)
        for t in indexed:
            if st.button(f"{'▶ ' if t == st.session_state.current_ticker else ''}{t}", key=f"btn_{t}", use_container_width=True):
                st.session_state.current_ticker = t
                st.session_state.chain = None
                st.session_state.messages = []
                st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────

active_ticker = st.session_state.current_ticker or (ticker if already_indexed else None)

if not active_ticker:
    # ── Welcome screen ────────────────────────────────────────────────────
    st.markdown("""
    <div class="bb-header">
        <div>
            <div class="bb-logo">◈ SEC FILING Q&A ENGINE</div>
            <div class="bb-subtitle">AI-powered financial intelligence from official SEC filings</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    for col, icon, title, desc in [
        (col1, "📥", "INGEST", "Enter any US ticker and fetch official 10-K/10-Q filings directly from SEC EDGAR — free, no API key needed"),
        (col2, "🧠", "INDEX", "Filings are chunked, embedded via Ollama, and stored in a local ChromaDB vector database on your machine"),
        (col3, "💬", "QUERY", "Ask natural language questions — the RAG pipeline retrieves relevant chunks and generates answers using local AI"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:1.5rem">{icon}</div>
                <div style="font-size:0.65rem;font-weight:700;color:#00a8ff;letter-spacing:0.1em;margin:0.5rem 0 0.3rem">{title}</div>
                <div style="font-size:0.78rem;color:#9ca3af;line-height:1.5">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Sample Questions</div>', unsafe_allow_html=True)
    questions = [
        ("Risk Analysis", "What are the main risk factors disclosed?"),
        ("Financial Performance", "How did revenue and profit change year over year?"),
        ("Business Strategy", "What are the company's key strategic priorities?"),
        ("Market Position", "How does the company describe its competitive position?"),
        ("Legal & Regulatory", "Are there any significant legal proceedings or regulatory risks?"),
    ]
    c1, c2 = st.columns(2)
    for i, (category, q) in enumerate(questions):
        col = c1 if i % 2 == 0 else c2
        with col:
            st.markdown(f"""
            <div class="metric-card" style="padding:0.7rem 1rem">
                <div style="font-size:0.6rem;color:#00a8ff;font-weight:600;letter-spacing:0.08em">{category}</div>
                <div style="font-size:0.8rem;color:#d1d5db;margin-top:0.2rem">{q}</div>
            </div>
            """, unsafe_allow_html=True)

else:
    # ── Active Q&A ────────────────────────────────────────────────────────
    chunk_count = get_chunk_count(active_ticker)

    st.markdown(f"""
    <div class="bb-header">
        <span class="bb-ticker-badge">{active_ticker}</span>
        <div>
            <div class="bb-logo" style="font-size:1rem">{form_type} FILING ANALYSIS</div>
            <div class="bb-subtitle">{chunk_count} indexed chunks · Local AI · SEC EDGAR</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        ("INDEXED CHUNKS", str(chunk_count), "Vector DB entries"),
        ("DATA SOURCE", "SEC EDGAR", "Official filings"),
        ("AI MODEL", "phi3", "Local · No API cost"),
        ("FILING TYPE", form_type, "Selected form"),
    ]
    for col, (label, value, sub) in zip([m1, m2, m3, m4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="font-size:1.1rem">{value}</div>
                <div style="font-size:0.68rem;color:#4b5563;margin-top:0.2rem">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Initialize chain
    if st.session_state.chain is None or st.session_state.current_ticker != active_ticker:
        with st.spinner("Initializing analysis engine..."):
            try:
                st.session_state.chain = SECQueryChain(ticker=active_ticker)
                st.session_state.current_ticker = active_ticker
            except Exception as e:
                st.error(f"Failed to load: {e}")
                st.stop()

    # Quick questions
    st.markdown('<div class="section-header">Quick Analysis</div>', unsafe_allow_html=True)
    qcols = st.columns(4)
    quick_qs = ["Main risk factors?", "Revenue trend?", "Business strategy?", "Competitive position?"]
    for i, q in enumerate(quick_qs):
        if qcols[i].button(q, key=f"qq{i}", use_container_width=True):
            st.session_state.pending_question = f"What is {active_ticker}'s {q.lower()}"

    st.markdown("<br>", unsafe_allow_html=True)

    # Chat history
    if st.session_state.messages:
        st.markdown('<div class="section-header">Analysis History</div>', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and "sources" in msg:
                    with st.expander(f"📎 {len(msg['sources'])} source chunks retrieved"):
                        for i, src in enumerate(msg["sources"]):
                            st.markdown(f"""
                            <span class="source-chip">{src['section'].upper()}</span>
                            <span class="source-chip">📅 {src['filing_date']}</span>
                            <div class="source-text" style="margin-top:0.5rem">{src['preview']}</div>
                            """, unsafe_allow_html=True)
                            if i < len(msg["sources"]) - 1:
                                st.divider()

    # Handle quick question
    if "pending_question" in st.session_state:
        question = st.session_state.pop("pending_question")
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("⟳ Analyzing filing..."):
            result = st.session_state.chain.ask(question)
        st.session_state.messages.append({"role": "assistant", "content": result["answer"], "sources": result["sources"]})
        st.rerun()

    # Chat input
    st.markdown('<div class="section-header">Ask a Question</div>', unsafe_allow_html=True)
    if question := st.chat_input(f"Query {active_ticker} SEC filing..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("⟳ Analyzing..."):
                result = st.session_state.chain.ask(question)
            st.markdown(result["answer"])
            with st.expander(f"📎 {len(result['sources'])} source chunks"):
                for i, src in enumerate(result["sources"]):
                    st.markdown(f"""
                    <span class="source-chip">{src['section'].upper()}</span>
                    <span class="source-chip">📅 {src['filing_date']}</span>
                    <div class="source-text" style="margin-top:0.5rem">{src['preview']}</div>
                    """, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": result["answer"], "sources": result["sources"]})

    if st.session_state.messages:
        if st.button("✕ Clear History", type="secondary"):
            st.session_state.messages = []
            st.rerun()
