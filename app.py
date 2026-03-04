"""
AI-Based Academic Answer Evaluation System
Streamlit UI — main entry point
"""

import os
import time
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv

from rag_engine  import RAGEngine
from evaluator   import Evaluator

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Answer Evaluator",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()  # loads GROQ_API_KEY from .env if present

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .grade-badge {
        font-size: 3rem;
        font-weight: 900;
        text-align: center;
        padding: 1rem;
        border-radius: 50%;
        width: 90px;
        height: 90px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: auto;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #0f3460;
    }
    .concept-chip {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        margin: 3px;
        font-size: 0.82rem;
    }
    .chip-green  { background: #d4edda; color: #155724; }
    .chip-red    { background: #f8d7da; color: #721c24; }
    .chip-blue   { background: #d1ecf1; color: #0c5460; }
    .feedback-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .improved-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
if "rag" not in st.session_state:
    rag = RAGEngine()
    rag.load_index()          # load persisted index if it exists
    st.session_state.rag = rag

if "history" not in st.session_state:
    st.session_state.history = []


# ── Helpers ───────────────────────────────────────────────────────────────────
GRADE_COLORS = {
    "A+": "#28a745", "A": "#5cb85c", "B": "#17a2b8",
    "C":  "#ffc107", "D": "#fd7e14", "F": "#dc3545",
}

def grade_color(grade: str) -> str:
    return GRADE_COLORS.get(grade, "#6c757d")

def render_chips(items: list, css_class: str):
    if not items:
        st.markdown("*None identified*")
        return
    html = " ".join(
        f'<span class="concept-chip {css_class}">{item}</span>'
        for item in items
    )
    st.markdown(html, unsafe_allow_html=True)

def score_gauge(marks: int, max_marks: int = 10):
    pct = marks / max_marks * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=marks,
        number={"suffix": f"/{max_marks}", "font": {"size": 36}},
        gauge={
            "axis":  {"range": [0, max_marks], "tickwidth": 1},
            "bar":   {"color": grade_color(result_grade(marks, max_marks))},
            "steps": [
                {"range": [0,  2],  "color": "#f8d7da"},
                {"range": [2,  5],  "color": "#fff3cd"},
                {"range": [5,  7],  "color": "#d1ecf1"},
                {"range": [7,  9],  "color": "#d4edda"},
                {"range": [9, 10],  "color": "#c3e6cb"},
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "value": marks},
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(height=200, margin=dict(t=20, b=10, l=20, r=20))
    return fig

def result_grade(marks: int, max_marks: int) -> str:
    pct = marks / max_marks * 100
    if pct >= 90: return "A+"
    if pct >= 80: return "A"
    if pct >= 70: return "B"
    if pct >= 50: return "C"
    if pct >= 30: return "D"
    return "F"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # API key
    api_key = st.text_input(
        "🔑 Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Get your free key at https://console.groq.com",
    )
    if api_key:
        st.success("API key loaded ✓")
    else:
        st.warning("Enter your Groq API key to enable evaluation.")
        st.markdown("[Get free key →](https://console.groq.com)")

    st.markdown("---")

    # PDF upload
    st.markdown("### 📚 Knowledge Base")
    rag: RAGEngine = st.session_state.rag
    st.info(f"Chunks indexed: **{rag.total_chunks}**")

    uploaded = st.file_uploader(
        "Upload syllabus / reference PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if uploaded:
        if st.button("📥 Index PDFs", use_container_width=True):
            progress = st.progress(0)
            paths    = []
            os.makedirs("uploaded_pdfs", exist_ok=True)
            for i, f in enumerate(uploaded):
                dest = f"uploaded_pdfs/{f.name}"
                with open(dest, "wb") as out:
                    out.write(f.read())
                paths.append(dest)
                progress.progress((i + 1) / len(uploaded) * 0.4)

            with st.spinner("Generating embeddings…"):
                n = rag.add_documents(paths)
                progress.progress(1.0)
            st.success(f"Indexed {n} chunks from {len(paths)} PDF(s)!")
            st.rerun()

    if rag.total_chunks > 0:
        if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
            rag.clear_index()
            st.success("Knowledge base cleared.")
            st.rerun()

    st.markdown("---")
    st.markdown("### 📝 Evaluation History")
    if st.session_state.history:
        for i, h in enumerate(reversed(st.session_state.history[-5:])):
            st.markdown(
                f"**{i+1}.** {h['question'][:40]}… → "
                f"<span style='color:{grade_color(h['grade'])};font-weight:bold'>{h['grade']} ({h['marks']}/10)</span>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown("*No evaluations yet.*")


# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎓 AI Academic Answer Evaluator</h1>
    <p style='opacity:0.85;margin:0'>Powered by RAG · FAISS · Groq LLaMA-3 · Sentence Transformers</p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["📝 Evaluate Answer", "📊 Batch Evaluation", "ℹ️ How It Works"])

# ─────────────────────────────────────────────────────────────────
# Tab 1 — Single answer evaluation
# ─────────────────────────────────────────────────────────────────
with tabs[0]:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 📋 Input")
        question = st.text_area(
            "Question",
            placeholder="e.g. Explain the concept of Retrieval-Augmented Generation (RAG).",
            height=120,
        )
        student_answer = st.text_area(
            "Student's Answer",
            placeholder="Paste the student's descriptive answer here…",
            height=220,
        )

        max_marks = st.slider("Max Marks", min_value=5, max_value=20, value=10, step=5)

        evaluate_btn = st.button(
            "🚀 Evaluate Answer",
            use_container_width=True,
            disabled=not api_key,
            type="primary",
        )

    with col2:
        st.markdown("### 🔍 Retrieved Context")
        if question:
            hits = st.session_state.rag.retrieve(question, top_k=3)
            if hits:
                for h in hits:
                    with st.expander(f"📄 {h['source']} — p.{h['page']}  (relevance: {h['score']:.2f})"):
                        st.write(h["text"][:500] + ("…" if len(h["text"]) > 500 else ""))
            else:
                st.info("No reference material indexed yet. Upload PDFs in the sidebar, or evaluation will use general LLM knowledge.")
        else:
            st.info("Enter a question to preview retrieved context.")

    # ── Evaluation result ──────────────────────────────────────────
    if evaluate_btn:
        if not question.strip():
            st.error("Please enter a question.")
        elif not student_answer.strip():
            st.error("Please enter the student's answer.")
        else:
            with st.spinner("Evaluating… retrieving context and running LLM…"):
                context   = st.session_state.rag.get_context(question)
                evaluator = Evaluator(api_key)
                t0        = time.time()
                result    = evaluator.evaluate(question, student_answer, context)
                elapsed   = round(time.time() - t0, 1)

            if "error" in result:
                st.error(f"Evaluation failed: {result['error']}")
            else:
                # Scale to chosen max_marks
                scale = max_marks / 10
                result["marks_awarded"] = round(result["marks_awarded"] * scale)
                result["max_marks"]     = max_marks

                # Save to history
                st.session_state.history.append({
                    "question": question,
                    "grade":    result["grade"],
                    "marks":    result["marks_awarded"],
                })

                st.markdown("---")
                st.markdown("## 📊 Evaluation Results")

                # Top metrics row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Marks", f"{result['marks_awarded']} / {max_marks}")
                m2.metric("Percentage", f"{result['percentage']}%")
                m3.metric("Grade", result["grade"])
                m4.metric("Eval Time", f"{elapsed}s")

                r1, r2 = st.columns([1, 1], gap="large")

                with r1:
                    st.plotly_chart(score_gauge(result["marks_awarded"], max_marks), use_container_width=True)

                    st.markdown("#### ✅ Concepts Covered")
                    render_chips(result["concepts_covered"], "chip-green")

                    st.markdown("#### ❌ Concepts Missing")
                    render_chips(result["concepts_missing"], "chip-red")

                with r2:
                    st.markdown("#### 💪 Strengths")
                    for s in result.get("strengths", []):
                        st.markdown(f"- {s}")

                    st.markdown("#### 📉 Areas to Improve")
                    for w in result.get("weaknesses", []):
                        st.markdown(f"- {w}")

                    st.markdown("#### 💬 Detailed Feedback")
                    st.markdown(
                        f'<div class="feedback-box">{result["detailed_feedback"]}</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("#### 💡 Model / Improved Answer")
                st.markdown(
                    f'<div class="improved-box">{result["improved_answer"]}</div>',
                    unsafe_allow_html=True,
                )

                if context:
                    with st.expander("📚 View Retrieved Reference Context"):
                        st.text(context)


# ─────────────────────────────────────────────────────────────────
# Tab 2 — Batch evaluation
# ─────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("### 📦 Batch Evaluation")
    st.info("Upload a CSV with columns: `question`, `student_answer` (optional: `student_name`)")

    import pandas as pd
    import io

    sample_csv = "student_name,question,student_answer\nAlice,What is RAG?,RAG stands for Retrieval Augmented Generation. It retrieves relevant documents and feeds them to an LLM.\nBob,What is FAISS?,FAISS is a library for fast similarity search."
    st.download_button("📥 Download Sample CSV", sample_csv, "sample_batch.csv", "text/csv")

    batch_file = st.file_uploader("Upload batch CSV", type=["csv"], key="batch_csv")
    if batch_file and api_key:
        df = pd.read_csv(batch_file)
        st.dataframe(df.head(), use_container_width=True)

        if st.button("🚀 Run Batch Evaluation", type="primary"):
            evaluator = Evaluator(api_key)
            results   = []
            progress  = st.progress(0)
            status    = st.empty()

            for i, row in df.iterrows():
                status.text(f"Evaluating {i+1}/{len(df)}…")
                q   = str(row.get("question", ""))
                ans = str(row.get("student_answer", ""))
                ctx = st.session_state.rag.get_context(q)
                res = evaluator.evaluate(q, ans, ctx)
                results.append({
                    "Student":         row.get("student_name", f"Student {i+1}"),
                    "Question":        q[:60] + "…",
                    "Marks":           res.get("marks_awarded", 0),
                    "Grade":           res.get("grade", "N/A"),
                    "Percentage":      res.get("percentage", 0),
                    "Key Feedback":    res.get("detailed_feedback", "")[:120] + "…",
                })
                progress.progress((i + 1) / len(df))

            status.empty()
            result_df = pd.DataFrame(results)
            st.dataframe(result_df, use_container_width=True)

            csv_out = result_df.to_csv(index=False)
            st.download_button("📥 Download Results", csv_out, "evaluation_results.csv", "text/csv")

            # Summary chart
            grade_counts = result_df["Grade"].value_counts()
            fig = go.Figure(go.Bar(
                x=grade_counts.index.tolist(),
                y=grade_counts.values.tolist(),
                marker_color=[grade_color(g) for g in grade_counts.index],
            ))
            fig.update_layout(title="Grade Distribution", xaxis_title="Grade", yaxis_title="Count", height=300)
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────
# Tab 3 — How It Works
# ─────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("### 🏗️ System Architecture")
    st.markdown("""
```
┌─────────────────────────────────────────────────────────────┐
│                   OFFLINE — Knowledge Base Setup            │
│                                                             │
│  Syllabus PDFs  →  Text Chunks  →  Sentence Embeddings     │
│                                         ↓                   │
│                                    FAISS Index              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   ONLINE — Evaluation Pipeline              │
│                                                             │
│  Question + Answer  →  Embed Question  →  FAISS Search     │
│                                               ↓             │
│                                     Top-K Relevant Chunks   │
│                                               ↓             │
│                              Prompt = Q + Answer + Context  │
│                                               ↓             │
│                              Groq LLaMA-3 (LLM Evaluation)  │
│                                               ↓             │
│              Marks | Grade | Concepts | Feedback | Model Ans │
└─────────────────────────────────────────────────────────────┘
```

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Embeddings** | Sentence Transformers (`all-MiniLM-L6-v2`) | Convert text to semantic vectors |
| **Vector DB** | FAISS (IndexFlatIP) | Fast cosine similarity search |
| **LLM** | Groq LLaMA-3 70B | Concept extraction + scoring + feedback |
| **RAG** | Custom pipeline | Grounds evaluation in syllabus material |
| **UI** | Streamlit | Interactive web interface |
| **PDF Parsing** | PyMuPDF | Extract text from uploaded PDFs |

### 📈 Evaluation Rubric (out of 10)

| Score | Grade | Criteria |
|-------|-------|---------|
| 9–10 | A+ | All key concepts covered, correct & clearly explained |
| 8    | A  | Most concepts covered with minor omissions |
| 7    | B  | Good coverage, some inaccuracies |
| 5–6  | C  | Partial coverage, basic understanding shown |
| 3–4  | D  | Minimal relevant content |
| 0–2  | F  | Incorrect or irrelevant answer |
""")
