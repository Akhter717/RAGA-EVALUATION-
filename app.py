import json
import time

import numpy as np
import pandas as pd
import streamlit as st
from groq import Groq, RateLimitError

st.set_page_config(page_title="RAGAS Evaluation Dashboard", page_icon="📊", layout="wide")

# ----------------------------------------------------------------------------
# Config (targets & low-thresholds come from Class 6 PDF, sections 6.1 and 6.2)
# ----------------------------------------------------------------------------
METRICS = {
    "faithfulness": {
        "label": "Faithfulness", "target": 0.90, "low": 0.80,
        "asks": "Is the answer grounded in the retrieved context?",
        "diagnosis": "LLM is HALLUCINATING (adding facts not in context).",
        "fix": "Strengthen system prompt: 'Answer ONLY from context'. Set temperature=0. Add an 'I don't know' fallback.",
    },
    "answer_relevancy": {
        "label": "Answer Relevancy", "target": 0.85, "low": 0.80,
        "asks": "Does the answer actually address the question?",
        "diagnosis": "Answers are OFF-TOPIC (not addressing what the user asked).",
        "fix": "Improve prompt template. Add: 'Answer the question directly and specifically.'",
    },
    "context_precision": {
        "label": "Context Precision", "target": 0.80, "low": 0.70,
        "asks": "Are the retrieved chunks actually relevant?",
        "diagnosis": "Retrieval is returning JUNK chunks.",
        "fix": "Reduce chunk size, try hybrid retrieval (BM25 + vector), add metadata filters, add reranking.",
    },
    "context_recall": {
        "label": "Context Recall", "target": 0.80, "low": 0.70,
        "asks": "Were ALL relevant chunks found?",
        "diagnosis": "Relevant docs exist but are MISSED by retrieval.",
        "fix": "Increase k from 3 to 5-7, add hybrid retrieval, add query rewriting or HyDE.",
    },
    "answer_correctness": {
        "label": "Answer Correctness", "target": 0.85, "low": 0.80,
        "asks": "Is the answer factually correct vs ground truth?",
        "diagnosis": "Answer is WRONG compared to ground truth.",
        "fix": "Check retrieval first. If retrieval is good, fix the prompt or use a better LLM.",
    },
}

MODELS = {
    "OpenAI gpt-oss-120b (best quality)": "openai/gpt-oss-120b",
    "OpenAI gpt-oss-20b (faster, lighter)": "openai/gpt-oss-20b",
}

CTX_SEP = "|||"

DEMO = pd.DataFrame([
    {
        "question": "How many annual leaves do I get?",
        "answer": "Employees get 24 days of annual leave per year. [Source: HR_Policy.pdf, Page 3]",
        "contexts": "Every full-time employee is entitled to 24 days of paid annual leave per calendar year."
                    f" {CTX_SEP} Annual leave must be approved by the reporting manager at least 3 days in advance.",
        "ground_truth": "24 days per year",
    },
    {
        "question": "Can I carry forward unused leave?",
        "answer": "Yes, you can carry forward up to 10 days to the next year. You also receive a cash bonus for unused leave.",
        "contexts": "Unused annual leave up to 10 days can be carried forward to the next calendar year."
                    f" {CTX_SEP} Carried forward leave expires on March 31.",
        "ground_truth": "Up to 10 unused days can be carried forward, and they expire on March 31.",
    },
    {
        "question": "What is the work from home policy?",
        "answer": "Employees can work from home 2 days per week with manager approval.",
        "contexts": "The office cafeteria is open from 9 AM to 6 PM."
                    f" {CTX_SEP} Employees can work from home 2 days per week with manager approval."
                    f" {CTX_SEP} Parking passes are issued by the admin team.",
        "ground_truth": "Employees can work from home 2 days per week with manager approval.",
    },
    {
        "question": "What is the notice period and when is the final settlement paid?",
        "answer": "The notice period is 60 days.",
        "contexts": "The standard notice period for confirmed employees is 60 days.",
        "ground_truth": "The notice period is 60 days, and the final settlement is paid within 45 days of the last working day.",
    },
])


# ----------------------------------------------------------------------------
# LLM + embeddings helpers
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading free local embedding model (first time only)...")
def get_embedder():
    from fastembed import TextEmbedding
    return TextEmbedding("BAAI/bge-small-en-v1.5")


def embed(texts):
    vecs = np.array(list(get_embedder().embed(texts)))
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


class Judge:
    """Thin wrapper over Groq with JSON output, throttling and rate-limit retry."""

    def __init__(self, api_key, model, delay):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.delay = delay

    def ask(self, prompt):
        for attempt in range(5):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a strict evaluator. Reply with valid JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    reasoning_effort="low",
                    response_format={"type": "json_object"},
                )
                time.sleep(self.delay)
                return json.loads(resp.choices[0].message.content)
            except RateLimitError:
                time.sleep(15 * (attempt + 1))
            except json.JSONDecodeError:
                time.sleep(1)
        raise RuntimeError("LLM call failed after retries (rate limit or bad JSON).")


def fmt_ctx(contexts):
    return "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))


# ----------------------------------------------------------------------------
# The 5 RAGAS-style metrics (same logic as the PDF's table, section 6.1)
# ----------------------------------------------------------------------------
def faithfulness(j, q, a, ctx, gt):
    r = j.ask(
        "Break the ANSWER into separate factual claims. For each claim, say whether it is "
        "directly supported by the CONTEXT.\n"
        f"CONTEXT:\n{fmt_ctx(ctx)}\n\nANSWER: {a}\n\n"
        'Return: {"claims":[{"claim":"...","supported":true}]}'
    )
    claims = r.get("claims", [])
    if not claims:
        return np.nan
    return sum(bool(c.get("supported")) for c in claims) / len(claims)


def answer_relevancy(j, q, a, ctx, gt):
    r = j.ask(
        "Generate 3 different questions that the following ANSWER would be a good response to.\n"
        f"ANSWER: {a}\n\n"
        'Return: {"questions":["...","...","..."]}'
    )
    qs = [x for x in r.get("questions", []) if isinstance(x, str) and x.strip()]
    if not qs:
        return np.nan
    v = embed([q] + qs)
    return float(np.clip(np.mean(v[1:] @ v[0]), 0, 1))


def context_precision(j, q, a, ctx, gt):
    r = j.ask(
        "For each numbered CHUNK, decide if it is relevant for answering the QUESTION "
        "(use the reference answer as a guide).\n"
        f"QUESTION: {q}\nREFERENCE: {gt}\nCHUNKS:\n{fmt_ctx(ctx)}\n\n"
        'Return: {"verdicts":[{"chunk":1,"relevant":true}]}'
    )
    v = r.get("verdicts", [])
    if not v:
        return np.nan
    return sum(bool(x.get("relevant")) for x in v) / len(ctx)


def context_recall(j, q, a, ctx, gt):
    r = j.ask(
        "Split the REFERENCE answer into individual statements. For each, say whether it can be "
        "found in (attributed to) the CONTEXT.\n"
        f"CONTEXT:\n{fmt_ctx(ctx)}\n\nREFERENCE: {gt}\n\n"
        'Return: {"statements":[{"statement":"...","attributed":true}]}'
    )
    s = r.get("statements", [])
    if not s:
        return np.nan
    return sum(bool(x.get("attributed")) for x in s) / len(s)


def answer_correctness(j, q, a, ctx, gt):
    r = j.ask(
        "Compare the ANSWER with the REFERENCE. List facts as:\n"
        "TP = facts in both, FP = facts in the answer but not in the reference (or wrong), "
        "FN = facts in the reference missing from the answer.\n"
        f"QUESTION: {q}\nREFERENCE: {gt}\nANSWER: {a}\n\n"
        'Return: {"TP":["..."],"FP":["..."],"FN":["..."]}'
    )
    tp, fp, fn = len(r.get("TP", [])), len(r.get("FP", [])), len(r.get("FN", []))
    denom = tp + 0.5 * (fp + fn)
    f1 = tp / denom if denom else 0.0
    v = embed([a, gt])
    sim = float(np.clip(v[0] @ v[1], 0, 1))
    return 0.75 * f1 + 0.25 * sim  # same weighting RAGAS uses


FUNCS = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_precision": context_precision,
    "context_recall": context_recall,
    "answer_correctness": answer_correctness,
}


def parse_contexts(x):
    if isinstance(x, list):
        return [str(c) for c in x if str(c).strip()]
    x = str(x).strip()
    if x.startswith("["):
        try:
            return [str(c) for c in json.loads(x)]
        except Exception:
            pass
    return [c.strip() for c in x.split(CTX_SEP) if c.strip()]


def status(metric, score):
    if pd.isna(score):
        return "⚪ n/a"
    m = METRICS[metric]
    if score >= m["target"]:
        return "🟢 Pass"
    if score >= m["low"]:
        return "🟡 Needs work"
    return "🔴 Fix needed"


# ----------------------------------------------------------------------------
# Build a test dataset from a document (no CSV needed)
# ----------------------------------------------------------------------------
def chunk_text(text, size=500, overlap=50):
    """Chunking per the PDF's golden rule: ~500 chars with ~50 overlap."""
    text = " ".join(text.split())
    chunks, i = [], 0
    while i < len(text):
        end = min(i + size, len(text))
        if end < len(text):
            sp = text.rfind(" ", i, end)
            if sp > i + size // 2:
                end = sp
        chunks.append(text[i:end].strip())
        if end >= len(text):
            break
        i = max(end - overlap, i + 1)
    return [c for c in chunks if len(c) > 80]


def read_document(file):
    if file.name.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join((pg.extract_text() or "") for pg in PdfReader(file).pages)
    return file.getvalue().decode("utf-8", errors="ignore")


def build_dataset(judge, chunks, n, k, bar):
    """Mini RAG: make test questions from the document, retrieve top-k chunks, answer from context."""
    cv = embed(chunks)
    idx = np.linspace(0, len(chunks) - 1, num=min(n, len(chunks)), dtype=int)
    rows = []
    for step, i in enumerate(idx):
        bar.progress(step / len(idx), text=f"Building test question {step + 1}/{len(idx)}...")
        qa = judge.ask(
            f"PASSAGE:\n{chunks[i]}\n\n"
            "Write ONE clear, self-contained question that this passage answers (never say 'the passage'), "
            "and a short correct answer based only on the passage.\n"
            'Return: {"question":"...","ground_truth":"..."}'
        )
        q, gt = qa.get("question"), qa.get("ground_truth")
        if not q or not gt:
            continue
        top = np.argsort(-(cv @ embed([q])[0]))[:k]
        ctx = [chunks[t] for t in top]
        ans = judge.ask(
            "Answer the QUESTION using ONLY the CONTEXT. If the context does not contain the answer, "
            'reply exactly "I don\'t know". Be concise.\n'
            f"CONTEXT:\n{fmt_ctx(ctx)}\n\nQUESTION: {q}\n\n"
            'Return: {"answer":"..."}'
        ).get("answer", "")
        rows.append({"question": q, "answer": str(ans), "contexts": f" {CTX_SEP} ".join(ctx), "ground_truth": gt})
    bar.progress(1.0, text="Dataset ready")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    default_key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""
    try:
        default_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        default_key = ""
    api_key = st.text_input("Groq API key (free)", value=default_key, type="password",
                            help="Get a free key at console.groq.com/keys - no credit card needed.")
    model_label = st.selectbox("Judge LLM", list(MODELS.keys()))
    delay = st.slider("Delay between LLM calls (sec)", 0.0, 5.0, 2.0, 0.5,
                      help="Keeps you inside Groq's free rate limit (~30 requests/min).")
    selected = st.multiselect("Metrics to run", list(METRICS.keys()), default=list(METRICS.keys()),
                              format_func=lambda k: METRICS[k]["label"])
    st.caption("LLM judge: OpenAI gpt-oss via Groq (free tier)\n\nEmbeddings: fastembed `bge-small-en-v1.5` (free, runs locally)")

# ----------------------------------------------------------------------------
# Main page
# ----------------------------------------------------------------------------
st.title("📊 RAG Evaluation Dashboard")
st.caption("RAGAS-style evaluation of a RAG system: 5 key metrics, targets and fixes from the RAG Mastery Class 6 guide.")

tab_eval, tab_guide = st.tabs(["🧪 Evaluate", "📘 Metric Guide"])

with tab_guide:
    st.subheader("The 5 RAGAS metrics")
    st.dataframe(pd.DataFrame([
        {"Metric": m["label"], "What it asks": m["asks"], "Target": f"> {m['target']:.2f}",
         "Alarm below": f"< {m['low']:.2f}", "If low: diagnosis": m["diagnosis"], "How to fix": m["fix"]}
        for m in METRICS.values()
    ]), hide_index=True, use_container_width=True)
    st.info("If ALL scores are above target, the system is production-ready. Keep re-running this evaluation after every "
            "change to chunking, retrieval or prompts, and alert if any metric drops more than 10% from baseline.")

with tab_eval:
    st.subheader("1. Evaluation dataset")
    st.write("Each row = one test question with the answer your RAG produced, the chunks it retrieved, and the correct "
             "answer (ground truth). The PDF recommends 20-50 rows.")

    src = st.radio("Data source", ["Use demo data", "Build from my document (no CSV needed)", "Upload CSV"])
    df = None
    if src == "Upload CSV":
        up = st.file_uploader("CSV with columns: question, answer, contexts, ground_truth", type="csv")
        st.download_button("Download CSV template", DEMO.to_csv(index=False), "ragas_template.csv", "text/csv")
        df = pd.read_csv(up) if up else None
        st.caption(f"`contexts` column: separate multiple chunks with `{CTX_SEP}` (or use a JSON list).")
    elif src.startswith("Build"):
        st.write("Paste text or upload a PDF/TXT. The app will make test questions from it, run a small RAG "
                 "(retrieve top-k chunks, answer from context) and prepare the dataset for you.")
        doc_file = st.file_uploader("Document (PDF or TXT)", type=["pdf", "txt"])
        doc_text = st.text_area("...or paste text here", height=150)
        c1, c2 = st.columns(2)
        n_q = c1.slider("Number of test questions", 3, 10, 5)
        top_k = c2.slider("top_k chunks retrieved", 2, 8, 4)
        if st.button("🛠️ Generate dataset", disabled=not api_key):
            text = read_document(doc_file) if doc_file else doc_text
            chunks = chunk_text(text)
            if len(chunks) < 3:
                st.error("Document is too short. Please provide at least a few paragraphs.")
            else:
                bar = st.progress(0.0, text="Starting...")
                try:
                    st.session_state["gen_df"] = build_dataset(
                        Judge(api_key, MODELS[model_label], delay), chunks, n_q, top_k, bar)
                    st.session_state["gen_info"] = f"{len(chunks)} chunks (500 chars, 50 overlap), top_k={top_k}"
                except Exception as e:
                    st.error(f"Could not build dataset: {e}")
        if not api_key:
            st.warning("Add your free Groq API key in the sidebar first.")
        if st.session_state.get("gen_df") is not None:
            df = st.session_state["gen_df"]
            st.success(f"Dataset ready from {st.session_state.get('gen_info', '')}. You can edit any cell below.")
            st.caption("Note: questions are auto-generated from the document, so this is a quick demo test set. "
                       "For a real audit, use your own questions and ground truths.")
    else:
        df = DEMO.copy()

    if df is not None:
        missing = {"question", "answer", "contexts", "ground_truth"} - set(df.columns)
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
            df = None

    if df is not None:
        df = st.data_editor(df, num_rows="dynamic", use_container_width=True, key=f"editor_{src[:5]}")
        n_calls = len(df) * len(selected)
        st.caption(f"{len(df)} rows x {len(selected)} metrics = about {n_calls} LLM calls "
                   f"(~{int(n_calls * (delay + 1) / 60) + 1} min).")

        st.subheader("2. Run")
        if st.button("🚀 Run evaluation", type="primary", disabled=not (api_key and selected)):
            judge = Judge(api_key, MODELS[model_label], delay)
            rows, errors = [], []
            bar = st.progress(0.0, text="Starting...")
            total = len(df) * len(selected)
            done = 0
            for i, rec in df.reset_index(drop=True).iterrows():
                q, a, gt = str(rec["question"]), str(rec["answer"]), str(rec["ground_truth"])
                ctx = parse_contexts(rec["contexts"])
                out = {"question": q}
                for m in selected:
                    bar.progress(done / total, text=f"Row {i + 1}/{len(df)} - {METRICS[m]['label']}")
                    try:
                        out[m] = FUNCS[m](judge, q, a, ctx, gt)
                    except Exception as e:
                        out[m] = np.nan
                        errors.append(f"Row {i + 1} / {m}: {e}")
                    done += 1
                rows.append(out)
            bar.progress(1.0, text="Done")
            st.session_state["results"] = pd.DataFrame(rows)
            st.session_state["errors"] = errors
        if not api_key:
            st.warning("Add your free Groq API key in the sidebar to run the evaluation.")

    # ------------------------------------------------------------------ results
    res = st.session_state.get("results")
    if res is not None:
        st.divider()
        st.subheader("3. Results")
        ran = [m for m in METRICS if m in res.columns]
        avg = {m: res[m].mean() for m in ran}

        cols = st.columns(len(ran))
        for c, m in zip(cols, ran):
            v = avg[m]
            c.metric(METRICS[m]["label"], "n/a" if pd.isna(v) else f"{v:.2f}",
                     delta=None if pd.isna(v) else f"{v - METRICS[m]['target']:+.2f} vs target")
            c.caption(status(m, v))

        valid = [avg[m] for m in ran if not pd.isna(avg[m])]
        if valid and all(status(m, avg[m]).startswith("🟢") for m in ran):
            st.success("✅ All metrics meet their targets: this RAG system looks PRODUCTION-READY.")
        elif valid:
            st.warning("⚠️ Some metrics are below target. See diagnosis below.")

        chart_df = pd.DataFrame({
            "Metric": [METRICS[m]["label"] for m in ran],
            "Score": [avg[m] for m in ran],
            "Target": [METRICS[m]["target"] for m in ran],
        }).set_index("Metric")
        st.bar_chart(chart_df)

        st.markdown("#### Diagnosis & fixes")
        diag = []
        for m in ran:
            s = status(m, avg[m])
            if not s.startswith("🟢"):
                diag.append({"Status": s, "Metric": METRICS[m]["label"],
                             "Score": None if pd.isna(avg[m]) else round(avg[m], 2),
                             "Diagnosis": METRICS[m]["diagnosis"], "How to fix": METRICS[m]["fix"]})
        if diag:
            st.dataframe(pd.DataFrame(diag), hide_index=True, use_container_width=True)
        else:
            st.write("Nothing to fix. Continue weekly monitoring.")

        st.markdown("#### Per-question scores")
        show = res.copy()
        for m in ran:
            show[m] = show[m].round(2)
        st.dataframe(show, hide_index=True, use_container_width=True)
        st.download_button("⬇️ Download results CSV", res.to_csv(index=False), "ragas_results.csv", "text/csv")

        if st.session_state.get("errors"):
            with st.expander(f"{len(st.session_state['errors'])} warnings"):
                for e in st.session_state["errors"]:
                    st.write(e)

st.caption("Scores are computed with an LLM-as-judge following the RAGAS methodology, so small run-to-run variation is normal.")
