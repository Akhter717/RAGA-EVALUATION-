import os
import time

import pandas as pd
import streamlit as st

st.set_page_config(page_title="RAG Evaluation Framework", layout="wide")

# ---- secrets come from Streamlit Cloud: App settings -> Secrets ----
try:
    for _k in ("GROQ_API_KEY", "RAG_MODEL", "JUDGE_MODEL", "EMBED_MODEL"):
        if _k in st.secrets and not os.getenv(_k):
            os.environ[_k] = str(st.secrets[_k]).strip()
except Exception:
    pass

if not os.getenv("GROQ_API_KEY"):
    st.error('GROQ_API_KEY not found. Open the app -> Settings -> Secrets and add:\n\nGROQ_API_KEY = "gsk_..."')
    st.stop()

import evaluator
import framework
import rag

ICON = {"PASS": "✅ PASS", "FAIL": "❌ FAIL", "ERROR": "⚠️ ERROR"}


# ------------------------------------------------------------------ helpers
def run_question(kb, q, reference, pages, texts, negative, top_k, category="positive", on_progress=None):
    """Retrieve -> RAG answer -> 13-metric evaluation. Returns one result dict."""
    hits = kb.retrieve(q, top_k)
    chunks = [c for c, _ in hits]
    answer = rag.answer(q, chunks)
    ev = evaluator.evaluate_row(q, answer, chunks, reference, pages, texts, negative, on_progress)
    return {
        "question": q, "category": category, "negative": negative, "answer": answer,
        "reference": "" if negative else reference, "pages": pages, "texts": texts,
        "hits": [(c.source, c.page, s, c.text) for c, s in hits],
        "refused": ev["refused"], "metrics": ev["metrics"], "errors": ev["errors"], "exception": "",
    }


def failed_result(row, exc):
    return {
        "question": row["question"], "category": row["category"], "negative": row["negative"],
        "answer": "", "reference": row["reference"], "pages": row["pages"], "texts": row["texts"],
        "hits": [], "refused": False, "metrics": {m: None for m in framework.METRIC_NAMES},
        "errors": {}, "exception": f"{type(exc).__name__}: {str(exc)[:200]}",
    }


def judged(results, thresholds):
    """Pass/fail is decided here, with the CURRENT thresholds, so changing them updates every row."""
    out = []
    for r in results:
        r = dict(r)
        if r["exception"]:
            r["status"], r["reason"] = "ERROR", "The question could not be processed: " + r["exception"]
        else:
            r["status"], r["reason"] = framework.judge_row(
                r["negative"], r["refused"], r["metrics"], thresholds, bool(r["errors"]))
        out.append(r)
    return out


def metric_rows(res, thresholds):
    rows = []
    has_source = bool(res["pages"] or res["texts"])
    for name in framework.METRIC_NAMES:
        group, meaning, needs, _, direction = framework.METRICS[name]
        v = res["metrics"].get(name)
        th = thresholds[name]
        if framework.is_num(v):
            bad = bool(framework.check_thresholds({name: v}, thresholds))
            score, result, note = f"{v:.3f}", "❌" if bad else "✅", ""
        else:
            score, result = "-", "-"
            if name in res["errors"]:
                result, note = "error", res["errors"][name]
            elif needs == "correct answer" and not res["reference"]:
                note = "needs a correct answer"
            elif needs == "expected source" and not has_source:
                note = "needs an expected source"
            elif res["refused"] or res["negative"]:
                note = "skipped (the RAG did not answer)"
            else:
                note = ""
        rows.append({"Group": group, "Metric": name, "Score": score,
                     "Threshold": f"{'>=' if direction == '>=' else '<='} {th:.2f}",
                     "Result": result, "What it tells you": meaning, "Note": note})
    return pd.DataFrame(rows)


def show_result(res):
    msg = f"{ICON[res['status']]}: {res['reason']}"
    {"PASS": st.success, "FAIL": st.error, "ERROR": st.warning}[res["status"]](msg)
    st.markdown("**RAG answer**")
    st.write(res["answer"] or "(no answer)")
    if res["reference"]:
        st.markdown("**Correct answer (from the dataset or typed)**")
        st.write(res["reference"])
    if res["pages"] or res["texts"]:
        st.caption("Expected source: " + "; ".join(
            ([f"page {', '.join(map(str, res['pages']))}"] if res["pages"] else []) +
            ([f"text \"{t}\"" for t in res["texts"]])))
    if res["hits"]:
        with st.expander(f"Retrieved chunks ({len(res['hits'])})"):
            for i, (src, page, sim, text) in enumerate(res["hits"], 1):
                st.markdown(f"**[{i}] {src} - page {page}** (similarity {sim:.3f})")
                st.caption(text)
    st.markdown("**Metrics**")
    st.dataframe(res["table"], hide_index=True)


def results_frame(results):
    rows = []
    for i, r in enumerate(results, 1):
        row = {"#": i, "Question": r["question"], "Category": r["category"], "Status": ICON[r["status"]]}
        for m in framework.METRIC_NAMES:
            v = r["metrics"].get(m)
            row[m] = round(v, 3) if framework.is_num(v) else None
        row["Issue"] = r["reason"]
        row["RAG answer"] = r["answer"]
        row["Correct answer"] = r["reference"]
        rows.append(row)
    return pd.DataFrame(rows)


def pct(v):
    return "-" if v is None else f"{v * 100:.0f}%"


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Setup")
    files = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=True)
    top_k = st.slider("Chunks given to the RAG (top-k)", 1, 10, 4)
    st.caption(f"RAG model: `{rag.rag_model()}`")
    st.caption(f"Judge model: `{rag.judge_model()}`")
    st.caption(f"Embedding: `{rag.embed_model_name()}`")
    with st.expander("Pass/fail thresholds"):
        st.caption("A question passes when every calculated metric meets its threshold.")
        thresholds = {}
        for name in framework.METRIC_NAMES:
            d = framework.METRICS[name]
            label = f"{name} (maximum)" if d[4] == "<=" else f"{name} (minimum)"
            thresholds[name] = st.number_input(label, 0.0, 1.0, float(d[3]), 0.05, key=f"th_{name}")

# --------------------------------------------------------------- build index
if files:
    sig = tuple((f.name, f.size) for f in files)
    if st.session_state.get("sig") != sig:
        with st.spinner("Reading PDF and building the search index..."):
            chunks = []
            for f in files:
                chunks += rag.chunk_pdf(f.getvalue(), f.name)
            try:
                st.session_state.kb = rag.RagIndex(chunks)
                st.session_state.sig = sig
                st.session_state.pop("single", None)
                st.session_state.pop("dataset_run", None)
            except ValueError as e:
                st.session_state.kb = None
                st.session_state.sig = None
                st.error(str(e))
kb = st.session_state.get("kb") if files else None

# --------------------------------------------------------------------- main
st.title("RAG Evaluation Framework")
st.caption("Tests a RAG system on retrieval quality, answer quality, faithfulness and relevance, "
           "using 13 metrics and pass/fail thresholds.")

if kb is None:
    st.info("Upload a PDF in the sidebar to start.")
    st.stop()

st.success(f"Ready: {len(kb.chunks)} chunks indexed from {len(files)} PDF file(s).")
tab_one, tab_data = st.tabs(["Single question", "Dataset evaluation"])

# ------------------------------------------------------------ single question
with tab_one:
    question = st.text_input("Question", key="q_single")
    reference = st.text_area(
        "Correct answer (optional)", height=80, key="ref_single",
        help="Enables Context Recall, Correctness and Completeness.")
    with st.expander("Expected source (optional): enables Precision, Recall and Hit Rate"):
        pages_in = st.text_input("Expected page number(s), for example 2 or 2,3", key="pages_single")
        texts_in = st.text_input("Expected phrase(s) from the document, separate several with |", key="texts_single")
    negative = st.checkbox("This question is NOT in the document (the RAG should refuse)", key="neg_single")

    if st.button("Ask & Evaluate", type="primary", key="go_single"):
        if not question.strip():
            st.warning("Type a question first.")
        else:
            with st.status("Working...", expanded=True) as status:
                st.write("Retrieving chunks and writing the answer...")
                res = run_question(
                    kb, question.strip(), reference.strip(), framework.parse_pages(pages_in),
                    framework.parse_texts(texts_in), negative, top_k,
                    category="negative" if negative else "positive",
                    on_progress=lambda n: st.write(f"Scoring: {n}..."))
                status.update(label="Done", state="complete", expanded=False)
            st.session_state.single = res

    res = st.session_state.get("single")
    if res:
        res = judged([res], thresholds)[0]
        res["table"] = metric_rows(res, thresholds)
        st.divider()
        show_result(res)
        st.download_button("Download scores (CSV)", res["table"].to_csv(index=False),
                           file_name="scores.csv", mime="text/csv", key="dl_single")

# ----------------------------------------------------------- dataset evaluation
with tab_data:
    st.markdown(
        "Upload a test set and the app runs every question and scores all 13 metrics.\n\n"
        "**Columns:** `question` (required), `reference_answer` (write `NOT_IN_DOCUMENT` when the document "
        "has no answer), `expected_text` (short phrase from the source, several separated with `|`), "
        "`expected_page`, `category`.")
    st.download_button("Download a sample template (CSV)", framework.SAMPLE_TEMPLATE,
                       file_name="dataset_template.csv", mime="text/csv", key="dl_template")
    dfile = st.file_uploader("Test dataset (CSV or JSON)", type=["csv", "json"], key="dataset_file")
    rows = None
    if dfile:
        try:
            rows, warns = framework.load_dataset(dfile.getvalue(), dfile.name)
            for w in warns[:5]:
                st.warning(w)
        except ValueError as e:
            st.error(str(e))
    if rows:
        st.caption(f"{len(rows)} questions loaded "
                   f"({sum(1 for r in rows if r['negative'])} not in the document, "
                   f"{sum(1 for r in rows if r['reference'])} with a correct answer, "
                   f"{sum(1 for r in rows if r['pages'] or r['texts'])} with an expected source).")
        with st.expander("Preview"):
            st.dataframe(pd.DataFrame([{
                "question": r["question"], "reference_answer": r["reference"] or ("NOT_IN_DOCUMENT" if r["negative"] else ""),
                "expected_page": ", ".join(map(str, r["pages"])), "expected_text": " | ".join(r["texts"]),
                "category": r["category"]} for r in rows]), hide_index=True)
        c1, c2 = st.columns(2)
        n_run = c1.number_input("Questions to run", 1, len(rows), min(len(rows), 50), key="n_run")
        pause = c2.number_input("Pause between questions (seconds)", 0.0, 30.0, 1.0, 0.5, key="pause",
                                help="Free Groq accounts have rate limits. A short pause avoids errors.")
        st.caption("Each question needs about 10 to 15 judge calls, so 30 questions can take several minutes.")
        if st.button("Run evaluation", type="primary", key="go_data"):
            results = []
            bar = st.progress(0.0, text="Starting...")
            for i, row in enumerate(rows[:int(n_run)]):
                bar.progress(i / int(n_run), text=f"Question {i + 1} of {int(n_run)}: {row['question'][:70]}")
                try:
                    results.append(run_question(kb, row["question"], row["reference"], row["pages"],
                                                row["texts"], row["negative"], top_k, row["category"]))
                except Exception as e:  # one bad question must not stop the whole run
                    results.append(failed_result(row, e))
                if pause and i < int(n_run) - 1:
                    time.sleep(pause)
            bar.progress(1.0, text="Done")
            st.session_state.dataset_run = {
                "results": results,
                "config": {
                    "Documents": ", ".join(f.name for f in files), "Chunks indexed": str(len(kb.chunks)),
                    "RAG model": rag.rag_model(), "Judge model": rag.judge_model(),
                    "Embedding model": rag.embed_model_name(), "Chunks per question (top-k)": str(top_k),
                },
            }

    run = st.session_state.get("dataset_run")
    if run:
        results = judged(run["results"], thresholds)
        s = framework.summarize(results, thresholds)
        st.divider()
        st.subheader("Summary")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Questions", s["questions"])
        m2.metric("Passed", s["passed"])
        m3.metric("Failed", s["failed"] + s["errors"])
        m4.metric("Pass rate", pct(s["pass_rate"]))
        m5.metric("Correct refusals", f"{s['correct_refusals']} of {s['negative_total']}" if s["negative_total"] else "-")
        for title, group in (("Retrieval metrics", "Retrieval"), ("Generation metrics", "Generation")):
            st.markdown(f"**{title}**")
            st.dataframe(pd.DataFrame([{
                "Metric": m["Metric"], "Average": None if m["Average"] is None else round(m["Average"], 3),
                "Threshold": m["Threshold"], "Pass rate": pct(m["Pass rate"]),
                "Questions scored": m["Questions scored"]} for m in s["metric_rows"] if m["Group"] == group]),
                hide_index=True)
        if len(s["category_rows"]) > 1:
            st.markdown("**By category**")
            st.dataframe(pd.DataFrame([{**c, "Pass rate": pct(c["Pass rate"])} for c in s["category_rows"]]),
                         hide_index=True)

        st.subheader("Results per question")
        df = results_frame(results)
        st.dataframe(df, hide_index=True)

        st.markdown("**Look at one question in detail**")
        labels = [f"{i + 1}. {ICON[r['status']]}  {r['question'][:90]}" for i, r in enumerate(results)]
        pick = st.selectbox("Question", range(len(results)), format_func=lambda i: labels[i], key="pick")
        chosen = results[pick]
        chosen["table"] = metric_rows(chosen, thresholds)
        show_result(chosen)

        st.subheader("Download")
        d1, d2 = st.columns(2)
        d1.download_button("Results (CSV)", df.to_csv(index=False), file_name="rag_evaluation_results.csv",
                           mime="text/csv", key="dl_results")
        d2.download_button("Summary report (Markdown)",
                           framework.build_report(results, thresholds, {**run["config"], "Questions run": str(len(results))}),
                           file_name="rag_evaluation_report.md", mime="text/markdown", key="dl_report")
        st.caption("Scores come from an LLM judge, so they can vary slightly between runs. "
                   "Use them as signals, not proof.")
