"""Test framework pieces that need no LLM: dataset loading, retrieval metrics
(Precision, Recall, Hit Rate), pass/fail thresholds, summary and report."""
import csv
import io
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

NOT_IN_DOC = "NOT_IN_DOCUMENT"

# metric -> (group, what it tells you, needs, default threshold, direction)
# direction ">=": score must be at least the threshold; "<=": score must be at most the threshold.
METRICS: Dict[str, Tuple[str, str, str, float, str]] = {
    "Context Precision": ("Retrieval", "Were the retrieved chunks useful and ranked well? (RAGAS)", "-", 0.70, ">="),
    "Context Recall": ("Retrieval", "Do the retrieved chunks contain everything the correct answer needs? (RAGAS)", "correct answer", 0.70, ">="),
    "Precision": ("Retrieval", "Share of retrieved chunks that match the expected source", "expected source", 0.25, ">="),
    "Recall": ("Retrieval", "Share of the expected sources that were retrieved", "expected source", 0.70, ">="),
    "Hit Rate": ("Retrieval", "Was at least one expected chunk retrieved? (1 = yes)", "expected source", 1.00, ">="),
    "Correctness": ("Generation", "Do the answer's facts match the correct answer? (RAGAS Factual Correctness)", "correct answer", 0.60, ">="),
    "Relevance": ("Generation", "Does the answer address the question? (RAGAS Answer Relevancy)", "-", 0.70, ">="),
    "Completeness": ("Generation", "Does the answer cover all key points of the correct answer? (judge)", "correct answer", 0.70, ">="),
    "Coherence": ("Generation", "Is the answer clear and well structured? (judge)", "-", 0.70, ">="),
    "Conciseness": ("Generation", "Is the answer free of filler and repetition? (judge)", "-", 0.60, ">="),
    "Faithfulness": ("Generation", "Is every claim supported by the retrieved chunks? (RAGAS)", "-", 0.80, ">="),
    "Citation Accuracy": ("Generation", "Do the cited chunk numbers really support the statements? (judge)", "-", 0.70, ">="),
    "Hallucination Rate": ("Generation", "Share of the answer NOT supported by the chunks (1 - Faithfulness)", "-", 0.20, "<="),
}
METRIC_NAMES: List[str] = list(METRICS)
RETRIEVAL_NAMES = [m for m in METRIC_NAMES if METRICS[m][0] == "Retrieval"]
GENERATION_NAMES = [m for m in METRIC_NAMES if METRICS[m][0] == "Generation"]


def default_thresholds() -> Dict[str, float]:
    return {m: METRICS[m][3] for m in METRIC_NAMES}


def is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# ------------------------------------------------------------------ dataset
_ALIASES = {
    "question": ("question", "query", "user_input", "q"),
    "reference": ("reference_answer", "reference", "ground_truth", "ground_truth_answer", "expected_answer", "answer"),
    "pages": ("expected_page", "expected_pages", "page", "pages", "ground_truth_page"),
    "texts": ("expected_text", "expected_texts", "expected_phrase", "ground_truth_context", "source_text"),
    "category": ("category", "type", "tag"),
}


def _key(k) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(k or "").strip().lower()).strip("_")


def parse_pages(value) -> List[int]:
    """'2', '2,3', '2;3' and '2-4' all work."""
    pages: List[int] = []
    for part in re.split(r"[,;|\s]+", str(value or "").strip()):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            pages += list(range(min(a, b), min(max(a, b), min(a, b) + 50) + 1))
        elif part.isdigit():
            pages.append(int(part))
    return sorted(set(pages))


def parse_texts(value) -> List[str]:
    """Several phrases are separated with |"""
    return [t.strip() for t in str(value or "").split("|") if t.strip()]


def is_negative_reference(reference: str) -> bool:
    return _key(reference) in ("not_in_document", "not_in_doc", "no_answer", "unanswerable")


def _row_from_raw(raw: dict) -> dict:
    norm = {_key(k): v for k, v in raw.items()}

    def pick(field):
        for alias in _ALIASES[field]:
            if alias in norm and str(norm[alias] or "").strip():
                return str(norm[alias]).strip()
        return ""

    ref = pick("reference")
    category = pick("category")
    negative = is_negative_reference(ref) or _key(category) in ("negative", "out_of_document", "unanswerable")
    return {
        "question": pick("question"),
        "reference": "" if negative else ref,
        "pages": parse_pages(pick("pages")),
        "texts": parse_texts(pick("texts")),
        "category": category or ("negative" if negative else "positive"),
        "negative": negative,
    }


def load_dataset(file_bytes: bytes, filename: str) -> Tuple[List[dict], List[str]]:
    """Read a CSV or JSON test set. Returns (rows, warnings). Raises ValueError for unusable files."""
    warnings: List[str] = []
    name = (filename or "").lower()
    text = file_bytes.decode("utf-8-sig", errors="replace")
    if name.endswith(".json"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"The JSON file is not valid: {e}")
        if isinstance(data, dict):
            data = data.get("questions") or data.get("data") or data.get("dataset") or []
        if not isinstance(data, list):
            raise ValueError("The JSON file must contain a list of questions.")
        raws = [d for d in data if isinstance(d, dict)]
    else:
        raws = list(csv.DictReader(io.StringIO(text)))
    rows: List[dict] = []
    for i, raw in enumerate(raws, 1):
        row = _row_from_raw(raw)
        if not row["question"]:
            warnings.append(f"Row {i} has no question and was skipped.")
            continue
        rows.append(row)
    if not rows:
        raise ValueError("No questions found. The file needs a 'question' column (or key).")
    return rows, warnings


SAMPLE_TEMPLATE = (
    "question,reference_answer,expected_text,expected_page,category\n"
    "How many test cases are in the smoke suite?,The smoke suite has 25 test cases and must finish in under 15 minutes.,25 test cases,,positive\n"
    "What is the fix deadline for a Critical defect?,Within 24 hours.,,4,positive\n"
    "What is the annual QA budget?,NOT_IN_DOCUMENT,,,negative\n"
)


# ------------------------------------------------------ small LLM-output helpers
def likert_to_unit(n) -> Optional[float]:
    """Judge scores 1 to 5 -> 0 to 1. Anything else -> None."""
    try:
        x = float(n)
    except (TypeError, ValueError):
        return None
    return (x - 1) / 4 if 1 <= x <= 5 else None


_CITE_RE = re.compile(r"\[\s*(\d+(?:\s*[,;]\s*\d+)*)\s*\]")


def cited_numbers(text: str) -> List[int]:
    """Chunk numbers cited in an answer, e.g. 'x [1][2, 3]' -> [1, 2, 3]."""
    nums = set()
    for m in _CITE_RE.finditer(text or ""):
        nums.update(int(n) for n in re.findall(r"\d+", m.group(1)))
    return sorted(nums)


# ------------------------------------------------ retrieval metrics (no LLM)
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def retrieval_metrics(hits: List[Tuple[int, str]], pages: List[int], texts: List[str]) -> Dict[str, Optional[float]]:
    """hits = [(page, chunk_text), ...] in retrieval order.
    A chunk is relevant if it is on an expected page OR contains an expected phrase.
    Precision = relevant chunks / retrieved chunks
    Recall    = expected items (pages and phrases) found in the retrieved chunks / all expected items
    Hit Rate  = 1 if at least one chunk is relevant, else 0.
    Returns None for all three when the row has no expected source."""
    items = [("p", p) for p in pages] + [("t", _norm(t)) for t in texts]
    if not items:
        return {"Precision": None, "Recall": None, "Hit Rate": None}
    norm_chunks = [(page, _norm(text)) for page, text in hits]

    def matches(chunk, item):
        page, text = chunk
        return page == item[1] if item[0] == "p" else item[1] in text

    relevant = sum(1 for c in norm_chunks if any(matches(c, it) for it in items))
    covered = sum(1 for it in items if any(matches(c, it) for c in norm_chunks))
    return {
        "Precision": relevant / len(norm_chunks) if norm_chunks else 0.0,
        "Recall": covered / len(items),
        "Hit Rate": 1.0 if relevant else 0.0,
    }


# ------------------------------------------------------------ pass / fail
def check_thresholds(metrics: Dict[str, object], thresholds: Dict[str, float]) -> List[str]:
    """Names of metrics that miss their threshold. Metrics without a number are ignored."""
    failed = []
    for name in METRIC_NAMES:
        v = metrics.get(name)
        th = thresholds.get(name)
        if not is_num(v) or th is None:
            continue
        ok = v >= th - 1e-9 if METRICS[name][4] == ">=" else v <= th + 1e-9
        if not ok:
            failed.append(name)
    return failed


def judge_row(negative: bool, refused: bool, metrics: Dict[str, object],
              thresholds: Dict[str, float], had_error: bool = False,
              has_ground_truth: bool = True) -> Tuple[str, str]:
    """Return (status, reason). Status is PASS, FAIL, ERROR or NOTE.
    NOTE = the RAG refused and nothing tells us whether the answer exists in the document."""
    if negative:
        if refused:
            return "PASS", "Correct refusal: the question is not in the document and the RAG said so."
        return "FAIL", "The question is not in the document, but the RAG gave an answer (possible hallucination)."
    if refused and not has_ground_truth:
        return "NOTE", ("The RAG said it does not know. No correct answer or expected source was given, so this "
                        "cannot be judged. If the question is not in the document, mark it that way (checkbox, or "
                        "NOT_IN_DOCUMENT in the dataset): that counts as a correct refusal.")
    if refused:
        hit = metrics.get("Hit Rate")
        if is_num(hit) and hit == 0.0:
            return "FAIL", "The RAG refused and the expected chunk was not retrieved (retrieval miss)."
        return "FAIL", "The RAG refused although the answer exists in the document."
    if not any(is_num(metrics.get(m)) for m in METRIC_NAMES):
        return "ERROR", "No metric could be calculated (see the error details)."
    failed = check_thresholds(metrics, thresholds)
    if failed:
        return "FAIL", "Below threshold: " + ", ".join(failed)
    return "PASS", "All calculated metrics meet their thresholds."


# ----------------------------------------------------------------- summary
def summarize(results: List[dict], thresholds: Dict[str, float]) -> dict:
    n = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    errors = sum(1 for r in results if r["status"] == "ERROR")
    notes = sum(1 for r in results if r["status"] == "NOTE")
    neg = [r for r in results if r["negative"]]
    metric_rows = []
    for name in METRIC_NAMES:
        vals = [r["metrics"][name] for r in results if is_num(r["metrics"].get(name))]
        th = thresholds.get(name)
        if vals:
            ok = [v >= th - 1e-9 if METRICS[name][4] == ">=" else v <= th + 1e-9 for v in vals]
            avg, rate = sum(vals) / len(vals), sum(ok) / len(vals)
        else:
            avg, rate = None, None
        metric_rows.append({"Metric": name, "Group": METRICS[name][0], "Average": avg,
                            "Threshold": f"{'>=' if METRICS[name][4] == '>=' else '<='} {th:.2f}",
                            "Pass rate": rate, "Questions scored": len(vals)})
    cats: Dict[str, List[dict]] = {}
    for r in results:
        cats.setdefault(r["category"], []).append(r)
    cat_rows = [{"Category": c, "Questions": len(rs),
                 "Passed": sum(1 for r in rs if r["status"] == "PASS"),
                 "Pass rate": sum(1 for r in rs if r["status"] == "PASS") / len(rs)}
                for c, rs in cats.items()]
    return {
        "questions": n, "passed": passed, "failed": failed, "errors": errors, "notes": notes,
        "pass_rate": passed / (n - notes) if n - notes else 0.0,
        "negative_total": len(neg),
        "correct_refusals": sum(1 for r in neg if r["status"] == "PASS"),
        "metric_rows": metric_rows, "category_rows": cat_rows,
    }


def _fmt(v, pct=False) -> str:
    if v is None:
        return "-"
    return f"{v * 100:.0f}%" if pct else f"{v:.3f}"


def build_report(results: List[dict], thresholds: Dict[str, float], config: Dict[str, str]) -> str:
    """Markdown summary report."""
    s = summarize(results, thresholds)
    L = ["# RAG Evaluation Report", "",
         f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}", "", "## Configuration", ""]
    L += [f"- {k}: {v}" for k, v in config.items()]
    L += ["", "## Overall result", "",
          f"- Questions evaluated: {s['questions']}",
          f"- Passed: {s['passed']}  |  Failed: {s['failed']}  |  Errors: {s['errors']}",
          f"- Pass rate: {s['pass_rate'] * 100:.0f}%"]
    if s["notes"]:
        L.append(f"- Not judged (the RAG refused and no ground truth was given): {s['notes']}")
    if s["negative_total"]:
        L.append(f"- Correct refusals on questions not in the document: {s['correct_refusals']} of {s['negative_total']}")
    for title, group in (("Retrieval metrics", "Retrieval"), ("Generation metrics", "Generation")):
        L += ["", f"## {title}", "", "| Metric | Average | Threshold | Pass rate | Questions scored |", "|---|---|---|---|---|"]
        for m in s["metric_rows"]:
            if m["Group"] == group:
                L.append(f"| {m['Metric']} | {_fmt(m['Average'])} | {m['Threshold']} | "
                         f"{_fmt(m['Pass rate'], True)} | {m['Questions scored']} |")
    if len(s["category_rows"]) > 1:
        L += ["", "## By category", "", "| Category | Questions | Passed | Pass rate |", "|---|---|---|---|"]
        L += [f"| {c['Category']} | {c['Questions']} | {c['Passed']} | {c['Pass rate'] * 100:.0f}% |"
              for c in s["category_rows"]]
    bad = [r for r in results if r["status"] != "PASS"]
    L += ["", "## Questions that did not pass", ""]
    if bad:
        for r in bad:
            L.append(f"- [{r['status']}] {r['question']}  \n  {r['reason']}")
    else:
        L.append("None. Every question passed.")
    L += ["", "Scores come from an LLM judge and can vary slightly between runs. "
              "Use them as signals, not proof."]
    return "\n".join(L)
