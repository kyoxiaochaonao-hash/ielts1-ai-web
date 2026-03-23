import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import requests
from dotenv import load_dotenv
import json
from pathlib import Path
import re
import ast
import traceback
import threading
import uuid
import time
import hashlib
import unicodedata

load_dotenv()

app = Flask(__name__)
CORS(app)

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

IELTS_LIBRARY_DIR = Path(__file__).resolve().parent / "雅思资料库"
_library_cache = {"sig": None, "chunks": []}
APP_STATE = {
    "mode": "practice",
    "part": 1,
    "question_index": 0,
    "answers": [],
    "total_questions": 0,
    "prompts": {},
    "prep_seconds": 30,
    "recent_questions": {"part1": [], "part2": [], "part3": []},
    "round_history": {"part1": [], "part2": [], "part3": []},
}
EXAM_STEPS = [
    {"part": "part1", "type": "question", "in_part_index": 1, "in_part_total": 3},
    {"part": "part1", "type": "question", "in_part_index": 2, "in_part_total": 3},
    {"part": "part1", "type": "question", "in_part_index": 3, "in_part_total": 3},
    {"part": "part2", "type": "cue_card", "in_part_index": 1, "in_part_total": 1},
    {"part": "part3", "type": "question", "in_part_index": 1, "in_part_total": 2},
    {"part": "part3", "type": "question", "in_part_index": 2, "in_part_total": 2},
]
TOTAL_STEPS = len(EXAM_STEPS)
JOBS = {}
JOBS_LOCK = threading.Lock()
JOB_TTL_SECONDS = 10 * 60

def _normalize_part(part) -> str:
    if part is None:
        return "part1"
    if isinstance(part, int):
        return f"part{part}" if part in (1, 2, 3) else "part1"
    s = str(part).strip().lower()
    if s in {"1", "part1", "p1"}:
        return "part1"
    if s in {"2", "part2", "p2"}:
        return "part2"
    if s in {"3", "part3", "p3"}:
        return "part3"
    return "part1"

def _is_api_path(path: str) -> bool:
    return path in {
        "/set_mode",
        "/submit_answer",
        "/generate_question",
        "/analyze_answer",
        "/pronunciation_score",
    }

@app.errorhandler(Exception)
def handle_exception(e):
    path = getattr(request, "path", "")
    if _is_api_path(path):
        print("Unhandled exception:", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e), "path": path}), 500
    raise e

def _cleanup_jobs():
    now = time.time()
    with JOBS_LOCK:
        expired = [job_id for job_id, job in JOBS.items() if now - job.get("created_at", now) > JOB_TTL_SECONDS]
        for job_id in expired:
            JOBS.pop(job_id, None)

def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _chunk_text(text: str, max_chars: int = 1200, min_chars: int = 250):
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    chunks = []
    buf = ""
    for b in blocks:
        if not buf:
            buf = b
            continue
        if len(buf) + 2 + len(b) <= max_chars:
            buf = f"{buf}\n\n{b}"
        else:
            if len(buf) >= min_chars:
                chunks.append(buf)
            buf = b
    if buf and len(buf) >= min_chars:
        chunks.append(buf)
    if not chunks and text:
        return [text[:max_chars]]
    return chunks

def _read_txt(path: Path) -> str:
    try:
        return _normalize_text(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        try:
            return _normalize_text(path.read_text(encoding="gbk", errors="ignore"))
        except Exception:
            return ""

def _read_docx(path: Path) -> str:
    try:
        from docx import Document
        doc = Document(str(path))
        paras = []
        for p in doc.paragraphs:
            t = (p.text or "").strip()
            if t:
                paras.append(t)
        return _normalize_text("\n".join(paras))
    except Exception as e:
        print(f"Failed to read docx: {path.name}: {str(e)}")
        return ""

def load_ielts_library():
    if not IELTS_LIBRARY_DIR.exists():
        _library_cache["sig"] = None
        _library_cache["chunks"] = []
        return []

    files = sorted([p for p in IELTS_LIBRARY_DIR.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".docx"}])
    sig = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files)
    if _library_cache["sig"] == sig:
        return _library_cache["chunks"]

    chunks = []
    for p in files:
        if p.suffix.lower() == ".txt":
            text = _read_txt(p)
        else:
            text = _read_docx(p)
        if not text:
            continue
        for c in _chunk_text(text):
            chunks.append({"source": p.name, "text": c})

    _library_cache["sig"] = sig
    _library_cache["chunks"] = chunks
    print(f"IELTS library loaded: {len(files)} files, {len(chunks)} chunks")
    return chunks

def _tokens(s: str):
    s = s.lower()
    words = re.findall(r"[a-z]{2,}", s)
    cjk = [ch for ch in s if "\u4e00" <= ch <= "\u9fff"]
    return set(words + cjk)

def retrieve_library_snippets(query: str, k: int = 6):
    chunks = load_ielts_library()
    if not chunks:
        return []
    q = _tokens(query)
    scored = []
    for ch in chunks:
        t = _tokens(ch["text"])
        overlap = len(q.intersection(t))
        bonus = 8 if ("评分" in ch["source"] or "标准" in ch["source"]) else 0
        scored.append((overlap + bonus, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [ch for score, ch in scored[:k] if score > 0] or [scored[0][1]]
    return top

def _part_to_number(part: str) -> int:
    if part == "part2":
        return 2
    if part == "part3":
        return 3
    return 1

def _reset_state(mode: str, part: str | None = None):
    mode = (mode or "practice").strip().lower()
    if mode not in {"practice", "exam"}:
        mode = "practice"

    if mode == "exam":
        APP_STATE["mode"] = "exam"
        APP_STATE["question_index"] = 0
        APP_STATE["answers"] = []
        APP_STATE["total_questions"] = TOTAL_STEPS
        APP_STATE["part"] = 1
        APP_STATE["prompts"] = {}
        APP_STATE["round_history"] = {"part1": [], "part2": [], "part3": []}
        return

    APP_STATE["mode"] = "practice"
    APP_STATE["question_index"] = 0
    APP_STATE["answers"] = []
    APP_STATE["total_questions"] = TOTAL_STEPS
    APP_STATE["part"] = _part_to_number(part or "part1")
    APP_STATE["prompts"] = {}
    APP_STATE["round_history"] = {"part1": [], "part2": [], "part3": []}

    if (part or "part1").strip().lower() == "part2":
        APP_STATE["question_index"] = 3
        APP_STATE["part"] = 2
    elif (part or "part1").strip().lower() == "part3":
        APP_STATE["question_index"] = 4
        APP_STATE["part"] = 3

def _generate_question_text(ielts_part: str, api_key: str) -> str:
    ielts_part = _normalize_part(ielts_part)
    avoid = []
    avoid.extend(APP_STATE.get("recent_questions", {}).get(ielts_part, []))
    avoid.extend(APP_STATE.get("round_history", {}).get(ielts_part, []))
    recent_topics = [_infer_topic(x) for x in avoid][-4:]
    q, warning = _generate_question_json(ielts_part, api_key, avoid=avoid, avoid_topics=recent_topics)
    if warning:
        print(f"Question warning: {warning}")
    _remember_question(ielts_part, q)
    return q

def _normalize_question_for_compare(q: str) -> str:
    q = (q or "").lower().strip()
    q = re.sub(r"['\"\u201c\u201d\u2018\u2019]", "", q)
    q = re.sub(r"\s+", " ", q)
    q = re.sub(r"[^a-z0-9 ?!,-]", "", q)
    return q

def _sanitize_question(q: str) -> str:
    q = (q or "").strip()
    q = q.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    q = q.replace("\"", "").replace("'", "").replace("\u201c", "").replace("\u201d", "").replace("\u2018", "").replace("\u2019", "")
    q = re.sub(r"^\s*[\-\*\u2022]\s+", "", q)
    q = re.sub(r"^\s*\(?\d+\)?[.)]\s*", "", q)
    q = re.sub(r"^\s*Q\d+\s*[:\-]\s*", "", q, flags=re.IGNORECASE)
    q = re.sub(r"\s+", " ", q).strip()
    return q

def _remember_question(part: str, q: str):
    part = _normalize_part(part)
    q = _sanitize_question(q)
    if not q:
        return
    if part not in APP_STATE["recent_questions"]:
        APP_STATE["recent_questions"][part] = []
    normalized = _normalize_question_for_compare(q)
    existing = [_normalize_question_for_compare(x) for x in APP_STATE["recent_questions"][part]]
    if normalized in existing:
        return
    APP_STATE["recent_questions"][part].append(q)
    if "round_history" in APP_STATE:
        if part not in APP_STATE["round_history"]:
            APP_STATE["round_history"][part] = []
        APP_STATE["round_history"][part].append(q)
    if len(APP_STATE["recent_questions"][part]) > 30:
        APP_STATE["recent_questions"][part] = APP_STATE["recent_questions"][part][-30:]
    if "round_history" in APP_STATE and len(APP_STATE["round_history"].get(part, [])) > 30:
        APP_STATE["round_history"][part] = APP_STATE["round_history"][part][-30:]

def _canonicalize_for_similarity(q: str) -> str:
    q = (q or "").lower().strip()
    q = re.sub(r"[\"'’‘“”]", "", q)
    q = re.sub(r"[^a-z0-9 ?!,-]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    synonyms = {
        "free time": "leisure",
        "spare time": "leisure",
        "hometown": "home town",
        "job": "work",
        "career": "work",
        "studying": "study",
        "studies": "study",
        "technology": "tech",
        "smartphone": "phone",
        "mobile phone": "phone",
    }
    for k, v in synonyms.items():
        q = q.replace(k, v)
    return q

def _question_tokens(q: str):
    q = _canonicalize_for_similarity(q)
    tokens = re.findall(r"[a-z0-9]+", q)
    stop = {
        "do","you","your","usually","often","what","why","how","when","where","which",
        "is","are","was","were","a","an","the","to","of","in","on","for","and","or",
        "would","could","can","like","think","about","tell","me","please",
    }
    return {t for t in tokens if t not in stop}

def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def _is_semantically_similar(q1: str, q2: str) -> bool:
    t1 = _question_tokens(q1)
    t2 = _question_tokens(q2)
    sim = _jaccard(t1, t2)
    return sim >= 0.72

def _infer_topic(q: str) -> str:
    ql = (q or "").lower()
    topics = {
        "work": ["work", "job", "career", "boss", "colleague", "office"],
        "study": ["study", "school", "university", "class", "teacher", "exam"],
        "hometown": ["hometown", "home town", "where you grew up", "city", "town", "village"],
        "hobbies": ["hobby", "free time", "spare time", "leisure", "weekend", "sports", "music", "reading"],
        "technology": ["technology", "tech", "internet", "phone", "social media", "app"],
        "travel": ["travel", "trip", "holiday", "vacation", "tourism"],
        "food": ["food", "cook", "cooking", "restaurant", "meal"],
    }
    for topic, keys in topics.items():
        if any(k in ql for k in keys):
            return topic
    return "general"

def _generate_question_json(ielts_part: str, api_key: str, avoid=None, avoid_topics=None, index: int = 0):
    ielts_part = _normalize_part(ielts_part)
    avoid = avoid or []
    avoid_clean = [_sanitize_question(a) for a in avoid if a]
    avoid_clean = [a for a in avoid_clean if a]
    avoid_text = " | ".join(avoid_clean[:12])
    avoid_topics = avoid_topics or []

    system_prompt = (
        f"You are an IELTS speaking examiner. Generate ONE concise IELTS speaking {ielts_part} prompt.\n"
        "Output format must be STRICT JSON ONLY: {\"question\":\"...\"}\n"
        "Rules:\n"
        "- Return only JSON. No markdown. No extra text.\n"
        "- The question field must NOT contain any quote characters (\" or ').\n"
        "- No numbering (no '1.' '2.' 'Q1:' etc).\n"
        "- No extra newlines; single line only.\n"
        "- Natural IELTS speaking style.\n"
    )
    if avoid_text:
        system_prompt += f"\nAvoid repeating any of these questions (do not paraphrase them closely): {avoid_text}\n"
    if avoid_topics:
        system_prompt += f"\nTry to avoid these topics in this question: {', '.join(avoid_topics[:6])}\n"

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Generate a new question now."},
        ],
        "stream": False,
        "max_tokens": 120,
        "temperature": 0.8,
    }

    last_error = None
    for _ in range(3):
        try:
            response = requests.post(
                DEEPSEEK_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                json=payload,
                timeout=20,
            )
            if response.status_code != 200:
                last_error = f"DeepSeek API returned {response.status_code}: {response.text}"
                continue
            data = response.json()
            raw = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
            obj = _parse_model_output(raw)
            q = _sanitize_question(str(obj.get("question", "")))
            if not q:
                last_error = "Empty question from model"
                continue
            if "\"" in q or "'" in q:
                q = _sanitize_question(q)
            normalized = _normalize_question_for_compare(q)
            if any(_normalize_question_for_compare(a) == normalized for a in avoid_clean):
                last_error = "Repeated question (exact/normalized)"
                payload["temperature"] = min(1.2, (payload.get("temperature") or 0.8) + 0.2)
                continue
            if any(_is_semantically_similar(q, a) for a in avoid_clean):
                last_error = "Repeated question (semantic)"
                payload["temperature"] = min(1.2, (payload.get("temperature") or 0.8) + 0.2)
                continue
            topic = _infer_topic(q)
            if avoid_topics and topic in avoid_topics and topic != "general":
                last_error = "Topic repetition"
                payload["temperature"] = min(1.2, (payload.get("temperature") or 0.8) + 0.2)
                continue
            if len(q) > 180:
                q = q[:180].rstrip()
            return q, None
        except Exception as e:
            last_error = str(e)
            payload["temperature"] = min(1.2, (payload.get("temperature") or 0.8) + 0.2)

    q = _sanitize_question(_fallback_question(ielts_part, index=index))
    return q, last_error or "Fallback used"

def _generate_part2_cue_card(api_key: str, history_questions=None) -> str:
    avoid = []
    avoid.extend(APP_STATE.get("recent_questions", {}).get("part2", []))
    avoid.extend(APP_STATE.get("round_history", {}).get("part2", []))
    if history_questions:
        avoid.extend(history_questions)
    recent_topics = [_infer_topic(x) for x in APP_STATE.get("round_history", {}).get("part2", [])][-3:]
    q, warning = _generate_question_json("part2", api_key, avoid=avoid, avoid_topics=recent_topics, index=3)
    if warning:
        print(f"Part2 warning: {warning}")
    _remember_question("part2", q)
    return q

def _fallback_question(part: str, index: int = 0) -> str:
    part = (part or "part1").strip().lower()
    if part == "part2":
        return (
            "Describe a time you helped someone.\n"
            "You should say:\n"
            "- who the person was\n"
            "- what you did\n"
            "- why you helped\n"
            "- and explain how you felt about it"
        )
    if part == "part3":
        questions = [
            "Why do some people find it difficult to help others?",
            "How can schools encourage students to support each other?",
            "Do you think communities are more supportive now than in the past? Why?",
        ]
        return questions[index % len(questions)]
    questions = [
        "Do you like helping other people? Why or why not?",
        "When was the last time someone helped you?",
        "What kinds of help do people usually need in daily life?",
    ]
    return questions[index % len(questions)]

def _safe_generate_question(part: str, api_key: str, index: int = 0, history_questions=None):
    try:
        part = _normalize_part(part)
        avoid = []
        avoid.extend(APP_STATE.get("recent_questions", {}).get(part, []))
        avoid.extend(APP_STATE.get("round_history", {}).get(part, []))
        if history_questions:
            avoid.extend(history_questions)
        recent_topics = [_infer_topic(x) for x in avoid][-4:]
        q, warning = _generate_question_json(part, api_key, avoid=avoid, avoid_topics=recent_topics, index=index)
        _remember_question(part, q)
        return q, warning
    except Exception as e:
        part = _normalize_part(part)
        q = _sanitize_question(_fallback_question(part, index=index))
        _remember_question(part, q)
        return q, str(e)

def _get_step(step_index: int):
    if step_index < 0:
        step_index = 0
    if step_index >= TOTAL_STEPS:
        step_index = TOTAL_STEPS - 1
    step = dict(EXAM_STEPS[step_index])
    if step.get("type") == "cue_card":
        step["prep_seconds"] = int(APP_STATE.get("prep_seconds") or 30)
    return step

def _progress_label(step_index: int) -> str:
    step = _get_step(step_index)
    part_num = _part_to_number(step["part"])
    if step["part"] == "part2":
        return f"Part {part_num} - Cue Card"
    return f"Part {part_num} - Q{step['in_part_index']}/{step['in_part_total']}"

def _get_or_generate_prompt(step_index: int, api_key: str, history_questions=None):
    step = _get_step(step_index)
    cached = APP_STATE.get("prompts", {}).get(step_index)
    if cached:
        return cached, None

    if step["type"] == "cue_card":
        try:
            text = _generate_part2_cue_card(api_key, history_questions=history_questions)
            APP_STATE["prompts"][step_index] = text
            return text, None
        except Exception as e:
            text = _fallback_question("part2")
            APP_STATE["prompts"][step_index] = text
            return text, str(e)

    text, warning = _safe_generate_question(step["part"], api_key, index=step_index, history_questions=history_questions)
    APP_STATE["prompts"][step_index] = text
    return text, warning

def _score_transcript(question: str, transcript: str, api_key: str):
    transcript = (transcript or "").strip()
    question = (question or "").strip()
    transcript_for_model = _clip(transcript, 1300)

    query = f"{question}\n{transcript_for_model}"
    refs = retrieve_library_snippets(query, k=1)
    refs_text = _format_refs_for_prompt(refs, max_total_chars=520, max_each=420)

    evidence_quotes = _extract_quotes(transcript_for_model, max_quotes=5)
    vocab = _vocab_candidates(transcript_for_model, max_items=8)
    markers = _grammar_markers(transcript_for_model)
    topic_ratio = _topic_overlap_ratio(question, transcript_for_model)
    word_count = len(re.findall(r"[a-zA-Z']+", transcript_for_model))

    filler_words = ["um", "uh", "like", "you know", "kind of", "sort of"]
    t_low = (transcript_for_model or "").lower()
    filler_counts = {w: t_low.count(w) for w in filler_words if w in t_low}

    tokens = _keywords(transcript_for_model)
    freq = {}
    for tok in tokens:
        freq[tok] = freq.get(tok, 0) + 1
    repeated = sorted([(k, v) for k, v in freq.items() if v >= 2], key=lambda x: (-x[1], x[0]))[:6]

    sentences = [s.strip() for s in re.split(r"[.!?]+", transcript_for_model) if s.strip()]
    sentence_count = len(sentences)
    avg_words = (word_count / sentence_count) if sentence_count else float(word_count)

    question_keywords = _keywords(question)[:8]
    rewrite_target = evidence_quotes[-1] if evidence_quotes else _clip(transcript_for_model, 120)

    issue_hints = []
    if topic_ratio < 0.18:
        issue_hints.append("off_topic")
    if word_count < 35:
        issue_hints.append("too_short")
    if filler_counts:
        issue_hints.append("filler_words")
    if repeated:
        issue_hints.append("repetition")
    if not any(m in (markers or []) for m in ["for example", "because"]):
        issue_hints.append("missing_reason_example")
    if sentence_count >= 2 and avg_words < 8:
        issue_hints.append("very_short_sentences")

    evidence = {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_words_per_sentence": round(avg_words, 1),
        "on_topic_ratio": round(topic_ratio, 2),
        "question_keywords": question_keywords[:6],
        "vocabulary_candidates": vocab[:8],
        "grammar_markers_found": markers[:8],
        "filler_words_found": filler_counts,
        "repeated_words": repeated,
        "evidence_quotes": evidence_quotes[:5],
        "rewrite_target": rewrite_target,
        "possible_issue_hints": issue_hints,
    }
    evidence_sig_src = json.dumps(evidence, ensure_ascii=False, sort_keys=True).encode("utf-8", errors="ignore")
    evidence_signature = hashlib.sha1(evidence_sig_src).hexdigest()[:10]

    system_prompt = f"""You are an IELTS Speaking examiner. Grade the candidate's answer and give actionable feedback.

Reference notes (use when helpful, don't copy):
{refs_text}

Evidence (must use; keep feedback grounded in what the candidate actually said):
{json.dumps(evidence, ensure_ascii=False)}

Assessment criteria:
- Fluency and Coherence
- Lexical Resource
- Grammatical Range and Accuracy

Requirements (avoid template tone, be specific):
1) You must cite or closely paraphrase at least 2 quotes from evidence_quotes, and explain why each is good or problematic.
2) You must provide at least 1 rewrite example based on rewrite_target, and explain the reason for your changes (e.g. adding reason/example, fixing grammar, improving precision).
3) No generic lines like "good but can improve". If you praise something, point to a specific word/phrase/sentence.
4) Make the feedback noticeably different for weak vs strong answers.

Output format (JSON only, no markdown, no extra text):
{{
  "fluency": number,
  "vocabulary": number,
  "grammar": number,
  "overall": number,
  "feedback_en": "Detailed natural feedback in English. You may use \\n for paragraphs. Must include at least 2 quotes/paraphrases from evidence_quotes.",
  "feedback_zh": "中文详细评价（可用 \\n 分段），必须包含至少2处来自 evidence_quotes 的引用/改写。",
  "improvement_en": "Actionable improvement steps in English. Must include at least 1 rewrite example based on rewrite_target.",
  "improvement_zh": "中文可执行建议，必须包含至少1条基于 rewrite_target 的改写示例。"
}}

Strict constraints:
- Return JSON only, no markdown.
- Use double quotes for all keys and string values.
- Do not include raw newlines in JSON strings; use \\n instead.
- feedback_en and improvement_en must be English only (no Chinese characters).
- feedback_zh and improvement_zh must be Chinese only (no English sentences).
- Do not use markdown symbols like **, __, or backticks.`"""

    user_content = f"Question: {question}\nTranscript: {transcript_for_model}"

    response = requests.post(
        DEEPSEEK_API_URL,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "max_tokens": 600,
            "temperature": 0.35,
        },
        timeout=35,
    )
    if response.status_code != 200:
        fallback = _fallback_result(transcript_for_model)
        fallback["warning"] = f"DeepSeek API Error: {response.text}"
        return fallback

    response_data = response.json()
    content = (response_data["choices"][0]["message"]["content"] or "").strip()
    try:
        final_result = _parse_model_output(content)
    except Exception as e:
        fallback = _fallback_result(transcript_for_model)
        snippet = _clip(_extract_json_candidate(content), 260)
        fallback["warning"] = f"Model output parse error: {str(e)}"
        fallback["raw_snippet"] = snippet
        return fallback
    final_result["_meta"] = {"source": "model", "prompt_version": "evidence_rich_v3", "evidence_signature": evidence_signature}
    if "feedback_en" in final_result and "feedback" not in final_result:
        final_result["feedback"] = final_result["feedback_en"]
    if "suggestions_en" in final_result and "suggestions" not in final_result:
        final_result["suggestions"] = final_result["suggestions_en"]
    if "overall" in final_result and "overall_band" not in final_result:
        final_result["overall_band"] = final_result.get("overall")
    if "fluency" in final_result and "fluency_score" not in final_result:
        final_result["fluency_score"] = final_result.get("fluency")
    if "vocabulary" in final_result and "vocabulary_score" not in final_result:
        final_result["vocabulary_score"] = final_result.get("vocabulary")
    if "grammar" in final_result and "grammar_score" not in final_result:
        final_result["grammar_score"] = final_result.get("grammar")
    if "feedback" in final_result:
        if "feedback_en" not in final_result:
            final_result["feedback_en"] = final_result.get("feedback")
        if "feedback_zh" not in final_result:
            final_result["feedback_zh"] = final_result.get("feedback")
    if "improvement" in final_result:
        if "suggestions_en" not in final_result:
            final_result["suggestions_en"] = final_result.get("improvement")
        if "suggestions_zh" not in final_result:
            final_result["suggestions_zh"] = final_result.get("improvement")
    if "improvement_en" in final_result and "suggestions_en" not in final_result:
        final_result["suggestions_en"] = final_result.get("improvement_en")
    if "improvement_zh" in final_result and "suggestions_zh" not in final_result:
        final_result["suggestions_zh"] = final_result.get("improvement_zh")
    if "feedback_en" in final_result:
        final_result["feedback_en"] = str(final_result.get("feedback_en") or "")
    if "feedback_zh" in final_result:
        final_result["feedback_zh"] = str(final_result.get("feedback_zh") or "")
    final_result = _ensure_bilingual_and_clean(final_result, api_key)
    final_result["pronunciation"] = _simulate_pronunciation(transcript_for_model)
    return final_result

def _clip(s: str, max_chars: int) -> str:
    s = s or ""
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1].rstrip() + "…"

def _format_refs_for_prompt(refs, max_total_chars: int = 1800, max_each: int = 520):
    parts = []
    total = 0
    for i, r in enumerate(refs):
        text = _clip(r.get("text", ""), max_each)
        header = f"[{i+1}] {r.get('source', 'ref')}\n"
        block = header + text
        if total + len(block) > max_total_chars:
            break
        parts.append(block)
        total += len(block) + 2
    return "\n\n".join(parts)

def _clean_display_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\\n", "\n")
    s = s.replace('\\"', '"')
    s = s.replace("**", "")
    s = s.replace("__", "")
    s = s.replace("`", "")
    return s.strip()

def _translate_text(text: str, target_lang: str, api_key: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    target_lang = (target_lang or "").strip().lower()
    if target_lang not in {"zh", "en"}:
        return text

    sys = "Translate the following text into Chinese. Keep line breaks. Output only the translated text." if target_lang == "zh" else "Translate the following text into natural English. Keep line breaks. Output only the translated text."
    try:
        r = requests.post(
            DEEPSEEK_API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": text},
                ],
                "stream": False,
                "max_tokens": 420,
                "temperature": 0.2,
            },
            timeout=20,
        )
        if r.status_code != 200:
            return text
        data = r.json()
        out = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
        return out or text
    except Exception:
        return text

def _ensure_bilingual_and_clean(result: dict, api_key: str) -> dict:
    if not isinstance(result, dict):
        return result

    feedback_en = _clean_display_text(result.get("feedback_en") or "")
    feedback_zh = _clean_display_text(result.get("feedback_zh") or "")
    improve_en = _clean_display_text(result.get("improvement_en") or result.get("suggestions_en") or "")
    improve_zh = _clean_display_text(result.get("improvement_zh") or result.get("suggestions_zh") or "")

    if not feedback_zh and feedback_en:
        feedback_zh = _clean_display_text(_translate_text(feedback_en, "zh", api_key))
    if not feedback_en and feedback_zh:
        feedback_en = _clean_display_text(_translate_text(feedback_zh, "en", api_key))

    if not improve_zh and improve_en:
        improve_zh = _clean_display_text(_translate_text(improve_en, "zh", api_key))
    if not improve_en and improve_zh:
        improve_en = _clean_display_text(_translate_text(improve_zh, "en", api_key))

    result["feedback_en"] = feedback_en
    result["feedback_zh"] = feedback_zh
    result["suggestions_en"] = improve_en
    result["suggestions_zh"] = improve_zh
    if "improvement_en" in result:
        result["improvement_en"] = improve_en
    if "improvement_zh" in result:
        result["improvement_zh"] = improve_zh
    return result

def _simulate_pronunciation(transcript: str) -> dict:
    t = (transcript or "").strip()
    t_low = t.lower()
    word_count = len(re.findall(r"[a-zA-Z']+", t))
    sentences = [s.strip() for s in re.split(r"[.!?]+", t) if s.strip()]
    sentence_count = len(sentences)

    filler_words = ["um", "uh", "like", "you know", "kind of", "sort of"]
    filler_total = sum(t_low.count(w) for w in filler_words)
    endings = len(re.findall(r"\b[a-zA-Z]+(?:ed|s)\b", t_low))

    def clamp(x, lo, hi):
        try:
            x = float(x)
        except Exception:
            x = float(lo)
        return max(float(lo), min(float(hi), x))

    completeness = 46 + min(46, word_count * 0.8)
    if sentence_count >= 3:
        completeness += 4
    if word_count < 18:
        completeness -= 10
    completeness = clamp(completeness, 40, 92)

    fluency = 56 + min(18, sentence_count * 3.0)
    fluency -= min(14, filler_total * 3.0)
    if word_count < 25:
        fluency -= 6
    fluency = clamp(fluency, 40, 92)

    accuracy = 60 + min(10, word_count / 20.0)
    if word_count >= 35 and endings <= 1:
        accuracy -= 4
    if re.search(r"\b(i|he|she|they)\s+go\b", t_low):
        accuracy -= 3
    accuracy = clamp(accuracy, 42, 92)

    pronunciation_score = clamp(0.36 * accuracy + 0.34 * fluency + 0.30 * completeness, 40, 92)

    feedback_en = "Focus on clearer word endings (-s/-ed) and smoother linking between words. Shadow 2–3 short sentences from your answer at a steady pace, then record and compare."
    feedback_zh = "优先练清词尾（-s/-ed）和连读。建议从你的回答里选2–3个短句做跟读：先慢后快，录音对比改进。"

    return {
        "enabled": False,
        "provider": "ai_simulated",
        "accuracy": round(accuracy),
        "fluency": round(fluency),
        "completeness": round(completeness),
        "pronunciation_score": round(pronunciation_score),
        "pronunciation_feedback_en": feedback_en,
        "pronunciation_feedback_zh": feedback_zh,
        "focus_points": [
            {"point_en": "Word endings", "point_zh": "词尾发音", "example_words": ["worked", "likes", "watched"]},
            {"point_en": "Linking", "point_zh": "连读", "example_words": ["kind of", "a lot of", "going to"]},
            {"point_en": "Sentence stress", "point_zh": "句子重音", "example_words": ["really important", "main reason", "I would say"]},
        ],
    }

def _fallback_result(transcript: str):
    word_count = len(re.findall(r"[a-zA-Z']+", transcript or ""))
    overall = 5.5 if word_count >= 70 else 5.0 if word_count >= 35 else 4.5
    return {
        "_meta": {"source": "fallback", "prompt_version": "fallback_v1"},
        "overall_band": overall,
        "grammar_score": max(4.0, overall - 0.5),
        "vocabulary_score": max(4.0, overall - 0.5),
        "fluency_score": max(4.0, overall - 0.5),
        "feedback_en": "1. Your answer is generally on-topic, but the response is a bit short, which limits development.\n2. Vocabulary: try to add 2–3 topic-specific collocations instead of general words.\n3. Grammar: aim for at least two complex sentences (e.g., relative/conditional clauses).\n4. Issues: some ideas may sound list-like; add linking phrases and a clear example.\n5. Band positioning: this sits around the current band shown; to move up, extend ideas with reasons + examples and improve sentence variety.",
        "feedback_zh": "1. 回答基本切题，但整体偏短，观点展开不足。\n2. 词汇：建议加入2–3个话题相关的固定搭配，减少泛泛用词。\n3. 语法：至少使用两句复杂句（如定语从句/条件句）来提升句式层次。\n4. 问题：观点可能偏罗列，建议用连接词并补充一个具体例子。\n5. 分数定位：大致处于当前分段；想提升需要“理由+例子”的展开，以及更丰富的句式。",
        "suggestions_en": "- Fluency & Coherence: Speak for 60–90 seconds with one clear main idea, then add a reason and one example.\n- Lexical Resource: Build a mini word bank for this topic (5 collocations) and force yourself to use 2 each time.\n- Grammatical Range: Rewrite 3 simple sentences into complex ones using because/although/if or relative clauses.",
        "suggestions_zh": "- 流利度与连贯性：围绕一个主观点连续说60–90秒，再补充理由与一个例子。\n- 词汇：为该话题整理5个常用搭配，每次练习强制使用其中2个。\n- 语法：把3句简单句改写为复杂句（because/although/if/定语从句等）。",
        "pronunciation": _simulate_pronunciation(transcript or ""),
    }

def _keywords(text: str):
    tokens = re.findall(r"[a-zA-Z]{2,}", (text or "").lower())
    stop = {
        "do","you","your","usually","often","what","why","how","when","where","which",
        "is","are","was","were","a","an","the","to","of","in","on","for","and","or",
        "would","could","can","like","think","about","tell","me","please","i","we","they",
        "he","she","it","my","our","their","this","that","these","those","because",
    }
    return [t for t in tokens if t not in stop]

def _topic_overlap_ratio(question: str, transcript: str) -> float:
    q = set(_keywords(question))
    a = set(_keywords(transcript))
    if not q or not a:
        return 0.0
    return len(q & a) / max(1, len(q))

def _extract_quotes(transcript: str, max_quotes: int = 5):
    t = _normalize_text(transcript or "")
    t = t.replace("\n", " ")
    parts = re.split(r"[.!?;]+|\bbut\b|\band\b|\bbecause\b|\bso\b|\bhowever\b", t, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]
    parts.sort(key=len, reverse=True)
    quotes = []
    seen = set()
    for p in parts:
        p = re.sub(r"\s+", " ", p).strip()
        if len(p) < 18:
            continue
        if len(p) > 120:
            p = p[:120].rstrip()
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        quotes.append(p)
        if len(quotes) >= max_quotes:
            break
    return quotes

def _vocab_candidates(transcript: str, max_items: int = 8):
    words = re.findall(r"[a-zA-Z]{4,}", (transcript or ""))
    words = [w.lower() for w in words]
    stop = set(_keywords("do you what why how when where which is are was were a an the to of in on for and or would could can like think about tell me please because"))
    cands = []
    for w in words:
        if w in stop:
            continue
        if len(w) >= 7:
            cands.append(w)
    uniq = []
    seen = set()
    for w in cands:
        if w in seen:
            continue
        seen.add(w)
        uniq.append(w)
        if len(uniq) >= max_items:
            break
    return uniq

def _grammar_markers(transcript: str):
    t = (transcript or "").lower()
    markers = []
    for m in ["because", "although", "however", "for example", "if", "when", "which", "that", "while", "even though"]:
        if m in t:
            markers.append(m)
    return markers[:8]

def _light_feedback(question: str, transcript: str, api_key: str) -> str:
    transcript = (transcript or "").strip()
    if not transcript:
        return "On-topic unclear—please speak longer and add one reason."

    wc = len(re.findall(r"[a-zA-Z']+", transcript))
    ratio = _topic_overlap_ratio(question or "", transcript)

    if ratio < 0.18:
        if wc < 25:
            return "A bit off-topic and too short—answer the question directly, then add one clear reason."
        return "Slightly off-topic—refocus on the question and add one specific example."

    if wc < 22:
        return "Good start, but speak longer—add one reason and one example."
    if wc < 45:
        return "On-topic, but extend more—use a clear reason and a short example."
    return "On-topic and clear—improve by adding one stronger example and a more complex sentence."

def _to_heavy_schema(result: dict):
    if not isinstance(result, dict):
        return {}
    overall = result.get("overall_band", result.get("overall"))
    grammar = result.get("grammar_score", result.get("grammar"))
    vocab = result.get("vocabulary_score", result.get("vocabulary"))
    fluency = result.get("fluency_score", result.get("fluency"))
    feedback = result.get("feedback_en") or result.get("feedback_zh") or result.get("feedback") or ""
    improvement = result.get("suggestions_en") or result.get("suggestions_zh") or result.get("improvement") or result.get("suggestions") or ""
    return {
        "overall": overall,
        "grammar": grammar,
        "vocabulary": vocab,
        "fluency": fluency,
        "feedback": feedback,
        "improvement": improvement,
    }

def _start_heavy_job(job_id: str, question: str, transcript: str, api_key: str):
    def run():
        try:
            result = _score_transcript(question, transcript, api_key)
            heavy_schema = _to_heavy_schema(result)
            with JOBS_LOCK:
                job = JOBS.get(job_id, {})
                job["status"] = "done"
                job["result"] = result
                job["heavy"] = heavy_schema
                JOBS[job_id] = job
        except Exception as e:
            with JOBS_LOCK:
                job = JOBS.get(job_id, {})
                job["status"] = "error"
                job["error"] = str(e)
                JOBS[job_id] = job

    t = threading.Thread(target=run, daemon=True)
    t.start()

@app.route('/get_result', methods=['GET'])
def get_result():
    _cleanup_jobs()
    job_id = (request.args.get("job_id") or "").strip()
    if not job_id:
        return jsonify({"error": "Missing job_id"}), 400
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job), 200

def _extract_json_candidate(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    t = t.strip()
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start:end + 1].strip()
    return t

def _normalize_punct(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u201c", "\"").replace("\u201d", "\"").replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("、", ",")
    s = s.replace("。", ".").replace("；", ";")
    s = s.replace("【", "[").replace("】", "]")
    s = s.replace("\u00a0", " ")
    return s

def _escape_newlines_in_json_strings(text: str) -> str:
    if not text:
        return ""
    out = []
    in_str = False
    escape = False
    for ch in text:
        if in_str:
            if escape:
                out.append(ch)
                escape = False
                continue
            if ch == "\\":
                out.append(ch)
                escape = True
                continue
            if ch == "\"":
                out.append(ch)
                in_str = False
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                continue
            if ch == "\t":
                out.append("\\t")
                continue
            out.append(ch)
            continue
        if ch == "\"":
            out.append(ch)
            in_str = True
            escape = False
            continue
        out.append(ch)
    return "".join(out)

def _parse_key_value_object(text: str):
    if not text:
        return None

    t = _normalize_text(_normalize_punct(text))
    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    if not lines:
        return None

    key_alias = {
        "fluency": "fluency",
        "vocabulary": "vocabulary",
        "grammar": "grammar",
        "overall": "overall",
        "feedback": "feedback",
        "feedback en": "feedback_en",
        "feedback_en": "feedback_en",
        "feedback zh": "feedback_zh",
        "feedback_zh": "feedback_zh",
        "improvement": "improvement",
        "improvement en": "improvement_en",
        "improvement_en": "improvement_en",
        "improvement zh": "improvement_zh",
        "improvement_zh": "improvement_zh",
        "建议": "improvement",
        "改进": "improvement",
        "改进建议": "improvement",
        "反馈": "feedback",
        "流利度": "fluency",
        "词汇": "vocabulary",
        "语法": "grammar",
        "总分": "overall",
        "overall band": "overall",
    }
    wanted = {"fluency", "vocabulary", "grammar", "overall", "feedback", "improvement", "feedback_en", "feedback_zh", "improvement_en", "improvement_zh"}

    def norm_key(k: str) -> str:
        k = (k or "").strip().lower()
        k = k.strip("{}[]()\"'`")
        k = re.sub(r"\s+", " ", k)
        return key_alias.get(k, "")

    def parse_number(v: str):
        m = re.search(r"(\d+(?:\.\d+)?)", v or "")
        return float(m.group(1)) if m else None

    out = {}
    current_key = None
    buf = []

    def flush():
        nonlocal current_key, buf
        if not current_key:
            buf = []
            return
        text_val = " ".join([re.sub(r"\s+", " ", x).strip() for x in buf if x.strip()]).strip()
        if text_val:
            out[current_key] = text_val
        buf = []

    for ln in lines:
        ln = re.sub(r"^\s*[\-\*\u2022]\s*", "", ln)
        ln = re.sub(r"^\s*\(?\d+\)?[.)]\s*", "", ln)
        ln = ln.strip().strip(",")
        if ":" in ln:
            k, v = ln.split(":", 1)
            nk = norm_key(k)
            if nk in wanted:
                flush()
                current_key = nk
                v = v.strip()
                v = v.strip().strip(",")
                if nk in {"fluency", "vocabulary", "grammar", "overall"}:
                    num = parse_number(v)
                    if num is not None:
                        out[nk] = num
                        current_key = None
                        buf = []
                    else:
                        current_key = None
                        buf = []
                else:
                    if v:
                        buf = [v.strip().strip("\"'")]
                    else:
                        buf = []
                continue
        if current_key:
            buf.append(ln.strip().strip("\"'"))

    flush()

    if not any(k in out for k in {"feedback", "feedback_en", "feedback_zh", "improvement", "improvement_en", "improvement_zh"}):
        return None
    return out

def _coerce_schema(obj: dict):
    if not isinstance(obj, dict):
        return obj
    for k in ["fluency", "vocabulary", "grammar", "overall"]:
        if k in obj and isinstance(obj[k], str):
            m = re.search(r"(\d+(?:\.\d+)?)", obj[k])
            if m:
                obj[k] = float(m.group(1))
    for k in ["feedback", "improvement"]:
        if k in obj and obj[k] is not None and not isinstance(obj[k], str):
            obj[k] = str(obj[k])
    return obj

def _parse_model_output(text: str):
    candidate = _extract_json_candidate(text)
    if not candidate:
        raise ValueError("Empty model output")

    candidate = _normalize_punct(candidate)
    candidate = re.sub(r",\s*(\}|\])", r"\1", candidate)
    candidate = _escape_newlines_in_json_strings(candidate)

    try:
        return _coerce_schema(json.loads(candidate))
    except Exception:
        pass

    try:
        obj = ast.literal_eval(candidate)
    except Exception as e:
        kv = _parse_key_value_object(candidate)
        if kv is not None:
            return _coerce_schema(kv)
        raise ValueError(f"Failed to parse as JSON: {str(e)}")
    if not isinstance(obj, dict):
        raise ValueError("Model output is not an object")
    return _coerce_schema(obj)

@app.route('/pronunciation_score', methods=['POST'])
def pronunciation_score():
    return jsonify({
        "enabled": False,
        "message": "Pronunciation scoring API not connected. Using AI-generated pronunciation guidance instead."
    }), 501

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_question', methods=['POST'])
def generate_question():
    data = request.get_json(silent=True) or {}
    ielts_part = _normalize_part(data.get('part', 'part1'))
    history_questions = data.get('history_questions') or []
    api_key = data.get('api_key') or os.getenv("DEEPSEEK_API_KEY")
    
    if not api_key:
        print("Error: Missing API Key")
        return jsonify({"error": "Missing API Key"}), 400

    try:
        print(f"Generating question for {ielts_part}...")
        question, warning = _safe_generate_question(ielts_part, api_key, index=APP_STATE.get("question_index", 0), history_questions=history_questions)
        print("Question generated successfully.")
        payload = {"question": question}
        if warning:
            payload["warning"] = warning
        return jsonify(payload)
    except Exception as e:
        print(f"Exception in generate_question: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/set_mode', methods=['POST'])
def set_mode():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "practice")
    part = data.get("part", "part1")
    prep_seconds = data.get("prep_seconds", None)
    history_questions = data.get("history_questions") or []
    api_key = data.get('api_key') or os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        return jsonify({"error": "Missing DeepSeek API Key"}), 400

    if prep_seconds is not None:
        try:
            ps = int(prep_seconds)
            if ps < 0:
                ps = 0
            if ps > 180:
                ps = 180
            APP_STATE["prep_seconds"] = ps
        except Exception:
            pass

    _reset_state(mode, part)

    idx = APP_STATE["question_index"]
    prompt, warning = _get_or_generate_prompt(idx, api_key, history_questions=history_questions)
    step = _get_step(idx)
    payload = {
        "mode": APP_STATE["mode"],
        "question_index": idx,
        "total_questions": APP_STATE["total_questions"],
        "part": step["part"],
        "step_type": step["type"],
        "progress_label": _progress_label(idx),
        "question": prompt,
    }
    if step.get("prep_seconds"):
        payload["prep_seconds"] = step["prep_seconds"]
    if warning:
        payload["warning"] = warning
    return jsonify(payload), 200

@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    data = request.get_json(silent=True) or {}
    transcript = (data.get("transcript") or "").strip()
    question = (data.get("question") or "").strip()
    part = (data.get("part") or "part1").strip()
    history_questions = data.get("history_questions") or []
    api_key = data.get('api_key') or os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        return jsonify({"error": "Missing DeepSeek API Key"}), 400

    if not transcript:
        return jsonify({"error": "Empty transcript"}), 400

    if APP_STATE["mode"] == "practice":
        APP_STATE["part"] = _part_to_number(part)
        APP_STATE["answers"].append(transcript)
        APP_STATE["question_index"] += 1

        light = _light_feedback(question, transcript, api_key)
        job_id = str(uuid.uuid4())
        with JOBS_LOCK:
            JOBS[job_id] = {
                "job_id": job_id,
                "status": "pending",
                "created_at": time.time(),
                "mode": "practice",
            }
        _start_heavy_job(job_id, question, transcript, api_key)

        next_index = APP_STATE["question_index"]
        next_prompt, warning = _get_or_generate_prompt(next_index, api_key, history_questions=history_questions)
        next_step = _get_step(next_index)

        return jsonify({
            "mode": "practice",
            "phase": "light",
            "question_index": APP_STATE["question_index"],
            "total_questions": APP_STATE["total_questions"],
            "job_id": job_id,
            "light_feedback": light,
            "next_part": next_step["part"],
            "next_step_type": next_step["type"],
            "next_progress_label": _progress_label(next_index),
            "next_question": next_prompt,
            "next_prep_seconds": next_step.get("prep_seconds", 0),
            "warning": warning
        }), 200

    APP_STATE["answers"].append(transcript)
    current_index = APP_STATE["question_index"]
    APP_STATE["question_index"] = current_index + 1

    if APP_STATE["question_index"] >= APP_STATE["total_questions"]:
        full_answer = " ".join(APP_STATE["answers"])
        light = _light_feedback("Full IELTS Speaking Exam (Parts 1/2/3)", full_answer, api_key)
        job_id = str(uuid.uuid4())
        with JOBS_LOCK:
            JOBS[job_id] = {
                "job_id": job_id,
                "status": "pending",
                "created_at": time.time(),
                "mode": "exam_final",
            }
        _start_heavy_job(job_id, "Full IELTS Speaking Exam (Parts 1/2/3)", full_answer, api_key)
        return jsonify({
            "mode": "exam",
            "done": True,
            "phase": "light",
            "total_questions": APP_STATE["total_questions"],
            "job_id": job_id,
            "light_feedback": light
        }), 200

    next_index = APP_STATE["question_index"]
    next_prompt, warning = _get_or_generate_prompt(next_index, api_key, history_questions=history_questions)
    next_step = _get_step(next_index)
    APP_STATE["part"] = _part_to_number(next_step["part"])
    payload = {
        "mode": "exam",
        "done": False,
        "question_index": next_index,
        "total_questions": APP_STATE["total_questions"],
        "next_part": next_step["part"],
        "next_step_type": next_step["type"],
        "next_progress_label": _progress_label(next_index),
        "next_question": next_prompt,
        "next_prep_seconds": next_step.get("prep_seconds", 0),
    }
    if warning:
        payload["warning"] = warning
    return jsonify(payload), 200

@app.route('/analyze_answer', methods=['POST'])
def analyze_answer():
    data = request.get_json(silent=True) or {}
    transcript = data.get('transcript')
    question = data.get('question')
    api_key = data.get('api_key') or os.getenv("DEEPSEEK_API_KEY")
    
    if not api_key:
        print("Error: Missing DeepSeek API Key")
        return jsonify({"error": "Missing DeepSeek API Key"}), 400
    
    try:
        result = _score_transcript(question or "", transcript or "", api_key)
        return jsonify(result)
    except Exception as e:
        fallback = _fallback_result(transcript or "")
        fallback["warning"] = str(e)
        return jsonify(fallback), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
