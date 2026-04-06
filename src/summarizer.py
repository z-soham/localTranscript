"""LLM-based transcript summarisation.

Uses the OpenAI-compatible chat-completions endpoint so it works with:
  - OpenRouter  (https://openrouter.ai/api/v1)
  - Ollama      (http://localhost:11434/v1)
  - Any other OpenAI-compatible local or hosted LLM.

No third-party packages are required — only stdlib urllib is used.
"""

import json
import urllib.error
import urllib.request
from pathlib import Path


SUMMARY_MODES = ["Meeting", "General Video"]


def summarize_transcript(
    file_path: Path,
    llm_url: str,
    api_key: str,
    llm_model: str,
    mode: str = "Meeting",
) -> str:
    """Read a .txt or .srt transcript file and return a formatted summary.

    mode: "Meeting" for MoM-style output, "General Video" for takeaways/suggestions.
    """
    text = file_path.read_text(encoding="utf-8", errors="replace")

    if file_path.suffix.lower() == ".srt":
        text = _strip_srt_timestamps(text)

    if not text.strip():
        raise ValueError("The transcript file appears to be empty.")

    if mode == "General Video":
        system_msg = (
            "You are a helpful content analyst. "
            "Produce well-structured, concise video summaries in Markdown."
        )
        prompt = _build_general_prompt(text)
    else:
        system_msg = (
            "You are a professional meeting assistant. "
            "Produce well-structured, concise meeting summaries in Markdown."
        )
        prompt = _build_meeting_prompt(text)

    base_url = llm_url.rstrip("/")
    endpoint = base_url if base_url.endswith("/chat/completions") else base_url + "/chat/completions"

    payload = {
        "model": llm_model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
    }

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from LLM endpoint: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach LLM endpoint ({endpoint}): {exc.reason}") from exc


def _strip_srt_timestamps(text: str) -> str:
    """Remove SRT sequence numbers and timestamp lines, keeping only dialogue."""
    lines = text.splitlines()
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.isdigit():
            continue
        if "-->" in stripped:
            continue
        result.append(stripped)
    # Collapse multiple blank lines to a single one
    cleaned: list[str] = []
    prev_blank = False
    for line in result:
        if line == "":
            if not prev_blank:
                cleaned.append(line)
            prev_blank = True
        else:
            cleaned.append(line)
            prev_blank = False
    return "\n".join(cleaned).strip()


def _build_meeting_prompt(transcript_text: str) -> str:
    return f"""\
You are given a meeting transcript below. Analyse it thoroughly and produce a structured summary using **exactly** the Markdown sections shown. Use bullet points and be concise.

---

## Minutes of Meeting

_A brief narrative of what was discussed — key topics, context, and outcomes._

## Key Decisions

- _Decision 1_
- _Decision 2_

## Action Items

| # | Task | Owner | Due Date |
|---|------|-------|----------|
| 1 | _description_ | _name or team_ | _date or "TBD"_ |

## Participants

_List names, roles, or teams mentioned in the transcript._

## Next Steps

_What happens after this meeting — follow-ups, upcoming meetings, deadlines._

---

**Transcript:**

{transcript_text}
"""


def _build_general_prompt(transcript_text: str) -> str:
    return f"""\
You are given a video transcript below. Analyse it thoroughly and produce a structured summary using **exactly** the Markdown sections shown. Use bullet points and be concise.

---

## Overview

_A brief description of what the video is about — topic, format, and purpose._

## Key Takeaways

- _Main point 1_
- _Main point 2_

## Important Details

_Notable facts, data points, examples, or explanations covered in the video._

## Suggestions & Recommendations

_Any advice, tips, or recommendations given in the video. Omit this section if none._

## Further Exploration

_Related topics, resources, or questions the viewer might want to explore after watching._

---

**Transcript:**

{transcript_text}
"""
