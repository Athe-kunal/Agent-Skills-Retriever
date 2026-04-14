from __future__ import annotations

import datetime
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import tiktoken
import yaml
from tqdm import tqdm

# Matches any character from a non-Latin Unicode script.
# Using a blocklist of known non-English scripts is more reliable than an
# ASCII-ratio heuristic because technical docs legitimately contain many
# non-letter ASCII characters (code, URLs, punctuation).
_NON_ENGLISH_SCRIPT_RE = re.compile(
    "["
    "\u0400-\u052F"  # Cyrillic + Cyrillic Supplement (Russian, Bulgarian, ...)
    "\u0590-\u05FF"  # Hebrew
    "\u0600-\u06FF"  # Arabic
    "\u0750-\u077F"  # Arabic Supplement
    "\u0900-\u097F"  # Devanagari (Hindi, Sanskrit)
    "\u0980-\u09FF"  # Bengali
    "\u0A00-\u0A7F"  # Gurmukhi (Punjabi)
    "\u0A80-\u0AFF"  # Gujarati
    "\u0B00-\u0B7F"  # Odia
    "\u0B80-\u0BFF"  # Tamil
    "\u0C00-\u0C7F"  # Telugu
    "\u0C80-\u0CFF"  # Kannada
    "\u0D00-\u0D7F"  # Malayalam
    "\u0E00-\u0E7F"  # Thai
    "\u0E80-\u0EFF"  # Lao
    "\u0F00-\u0FFF"  # Tibetan
    "\u1000-\u109F"  # Myanmar
    "\u10A0-\u10FF"  # Georgian
    "\u1100-\u11FF"  # Hangul Jamo (Korean)
    "\u3040-\u309F"  # Hiragana (Japanese)
    "\u30A0-\u30FF"  # Katakana (Japanese)
    "\u3400-\u4DBF"  # CJK Extension A
    "\u4E00-\u9FFF"  # CJK Unified Ideographs (Chinese / Japanese / Korean)
    "\uA960-\uA97F"  # Hangul Jamo Extended-A
    "\uAC00-\uD7AF"  # Hangul Syllables (Korean)
    "]"
)

encoding = tiktoken.encoding_for_model("gpt-4o-mini")  # or similar

threshold = 63_000

# Fenced frontmatter with both delimiters: --- ... ---
_FRONTMATTER_BOTH_RE = re.compile(r"\A---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|$)", re.DOTALL)
# Fenced frontmatter with only the opening ---  (no closing delimiter found)
_FRONTMATTER_OPEN_RE = re.compile(r"\A---\s*\r?\n(.*)", re.DOTALL)
# Direct key extraction: value runs until the next bare "key:" line, a "---"
# delimiter, or end of string.  The \n--- guard prevents bleeding into fenced
# block delimiters when the regex is applied to the full raw text.
_NAME_RE = re.compile(r"^name:\s*(.+)$", re.MULTILINE)
_DESCRIPTION_RE = re.compile(
    r"^description:\s*(.*?)(?=\n\w[\w -]*:|\n---|\Z)",
    re.MULTILINE | re.DOTALL,
)
# Lone surrogates cannot be encoded to UTF-8 for JSONL output.
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")

# OpenAI Batch ``custom_id`` should stay bounded; long metadata otherwise blows past limits.
MAX_OPENAI_BATCH_CUSTOM_ID_LEN = 500


@dataclass(frozen=True)
class SkillMdRecord:
    relative_path: str
    content: str
    metadata: dict[str, str]


class SkillMdBatchCustomIdPayload(NamedTuple):
    """Decoded values from ``encode_skill_md_record_batch_custom_id``."""

    relative_path: str
    metadata: dict[str, str]
    record_index: int


def encode_skill_md_record_batch_custom_id(
    record: SkillMdRecord,
    record_index: int,
) -> str:
    """
    Build a compact and stable Batch API ``custom_id``.

    The id embeds only the record index because batch-output correlation already happens via
    the ``skill_md_records.jsonl`` rows written alongside batch input.
    """
    del record
    custom_id = f"sm-{record_index}"
    if len(custom_id) > MAX_OPENAI_BATCH_CUSTOM_ID_LEN:
        raise ValueError(
            f"custom_id exceeds {MAX_OPENAI_BATCH_CUSTOM_ID_LEN=}; {len(custom_id)=}"
        )
    return custom_id


def decode_skill_md_batch_custom_id(custom_id: str) -> SkillMdBatchCustomIdPayload:
    """Recover record index from a compact ``sm-<index>`` custom id."""
    if not custom_id.startswith("sm-"):
        raise ValueError(f"Expected custom_id starting with 'sm-'; got {custom_id!r}")
    record_index = int(custom_id[3:])
    return SkillMdBatchCustomIdPayload(
        relative_path="",
        metadata={},
        record_index=record_index,
    )


def contains_non_english_script(text: str) -> bool:
    return _NON_ENGLISH_SCRIPT_RE.search(text) is not None


def scrub_surrogate_codepoints(text: str) -> str:
    """Replace lone surrogate code points (U+D800–U+DFFF) so the string is valid UTF-8."""
    return _SURROGATE_RE.sub("\ufffd", text)


def _metadata_value_to_str(value: object) -> str:
    """Coerce a YAML value to a string for dict[str, str] storage."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    if isinstance(value, datetime.date):
        return value.isoformat()
    if isinstance(value, (dict, list)):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    return str(value)


def _extract_fenced_block(text: str) -> tuple[str, str]:
    """
    Return (yaml_block, body) for a fenced --- frontmatter.

    Handles three cases:
      - Both opening and closing ``---`` present → yaml_block is the content
        between them; body is everything after the closing delimiter.
      - Only the opening ``---`` present → yaml_block is everything after it;
        body is empty.
      - No ``---`` at all → both are empty strings.
    """
    match = _FRONTMATTER_BOTH_RE.match(text)
    if match:
        return match.group(1), text[match.end() :]

    match = _FRONTMATTER_OPEN_RE.match(text)
    if match:
        return match.group(1), ""

    return "", ""


def _parse_yaml_block(yaml_text: str) -> dict[str, str]:
    """
    Parse a YAML string into a dict[str, str].

    Returns an empty dict on parse failure or if the result is not a mapping.
    """
    try:
        loaded = yaml.safe_load(yaml_text)
    except yaml.YAMLError:
        return {}

    if not isinstance(loaded, dict):
        return {}

    return {str(key): _metadata_value_to_str(val) for key, val in loaded.items()}


def _extract_name(text: str) -> str:
    """Return the value of the first ``name:`` line, or empty string."""
    match = _NAME_RE.search(text)
    return match.group(1).strip() if match else ""


def _extract_description(text: str) -> str:
    """
    Return the value of the first ``description:`` field.

    The value continues until the next bare ``key:`` line or end of string,
    so multi-line descriptions are captured correctly.
    """
    match = _DESCRIPTION_RE.search(text)
    return match.group(1).strip() if match else ""


def parse_skill_md_frontmatter(raw_text: str) -> tuple[dict[str, str], str]:
    """
    Split SKILL.md into frontmatter metadata and markdown body.

    Strategy:
      1. ``name`` and ``description`` are always extracted directly by regex so
         they are found even when no ``---`` delimiters exist.
      2. All other metadata keys come from the fenced ``---`` / ``---`` block
         when present (missing closing delimiter is tolerated).
      3. The body is whatever follows the closing ``---``; empty if the closing
         delimiter was absent.

    All values are stored as strings (nested YAML structures are JSON-encoded).
    ``name`` and ``description`` are always present (empty string if missing).
    """
    text = raw_text.lstrip("\ufeff")

    yaml_block, body = _extract_fenced_block(text)

    if yaml_block:
        # Fenced block found — parse it for all keys; body is what follows.
        meta = _parse_yaml_block(yaml_block)
    else:
        # No --- delimiters — try parsing the whole text as raw YAML to pick
        # up extra keys (version, author, …) beyond name / description.
        meta = _parse_yaml_block(text)
        if meta:
            body = ""
        else:
            # Pure markdown: no structured metadata found at all.
            body = text

    # Direct regex extraction always wins for name / description — it tolerates
    # malformed YAML and files without --- delimiters.
    name = _extract_name(text)
    description = _extract_description(text)

    if name:
        meta["name"] = name
    if description:
        meta["description"] = description

    meta.setdefault("name", "")
    meta.setdefault("description", "")
    return meta, body


def coerce_skill_md_metadata(raw: object) -> dict[str, str]:
    """
    Normalize metadata loaded from JSON (e.g. JSONL) to dict[str, str].

    Nested dict/list values are JSON-encoded (same idea as YAML frontmatter).
    name and description are always present (empty string if missing).
    """
    if not isinstance(raw, dict):
        return {"name": "", "description": ""}
    out: dict[str, str] = {}
    for key, val in raw.items():
        k = str(key)
        if isinstance(val, str):
            out[k] = val
        elif val is None:
            out[k] = ""
        elif isinstance(val, (dict, list)):
            out[k] = json.dumps(val, ensure_ascii=False, sort_keys=True, default=str)
        else:
            out[k] = str(val)
    out.setdefault("name", "")
    out.setdefault("description", "")
    return out


def read_skill_md_records_jsonl(records_path: str | Path) -> list[SkillMdRecord]:
    """Load SkillMdRecord rows from JSONL (relative_path, content, optional metadata).

    Ignores optional keys such as ``custom_id`` and ``record_index`` used for batch correlation.
    """
    path = Path(records_path)
    records: list[SkillMdRecord] = []
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            line = line.strip()
            if not line:
                continue
            raw_record = json.loads(line)
            metadata = coerce_skill_md_metadata(raw_record.get("metadata", {}))
            records.append(
                SkillMdRecord(
                    relative_path=raw_record["relative_path"],
                    content=raw_record["content"],
                    metadata=metadata,
                )
            )
    return records


def write_skill_md_records_jsonl(
    records: list[SkillMdRecord],
    output_path: str | Path,
    *,
    include_openai_batch_custom_id: bool = True,
) -> None:
    """Write SkillMdRecord rows to JSONL (relative_path, content, metadata per line).

    If ``include_openai_batch_custom_id`` is True, each line also has ``record_index`` (0-based
    row index) and ``custom_id`` (same value sent to OpenAI Batch; length at most
    ``MAX_OPENAI_BATCH_CUSTOM_ID_LEN``).
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for record_index, record in enumerate(records):
            row = {
                "relative_path": scrub_surrogate_codepoints(record.relative_path),
                "content": scrub_surrogate_codepoints(record.content),
                "metadata": {
                    scrub_surrogate_codepoints(k): scrub_surrogate_codepoints(v)
                    for k, v in record.metadata.items()
                },
            }
            if include_openai_batch_custom_id:
                row["record_index"] = record_index
                row["custom_id"] = encode_skill_md_record_batch_custom_id(
                    record,
                    record_index,
                )
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_english_skill_md_records(root: str | Path) -> list[SkillMdRecord]:
    root_path = Path(root).resolve()
    records: list[SkillMdRecord] = []

    paths = list(root_path.rglob("SKILL.md"))
    for path in tqdm(paths, desc="Scanning SKILL.md files"):
        if not path.is_file():
            continue
        raw_text = path.read_text(encoding="utf-8", errors="replace")
        token_len = len(encoding.encode(raw_text, disallowed_special=set()))
        if token_len > threshold or token_len == 0:
            continue
        if contains_non_english_script(raw_text):
            continue
        rel = path.relative_to(root_path).as_posix()
        metadata, body = parse_skill_md_frontmatter(raw_text)
        records.append(
            SkillMdRecord(relative_path=rel, content=body, metadata=metadata)
        )

    records.sort(key=lambda r: r.relative_path)
    return records


if __name__ == "__main__":
    root = "skills/skills"
    records = collect_english_skill_md_records(root)
    print(f"{root=}")
    print(f"{len(records)=}")
    for r in records[:5]:
        print("---")
        print(r.relative_path)
        print(f"{r.metadata=}")
        print(r.content[:200] + ("…" if len(r.content) > 200 else ""))
