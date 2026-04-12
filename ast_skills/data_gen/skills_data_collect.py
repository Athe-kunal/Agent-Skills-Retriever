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

# Primary Han block (simplified + traditional in common use).
_HAN_RE = re.compile(r"[\u4e00-\u9fff]")
# Optional: also match CJK Extension A/B etc. if you need stricter “any CJK ideograph”:
# _HAN_RE = re.compile(
#     r"[\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df\u2a700-\u2b73f\u2b740-\u2b81f\u2b820-\u2ceaf]"
# )

encoding = tiktoken.encoding_for_model("gpt-4o-mini")  # or similar

threshold = 63_000

# Leading YAML block: --- ... --- then body (Agent Skills / Cursor convention).
_FRONTMATTER_RE = re.compile(r"\A---\s*\r?\n(.*?)\r?\n---\s*\r?\n", re.DOTALL)
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


def contains_chinese(text: str) -> bool:
    return _HAN_RE.search(text) is not None


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


def parse_skill_md_frontmatter(raw_text: str) -> tuple[dict[str, str], str]:
    """
    Split SKILL.md into frontmatter metadata (--- ... ---) and markdown body.

    All keys from the YAML block are stored in metadata with string values
    (nested structures are JSON-encoded). name and description are always
    present (empty string if missing).
    """
    text = raw_text.lstrip("\ufeff")
    match = _FRONTMATTER_RE.match(text)
    if not match:
        body = text
        meta: dict[str, str] = {"name": "", "description": ""}
        return meta, body

    frontmatter_block = match.group(1)
    body = text[match.end() :]

    try:
        loaded = yaml.safe_load(frontmatter_block)
    except yaml.YAMLError:
        loaded = None

    if not isinstance(loaded, dict):
        meta = {}
    else:
        meta = {str(key): _metadata_value_to_str(val) for key, val in loaded.items()}

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


def collect_skill_md_without_chinese(root: str | Path) -> list[SkillMdRecord]:
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
        if contains_chinese(raw_text):
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
    records = collect_skill_md_without_chinese(root)
    print(f"{root=}")
    print(f"{len(records)=}")
    for r in records[:5]:
        print("---")
        print(r.relative_path)
        print(f"{r.metadata=}")
        print(r.content[:200] + ("…" if len(r.content) > 200 else ""))
