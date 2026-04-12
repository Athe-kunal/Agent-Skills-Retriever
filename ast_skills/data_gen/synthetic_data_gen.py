"""
Seed question generator for aspect-based Skill retrieval.

For each annotated Skill, generates 5 types of seed questions:
  1. Direct     — mentions the tool/API by name (trains 'what' matching)
  2. Implicit   — describes the need without naming the tool (trains 'why' matching)
  3. Cross-domain — applies the Skill pattern to a different domain context
  4. Adversarial — uses ambiguous phrasing that could confuse keyword matching
  5. Negative   — keyword-overlapping queries that should NOT retrieve this Skill

Each Skill produces ~20 seed questions → ~40 triplets (2 hard negatives each).
At 47K Skills, this yields ~1.9M training triplets.
"""

import json
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import fire
import pydantic
from loguru import logger
from openai import OpenAI

from ast_skills.data_gen.skills_data_collect import (
    SkillMdRecord,
    collect_skill_md_without_chinese,
    encode_skill_md_record_batch_custom_id,
    read_skill_md_records_jsonl,
    write_skill_md_records_jsonl,
)

# ──────────────────────────────────────────────
# SEED QUESTION SCHEMA
# ──────────────────────────────────────────────


@dataclass
class SeedQuestion:
    question: str
    question_type: str  # direct | implicit | cross_domain | adversarial | negative
    label: int  # 1 = positive (should retrieve this Skill), 0 = negative
    skill_id: str
    reasoning: str  # Why this question maps (or doesn't) to the Skill


# ──────────────────────────────────────────────
# LLM PROMPT FOR SEED GENERATION
# ──────────────────────────────────────────────

SEED_GENERATION_PROMPT = """You are generating training data for an embedding model that retrieves
Agent Skills given a task description. You must generate diverse seed questions for ONE Skill.

## Skill being annotated
- Skill ID: {skill_id}
- WHAT (capability): {what}
- WHY (situational trigger): {why}
- Tools: {tools}
- Domain: {domain}
- NOT WHAT (anti-patterns): {not_what}

## Generate exactly 20 questions as JSON array

For each question, output:
{{"q": "<task instruction 1-3 sentences>", "type": "<type>", "label": <1 or 0>, "reasoning": "<why>"}}

### Types to generate (4 of each):

**direct** (label: 1) — The query explicitly mentions the tool, API, or technique.
The embedding model should learn simple keyword-to-Skill matching.
Example pattern: "Use [tool] to [do thing] with [data]"

**implicit** (label: 1) — The query describes the NEED but never names the tool.
This is the most important type — it trains the 'why' aspect.
Example pattern: "I need to [outcome] from [input data]" where the Skill's procedure is the right approach.

**cross_domain** (label: 1) — Apply the Skill's capability to a domain the Skill
wasn't written for. Tests whether the embedding generalizes beyond the Skill's native domain.
Example: A pivot table Skill for sales data → generate a query about clinical trial data pivoting.

**adversarial** (label: 1) — Ambiguous phrasing where keyword matching would fail but
intent matching should succeed. Uses synonyms, indirect descriptions, or domain jargon.
Example: "Build a summary matrix" instead of "Create a pivot table"

**negative** (label: 0) — Queries that share vocabulary with the Skill but should NOT
retrieve it. The task is genuinely different despite surface similarity.
Use the NOT WHAT anti-patterns as inspiration.
Example: "Read the Excel file" (keyword: Excel) vs "Create pivot tables in Excel" (actual match)

### Quality rules
- Each question should be 1-3 sentences, realistic task instruction style
- Vary complexity: some simple ("Create X from Y"), some compound ("First extract A, then B, finally C")
- Include specific data types: "from this CSV", "using the PDF report", "across 3 Excel sheets"
- Negative questions must be genuinely different tasks, not just slightly rephrased positives
- Cross-domain questions should use domain-specific terminology from the target domain

Output ONLY a JSON array of 20 objects, no markdown fences, no explanation.
"""

SEED_OPENAI_SYSTEM_MESSAGE = (
    "You follow instructions precisely. When asked for JSON, respond with a "
    "single valid JSON array only — no markdown fences or commentary."
)

DEFAULT_OPENAI_BATCH_MODEL = "gpt-4o-mini"


class SkillMdExtraction(pydantic.BaseModel):
    """All fields produced from a single SKILL.md body — nothing else is extracted."""

    reasoning: str
    what: str
    why: str
    seed_questions: list[str]


SKILL_MD_EXTRACTION_SYSTEM_MESSAGE = """You read Agent Skill files (SKILL.md). Return exactly one JSON object with keys in this order: reasoning, what, why, seed_questions. No markdown fences, no extra keys, and no commentary outside JSON.

Ground every statement in the provided SKILL.md. Use all relevant evidence from both narrative sections and implementation sections (for example: commands, scripts, API/tool usage, configuration, inputs/outputs, constraints, caveats, workflow steps, examples, and failure handling).

Writing requirements:
- `what`: comprehensive capability description. Explain the full scope of what the skill enables, key operations, required inputs, expected outputs, important options, and notable constraints.
- `why`: comprehensive situational trigger. Explain when to use the skill, the user problems it solves, operational context, trade-offs, and why this workflow is preferred over generic alternatives.
- `reasoning`: concise justification that maps the extracted `what` and `why` to explicit evidence in the SKILL.md and explains why each seed question is an appropriate retrieval target.
- `seed_questions`: exactly 5 realistic user tasks that are materially different in intent and wording and together cover the main use-cases implied by the SKILL.md.

Do not invent tools or behaviors that are not present in the SKILL.md. If something is missing, explicitly call out the gap instead of guessing. Before returning, validate that the output is valid JSON and that every required field is present.
"""


SKILL_MD_EXTRACTION_PROMPT = """You are given the complete markdown source of one SKILL.md file.

----- SKILL.md markdown -----
{content}
----- end -----
"""


SKILL_MD_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "skill_md_extraction_response",
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                },
                "what": {
                    "type": "string",
                },
                "why": {
                    "type": "string",
                },
                "seed_questions": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                    "minItems": 5,
                    "maxItems": 5,
                },
            },
            "required": ["reasoning", "what", "why", "seed_questions"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


# ──────────────────────────────────────────────
# PROMPT + OPENAI BATCH INPUT
# ──────────────────────────────────────────────


def build_seed_generation_user_content(annotation: dict) -> str:
    """User message text for seed question generation."""
    return SEED_GENERATION_PROMPT.format(
        skill_id=annotation["skill_id"],
        what=annotation["what"],
        why=annotation["why"],
        tools=", ".join(annotation.get("tools", [])),
        domain=annotation.get("domain", "general"),
        not_what="; ".join(annotation.get("not_what", ["none specified"])),
    )


def _sanitize_custom_id(skill_id: str, index: int) -> str:
    """Batch custom_id must be unique; keep alphanumeric-ish for portability."""
    base = re.sub(r"[^a-zA-Z0-9._-]+", "_", skill_id).strip("_")
    if not base:
        base = "skill"
    return f"{base}-{index}"


def openai_batch_chat_completion_request(
    custom_id: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    response_format: dict | None = None,
) -> dict:
    """
    One OpenAI Batch API input line for POST /v1/chat/completions.

    See https://developers.openai.com/api/docs/guides/batch
    """
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        body["response_format"] = response_format

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


DEFAULT_BATCH_ENDPOINT = "/v1/chat/completions"
DEFAULT_BATCH_COMPLETION_WINDOW = "24h"
OPENAI_BATCH_MAX_INPUT_FILE_BYTES = 200 * 1024 * 1024
OPENAI_BATCH_SAFE_INPUT_FILE_BYTES = (
    OPENAI_BATCH_MAX_INPUT_FILE_BYTES - (10 * 1024 * 1024)
)


def submit_openai_batch_job(
    batch_input_jsonl_path: str,
    *,
    endpoint: str = DEFAULT_BATCH_ENDPOINT,
    completion_window: str = DEFAULT_BATCH_COMPLETION_WINDOW,
    metadata_description: str | None = None,
) -> str:
    """
    Upload a Batch input .jsonl (purpose=batch) and create a batch job.

    Requires OPENAI_API_KEY. See https://developers.openai.com/api/docs/guides/batch

    ``metadata_description`` sets batch-level metadata on the Batch object (visible
    when you retrieve the job), not per-request fields in completion output lines.

    Returns:
      The batch id (e.g. batch_...) for status checks and result download.
    """
    path = Path(batch_input_jsonl_path).expanduser().resolve()
    logger.info(f"{path=}")
    if not path.is_file():
        raise FileNotFoundError(f"Batch input not found: {path}")

    client = OpenAI()

    with path.open("rb") as batch_file:
        uploaded = client.files.create(file=batch_file, purpose="batch")
    logger.info(f"{uploaded.id=}")

    create_kwargs: dict = {
        "input_file_id": uploaded.id,
        "endpoint": endpoint,
        "completion_window": completion_window,
    }
    if metadata_description:
        create_kwargs["metadata"] = {"description": metadata_description}

    batch = client.batches.create(**create_kwargs)
    logger.info(f"{batch.id=}")
    logger.info(f"{batch.status=}")
    logger.info(f"{batch.endpoint=}")
    logger.info(f"{batch.input_file_id=}")
    return batch.id


def build_skill_md_extraction_user_content(skill_md_record: SkillMdRecord) -> str:
    """Build user prompt for extracting what/why/seed questions from SKILL.md."""
    return SKILL_MD_EXTRACTION_PROMPT.format(
        content=skill_md_record.content,
    )


def _jsonl_line_size_bytes(row: dict) -> int:
    """Return UTF-8 byte size for one JSONL row (including newline)."""
    return len((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))


def _would_exceed_batch_file_size(
    current_size_bytes: int,
    next_line_size_bytes: int,
    max_file_size_bytes: int,
) -> bool:
    """Check whether appending a row would exceed the batch input file size limit."""
    return current_size_bytes + next_line_size_bytes > max_file_size_bytes


def write_openai_batch_skill_md_extraction_jsonl_for_records(
    skill_md_records: list[SkillMdRecord],
    output_path: str,
    *,
    model: str = DEFAULT_OPENAI_BATCH_MODEL,
    max_tokens: int = 1024,
    max_file_size_bytes: int = OPENAI_BATCH_SAFE_INPUT_FILE_BYTES,
) -> int:
    """
    Emit OpenAI Batch JSONL for SKILL.md extraction, from in-memory records.

    Each request ``custom_id`` is compact (``sm-<record_index>``) and aligns to the
    corresponding record row in ``skill_md_records.jsonl``.

    ``batches.create(..., metadata={...})`` is separate: job-level tags on the batch
    object, not echoed per completion line.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written_count = 0
    current_size_bytes = 0

    with out_path.open("w", encoding="utf-8") as output_file:
        for index, skill_md_record in enumerate(skill_md_records):
            custom_id = encode_skill_md_record_batch_custom_id(
                skill_md_record,
                index,
            )
            messages = [
                {"role": "system", "content": SKILL_MD_EXTRACTION_SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": build_skill_md_extraction_user_content(skill_md_record),
                },
            ]
            request_line = openai_batch_chat_completion_request(
                custom_id=custom_id,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                response_format=SKILL_MD_RESPONSE_FORMAT,
            )
            line_size_bytes = _jsonl_line_size_bytes(request_line)
            if _would_exceed_batch_file_size(
                current_size_bytes=current_size_bytes,
                next_line_size_bytes=line_size_bytes,
                max_file_size_bytes=max_file_size_bytes,
            ):
                logger.info(
                    "Stopping batch row writes at size limit: "
                    f"{current_size_bytes=} {line_size_bytes=} {max_file_size_bytes=}"
                )
                break

            output_file.write(json.dumps(request_line, ensure_ascii=False) + "\n")
            current_size_bytes += line_size_bytes
            written_count += 1

    logger.info(f"{output_path=}")
    logger.info(f"{len(skill_md_records)=}")
    logger.info(f"{written_count=}")
    logger.info(f"{current_size_bytes=}")
    logger.info(f"{max_file_size_bytes=}")
    return written_count


def write_openai_batch_skill_md_extraction_jsonl(
    records_path: str,
    output_path: str,
    *,
    model: str = DEFAULT_OPENAI_BATCH_MODEL,
    max_records: int | None = None,
    max_tokens: int = 1024,
    max_file_size_bytes: int = OPENAI_BATCH_SAFE_INPUT_FILE_BYTES,
) -> int:
    """Emit OpenAI Batch JSONL for SkillMdRecord extraction (read records from JSONL)."""
    skill_md_records = read_skill_md_records_jsonl(records_path)
    logger.info(f"{records_path=}")
    logger.info(f"{len(skill_md_records)=}")
    if max_records is not None:
        skill_md_records = skill_md_records[:max_records]
    return write_openai_batch_skill_md_extraction_jsonl_for_records(
        skill_md_records,
        output_path,
        model=model,
        max_tokens=max_tokens,
        max_file_size_bytes=max_file_size_bytes,
    )


def write_openai_batch_seed_jsonl(
    annotations_path: str,
    output_path: str,
    *,
    model: str = DEFAULT_OPENAI_BATCH_MODEL,
    max_skills: int | None = None,
    max_tokens: int = 4096,
    max_file_size_bytes: int = OPENAI_BATCH_SAFE_INPUT_FILE_BYTES,
) -> int:
    """
    Emit a .jsonl file suitable for OpenAI Batch upload (purpose=batch).

    One request per annotation; map results back with custom_id. All lines use
    the same model (required by Batch API).
    """
    annotations: list[dict] = []
    with open(annotations_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            annotations.append(json.loads(line))

    if max_skills is not None:
        annotations = annotations[:max_skills]

    written_count = 0
    current_size_bytes = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for i, ann in enumerate(annotations):
            custom_id = _sanitize_custom_id(ann["skill_id"], i)
            user_content = build_seed_generation_user_content(ann)
            messages = [
                {"role": "system", "content": SEED_OPENAI_SYSTEM_MESSAGE},
                {"role": "user", "content": user_content},
            ]
            req = openai_batch_chat_completion_request(
                custom_id=custom_id,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
            )
            line_size_bytes = _jsonl_line_size_bytes(req)
            if _would_exceed_batch_file_size(
                current_size_bytes=current_size_bytes,
                next_line_size_bytes=line_size_bytes,
                max_file_size_bytes=max_file_size_bytes,
            ):
                logger.info(
                    "Stopping seed batch row writes at size limit: "
                    f"{current_size_bytes=} {line_size_bytes=} {max_file_size_bytes=}"
                )
                break

            out.write(json.dumps(req, ensure_ascii=False) + "\n")
            current_size_bytes += line_size_bytes
            written_count += 1

    logger.info(f"{written_count=}")
    logger.info(f"{current_size_bytes=}")
    logger.info(f"{max_file_size_bytes=}")
    return written_count


# ──────────────────────────────────────────────
# GENERATION FUNCTION
# ──────────────────────────────────────────────


def generate_seed_questions(
    annotation: dict,
    client: OpenAI,
    *,
    model: str = DEFAULT_OPENAI_BATCH_MODEL,
    max_tokens: int = 4096,
) -> list[SeedQuestion]:
    """Generate 20 seed questions for a single annotated Skill via Chat Completions."""

    user_content = build_seed_generation_user_content(annotation)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SEED_OPENAI_SYSTEM_MESSAGE},
            {"role": "user", "content": user_content},
        ],
        max_tokens=max_tokens,
    )

    choice = response.choices[0]
    text = (choice.message.content or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]

    raw_questions = json.loads(text)

    seeds = []
    for rq in raw_questions:
        seeds.append(
            SeedQuestion(
                question=rq["q"],
                question_type=rq["type"],
                label=rq["label"],
                skill_id=annotation["skill_id"],
                reasoning=rq.get("reasoning", ""),
            )
        )

    return seeds


# ──────────────────────────────────────────────
# TRIPLET CONSTRUCTION FROM SEEDS
# ──────────────────────────────────────────────


def seeds_to_triplets(
    seeds: list[SeedQuestion],
    annotation: dict,
    all_annotations: list[dict],
    n_negatives_per_query: int = 2,
) -> list[dict]:
    """
    Convert seed questions into training triplets.

    For positive seeds (label=1):
      anchor = seed question
      positive = this Skill's aspect-formatted document
      negative = hard negative Skill's aspect-formatted document

    For negative seeds (label=0):
      These become hard negatives for OTHER Skills' positive queries.
      Store them separately for cross-Skill negative mining.
    """

    def format_aspect_doc(ann):
        return (
            f"[WHAT] {ann['what']} "
            f"[WHY] {ann['why']} "
            f"[TOOLS] {', '.join(ann.get('tools', []))} "
            f"[DOMAIN] {ann.get('domain', 'general')}"
        )

    positive_doc = format_aspect_doc(annotation)
    others = [a for a in all_annotations if a["skill_id"] != annotation["skill_id"]]

    triplets = []
    positive_seeds = [s for s in seeds if s.label == 1]

    for seed in positive_seeds:
        # Select hard negatives using different strategies per question type
        negatives = _select_negatives(seed, annotation, others, n_negatives_per_query)

        for neg_ann, neg_type in negatives:
            triplets.append(
                {
                    "anchor": seed.question,
                    "positive": positive_doc,
                    "negative": format_aspect_doc(neg_ann),
                    "positive_skill_id": annotation["skill_id"],
                    "negative_skill_id": neg_ann["skill_id"],
                    "question_type": seed.question_type,
                    "negative_type": neg_type,
                }
            )

    return triplets


def _select_negatives(
    seed: SeedQuestion,
    pos_annotation: dict,
    all_others: list[dict],
    n: int,
) -> list[tuple[dict, str]]:
    """Select hard negatives appropriate for the seed question type."""

    negatives = []
    pos_tools = set(pos_annotation.get("tools", []))
    pos_domain = pos_annotation.get("domain", "")

    if seed.question_type == "direct":
        # For direct queries, use tool-confusable negatives
        # (same tool mentioned, different task pattern)
        tool_overlap = [a for a in all_others if set(a.get("tools", [])) & pos_tools]
        if tool_overlap:
            negatives.append((random.choice(tool_overlap), "tool_confusable"))

    elif seed.question_type == "implicit":
        # For implicit queries, use in-domain negatives
        # (same domain, but different procedure needed)
        in_domain = [a for a in all_others if a.get("domain") == pos_domain]
        if in_domain:
            negatives.append((random.choice(in_domain), "in_domain"))

    elif seed.question_type == "cross_domain":
        # For cross-domain, use same-domain-as-query negatives
        # (the model should pick the procedural match, not the domain match)
        if all_others:
            negatives.append((random.choice(all_others), "cross_domain_distractor"))

    elif seed.question_type == "adversarial":
        # For adversarial, use the closest keyword-matching negative
        # (highest surface similarity, wrong intent)
        pos_concepts = set(pos_annotation.get("concepts", []))
        concept_overlap = [
            a for a in all_others if set(a.get("concepts", [])) & pos_concepts
        ]
        if concept_overlap:
            negatives.append((random.choice(concept_overlap), "keyword_overlap"))

    # Fill remaining slots with random hard negatives
    used_ids = {neg[0]["skill_id"] for neg in negatives}
    remaining = [a for a in all_others if a["skill_id"] not in used_ids]
    while len(negatives) < n and remaining:
        neg = random.choice(remaining)
        negatives.append((neg, "random"))
        remaining = [a for a in remaining if a["skill_id"] != neg["skill_id"]]

    return negatives[:n]


# ──────────────────────────────────────────────
# BATCH PROCESSING
# ──────────────────────────────────────────────


def generate_full_dataset(
    annotations_path: str,
    output_path: str,
    max_skills: int | None = None,
    questions_per_skill: int = 20,
    *,
    model: str = DEFAULT_OPENAI_BATCH_MODEL,
    max_tokens: int = 4096,
):
    """
    End-to-end: annotations → seed questions → triplets.

    Output: JSONL file with training triplets ready for the embedding trainer.
    """
    client = OpenAI()

    # Load annotations
    annotations = []
    with open(annotations_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            annotations.append(json.loads(line))

    if max_skills is not None:
        annotations = annotations[:max_skills]

    logger.info(f"Generating seeds for {len(annotations)} Skills (model={model})...")

    all_seeds = []
    all_triplets = []

    for i, ann in enumerate(annotations):
        try:
            seeds = generate_seed_questions(
                ann,
                client,
                model=model,
                max_tokens=max_tokens,
            )
            all_seeds.extend(seeds)

            triplets = seeds_to_triplets(seeds, ann, annotations)
            all_triplets.extend(triplets)

            if (i + 1) % 25 == 0:
                logger.info(
                    f"  {i+1}/{len(annotations)} Skills processed "
                    f"| {len(all_seeds)} seeds | {len(all_triplets)} triplets"
                )

        except Exception as e:
            logger.info(f"  Failed on {ann['skill_id']}: {e}")
            continue

    # Save triplets
    with open(output_path, "w", encoding="utf-8") as f:
        for t in all_triplets:
            f.write(json.dumps(t) + "\n")

    # Save seeds separately for analysis
    seeds_path = output_path.replace(".jsonl", "_seeds.jsonl")
    with open(seeds_path, "w", encoding="utf-8") as f:
        for s in all_seeds:
            f.write(json.dumps(asdict(s)) + "\n")

    # Summary stats
    logger.info("Dataset generated:")
    logger.info(f"  Skills:    {len(annotations)}")
    logger.info(f"  Seeds:     {len(all_seeds)}")
    logger.info(f"  Triplets:  {len(all_triplets)}")
    logger.info(f"  Saved to:  {output_path}")
    logger.info(f"  Seeds at:  {seeds_path}")

    # Type distribution
    type_counts = {}
    for s in all_seeds:
        type_counts[s.question_type] = type_counts.get(s.question_type, 0) + 1
    logger.info("  Seed type distribution:")
    for qtype, count in sorted(type_counts.items()):
        pct = count / len(all_seeds) * 100 if all_seeds else 0.0
        logger.info(f"    {qtype:15s} {count:6d} ({pct:.1f}%)")


class SyntheticDataGenCli:
    """Entry point for `python synthetic_data_gen.py <command> ...` via python-fire."""

    def extract_skill_md_batch(
        self,
        skills_root: str,
        batch_output_path: str = "data/openai_skill_md_batch_input.jsonl",
        records_jsonl_path: str = "data/skill_md_records.jsonl",
        model: str = DEFAULT_OPENAI_BATCH_MODEL,
        max_records: int | None = None,
        max_tokens: int = 1024,
        max_file_size_bytes: int = OPENAI_BATCH_SAFE_INPUT_FILE_BYTES,
    ) -> None:
        """
        Scan SKILL.md under skills_root, save records JSONL, then write OpenAI Batch input JSONL.

        By default both outputs go under ``data/``. Set ``records_jsonl_path`` to a falsey
        string (e.g. empty ``""``) to skip writing the records file.

        Each records line includes ``record_index`` and ``custom_id`` matching the batch job.
        Each ``custom_id`` follows ``sm-<record_index>`` and is capped at 500 characters.
        Decode with ``decode_skill_md_batch_custom_id`` in ``skills_data_collect``, or join
        batch lines to this file on exact ``custom_id``.
        """
        logger.info(f"{skills_root=}")
        records = collect_skill_md_without_chinese(skills_root)
        logger.info(f"collected {len(records)} SKILL.md records")

        if max_records is not None:
            records = records[:max_records]

        if records_jsonl_path:
            write_skill_md_records_jsonl(
                records,
                records_jsonl_path,
                include_openai_batch_custom_id=True,
            )
            logger.info(f"{records_jsonl_path=}")
            logger.info(f"{len(records)=}")

        record_count = write_openai_batch_skill_md_extraction_jsonl_for_records(
            records,
            batch_output_path,
            model=model,
            max_tokens=max_tokens,
            max_file_size_bytes=max_file_size_bytes,
        )
        logger.info(
            f"Wrote {record_count} batch lines to {batch_output_path} "
            f"(model={model})"
        )

    def extract_skill_md_batch_from_jsonl(
        self,
        records_path: str,
        batch_output_path: str = "data/openai_skill_md_batch_input.jsonl",
        records_jsonl_path: str = "data/skill_md_records.jsonl",
        model: str = DEFAULT_OPENAI_BATCH_MODEL,
        max_records: int | None = None,
        max_tokens: int = 4096,
        max_file_size_bytes: int = OPENAI_BATCH_SAFE_INPUT_FILE_BYTES,
    ) -> None:
        """
        Same as extract_skill_md_batch but read pre-built records JSONL instead of scanning.

        Input JSONL format per line:
          {"relative_path":"...", "content":"...", "metadata": {...}}
          (metadata optional; used for storage only, not sent in the extraction prompt.)

        After optional ``max_records`` slicing, writes ``records_jsonl_path`` (default under
        ``data/``) with ``record_index`` and ``custom_id`` aligned to the batch file. Pass
        ``records_jsonl_path=""`` to skip that write.
        """
        skill_md_records = read_skill_md_records_jsonl(records_path)
        logger.info(f"{records_path=}")
        logger.info(f"{len(skill_md_records)=}")
        if max_records is not None:
            skill_md_records = skill_md_records[:max_records]

        if records_jsonl_path:
            write_skill_md_records_jsonl(
                skill_md_records,
                records_jsonl_path,
                include_openai_batch_custom_id=True,
            )
            logger.info(f"{records_jsonl_path=}")

        record_count = write_openai_batch_skill_md_extraction_jsonl_for_records(
            skill_md_records,
            batch_output_path,
            model=model,
            max_tokens=max_tokens,
            max_file_size_bytes=max_file_size_bytes,
        )
        logger.info(
            f"Wrote {record_count} batch lines to {batch_output_path} "
            f"(model={model})"
        )

    def openai_batch(
        self,
        annotations_path: str,
        batch_output_path: str,
        model: str = DEFAULT_OPENAI_BATCH_MODEL,
        max_skills: int | None = None,
        max_tokens: int = 4096,
        max_file_size_bytes: int = OPENAI_BATCH_SAFE_INPUT_FILE_BYTES,
    ) -> None:
        """
        Write OpenAI Batch API input JSONL (one chat completion per annotation).

        Example:
          python synthetic_data_gen.py openai_batch fixtures/a.jsonl out.jsonl \\
            --model=gpt-3.5-turbo-0125 --max-skills=100
        """
        n = write_openai_batch_seed_jsonl(
            annotations_path,
            batch_output_path,
            model=model,
            max_skills=max_skills,
            max_tokens=max_tokens,
            max_file_size_bytes=max_file_size_bytes,
        )
        logger.info(f"Wrote {n} batch lines to {batch_output_path} (model={model})")

    def submit_batch(
        self,
        batch_input_jsonl_path: str,
        endpoint: str = DEFAULT_BATCH_ENDPOINT,
        completion_window: str = DEFAULT_BATCH_COMPLETION_WINDOW,
        metadata_description: str | None = None,
    ) -> None:
        """
        Upload batch_input_jsonl_path to OpenAI (purpose=batch) and enqueue a batch job.

        Set OPENAI_API_KEY in the environment (e.g. from .env in your shell).

        Example:
          uv run python ast_skills/data_gen/synthetic_data_gen.py submit_batch batch_input.jsonl
          uv run python ... submit_batch batch_input.jsonl \\
            --endpoint=/v1/embeddings --metadata-description="skill-md extraction"
        """
        batch_id = submit_openai_batch_job(
            batch_input_jsonl_path,
            endpoint=endpoint,
            completion_window=completion_window,
            metadata_description=metadata_description,
        )
        logger.info(
            "Poll with the API or dashboard; retrieve output when status is completed. "
            f"{batch_id=}"
        )

    def full_dataset(
        self,
        annotations_path: str,
        output_path: str,
        max_skills: int | None = None,
        questions_per_skill: int = 20,
        model: str = DEFAULT_OPENAI_BATCH_MODEL,
        max_tokens: int = 4096,
    ) -> None:
        """
        Run OpenAI Chat Completions seed generation and write triplets + seeds JSONL.

        Example:
          python synthetic_data_gen.py full_dataset annotations.jsonl triplets.jsonl \\
            --model=gpt-4o-mini
        """
        generate_full_dataset(
            annotations_path=annotations_path,
            output_path=output_path,
            max_skills=max_skills,
            questions_per_skill=questions_per_skill,
            model=model,
            max_tokens=max_tokens,
        )


if __name__ == "__main__":
    fire.Fire(SyntheticDataGenCli)
