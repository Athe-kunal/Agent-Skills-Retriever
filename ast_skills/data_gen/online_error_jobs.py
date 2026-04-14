"""Rerun online jobs for rows that previously failed (errors.jsonl custom_id)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, NamedTuple

import fire
from loguru import logger

from .batch_jobs import (
    ONLINE_OUTPUT_DIR,
    _OnlineOutcome,
    build_async_client,
    build_online_tasks,
    execute_online_tasks,
    load_jsonl_lines,
    load_online_config,
    outcomes_for_source_file,
    save_text,
)


class _RetryPlanEntry(NamedTuple):
    """One input JSONL file and the subset of rows to retry."""

    source_path: Path
    records: list[dict[str, Any]]


def validate_smoke_results_subdir(name: str) -> str:
    """Rejects path traversal and separators in the smoke-test folder name."""
    if not name:
        raise ValueError("smoke_results_subdir must be a non-empty directory name.")
    if "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError(
            f"Invalid smoke_results_subdir={name!r}: use a single folder name "
            "under online_results (no slashes)."
        )
    return name


def load_failed_custom_ids(
    online_results_dir: Path,
    smoke_results_subdir: str | None,
) -> set[str]:
    """Loads custom_id values from errors.jsonl under online_results_dir."""
    failed: set[str] = set()
    if smoke_results_subdir:
        validate_smoke_results_subdir(smoke_results_subdir)
        error_paths = [online_results_dir / smoke_results_subdir / "errors.jsonl"]
        logger.info(f"{smoke_results_subdir=} single-folder smoke mode")
    else:
        error_paths = sorted(online_results_dir.glob("**/errors.jsonl"))

    logger.info(f"{online_results_dir=} {len(error_paths)=}")

    for err_path in error_paths:
        if not err_path.is_file():
            logger.warning(f"{err_path=} missing; skipping")
            continue
        try:
            lines = load_jsonl_lines(err_path)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception(f"{err_path=} {exc=}")
            continue
        for row in lines:
            raw = row.get("custom_id")
            if raw is None:
                logger.info(f"{err_path=} missing custom_id in row={row!r}")
                continue
            failed.add(str(raw))
        logger.info(f"{err_path=} {len(lines)=}")

    return failed


def build_retry_plan(
    input_dir: Path,
    failed_ids: set[str],
) -> list[_RetryPlanEntry]:
    """Maps each input JSONL to retry rows whose custom_id is in failed_ids."""
    plan: list[_RetryPlanEntry] = []
    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    logger.info(f"{input_dir=} {len(jsonl_files)=} {len(failed_ids)=}")

    for path in jsonl_files:
        try:
            records = load_jsonl_lines(path)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception(f"{path=} {exc=}")
            continue
        retry_rows = [r for r in records if str(r.get("custom_id", "")) in failed_ids]
        if not retry_rows:
            continue
        plan.append(
            _RetryPlanEntry(
                source_path=path,
                records=retry_rows,
            )
        )
        logger.info(f"{path=} {len(records)=} {len(retry_rows)=}")

    return plan


def report_orphans(failed_ids: set[str], plan: list[_RetryPlanEntry]) -> None:
    """Logs custom_ids that appear in errors but were not found in any input file."""
    found: set[str] = set()
    for entry in plan:
        for row in entry.records:
            found.add(str(row.get("custom_id", "")))
    missing = sorted(failed_ids - found)
    if missing:
        logger.warning(f"{len(missing)=} sample={missing[:20]!r}")


def read_raw_jsonl_lines(path: Path) -> list[str]:
    """Reads non-empty stripped lines from a JSONL file."""
    if not path.is_file():
        return []
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def read_total_requests(canonical_dir: Path, source_path: Path) -> int:
    """Returns total_requests from summary.json or falls back to full input row count."""
    summary_path = canonical_dir / "summary.json"
    if summary_path.is_file():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            total = payload.get("total_requests")
            if isinstance(total, int) and total >= 0:
                return total
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"{summary_path=} {exc=}")
    return len(load_jsonl_lines(source_path))


def merge_retry_into_canonical_results(
    *,
    canonical_dir: Path,
    source_path: Path,
    retry_outcomes: list[_OnlineOutcome],
    model: str,
) -> None:
    """Merges retry outcomes into canonical output.jsonl / errors.jsonl / summary.json."""
    canonical_dir.mkdir(parents=True, exist_ok=True)

    retry_success: dict[str, str] = {}
    retry_fail: dict[str, str] = {}
    for outcome in retry_outcomes:
        if outcome.error is None and outcome.response_json is not None:
            retry_success[outcome.custom_id] = outcome.response_json
        else:
            retry_fail[outcome.custom_id] = outcome.error or ""

    retried_ids = set(retry_success) | set(retry_fail)
    logger.info(
        f"{canonical_dir=} {len(retry_success)=} {len(retry_fail)=} "
        f"{len(retried_ids)=}"
    )

    output_path = canonical_dir / "output.jsonl"
    errors_path = canonical_dir / "errors.jsonl"

    merged_output_lines = read_raw_jsonl_lines(path=output_path)
    for outcome in retry_outcomes:
        if outcome.error is None and outcome.response_json is not None:
            merged_output_lines.append(outcome.response_json)

    merged_error_dicts: list[dict[str, Any]] = []
    if errors_path.is_file():
        for row in load_jsonl_lines(errors_path):
            cid = str(row.get("custom_id", ""))
            if cid in retry_success:
                continue
            if cid in retry_fail:
                continue
            merged_error_dicts.append(row)

    for cid in sorted(retry_fail.keys()):
        merged_error_dicts.append(
            {"custom_id": cid, "error": retry_fail[cid]},
        )

    if merged_output_lines:
        save_text(
            path=output_path,
            text="\n".join(merged_output_lines) + "\n",
        )
    elif output_path.is_file():
        output_path.unlink()

    if merged_error_dicts:
        error_lines = [
            json.dumps(row, ensure_ascii=False) for row in merged_error_dicts
        ]
        save_text(
            path=errors_path,
            text="\n".join(error_lines) + "\n",
        )
    elif errors_path.is_file():
        errors_path.unlink()

    total_requests = read_total_requests(
        canonical_dir=canonical_dir,
        source_path=source_path,
    )
    summary = {
        "source_file": source_path.name,
        "total_requests": total_requests,
        "succeeded": len(merged_output_lines),
        "failed": len(merged_error_dicts),
        "model": model,
    }
    save_text(
        path=canonical_dir / "summary.json",
        text=json.dumps(summary, indent=2),
    )
    logger.info(f"{summary=}")


async def run_retry_failures_online_async(
    online_results_dir: Path,
    input_dir: Path,
    smoke_results_subdir: str | None,
) -> None:
    """Runs failed custom_ids through the same online pool as batch_jobs online mode."""
    failed_ids = load_failed_custom_ids(
        online_results_dir=online_results_dir,
        smoke_results_subdir=smoke_results_subdir,
    )
    if not failed_ids:
        logger.info("No failed custom_id values found; nothing to do.")
        return

    plan = build_retry_plan(
        input_dir=input_dir,
        failed_ids=failed_ids,
    )
    report_orphans(failed_ids=failed_ids, plan=plan)
    if not plan:
        logger.info("No matching rows in input JSONL files; nothing to do.")
        return

    ONLINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    online_config = load_online_config()
    logger.info(f"{online_config.base_url=}")
    logger.info(f"{online_config.model=}")
    logger.info(f"{online_config.concurrency=}")

    client = build_async_client(
        api_key=online_config.api_key,
        base_url=online_config.base_url,
    )

    all_tasks = []
    invalid_outcomes = []
    total_records_by_path: dict[Path, int] = {}

    for entry in plan:
        try:
            built = build_online_tasks(
                records=entry.records,
                model=online_config.model,
                source_path=entry.source_path,
            )
            all_tasks.extend(built.tasks)
            invalid_outcomes.extend(built.invalid_outcomes)
            total_records_by_path[entry.source_path] = built.total_records
            logger.info(
                f"{entry.source_path=} {built.total_records=} {len(built.tasks)=} "
                f"{len(built.invalid_outcomes)=}"
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception(f"{entry.source_path=} {exc=}")

    try:
        api_outcomes = await execute_online_tasks(
            client=client,
            tasks=all_tasks,
            concurrency=online_config.concurrency,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception(f"{exc=}")
        api_outcomes = []

    combined_outcomes = invalid_outcomes + api_outcomes

    for entry in plan:
        if entry.source_path not in total_records_by_path:
            continue
        try:
            file_outcomes = outcomes_for_source_file(
                outcomes=combined_outcomes,
                source_path=entry.source_path,
            )
            canonical_dir = ONLINE_OUTPUT_DIR / entry.source_path.stem
            merge_retry_into_canonical_results(
                canonical_dir=canonical_dir,
                source_path=entry.source_path,
                retry_outcomes=file_outcomes,
                model=online_config.model,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception(f"{entry.source_path=} {exc=}")

    await client.close()


def run_retry_failures_online(
    online_results_dir: str = "online_results",
    input_dir: str = "data",
    smoke_results_subdir: str = "",
) -> None:
    """CLI entry: collect errors, match rows under data/, rerun online, merge in place.

    Requires the same environment variables as batch online mode
    (OPENAI_API_KEY, OPENAI_MODEL, optional OPENAI_BASE_URL, OPENAI_ONLINE_CONCURRENCY).

    Successful retries are appended to online_results/<stem>/output.jsonl and removed
    from errors.jsonl. Still-failing retries replace their errors.jsonl row. summary.json
    is recomputed (total_requests preserved from summary when present).

    For a smoke run on one result folder only, pass smoke_results_subdir (e.g.
    ``batch_input_9``) so only ``online_results/batch_input_9/errors.jsonl`` is read.
    """
    smoke = smoke_results_subdir.strip() or None
    asyncio.run(
        run_retry_failures_online_async(
            online_results_dir=Path(online_results_dir),
            input_dir=Path(input_dir),
            smoke_results_subdir=smoke,
        )
    )


def main() -> None:
    """Runs the CLI entrypoint via python-fire."""
    fire.Fire(run_retry_failures_online)


if __name__ == "__main__":
    main()
