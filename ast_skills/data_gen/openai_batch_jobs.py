from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, NamedTuple

import fire
from loguru import logger
from openai import AsyncOpenAI, OpenAI
from openai.types import Batch


INPUT_DIR = Path("data")
BATCH_INPUT_DONE_DIR = Path("done")
OUTPUT_DIR = Path("batch_results")
ONLINE_OUTPUT_DIR = Path("online_results")
POLL_INTERVAL_SECONDS = 30
DEFAULT_ONLINE_CONCURRENCY = 32

DEFAULT_MODE = "batch"
ONLINE_MODE = "online"


class _OnlineConfig(NamedTuple):
    """Holds environment values needed for online execution."""

    api_key: str
    base_url: str | None
    model: str
    concurrency: int


class _OnlineTask(NamedTuple):
    """Represents one online API task."""

    custom_id: str
    payload: dict[str, Any]


_QueueItem = _OnlineTask | None


class _OnlineOutcome(NamedTuple):
    """Represents one online task result."""

    custom_id: str
    response_json: str | None
    error: str | None


class _TaskBuildResult(NamedTuple):
    """Stores prepared tasks and pre-validation failures."""

    tasks: list[_OnlineTask]
    invalid_outcomes: list[_OnlineOutcome]
    total_records: int


def configure_logging() -> None:
    """Configures process-wide logging using loguru."""
    logger.remove()
    logger.add(
        sink=lambda message: print(message, end=""),
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} - {message}",
    )


def validate_mode(mode: str) -> None:
    """Validates runtime execution mode."""
    valid_modes = {DEFAULT_MODE, ONLINE_MODE}
    if mode not in valid_modes:
        valid_values = ", ".join(sorted(valid_modes))
        raise ValueError(f"Invalid mode: {mode}. Expected one of: {valid_values}")


def build_client(api_key: str | None = None, base_url: str | None = None) -> OpenAI:
    """Builds a synchronous OpenAI client with optional overrides."""
    if api_key is None and base_url is None:
        return OpenAI()
    return OpenAI(api_key=api_key, base_url=base_url)


def build_async_client(
    api_key: str | None = None,
    base_url: str | None = None,
) -> AsyncOpenAI:
    """Builds an asynchronous OpenAI client with optional overrides."""
    if api_key is None and base_url is None:
        return AsyncOpenAI()
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


def save_text(path: Path, text: str) -> None:
    """Saves text content to a file path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_jsonl_lines(path: Path) -> list[dict[str, Any]]:
    """Loads JSON lines from a file."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {path}: {exc}"
                ) from exc
            records.append(record)
    return records


def validate_batch_record(record: dict[str, Any], path: Path, line_number: int) -> None:
    """Validates one batch JSONL record shape."""
    if "custom_id" not in record:
        raise ValueError(f"{path} line {line_number}: missing 'custom_id'")
    if record.get("method") != "POST":
        raise ValueError(f"{path} line {line_number}: expected method='POST'")
    if "url" not in record:
        raise ValueError(f"{path} line {line_number}: missing 'url'")
    if "body" not in record or not isinstance(record["body"], dict):
        raise ValueError(f"{path} line {line_number}: missing or invalid 'body'")


def validate_batch_jsonl(path: Path) -> list[dict[str, Any]]:
    """Validates batch JSONL record shape before processing."""
    records = load_jsonl_lines(path)
    if not records:
        raise ValueError(f"{path} is empty.")

    for idx, record in enumerate(records, start=1):
        validate_batch_record(record=record, path=path, line_number=idx)
    return records


def resolve_batch_endpoint(records: list[dict[str, Any]], path: Path) -> str:
    """Resolves a single endpoint value for a batch JSONL file."""
    endpoints = {str(record["url"]) for record in records}
    if len(endpoints) != 1:
        raise ValueError(f"{path} must contain exactly one unique 'url' value.")
    endpoint = endpoints.pop()
    logger.info(f"{endpoint=}")
    return endpoint


def upload_batch_file(client: OpenAI, path: Path) -> str:
    """Uploads a JSONL file for batch execution."""
    with path.open("rb") as file_handle:
        uploaded = client.files.create(
            file=file_handle,
            purpose="batch",
        )
    return uploaded.id


def create_batch(
    client: OpenAI, input_file_id: str, source_file: Path, endpoint: str
) -> str:
    """Creates a batch job and returns the batch ID."""
    batch = client.batches.create(
        input_file_id=input_file_id,
        endpoint=endpoint,
        completion_window="24h",
        metadata={
            "source_filename": source_file.name,
            "pipeline": "jsonl-sequential-runner",
        },
    )
    return batch.id


TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}
ACTIVE_STATUSES = {"validating", "in_progress", "finalizing", "cancelling"}


def wait_for_batch(client: OpenAI, batch_id: str) -> object:
    """Polls a batch job until it reaches a terminal state."""
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        logger.info(f"{batch_id=} {status=}")

        if status in TERMINAL_STATUSES:
            return batch

        time.sleep(POLL_INTERVAL_SECONDS)


def fetch_active_batches(client: OpenAI) -> list[Batch]:
    """Returns all batch jobs currently in a non-terminal state."""
    active: list[Batch] = []
    for batch in client.batches.list():
        if batch.status in ACTIVE_STATUSES:
            active.append(batch)
    return active


def wait_until_no_active_batches(client: OpenAI) -> None:
    """Blocks until OpenAI reports no batch jobs in a non-terminal state.

    This prevents submitting a new job while a previous one is still running,
    even across separate script invocations or concurrent runs.
    """
    while True:
        active = fetch_active_batches(client=client)
        if not active:
            return

        active_ids = [b.id for b in active]
        logger.info(f"Waiting for active batches to finish: {active_ids=}")
        time.sleep(POLL_INTERVAL_SECONDS)


def download_file_content(client: OpenAI, file_id: str) -> str:
    """Downloads file content from OpenAI Files API."""
    content = client.files.content(file_id)
    if hasattr(content, "text"):
        return content.text
    return str(content)


def build_batch_summary(batch: object) -> dict[str, Any]:
    """Builds a serializable summary dictionary for one batch."""
    return {
        "id": batch.id,
        "status": batch.status,
        "input_file_id": batch.input_file_id,
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
        "request_counts": getattr(batch, "request_counts", None),
        "metadata": getattr(batch, "metadata", None),
    }


def resolve_done_destination(done_dir: Path, filename: str) -> Path:
    """Returns a non-colliding path under the done directory for a filename."""
    destination = done_dir / filename
    if not destination.exists():
        return destination

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    counter = 1
    while True:
        candidate = done_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def move_batch_input_to_done(source: Path) -> None:
    """Moves a processed batch JSONL input file into the top-level done folder."""
    BATCH_INPUT_DONE_DIR.mkdir(parents=True, exist_ok=True)
    destination = resolve_done_destination(
        done_dir=BATCH_INPUT_DONE_DIR,
        filename=source.name,
    )
    shutil.move(src=str(source), dst=str(destination))
    logger.info(f"{source=} {destination=}")


def save_batch_outputs(client: OpenAI, batch: object, batch_dir: Path) -> None:
    """Saves output and error files for one completed batch."""
    output_file_id = getattr(batch, "output_file_id", None)
    if output_file_id:
        output_text = download_file_content(client=client, file_id=output_file_id)
        save_text(batch_dir / f"batch_{output_file_id}_output.jsonl", output_text)
        logger.info(f"{output_file_id=}")

    error_file_id = getattr(batch, "error_file_id", None)
    if error_file_id:
        error_text = download_file_content(client=client, file_id=error_file_id)
        save_text(batch_dir / "errors.jsonl", error_text)
        logger.info(f"{error_file_id=}")


def process_one_jsonl_batch(client: OpenAI, path: Path) -> None:
    """Processes one input JSONL file through the Batch API."""
    logger.info(f"{path=}")
    records = validate_batch_jsonl(path)
    endpoint = resolve_batch_endpoint(records=records, path=path)

    wait_until_no_active_batches(client=client)

    input_file_id = upload_batch_file(client=client, path=path)
    logger.info(f"{input_file_id=}")

    batch_id = create_batch(
        client=client,
        input_file_id=input_file_id,
        source_file=path,
        endpoint=endpoint,
    )
    logger.info(f"{batch_id=}")

    batch = wait_for_batch(client=client, batch_id=batch_id)
    batch_dir = OUTPUT_DIR / path.stem
    batch_dir.mkdir(parents=True, exist_ok=True)
    # batch_summary = build_batch_summary(batch=batch)
    # save_text(
    #     batch_dir / "batch_summary.json",
    #     json.dumps(batch_summary, indent=2, default=str),
    # )

    if batch.status == "completed":
        save_batch_outputs(client=client, batch=batch, batch_dir=batch_dir)
        move_batch_input_to_done(source=path)
        return

    logger.warning(f"{path=} {batch.status=}")
    move_batch_input_to_done(source=path)


def load_online_config() -> _OnlineConfig:
    """Loads required online configuration from environment variables."""
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    model = os.environ.get("OPENAI_MODEL")
    concurrency_text = os.environ.get("OPENAI_ONLINE_CONCURRENCY")
    concurrency = parse_concurrency(concurrency_text)

    missing_vars: list[str] = []
    if not api_key:
        missing_vars.append("OPENAI_API_KEY")
    if not model:
        missing_vars.append("OPENAI_MODEL")

    if missing_vars:
        missing_text = ", ".join(missing_vars)
        raise ValueError(f"Missing environment variables: {missing_text}")

    return _OnlineConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        concurrency=concurrency,
    )


def parse_concurrency(concurrency_text: str | None) -> int:
    """Parses concurrency from environment text."""
    if concurrency_text is None:
        return DEFAULT_ONLINE_CONCURRENCY

    try:
        concurrency = int(concurrency_text)
    except ValueError as exc:
        raise ValueError(
            f"OPENAI_ONLINE_CONCURRENCY must be an integer: {concurrency_text}"
        ) from exc

    if concurrency <= 0:
        raise ValueError("OPENAI_ONLINE_CONCURRENCY must be greater than zero.")
    return concurrency


def make_online_request_body(record: dict[str, Any], model: str) -> dict[str, Any]:
    """Creates a Responses API payload from a batch JSONL record."""
    body = record.get("body")
    if not isinstance(body, dict):
        raise ValueError("Each JSONL record must include a 'body' dictionary.")

    payload = dict(body)
    payload["model"] = model
    return payload


def build_online_tasks(records: list[dict[str, Any]], model: str) -> _TaskBuildResult:
    """Builds tasks for online execution and tracks invalid records."""
    tasks: list[_OnlineTask] = []
    invalid_outcomes: list[_OnlineOutcome] = []

    for index, record in enumerate(records, start=1):
        custom_id = str(record.get("custom_id", f"line_{index}"))
        try:
            payload = make_online_request_body(record=record, model=model)
        except Exception as exc:  # pylint: disable=broad-except
            invalid_outcomes.append(
                _OnlineOutcome(custom_id=custom_id, response_json=None, error=str(exc))
            )
            logger.exception(f"{custom_id=} {exc=}")
            continue
        tasks.append(_OnlineTask(custom_id=custom_id, payload=payload))

    return _TaskBuildResult(
        tasks=tasks,
        invalid_outcomes=invalid_outcomes,
        total_records=len(records),
    )


def build_online_queue(
    tasks: list[_OnlineTask], concurrency: int
) -> asyncio.Queue[_QueueItem]:
    """Builds a queue containing work items and worker stop sentinels."""
    queue: asyncio.Queue[_QueueItem] = asyncio.Queue()
    for task in tasks:
        queue.put_nowait(task)
    for _ in range(concurrency):
        queue.put_nowait(None)
    return queue


async def execute_online_task(client: AsyncOpenAI, task: _OnlineTask) -> _OnlineOutcome:
    """Executes one online task."""
    try:
        response = await client.responses.create(**task.payload)
        response_json = response.model_dump_json()
        return _OnlineOutcome(
            custom_id=task.custom_id,
            response_json=response_json,
            error=None,
        )
    except Exception as exc:  # pylint: disable=broad-except
        return _OnlineOutcome(
            custom_id=task.custom_id,
            response_json=None,
            error=str(exc),
        )


async def online_worker(
    worker_id: int,
    client: AsyncOpenAI,
    queue: asyncio.Queue[_QueueItem],
    outcomes: list[_OnlineOutcome],
) -> None:
    """Consumes tasks from queue and appends outcomes."""
    logger.info(f"{worker_id=}")
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            logger.info(f"{worker_id=} stopping")
            return

        task: _OnlineTask = item
        outcome = await execute_online_task(client=client, task=task)
        outcomes.append(outcome)
        logger.info(f"{task.custom_id=} {outcome.error=}")
        queue.task_done()


async def execute_online_tasks(
    client: AsyncOpenAI,
    tasks: list[_OnlineTask],
    concurrency: int,
) -> list[_OnlineOutcome]:
    """Executes tasks concurrently with an asyncio queue."""
    queue = build_online_queue(tasks=tasks, concurrency=concurrency)
    outcomes: list[_OnlineOutcome] = []

    worker_tasks = [
        asyncio.create_task(
            online_worker(
                worker_id=worker_id,
                client=client,
                queue=queue,
                outcomes=outcomes,
            )
        )
        for worker_id in range(concurrency)
    ]

    await queue.join()
    await asyncio.gather(*worker_tasks)
    return outcomes


def split_online_outcomes(
    outcomes: list[_OnlineOutcome],
) -> tuple[list[str], list[str]]:
    """Splits outcomes into output and error JSONL lines."""
    output_lines: list[str] = []
    error_lines: list[str] = []

    for outcome in outcomes:
        if outcome.error is None and outcome.response_json is not None:
            output_lines.append(outcome.response_json)
            continue

        error_record = {
            "custom_id": outcome.custom_id,
            "error": outcome.error,
        }
        error_lines.append(json.dumps(error_record, ensure_ascii=False))
    return output_lines, error_lines


def save_online_results(
    path: Path,
    outcomes: list[_OnlineOutcome],
    model: str,
    total_requests: int,
) -> None:
    """Saves online execution artifacts for one input file."""
    result_dir = ONLINE_OUTPUT_DIR / path.stem
    result_dir.mkdir(parents=True, exist_ok=True)

    output_lines, error_lines = split_online_outcomes(outcomes=outcomes)
    if output_lines:
        save_text(result_dir / "output.jsonl", "\n".join(output_lines) + "\n")

    if error_lines:
        save_text(result_dir / "errors.jsonl", "\n".join(error_lines) + "\n")

    summary = {
        "source_file": path.name,
        "total_requests": total_requests,
        "succeeded": len(output_lines),
        "failed": len(error_lines),
        "model": model,
    }
    save_text(result_dir / "summary.json", json.dumps(summary, indent=2))
    logger.info(f"{summary=}")


async def process_one_jsonl_online(
    client: AsyncOpenAI,
    path: Path,
    model: str,
    concurrency: int,
) -> None:
    """Processes one JSONL file with concurrent online Responses API calls."""
    logger.info(f"{path=} {concurrency=}")
    records = load_jsonl_lines(path)
    task_build_result = build_online_tasks(records=records, model=model)

    outcomes = await execute_online_tasks(
        client=client,
        tasks=task_build_result.tasks,
        concurrency=concurrency,
    )
    combined_outcomes = task_build_result.invalid_outcomes + outcomes

    save_online_results(
        path=path,
        outcomes=combined_outcomes,
        model=model,
        total_requests=task_build_result.total_records,
    )


def run_batch_mode(jsonl_files: list[Path]) -> None:
    """Runs all files through the Batch API mode."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = build_client()

    for path in jsonl_files:
        try:
            process_one_jsonl_batch(client=client, path=path)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception(f"{path=} {exc=}")


async def run_online_mode_async(jsonl_files: list[Path]) -> None:
    """Runs all files through concurrent online request mode."""
    ONLINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    online_config = load_online_config()
    logger.info(f"{online_config.base_url=}")
    logger.info(f"{online_config.model=}")
    logger.info(f"{online_config.concurrency=}")

    client = build_async_client(
        api_key=online_config.api_key,
        base_url=online_config.base_url,
    )

    for path in jsonl_files:
        try:
            await process_one_jsonl_online(
                client=client,
                path=path,
                model=online_config.model,
                concurrency=online_config.concurrency,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception(f"{path=} {exc=}")

    await client.close()


def run(mode: str = DEFAULT_MODE) -> None:
    """Runs batch jobs in batch or online mode.

    Args:
        mode: Execution mode, either "batch" or "online".
    """
    configure_logging()
    validate_mode(mode=mode)
    logger.info(f"{mode=}")

    jsonl_files = sorted(INPUT_DIR.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in {INPUT_DIR}")

    if mode == ONLINE_MODE:
        asyncio.run(run_online_mode_async(jsonl_files=jsonl_files))
        return

    run_batch_mode(jsonl_files=jsonl_files)


def main() -> None:
    """Runs the CLI entrypoint via python-fire."""
    fire.Fire(run)


if __name__ == "__main__":
    main()
