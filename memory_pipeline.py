"""
OpenAI-assisted memory dataset pipeline for the BDH demo.

This script keeps the workflow minimal:
1. Generate high-quality seed examples with normal Responses API calls.
2. Expand each seed into many paraphrases with the Batch API.
3. Finalize everything into JSONL files that `train.py` can consume directly.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from openai import OpenAI


DEFAULT_SEED_MODEL = "gpt-4o"
DEFAULT_BATCH_MODEL = "gpt-4o-mini"
MEMORY_TYPES = ["entity_fact"]
_E = r"(?P<entity>[A-Za-z0-9][A-Za-z0-9' -]{0,40})"
_V = r"(?P<value>[A-Za-z0-9][A-Za-z0-9' -]{0,40})"
_ART = r"(?:a|an|the) "
FACT_PATTERNS = [
    # "<Entity> is (a/an) <value>."
    re.compile(rf"^{_E} is (?:{_ART})?{_V}\.?$", re.IGNORECASE),
    # "<Entity> qualifies as (a/an) <value>."
    re.compile(rf"^{_E} qualifies as (?:{_ART})?{_V}\.?$", re.IGNORECASE),
    # "<Entity> classifies as (a/an) <value>."
    re.compile(rf"^{_E} classifies as (?:{_ART})?{_V}\.?$", re.IGNORECASE),
    # "<Entity> stands as (a/an) <value>."
    re.compile(rf"^{_E} stands as (?:{_ART})?{_V}\.?$", re.IGNORECASE),
    # "<Entity> represents (a/an) <value>."
    re.compile(rf"^{_E} represents (?:{_ART})?{_V}\.?$", re.IGNORECASE),
    # "<Entity> constitutes (a/an) <value>."
    re.compile(rf"^{_E} constitutes (?:{_ART})?{_V}\.?$", re.IGNORECASE),
    # "<Entity> functions as (a/an) <value>."
    re.compile(rf"^{_E} functions as (?:{_ART})?{_V}\.?$", re.IGNORECASE),
    # "<Entity> can be (described/categorized/defined/identified/called) as (a/an) <value>."
    re.compile(rf"^{_E} can be (?:described|categorized|defined|identified|called) as (?:{_ART})?{_V}\.?$", re.IGNORECASE),
    # "<Entity> falls (into the category of|under the category of) (a/an) <value>."
    re.compile(rf"^{_E} falls (?:into|under) (?:the category of )?(?:{_ART})?{_V}\.?$", re.IGNORECASE),
    # "<Entity> (predominantly )?appears as (a/an) <value>."
    re.compile(rf"^{_E} (?:predominantly )?appears as (?:{_ART})?{_V}\.?$", re.IGNORECASE),
    # "<Entity> holds the title of (a/an) <value>."
    re.compile(rf"^{_E} holds the title of (?:{_ART})?{_V}\.?$", re.IGNORECASE),
    # "<Entity> describes (a/an) <value>."
    re.compile(rf"^{_E} describes (?:{_ART})?{_V}\.?$", re.IGNORECASE),
    # "<Entity> exists in a <value> state."
    re.compile(rf"^{_E} exists in (?:{_ART})?{_V} state\.?$", re.IGNORECASE),
    # "<Entity> exists as a <value>."
    re.compile(rf"^{_E} exists as (?:{_ART})?{_V}\.?$", re.IGNORECASE),
    # "In terms of classification, <Entity> is (a/an) <value>."
    re.compile(rf"^In (?:terms of classification|its nature), {_E} is (?:{_ART})?{_V}\.?$", re.IGNORECASE),
    # "<Entity> fits the definition of (a/an) <value>."
    re.compile(rf"^{_E} fits the definition of (?:{_ART})?{_V}\.?$", re.IGNORECASE),
]
QUESTION_PATTERN = re.compile(r"^What is (?P<entity>[A-Za-z0-9][A-Za-z0-9' -]{0,40})\?$", re.IGNORECASE)

MEMORY_EXAMPLES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["examples"],
    "properties": {
        "examples": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["memory_type", "fact_statement", "ack", "question", "answer"],
                "properties": {
                    "memory_type": {
                        "type": "string",
                        "enum": MEMORY_TYPES,
                    },
                    "fact_statement": {"type": "string"},
                    "ack": {"type": "string"},
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                },
            },
        }
    },
}

SEED_SYSTEM_PROMPT = """
You generate short, high-quality training episodes for a language model memory demo.

Each episode must contain:
- one user fact statement
- one short assistant acknowledgement
- one user recall question
- one short assistant answer

Rules:
- Return JSON only.
- Keep episodes natural and short.
- Use exactly this memory pattern:
  - fact_statement: "<Entity> is a <value>." or "<Entity> is an <value>."
  - question: "What is <Entity>?"
  - ack: exactly "Noted."
  - answer: exactly the same string as fact_statement
- Use only one fact per episode.
- Keep memory_type equal to "entity_fact".
- Do not include explanations or extra fields.
""".strip()

PARAPHRASE_SYSTEM_PROMPT = """
You paraphrase memory-training episodes while preserving the exact semantics.

Rules:
- Return JSON only.
- Keep the same memory_type.
- Keep the same entity and the same answer string exactly.
- Keep the question in the exact form "What is <Entity>?"
- Keep the fact statement in the exact form "<Entity> is a/an <value>."
- Keep ack exactly "Noted."
- Keep answer exactly equal to fact_statement.
- Do not add extra facts, explanations, or metadata.
""".strip()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def dump_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def serialize_api_object(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return obj


def normalize_text(value: str) -> str:
    return " ".join(value.split()).strip()


def normalize_label(value: str) -> str:
    return re.sub(r"[^\w\s'-]", "", normalize_text(value)).strip().lower()


def extract_fact(fact_statement: str) -> tuple[str, str] | None:
    for pattern in FACT_PATTERNS:
        match = pattern.match(fact_statement)
        if match:
            entity = normalize_text(match.group("entity")).strip(" .?!")
            value = normalize_text(match.group("value")).strip(" .?!")
            if entity and value:
                return entity, value
    return None


_DROP_COUNTS: dict[str, int] = {}

def _drop(reason: str) -> None:
    _DROP_COUNTS[reason] = _DROP_COUNTS.get(reason, 0) + 1

def validate_example(example: dict[str, Any]) -> dict[str, Any] | None:
    required = ["memory_type", "fact_statement", "ack", "question", "answer"]
    if any(key not in example for key in required):
        _drop("missing_keys")
        return None

    memory_type = str(example["memory_type"]).strip()
    if memory_type not in MEMORY_TYPES:
        _drop(f"bad_memory_type:{memory_type}")
        return None

    fact_statement = normalize_text(str(example["fact_statement"]))
    question = normalize_text(str(example["question"]))
    answer = normalize_text(str(example["answer"]))
    ack = normalize_text(str(example["ack"]))

    parsed_fact = extract_fact(fact_statement)
    if parsed_fact is None:
        _drop("fact_no_parse")
        return None
    entity, value = parsed_fact

    question_match = QUESTION_PATTERN.match(question)
    if question_match is None:
        _drop("question_no_match")
        return None
    question_entity = normalize_text(question_match.group("entity")).strip(" .?!")
    if normalize_label(question_entity) != normalize_label(entity):
        _drop("question_entity_mismatch")
        return None

    # Accept if answer extracts to same entity/value as fact
    answer_parsed = extract_fact(answer)
    if answer_parsed is None:
        _drop("answer_no_parse")
        return None
    answer_entity, answer_value = answer_parsed
    if normalize_label(answer_entity) != normalize_label(entity) or normalize_label(answer_value) != normalize_label(value):
        _drop("answer_entity_value_mismatch")
        return None
    if ack != "Noted.":
        _drop(f"bad_ack:{ack[:30]}")
        return None

    cleaned = {
        "memory_type": memory_type,
        "fact_statement": fact_statement,
        "ack": ack,
        "question": question,
        "answer": answer,
        "entity": entity,
        "value": value,
    }

    if not all(cleaned[key] for key in ["fact_statement", "ack", "question", "answer"]):
        _drop("empty_field")
        return None
    if "\n" in cleaned["answer"] or "\n" in cleaned["ack"]:
        _drop("newline_in_answer_or_ack")
        return None
    return cleaned


def attach_metadata(example: dict[str, Any], source: str, example_id: str) -> dict[str, Any]:
    enriched = dict(example)
    enriched["example_id"] = example_id
    enriched["source"] = source
    return enriched


def response_output_text(response: Any) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    if isinstance(response, dict):
        output = response.get("output", [])
    else:
        output = getattr(response, "output", [])

    chunks: list[str] = []
    for item in output or []:
        item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        if item_type != "message":
            continue
        content = item.get("content") if isinstance(item, dict) else getattr(item, "content", [])
        for part in content or []:
            part_type = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
            if part_type == "output_text":
                text = part.get("text") if isinstance(part, dict) else getattr(part, "text", "")
                if text:
                    chunks.append(text)

    if not chunks:
        raise ValueError("Could not find output_text in model response.")
    return "".join(chunks)


def batch_result_output_text(record: dict[str, Any]) -> str:
    if record.get("error"):
        raise ValueError(f"Batch request failed: {record['error']}")
    response = record.get("response") or {}
    body = response.get("body") or {}
    return response_output_text(body)


def build_seed_user_prompt(num_examples: int) -> str:
    return (
        f"Generate {num_examples} memory episodes.\n"
        "Use memory_type = entity_fact for every example.\n"
        'Use ack exactly as "Noted." for every example.\n'
        "The answer field must be exactly identical to fact_statement.\n"
        "Keep each episode short, clean, and suitable for a hackathon memory demo."
    )


def build_paraphrase_user_prompt(seed_example: dict[str, Any], variants_per_seed: int) -> str:
    seed_json = json.dumps(seed_example, ensure_ascii=False, indent=2)
    return (
        f"Create {variants_per_seed} paraphrased variants of this seed example.\n"
        'Keep ack exactly "Noted.".\n'
        "Keep answer exactly identical to the new fact_statement.\n"
        "Vary the wording of the fact statement while preserving the same entity and value.\n"
        'Keep the recall question in the exact form "What is <Entity>?".\n'
        "Return JSON only.\n\n"
        f"Seed example:\n{seed_json}"
    )


def episode_to_text(example: dict[str, Any]) -> str:
    return (
        f"User: {example['fact_statement']}\n"
        f"Assistant: {example['ack']}\n\n"
        f"User: {example['question']}\n"
        f"Assistant: {example['answer']}\n"
    )


def build_training_rows(example: dict[str, Any]) -> list[dict[str, Any]]:
    # These two rows mirror the actual demo flow:
    # 1) store the fact
    # 2) ask for recall later without re-sending the fact
    store_input = f"User: {example['fact_statement']}\nAssistant:"
    store_target = " Noted."
    recall_input = f"User: {example['question']}\nAssistant:"
    recall_target = f" {example['fact_statement']}"
    return [
        {
            "type": "store",
            "memory_type": example["memory_type"],
            "example_id": f"{example['example_id']}-store",
            "source": example["source"],
            "entity": example["entity"],
            "value": example["value"],
            "input": store_input,
            "target": store_target,
            "text": store_input + store_target,
        },
        {
            "type": "recall",
            "memory_type": example["memory_type"],
            "example_id": f"{example['example_id']}-recall",
            "source": example["source"],
            "entity": example["entity"],
            "value": example["value"],
            "input": recall_input,
            "target": recall_target,
            "text": recall_input + recall_target,
        },
    ]


def call_structured_examples(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
) -> list[dict[str, Any]]:
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_output_tokens=max_output_tokens,
        text={
            "format": {
                "type": "json_schema",
                "name": "memory_examples",
                "strict": True,
                "schema": MEMORY_EXAMPLES_SCHEMA,
            }
        },
    )
    payload = json.loads(response_output_text(response))
    return payload["examples"]


def command_generate_seeds(args: argparse.Namespace) -> None:
    out_path = Path(args.out)
    if out_path.exists() and not args.overwrite and not args.append:
        raise FileExistsError(f"{out_path} already exists. Pass --overwrite to replace it or --append to add to it.")

    client = OpenAI()
    existing: list[dict[str, Any]] = []
    if args.append and out_path.exists():
        existing = list(read_jsonl(out_path))
        print(f"Loaded {len(existing)} existing seeds, appending {args.num_examples} more...")

    examples: list[dict[str, Any]] = []
    requested = args.num_examples
    cursor = len(existing)
    attempts = 0

    while len(examples) < requested and attempts < args.max_calls:
        attempts += 1
        remaining = requested - len(examples)
        batch_size = min(args.examples_per_call, remaining)
        raw_examples = call_structured_examples(
            client=client,
            model=args.model,
            system_prompt=SEED_SYSTEM_PROMPT,
            user_prompt=build_seed_user_prompt(batch_size),
            max_output_tokens=args.max_output_tokens,
        )
        for raw in raw_examples:
            example_id = f"seed-{cursor:05d}"
            validated = validate_example(raw)
            cursor += 1
            if validated is not None:
                examples.append(attach_metadata(validated, source="seed", example_id=example_id))
            if len(examples) >= requested:
                break

    if len(examples) < requested:
        raise RuntimeError(
            f"Only collected {len(examples)} valid examples after {attempts} calls. "
            "Try increasing --max-calls or lowering --examples-per-call."
        )

    all_examples = existing + examples[:requested]
    write_jsonl(out_path, all_examples)
    print(f"Saved {len(all_examples)} seed examples to {out_path} ({len(examples[:requested])} new)")


def command_build_batch(args: argparse.Namespace) -> None:
    seed_rows = read_jsonl(Path(args.seed_file))
    out_path = Path(args.out)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"{out_path} already exists. Pass --overwrite to replace it.")

    requests: list[dict[str, Any]] = []
    for row in seed_rows:
        request = {
            "custom_id": row["example_id"],
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": args.model,
                "input": [
                    {"role": "system", "content": PARAPHRASE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_paraphrase_user_prompt(
                            seed_example=row,
                            variants_per_seed=args.variants_per_seed,
                        ),
                    },
                ],
                "max_output_tokens": args.max_output_tokens,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "memory_examples",
                        "strict": True,
                        "schema": MEMORY_EXAMPLES_SCHEMA,
                    }
                },
            },
        }
        requests.append(request)

    write_jsonl(out_path, requests)
    print(f"Saved {len(requests)} batch requests to {out_path}")


def command_submit_batch(args: argparse.Namespace) -> None:
    client = OpenAI()
    input_path = Path(args.input_file)
    with input_path.open("rb") as handle:
        uploaded_file = client.files.create(file=handle, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"job": args.job_name},
    )

    payload = {
        "job_name": args.job_name,
        "input_file": str(input_path),
        "uploaded_file_id": uploaded_file.id,
        "batch": serialize_api_object(batch),
    }
    if args.metadata_out:
        dump_json(Path(args.metadata_out), payload)
    print(f"Submitted batch {batch.id}")
    print(f"Uploaded input file id: {uploaded_file.id}")


def resolve_batch_id(args: argparse.Namespace) -> str:
    if args.batch_id:
        return args.batch_id
    if not args.metadata_file:
        raise ValueError("Provide either --batch-id or --metadata-file.")
    metadata = json.loads(Path(args.metadata_file).read_text(encoding="utf-8"))
    batch = metadata.get("batch") or {}
    batch_id = batch.get("id")
    if not batch_id:
        raise ValueError(f"No batch id found in {args.metadata_file}")
    return batch_id


def command_batch_status(args: argparse.Namespace) -> None:
    client = OpenAI()
    batch_id = resolve_batch_id(args)
    batch = client.batches.retrieve(batch_id)
    payload = serialize_api_object(batch)
    print(json.dumps(payload, indent=2))


def command_download_batch(args: argparse.Namespace) -> None:
    client = OpenAI()
    batch_id = resolve_batch_id(args)
    batch = client.batches.retrieve(batch_id)
    if not batch.output_file_id:
        raise ValueError(f"Batch {batch_id} has no output_file_id yet. Current status: {batch.status}")

    content = client.files.content(batch.output_file_id)
    out_path = Path(args.out)
    ensure_parent(out_path)
    out_path.write_text(content.text, encoding="utf-8")

    if batch.error_file_id and args.errors_out:
        errors = client.files.content(batch.error_file_id)
        errors_out = Path(args.errors_out)
        ensure_parent(errors_out)
        errors_out.write_text(errors.text, encoding="utf-8")

    print(f"Downloaded batch output to {out_path}")


def dedupe_examples(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for example in examples:
        key = (
            example["memory_type"],
            example["fact_statement"],
            example["ack"],
            example["question"],
            example["answer"],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(example)
    return deduped


def command_finalize(args: argparse.Namespace) -> None:
    raw_seed_rows = read_jsonl(Path(args.seed_file))
    batch_rows = read_jsonl(Path(args.batch_output))

    combined: list[dict[str, Any]] = []
    for row in raw_seed_rows:
        validated = validate_example(row)
        if validated is not None:
            combined.append(attach_metadata(validated, source=row.get("source", "seed"), example_id=row.get("example_id", f"seed-{len(combined):05d}")))

    counter = 0
    failed_rows: list[dict[str, Any]] = []
    successful_rows = 0

    for batch_row in batch_rows:
        custom_id = batch_row.get("custom_id", "batch")
        try:
            parsed = json.loads(batch_result_output_text(batch_row))
        except Exception as exc:
            failed_rows.append({"custom_id": custom_id, "reason": str(exc)})
            continue

        for raw in parsed["examples"]:
            example_id = f"batch-{counter:06d}"
            validated = validate_example(raw)
            counter += 1
            if validated is not None:
                combined.append(attach_metadata(validated, source=f"batch:{custom_id}", example_id=example_id))
        successful_rows += 1

    combined = dedupe_examples(combined)
    rng = random.Random(args.shuffle_seed)
    rng.shuffle(combined)

    split_index = max(1, int(len(combined) * args.train_ratio))
    train_examples = combined[:split_index]
    valid_examples = combined[split_index:] or combined[-min(50, len(combined)) :]
    train_rows = [row for example in train_examples for row in build_training_rows(example)]
    valid_rows = [row for example in valid_examples for row in build_training_rows(example)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dump_json(out_dir / "summary.json", {
        "total_examples": len(combined),
        "train_examples": len(train_examples),
        "valid_examples": len(valid_examples),
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "failed_batch_rows": len(failed_rows),
        "memory_types": {memory_type: sum(ex["memory_type"] == memory_type for ex in combined) for memory_type in MEMORY_TYPES},
    })
    write_jsonl(out_dir / "all_examples.jsonl", combined)
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "valid.jsonl", valid_rows)
    if failed_rows:
        write_jsonl(out_dir / "failed_rows.jsonl", failed_rows)

    print(f"Finalized {len(combined)} examples into {out_dir}")
    print(f"Train split: {len(train_examples)} examples | {len(train_rows)} rows")
    print(f"Valid split: {len(valid_examples)} examples | {len(valid_rows)} rows")
    print(f"Processed: {len(batch_rows)} | सफल: {successful_rows} | Failed: {len(failed_rows)}")
    if _DROP_COUNTS:
        print("\nDrop reasons:")
        for reason, count in sorted(_DROP_COUNTS.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenAI-backed synthetic memory dataset pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_seeds = subparsers.add_parser("generate-seeds", help="Generate high-quality seed examples.")
    generate_seeds.add_argument("--out", default="memory_data/seeds.jsonl")
    generate_seeds.add_argument("--num-examples", type=int, default=120)
    generate_seeds.add_argument("--examples-per-call", type=int, default=20)
    generate_seeds.add_argument("--model", default=DEFAULT_SEED_MODEL)
    generate_seeds.add_argument("--max-output-tokens", type=int, default=4000)
    generate_seeds.add_argument("--max-calls", type=int, default=20)
    generate_seeds.add_argument("--overwrite", action="store_true")
    generate_seeds.add_argument("--append", action="store_true", help="Add new seeds to existing file instead of overwriting")
    generate_seeds.set_defaults(func=command_generate_seeds)

    build_batch = subparsers.add_parser("build-batch", help="Create a JSONL file for Batch API expansion.")
    build_batch.add_argument("--seed-file", default="memory_data/seeds.jsonl")
    build_batch.add_argument("--out", default="memory_data/batch_input.jsonl")
    build_batch.add_argument("--variants-per-seed", type=int, default=10)
    build_batch.add_argument("--model", default=DEFAULT_BATCH_MODEL)
    build_batch.add_argument("--max-output-tokens", type=int, default=4000)
    build_batch.add_argument("--overwrite", action="store_true")
    build_batch.set_defaults(func=command_build_batch)

    submit_batch = subparsers.add_parser("submit-batch", help="Upload a batch input file and create a batch job.")
    submit_batch.add_argument("--input-file", default="memory_data/batch_input.jsonl")
    submit_batch.add_argument("--job-name", default="bdh-memory-data")
    submit_batch.add_argument("--metadata-out", default="memory_data/batch_job.json")
    submit_batch.set_defaults(func=command_submit_batch)

    batch_status = subparsers.add_parser("batch-status", help="Fetch current status for a batch.")
    batch_status.add_argument("--batch-id")
    batch_status.add_argument("--metadata-file", default="memory_data/batch_job.json")
    batch_status.set_defaults(func=command_batch_status)

    download_batch = subparsers.add_parser("download-batch", help="Download completed batch outputs.")
    download_batch.add_argument("--batch-id")
    download_batch.add_argument("--metadata-file", default="memory_data/batch_job.json")
    download_batch.add_argument("--out", default="memory_data/batch_output.jsonl")
    download_batch.add_argument("--errors-out", default="memory_data/batch_errors.jsonl")
    download_batch.set_defaults(func=command_download_batch)

    finalize = subparsers.add_parser("finalize", help="Merge seeds and batch outputs into trainable JSONL files.")
    finalize.add_argument("--seed-file", default="memory_data/seeds.jsonl")
    finalize.add_argument("--batch-output", default="memory_data/batch_output.jsonl")
    finalize.add_argument("--out-dir", default="memory_data/final")
    finalize.add_argument("--train-ratio", type=float, default=0.9)
    finalize.add_argument("--shuffle-seed", type=int, default=7)
    finalize.set_defaults(func=command_finalize)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
