"""Post-processing utilities for saved model inference outputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_FINE_TUNED_RAW_PATH = Path("inference/fine-tuned-raw.jsonl")
DEFAULT_FINE_TUNED_OUTPUT_PATH = Path("inference/fine-tuned.jsonl")

_STRATEGY_PREFIX = re.compile(
    r"^\s*\[[^\]\r\n]+\]\s*(?P<response>.+?)\s*$",
    flags=re.DOTALL,
)


def extract_quoted_cot_response(output: str) -> tuple[str, bool]:
    """Extract the first quoted response while leaving unquoted output unchanged.

    Normal JSON decoding turns ``\"...\"`` in a JSONL file into ``"..."``.
    Literal backslash-quote delimiters are also supported in case a model emits
    them directly.
    """
    escaped_quote = r'\"'
    first = output.find(escaped_quote)
    if first != -1:
        second = output.find(escaped_quote, first + len(escaped_quote))
        if second != -1:
            response = output[first + len(escaped_quote) : second].strip()
            if response:
                return response, True

    first = output.find('"')
    if first != -1:
        second = output.find('"', first + 1)
        if second != -1:
            response = output[first + 1 : second].strip()
            if response:
                return response, True

    return output, False


def clean_cot_outputs(
    input_path: str | Path,
    output_path: str | Path,
) -> dict[str, int | str]:
    """Extract quoted ECoT/ESC-CoT responses into a separate JSONL file."""
    source = Path(input_path)
    destination = Path(output_path)
    if source.resolve() == destination.resolve():
        raise ValueError("input_path and output_path must be different")
    if not source.exists():
        raise FileNotFoundError(f"CoT output does not exist: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    record_count = 0
    extracted_count = 0
    with source.open("r", encoding="utf-8") as input_file, destination.open(
        "w", encoding="utf-8", newline="\n"
    ) as output_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("output"), str):
                raise ValueError(
                    f"Invalid CoT record at {source}:{line_number}: {record!r}"
                )

            response, extracted = extract_quoted_cot_response(record["output"])
            record["output"] = response
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            record_count += 1
            extracted_count += int(extracted)

    return {
        "input_path": str(source.resolve()),
        "output_path": str(destination.resolve()),
        "records": record_count,
        "extracted": extracted_count,
        "unchanged": record_count - extracted_count,
        "bytes": destination.stat().st_size,
    }


def clean_fine_tuned_outputs(
    input_path: str | Path,
    output_path: str | Path,
) -> dict[str, int | str]:
    """Remove leading strategy labels from fine-tuned JSONL responses.

    The raw model output is expected to start with ``[strategy]``. The cleaned
    response retains the remaining text and uppercases its first character.
    """
    source = Path(input_path)
    destination = Path(output_path)
    if source.resolve() == destination.resolve():
        raise ValueError("input_path and output_path must be different")
    if not source.exists():
        raise FileNotFoundError(f"Fine-tuned raw output does not exist: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    record_count = 0
    with source.open("r", encoding="utf-8") as input_file, destination.open(
        "w", encoding="utf-8", newline="\n"
    ) as output_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("output"), str):
                raise ValueError(
                    f"Invalid fine-tuned record at {source}:{line_number}: {record!r}"
                )

            match = _STRATEGY_PREFIX.fullmatch(record["output"])
            if match is None:
                raise ValueError(
                    f"Output does not start with a strategy label at "
                    f"{source}:{line_number}: {record['output']!r}"
                )
            response = match.group("response")
            record["output"] = response[:1].upper() + response[1:]
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            record_count += 1

    return {
        "input_path": str(source.resolve()),
        "output_path": str(destination.resolve()),
        "records": record_count,
        "bytes": destination.stat().st_size,
    }


def main(argv: list[str] | None = None) -> dict[str, int | str]:
    parser = argparse.ArgumentParser(
        description="Post-process saved fine-tuned and CoT inference outputs."
    )
    parser.add_argument(
        "--fine-tuned-raw",
        type=Path,
        default=DEFAULT_FINE_TUNED_RAW_PATH,
        help="Raw fine-tuned JSONL containing leading strategy labels",
    )
    parser.add_argument(
        "--fine-tuned-output",
        type=Path,
        default=DEFAULT_FINE_TUNED_OUTPUT_PATH,
        help="Destination for cleaned fine-tuned JSONL",
    )
    parser.add_argument(
        "--cot",
        action="append",
        nargs=2,
        type=Path,
        metavar=("INPUT", "OUTPUT"),
        help=(
            "Extract quoted CoT responses from INPUT into OUTPUT; repeat this "
            "option to process multiple files"
        ),
    )
    args = parser.parse_args(argv)

    results: dict[str, int | str] = {}
    fine_tuned_raw = args.fine_tuned_raw
    fine_tuned_output = args.fine_tuned_output
    if fine_tuned_output.exists():
        print(f"Fine-tuned output already exists; skipping: {fine_tuned_output.resolve()}")
    elif fine_tuned_raw.exists():
        summary = clean_fine_tuned_outputs(fine_tuned_raw, fine_tuned_output)
        results[str(fine_tuned_output)] = int(summary["records"])
        print(f"Cleaned {summary['records']} fine-tuned inference records")
        print(f"Output: {summary['output_path']}")
    else:
        print(f"Fine-tuned raw input not found; skipping: {fine_tuned_raw.resolve()}")

    for cot_input, cot_output in args.cot or []:
        summary = clean_cot_outputs(cot_input, cot_output)
        results[str(cot_output)] = int(summary["records"])
        print(
            f"Processed {summary['records']} CoT records: "
            f"extracted {summary['extracted']}, unchanged {summary['unchanged']}"
        )
        print(f"Output: {summary['output_path']}")
    return results


if __name__ == "__main__":
    main()
