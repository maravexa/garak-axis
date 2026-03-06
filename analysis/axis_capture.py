"""
axis_capture.py
---------------
Reads prompts from a garak report.jsonl, runs each through AxisAwareGenerator
to capture axis displacement, and writes a prompt-keyed lookup file.

Run this against any existing garak report — no need to re-run the scan.

Usage:
    python analysis/axis_capture.py \
        --report  results/garak_test.report.jsonl \
        --model   Qwen/Qwen2.5-1.5B-Instruct \
        --axis    axis/vectors/qwen2.5_1.5b_layer16_v2.pt \
        --output  results/axis_capture.jsonl

Output format (one JSON object per line):
    {
      "attempt_uuid": "...",
      "prompt": "...",
      "displacement": -0.42,
      "probe_classname": "dan.Dan_11_0"
    }
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def load_attempts(report_path: str) -> list:
    """Extract attempt records from a garak report.jsonl."""
    attempts = []
    with open(report_path) as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get("entry_type") == "attempt":
                    attempts.append(entry)
            except json.JSONDecodeError:
                continue
    log.info("Loaded %d attempt records from %s", len(attempts), report_path)
    return attempts


def deduplicate_prompts(attempts: list) -> list:
    """Deduplicate by prompt text — each unique prompt only needs one forward pass."""
    seen = {}
    for a in attempts:
        prompt = a.get("prompt", "")
        if prompt and prompt not in seen:
            seen[prompt] = {
                "attempt_uuid":   a.get("uuid", ""),
                "prompt":         prompt,
                "probe_classname": a.get("probe_classname", ""),
            }
    log.info("%d unique prompts after deduplication", len(seen))
    return list(seen.values())


def capture(report_path: str, model_name: str, axis_path: str, output_path: str):
    # Import here so the script is importable without torch installed
    from garak_axis_ext.generators.axis_aware import AxisAwareGenerator

    attempts = load_attempts(report_path)
    unique = deduplicate_prompts(attempts)

    log.info("Loading AxisAwareGenerator ...")
    gen = AxisAwareGenerator(
        name=model_name,
        axis_path=axis_path,
        device="cuda",
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    log.info("Capturing axis displacement for %d prompts ...", len(unique))
    with out.open("w") as fh:
        for i, record in enumerate(unique):
            gen.generate(record["prompt"])
            displacement = gen.store.flush()[0]
            result = {
                "attempt_uuid":    record["attempt_uuid"],
                "prompt":          record["prompt"],
                "displacement":    displacement,
                "probe_classname": record["probe_classname"],
            }
            fh.write(json.dumps(result) + "\n")

            if (i + 1) % 10 == 0 or (i + 1) == len(unique):
                log.info("  %d / %d  (latest displacement: %+.4f)",
                         i + 1, len(unique), displacement)

    log.info("Axis capture written to %s", output_path)


def _parse_args():
    p = argparse.ArgumentParser(description="Capture axis displacement for garak report prompts.")
    p.add_argument("--report", required=True, help="Path to garak report.jsonl")
    p.add_argument("--model",  required=True, help="HuggingFace model name or local path")
    p.add_argument("--axis",   required=True, help="Path to axis .pt checkpoint")
    p.add_argument("--output", default="results/axis_capture.jsonl",
                   help="Output path for displacement lookup")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    capture(
        report_path=args.report,
        model_name=args.model,
        axis_path=args.axis,
        output_path=args.output,
    )
