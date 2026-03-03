"""
run_scan.py
-----------
Example entry point wiring AxisAwareGenerator and AxisMonitor into a
Garak probe run. Adapt probe selection and detector list to your needs.

Usage:
    python run_scan.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --axis  axis/vectors/llama3_8b_layer22.pt \
        --probes garak.probes.dan,garak.probes.promptinject \
        --output results/
"""

import argparse
import importlib
import logging
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


def run(model_name: str, axis_path: str, probe_names: list, output_dir: str):
    from garak.generators.axis_aware import AxisAwareGenerator
    from garak.detectors.axis_monitor import AxisMonitor

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    log.info("Initialising AxisAwareGenerator ...")
    generator = AxisAwareGenerator(name=model_name, axis_path=axis_path)

    monitor = AxisMonitor(
        generator=generator,
        output_path=str(out / "axis_results.jsonl"),
    )

    for probe_name in probe_names:
        log.info("Running probe: %s", probe_name)
        try:
            module_path, class_name = probe_name.rsplit(".", 1)
            module = importlib.import_module(module_path)
            ProbeClass = getattr(module, class_name)
            probe = ProbeClass()
            attempts = probe.probe(generator)
            for attempt in attempts:
                monitor.detect(attempt)
        except Exception as exc:
            log.error("Probe %s failed: %s", probe_name, exc)

    summary = monitor.summary()
    log.info("--- Axis Monitor Summary ---")
    for sig, count in summary.items():
        log.info("  %-22s %d", sig, count)

    log.info("Results written to %s", output_dir)


def _parse_args():
    p = argparse.ArgumentParser(description="Garak scan with Assistant Axis monitoring.")
    p.add_argument("--model",   required=True, help="HuggingFace model name or local path")
    p.add_argument("--axis",    required=True, help="Path to axis .pt checkpoint")
    p.add_argument("--probes",  default="garak.probes.dan.Dan_11_0",
                   help="Comma-separated list of Garak probe class paths")
    p.add_argument("--output",  default="results/", help="Output directory")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        model_name=args.model,
        axis_path=args.axis,
        probe_names=args.probes.split(","),
        output_dir=args.output,
    )
