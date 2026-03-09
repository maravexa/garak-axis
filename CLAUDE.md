# CLAUDE.md

## Project Overview

**garak-axis** augments [Garak](https://github.com/leondz/garak) LLM vulnerability scanning with mechanistic persona monitoring based on Anthropic's [Assistant Axis](https://www.anthropic.com/research/assistant-axis) research. It tracks how far a model's internal persona drifts from its trained Assistant persona during adversarial probing, producing a four-way attack signature taxonomy (ALIGNMENT_FAILURE, PERSONA_JAILBREAK, NEAR_MISS, CLEAN).

## Repository Structure

```
garak-axis/
├── axis/                        # Core axis extraction
│   ├── axis_extractor.py        # PCA-based axis derivation (run once per model)
│   └── vectors/                 # Saved axis checkpoints (.pt files, gitignored)
├── garak_axis_ext/              # Local extension package (primary — avoids namespace collision)
│   ├── generators/
│   │   ├── axis_aware.py        # AxisAwareGenerator — HF generator with activation hook
│   │   └── axis_store.py        # Thread-local displacement buffer
│   └── detectors/
│       └── axis_monitor.py      # AxisMonitor — inline classifier (programmatic use)
├── garak/                       # Legacy mirror of garak_axis_ext (kept for compatibility)
│   ├── generators/
│   │   ├── axis_aware.py
│   │   └── axis_store.py
│   └── detectors/
│       └── axis_monitor.py
├── analysis/
│   ├── axis_capture.py          # Reads garak report, runs prompts through axis generator
│   └── axis_join.py             # Joins garak output + axis data, emits augmented JSONL
├── personas/
│   └── corpus.json              # Persona prompt corpus for PCA extraction
├── results/                     # Scan outputs (gitignored)
├── charts/                      # Generated visualization outputs
├── generate_charts.py           # Chart generation utility (per-family and comparison modes)
└── README.md
```

## Tech Stack

- **Python 3.10+**
- **PyTorch 2.5.1** (ROCm 6.2 for AMD GPU; also works with CUDA)
- **HuggingFace transformers** — model loading (AutoModelForCausalLM, AutoTokenizer)
- **scikit-learn** — PCA decomposition for axis extraction
- **garak 0.9.0.14** — LLM vulnerability scanner (pinned version; 0.14.x breaks probe loading)
- **accelerate** — distributed inference
- **matplotlib** — chart generation
- **numpy** — numerical operations

## Pipeline (Execution Order)

1. **Extract axis** — `python3 axis/axis_extractor.py` (once per model)
2. **Run garak scan** — `garak --model_type huggingface --model_name <model> --probes <probe>`
3. **Capture displacement** — `python3 analysis/axis_capture.py --report <report.jsonl> ...`
4. **Join and classify** — `python3 analysis/axis_join.py --report <report.jsonl> --capture <capture.jsonl> ...`
5. **Visualize** — `python3 generate_charts.py ...`

## Key Classes and Functions

- **`AxisAwareGenerator`** (`garak_axis_ext/generators/axis_aware.py`) — Garak-compatible generator wrapping HF models with an activation hook to capture displacement scalars. Two forward passes per turn: one for activation capture, one for text generation.
- **`AxisStore`** (`garak_axis_ext/generators/axis_store.py`) — Thread-local buffer for displacement scalars using `threading.local`.
- **`AxisMonitor`** (`garak_axis_ext/detectors/axis_monitor.py`) — Garak-compatible detector that collects axis data and classifies attack signatures.
- **`axis_extractor.py`** (`axis/`) — Standalone script deriving the axis vector via PCA over persona activations.

## Code Conventions

### Style
- **snake_case** for functions and variables, **PascalCase** for classes
- Type hints used selectively (`List`, `Optional`, `torch.Tensor`)
- Module-level docstrings with description and `Usage:` section
- Function docstrings with brief description, `Args:`, and `Returns:` in plain text
- Section dividers using `# ---------...` comments
- Constants at module top (e.g., `DEFAULT_DISPLACEMENT_THRESHOLD`, `DEFAULT_MAX_NEW_TOKENS`)

### Module Pattern
Every module follows: module docstring → imports (stdlib → third-party → local) → constants → functions/classes → CLI entry point (`_parse_args()` + `if __name__ == "__main__":`)

### Logging
```python
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
```

### Data Formats
- **JSONL** (one JSON object per line) for all pipeline outputs
- **PyTorch `.pt`** checkpoints for axis vectors
- **JSON** for persona corpus

## Build / Test / Lint

There is **no formal build system, test suite, or linter configuration**. This is research code.

- No `pyproject.toml`, `setup.py`, `Makefile`, or `tox.ini`
- No pytest, unittest, or test directories
- No flake8, ruff, mypy, or pre-commit hooks
- No CI/CD pipeline (no `.github/workflows/`)

**Validation is manual:** run the smoke test (step 7 in README) to verify axis quality by checking PC1 explained variance (>25%, typically ~98% for Qwen2.5).

## Installation Notes

- Dependencies are installed via individual `pip install` commands (no requirements.txt)
- The `garak_axis_ext` package is registered on the Python path via a `.pth` file in the venv's `site-packages/`
- Garak **must** be pinned to `0.9.0.14` — later versions break probe loading
- PyTorch must be installed first with `--index-url` for ROCm wheels, then torchvision/torchaudio with `--no-deps`

## Important Design Decisions

- **`garak_axis_ext` vs `garak/`**: The primary extension package is `garak_axis_ext` to avoid namespace collision with the installed garak package. The `garak/` directory is a legacy mirror.
- **Two forward passes**: Intentionally simple; a fused approach is possible but complicates generation.
- **Final token projection** in generator vs **mean pooling** in extractor — both valid; switch to final-token in extractor if results are noisy.
- **Deduplication**: `axis_capture.py` runs each unique prompt only once regardless of `generations_per_prompt`.
- **Garak as CLI only**: The pipeline uses garak purely via CLI and post-processes JSONL output, avoiding its unstable programmatic API.

## Git Conventions

- **Branch naming**: `claude/<feature-name>-<session-id>` for feature branches
- **Commit messages**: Imperative mood ("Add", "Fix", "Refactor"), concise title, detailed body with bullets for multi-part changes
- **Workflow**: Feature branches → PR → merge to main
