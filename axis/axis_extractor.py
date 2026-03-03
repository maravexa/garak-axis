"""
axis_extractor.py
-----------------
Offline, per-model utility to derive the Assistant Axis direction vector
via PCA over residual stream activations for a configurable persona corpus.

Run once per model; saves a .pt checkpoint consumed by AxisAwareGenerator.

Usage:
    python axis/axis_extractor.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --layer 22 \
        --corpus personas/corpus.json \
        --output axis/vectors/llama3_8b_layer22.pt
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus(path: str) -> list:
    """Load persona prompts from an external JSON config file.

    Expected format::

        {
          "meta": { ... },
          "personas": ["You are ...", "You are ...", ...]
        }
    """
    corpus_path = Path(path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Persona corpus not found: {path}")

    with corpus_path.open() as f:
        data = json.load(f)

    if "personas" not in data:
        raise ValueError(f"Corpus file must contain a 'personas' key: {path}")

    personas = data["personas"]
    if len(personas) < 20:
        log.warning(
            "Corpus has only %d entries; recommend 200+ for reliable axis extraction.",
            len(personas),
        )

    log.info("Loaded %d persona prompts from %s", len(personas), path)
    return personas


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def get_activation(model, tokenizer, prompt, layer, device):
    """Return mean residual stream activation at *layer* for *prompt*.

    Returns shape: [hidden_dim]
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=False,
    ).to(device)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    # hidden_states: tuple of (n_layers + 1) tensors, each [batch, seq, dim]
    hidden = outputs.hidden_states[layer]          # [1, seq_len, hidden_dim]
    return hidden[0].mean(dim=0).cpu().float().numpy()   # [hidden_dim]


# ---------------------------------------------------------------------------
# Axis extraction
# ---------------------------------------------------------------------------

def extract_axis(model_name, target_layer, corpus_path, output_path,
                 device="auto", n_components=10):
    """Derive the Assistant Axis via PCA over persona activation vectors.

    Saves a .pt checkpoint with keys:
        axis          - unit vector [hidden_dim]
        layer         - target layer index
        model         - model name string
        explained_var - fraction of variance explained by PC1
        corpus_path   - path to corpus used

    Returns the axis vector as a numpy array.
    """
    personas = load_corpus(corpus_path)

    log.info("Loading tokenizer and model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    if target_layer > n_layers:
        raise ValueError(
            f"target_layer={target_layer} exceeds model depth ({n_layers} layers). "
            f"Recommended range: {int(n_layers * 0.6)}-{int(n_layers * 0.75)}."
        )

    log.info("Extracting activations at layer %d / %d ...", target_layer, n_layers)
    activations = []
    for i, prompt in enumerate(personas):
        act = get_activation(model, tokenizer, prompt, target_layer, device)
        activations.append(act)
        if (i + 1) % 20 == 0:
            log.info("  %d / %d done", i + 1, len(personas))

    activation_matrix = np.array(activations)   # [n_personas, hidden_dim]
    log.info("Activation matrix shape: %s", activation_matrix.shape)

    n_components = min(n_components, len(personas))
    pca = PCA(n_components=n_components)
    pca.fit(activation_matrix)
    axis_vector = pca.components_[0]   # [hidden_dim], unit vector

    explained = pca.explained_variance_ratio_[0]
    log.info("PC1 explains %.1f%% of variance", explained * 100)
    if explained < 0.25:
        log.warning(
            "PC1 variance (%.1f%%) is below the recommended 25%% threshold. "
            "Consider expanding the persona corpus for a more reliable axis.",
            explained * 100,
        )

    # Ensure Assistant-like prompts project to the POSITIVE end
    assistant_act = get_activation(model, tokenizer, personas[0], target_layer, device)
    if np.dot(assistant_act, axis_vector) < 0:
        log.info("Flipping axis orientation so Assistant end is positive.")
        axis_vector = -axis_vector

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "axis": torch.tensor(axis_vector, dtype=torch.float32),
            "layer": target_layer,
            "model": model_name,
            "explained_var": float(explained),
            "corpus_path": str(corpus_path),
        },
        out,
    )
    log.info("Axis checkpoint saved to %s", output_path)
    return axis_vector


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Extract the Assistant Axis for a given model."
    )
    p.add_argument("--model",        required=True,
                   help="HuggingFace model name or local path")
    p.add_argument("--layer",        required=True, type=int,
                   help="Residual stream layer index to probe")
    p.add_argument("--corpus",       default="personas/corpus.json",
                   help="Path to persona corpus JSON (default: personas/corpus.json)")
    p.add_argument("--output",       default="axis/vectors/axis.pt",
                   help="Output path for axis checkpoint")
    p.add_argument("--device",       default="auto",
                   help="Device map (auto, cuda, cpu)")
    p.add_argument("--n-components", default=10, type=int,
                   help="Number of PCA components to compute")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    extract_axis(
        model_name=args.model,
        target_layer=args.layer,
        corpus_path=args.corpus,
        output_path=args.output,
        device=args.device,
        n_components=args.n_components,
    )
