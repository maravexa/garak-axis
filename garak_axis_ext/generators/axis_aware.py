"""
axis_aware.py
-------------
Garak generator subclass that intercepts HuggingFace forward passes to
project residual stream activations onto the pre-computed Assistant Axis.

Drop-in replacement for a HuggingFace-backed Garak generator. All existing
Garak probes and detectors work unchanged. Axis displacement data is
side-channelled into an AxisStore instance for consumption by AxisMonitor.

Usage in a Garak run script:
    from garak.generators.axis_aware import AxisAwareGenerator
    generator = AxisAwareGenerator(
        name="meta-llama/Llama-3.1-8B-Instruct",
        axis_path="axis/vectors/llama3_8b_layer22.pt",
    )
"""

import logging
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .axis_store import AxisStore

log = logging.getLogger(__name__)


class AxisAwareGenerator:
    """HuggingFace generator with Assistant Axis activation monitoring.

    Mirrors the interface expected by Garak probes::

        outputs = generator.generate(prompt)  # -> List[str]

    Axis displacement scalars are written to self.store after each call
    and consumed by AxisMonitor at the end of each probe attempt.
    """

    DEFAULT_MAX_NEW_TOKENS = 256
    DEFAULT_MAX_INPUT_TOKENS = 512

    def __init__(
        self,
        name: str,
        axis_path: str,
        device: str = "auto",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
    ):
        self.name = name
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.store = AxisStore()

        log.info("Loading axis checkpoint from %s", axis_path)
        checkpoint = torch.load(axis_path, map_location="cpu", weights_only=True)
        self.axis_vector: torch.Tensor = checkpoint["axis"].float()  # [hidden_dim]
        self.axis_layer: int = checkpoint["layer"]
        self._checkpoint_meta = {k: v for k, v in checkpoint.items() if k != "axis"}
        log.info(
            "Axis loaded | layer=%d | PC1 explained_var=%.1f%%",
            self.axis_layer,
            checkpoint.get("explained_var", float("nan")) * 100,
        )

        log.info("Loading tokenizer and model: %s", name)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(
            name,
            dtype=torch.float16,
            device_map=device,
        )
        self.model.eval()

        # Resolve device for tensor ops after model placement
        self._device = next(self.model.parameters()).device
        self.axis_vector = self.axis_vector.to(self._device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward_with_hook(self, input_ids: torch.Tensor) -> float:
        """Run a single forward pass and return axis displacement scalar.

        Uses output_hidden_states=True. Separate from generation to keep
        the generation loop clean and avoid any interference with sampling.
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )
        # hidden_states: tuple of (n_layers + 1) tensors, each [batch, seq, dim]
        hidden = outputs.hidden_states[self.axis_layer]  # [1, seq_len, hidden_dim]
        last_tok = hidden[0, -1, :].float()              # final token [hidden_dim]
        return torch.dot(last_tok, self.axis_vector).item()

    def _generate_text(self, input_ids: torch.Tensor) -> str:
        """Run the generation loop and decode the new tokens."""
        with torch.no_grad():
            gen_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = gen_ids[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Public interface (Garak-compatible)
    # ------------------------------------------------------------------

    def generate(self, prompt: str, generations_this_call: int = 1) -> List[str]:
        """Generate responses for *prompt*, recording axis displacement.

        Args:
            prompt: Input text. Garak passes a plain string.
            generations_this_call: Number of responses. AxisAwareGenerator
                returns one response per call for clean per-turn tracking.

        Returns:
            List of generated strings.
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self._device)

        input_ids = inputs["input_ids"]

        # --- Side-channel: capture axis displacement for this prompt ---
        displacement = self._forward_with_hook(input_ids)
        self.store.record(displacement)
        log.debug("Turn displacement: %.4f (store depth=%d)", displacement, self.store.depth())
        # ---------------------------------------------------------------

        return [self._generate_text(input_ids) for _ in range(generations_this_call)]

    def axis_info(self) -> dict:
        """Return metadata from the loaded axis checkpoint."""
        return dict(self._checkpoint_meta)
