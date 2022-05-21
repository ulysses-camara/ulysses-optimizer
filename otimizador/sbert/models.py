"""SBERT quantized model archirectures."""
from __future__ import annotations

import typing as t

import numpy as np
import numpy.typing as npt
import torch
import transformers

from .. import optional_import_utils


__all__ = [
    "ONNXSBERT",
]


class ONNXSBERT:
    """SBERT in ONNX format for inference in production.

    The ONNX format support faster inference, quantized and optimized models with
    hardware-specific instructions.

    Parameters
    ----------
    uri_model : str
        URI to load pretrained model from. If `local_files_only=True`, then it must
        be a local file.

    uri_tokenizer : str
        URI to pretrained text Tokenizer.

    local_files_only : bool, default=True
        If True, will search only for local pretrained model and tokenizers.
        If False, may download models from Huggingface HUB, if necessary.

    cache_dir_tokenizer : str, default='./cache/tokenizers'
        Cache directory for text tokenizer.
    """

    def __init__(
        self,
        uri_model: str,
        uri_tokenizer: str,
        local_files_only: bool = True,
        cache_dir_tokenizer: str = "./cache/tokenizers",
    ):
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            uri_tokenizer,
            local_files_only=local_files_only,
            cache_dir=cache_dir_tokenizer,
            use_fast=True,
        )

        optional_import_utils.load_required_module("onnxruntime")

        import onnxruntime  # pylint: disable='import-error'

        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )

        self._model: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            path_or_bytes=uri_model,
            sess_options=sess_options,
        )

    @property
    def model(self) -> onnxruntime.InferenceSession:  # type: ignore
        """Return ONNX SBERT model."""
        # pylint: disable='undefined-variable'
        return self._model

    def eval(self) -> "ONNXSBERT":
        """No-op method, created only to keep API consistent."""
        return self

    def train(self) -> "ONNXSBERT":
        """No-op method, created only to keep API consistent."""
        return self

    def encode(
        self,
        sequences: t.List[str],
        **kwargs: t.Any,
    ) -> npt.NDArray[np.float64]:
        """Predict a tokenized minibatch."""
        # pylint: disable='unused-argument'
        minibatch = self.tokenizer(
            sequences, return_tensors="np", truncation=True, padding="longest", max_length=512
        )

        minibatch = {key: np.atleast_2d(val) for key, val in minibatch.items()}

        model_out: t.List[npt.NDArray[np.float64]] = self._model.run(
            output_names=["sentence_embedding"],
            input_feed=minibatch,
            run_options=None,
        )

        logits: npt.NDArray[np.float64] = np.asfarray(model_out)

        if logits.ndim >= 3:
            logits = logits.squeeze(0)

        return logits

    def __call__(self, *args: t.Any, **kwargs: t.Any) -> npt.NDArray[np.float64]:
        return self.encode(*args, **kwargs)


class ONNXSBERTSurrogate(transformers.BertModel):
    """Create a temporary container class for SBERT + Average Pooler.

    This container is used only during the creation and quantization of ONNX files.
    This class should not be employed for production inference.
    """

    # pylint: disable='line-too-long,abstract-method,arguments-differ,no-member'
    # Adapted from: https://github.com/UKPLab/sentence-transformers/blob/cb08d92822ffcab9915564fd327e6579a5ed5830/examples/onnx_inference/onnx_inference.ipynb
    def __init__(self, config: transformers.BertConfig, *args: t.Any, **kwargs: t.Any):
        super().__init__(config, *args, **kwargs)
        self.sentence_embedding = torch.nn.Identity()

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor
    ) -> torch.Tensor:
        out: torch.Tensor = (
            super()  # type: ignore
            .forward(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            .last_hidden_state
        )

        # Average pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(out.size())
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        out = torch.sum(out * input_mask_expanded, 1)
        out = out / sum_mask
        out = self.sentence_embedding(out)

        return out
