"""SBERT quantized model archirectures."""
from __future__ import annotations

import typing as t
import os

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional
import transformers
import sentence_transformers
import optimum.onnxruntime
import tqdm.auto

from . import core


__all__ = [
    "ONNXSBERT",
]


class _SentenceEmbeddingPipeline(transformers.Pipeline):
    # pylint: disable='unused-argument'
    def __init__(
        self, *args: t.Any, postprocessing_layers: t.List[torch.nn.Module], **kwargs: t.Any
    ):
        super().__init__(*args, **kwargs)
        self.postprocessing_layers = postprocessing_layers

    def _sanitize_parameters(self, **kwargs: t.Any) -> t.Tuple[t.Dict[str, t.Any], ...]:
        preprocess_kwargs = {
            "max_length": int(kwargs.get("max_length", 512)),
        }
        return preprocess_kwargs, {}, {}

    def preprocess(
        self, input_: t.List[str], max_length: int = 512, **kwargs: t.Any
    ) -> t.Dict[str, torch.Tensor]:
        return self.tokenizer(  # type: ignore
            input_, padding="longest", truncation=True, return_tensors="pt", max_length=max_length
        )

    def _forward(
        self, input_tensors: t.Dict[str, torch.Tensor], **kwargs: t.Any
    ) -> t.Dict[str, torch.Tensor]:
        outputs = self.model(**input_tensors)
        return {"token_embeddings": outputs[0], "attention_mask": input_tensors["attention_mask"]}

    def postprocess(
        self, model_outputs: t.Dict[str, torch.Tensor], **kwargs: t.Any
    ) -> torch.Tensor:
        out = model_outputs

        for layer in self.postprocessing_layers:
            out = layer(out)

        return out["sentence_embedding"]


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
        uri_tokenizer: t.Optional[str] = None,
        local_files_only: bool = True,
        cache_dir_tokenizer: str = "./cache/tokenizers",
    ):
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            uri_tokenizer or uri_model,
            local_files_only=local_files_only,
            cache_dir=cache_dir_tokenizer,
            use_fast=True,
        )

        self._model = optimum.onnxruntime.ORTModelForFeatureExtraction.from_pretrained(
            uri_model,
            from_transformers=False,
            local_files_only=True,
        )

        self._emb_dim = 768

        if hasattr(self._model.config, "pooler_fc_size"):
            self._emb_dim = int(self._model.config.pooler_fc_size)

        elif hasattr(self._model.config, "hidden_size"):
            self._emb_dim = int(self._model.config.hidden_size)

        self._pipeline = _SentenceEmbeddingPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            postprocessing_layers=self._read_postprocessing_modules(uri_model),
        )

    @staticmethod
    def _read_postprocessing_modules(base_dir: str) -> t.List[torch.nn.Module]:
        submodules = core.read_additional_submodules(source_dir=base_dir)
        postprocessing_modules: t.List[torch.nn.Module] = []

        for submodule in submodules:
            submodule_type = os.path.basename(submodule).split("_")[-1]
            new_submodule = getattr(sentence_transformers.models, submodule_type).load(submodule)
            postprocessing_modules.append(new_submodule)

        return postprocessing_modules

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
        batch_size: int = 8,
        show_progress_bar: bool = True,
        normalize_embeddings: bool = False,
        **kwargs: t.Any,
    ) -> npt.NDArray[np.float64]:
        """Predict a tokenized minibatch."""
        # pylint: disable='unused-argument,invalid-name'
        n = len(sequences)
        logits = np.empty((n, self._emb_dim), dtype=float)

        with torch.no_grad():
            for i_start in tqdm.auto.tqdm(range(0, n, batch_size), disable=not show_progress_bar):
                i_end = i_start + batch_size
                batch = sequences[i_start:i_end]
                out = self._pipeline(batch)
                logits[i_start:i_end, :] = np.vstack([inst.to("cpu").numpy() for inst in out])

        if logits.size and normalize_embeddings:
            logits /= 1e-12 + np.linalg.norm(logits, axis=-1, ord=2, keepdims=True)

        return logits

    def __call__(self, *args: t.Any, **kwargs: t.Any) -> npt.NDArray[np.float64]:
        return self.encode(*args, **kwargs)
