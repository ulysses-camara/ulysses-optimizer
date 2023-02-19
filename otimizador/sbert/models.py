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
        self, *args: t.Any, postprocessing_modules: t.List[torch.nn.Module], **kwargs: t.Any
    ):
        super().__init__(*args, **kwargs)
        self.postprocessing_modules = postprocessing_modules

    def __len__(self) -> int:
        return 1 + len(self.postprocessing_modules)

    def _sanitize_parameters(self, **kwargs: t.Any) -> t.Tuple[t.Dict[str, t.Any], ...]:
        preprocess_kwargs = {
            "max_length": int(kwargs.get("max_length", 512)),
        }
        postprocess_kwargs = {
            "output_value": kwargs.get("output_value", "sentence_embedding"),
        }
        return preprocess_kwargs, {}, postprocess_kwargs

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
        self,
        model_outputs: t.Dict[str, torch.Tensor],
        output_value: str = "sentence_embedding",
        **kwargs: t.Any,
    ) -> torch.Tensor:
        out = model_outputs

        for layer in self.postprocessing_modules:
            out = layer(out)

        return out[output_value]


class ONNXSBERT:
    """SBERT in ONNX format for inference in production.

    The ONNX format support faster inference, quantized and optimized models with
    hardware-specific instructions.

    Parameters
    ----------
    uri_model : str
        URI to load pretrained model from. If `local_files_only=True`, then it must
        be a local file.

    uri_tokenizer : str or None, default=None
        URI to pretrained text Tokenizer.
        If None, will load tokenizer from `uri_model`.

    local_files_only : bool, default=True
        If True, will search only for local pretrained model and tokenizers.
        If False, may download models from Huggingface HUB, if necessary.

    cache_dir : str, default='./cache'
        Cache directory.
    """

    def __init__(
        self,
        uri_model: str,
        uri_tokenizer: t.Optional[str] = None,
        local_files_only: bool = True,
        cache_dir: str = "./cache",
    ):
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            uri_tokenizer or uri_model,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            use_fast=True,
        )

        self._model = optimum.onnxruntime.ORTModelForFeatureExtraction.from_pretrained(
            uri_model,
            from_transformers=False,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )

        self._emb_dim = 768

        if hasattr(self._model.config, "pooler_fc_size"):
            self._emb_dim = int(self._model.config.pooler_fc_size)

        elif hasattr(self._model.config, "hidden_size"):
            self._emb_dim = int(self._model.config.hidden_size)

        self._pipeline = _SentenceEmbeddingPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            postprocessing_modules=self._read_postprocessing_modules(uri_model),
        )

    def __len__(self) -> int:
        return len(self._pipeline)

    def __str__(self) -> str:
        parts: t.List[str] = [f"{self.__class__.__name__}("]
        parts.append(f"  (0): {str(self._model)}")
        for i, layer in enumerate(self._pipeline.postprocessing_modules, 1):
            parts.append(f"  ({i}): {layer}")
        parts.append(")")
        return "\n".join(parts)

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
    def model(self) -> optimum.onnxruntime.ORTModelForFeatureExtraction:
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
        sentences: t.Union[str, t.Sequence[str]],
        batch_size: int = 8,
        show_progress_bar: bool = True,
        output_value: str = "sentence_embedding",
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
        **kwargs: t.Any,
    ) -> t.Union[torch.Tensor, npt.NDArray[np.float64]]:
        """Predict a tokenized minibatch."""
        # pylint: disable='unused-argument,invalid-name'
        input_is_single_string = False

        if isinstance(sentences, str):
            sentences = [sentences]
            input_is_single_string = True

        if not isinstance(sentences, list):
            sentences = list(sentences)

        n = len(sentences)
        embeddings = torch.empty(n, self._emb_dim, requires_grad=False)
        length_sorted_idx = np.argsort([-len(item) for item in sentences])
        pbar = tqdm.auto.tqdm(
            range(0, n, batch_size), desc="Batches", disable=not show_progress_bar
        )

        with torch.no_grad():
            for i_start in pbar:
                i_end = i_start + batch_size
                batch = [str(sentences[k]) for k in length_sorted_idx[i_start:i_end]]
                batch_out = self._pipeline(batch, batch_size=len(batch), output_value=output_value)
                embeddings[i_start:i_end, :] = torch.vstack(batch_out)

        embeddings = embeddings.to("cpu")
        embeddings = embeddings[np.argsort(length_sorted_idx), :]

        if embeddings.numel() and normalize_embeddings:
            torch.nn.functional.normalize(embeddings, p=2, dim=-1, out=embeddings)

        out: t.Union[torch.Tensor, npt.NDArray[np.float64]] = embeddings

        if convert_to_numpy:
            out = np.asfarray(embeddings.numpy())

        if input_is_single_string:
            out = embeddings[0, :]

        return out

    def __call__(
        self, *args: t.Any, **kwargs: t.Any
    ) -> t.Union[torch.Tensor, npt.NDArray[np.float64]]:
        return self.encode(*args, **kwargs)
