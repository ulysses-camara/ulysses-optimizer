"""Resources to quantize SBERT models."""
import typing as t
import os
import warnings
import collections

import transformers
import sentence_transformers
import torch

from . import models
from .. import utils
from .. import optional_import_utils


__all__ = [
    "quantize_sbert_model_as_onnx",
]


def quantize_sbert_model_as_onnx(
    model: sentence_transformers.SentenceTransformer,
    quantized_model_filename: t.Optional[str] = None,
    intermediary_onnx_model_name: t.Optional[str] = None,
    intermediary_onnx_optimized_model_name: t.Optional[str] = None,
    quantized_model_dirpath: str = "./quantized_models",
    task_name: str = "unknown_task",
    optimization_level: int = 99,
    onnx_opset_version: int = 15,
    check_cached: bool = True,
    clean_intermediary_files: bool = True,
) -> utils.QuantizationOutputONNX:
    """Quantize SBERT as ONNX format.

    Parameters
    ----------
    model : sentence_transformers.SentenceTransformer
        Sentence Transformer Model to be quantized.

    quantized_model_filename : str or None, default=None
        Output filename. If None, a long and descriptive name will be derived from model's
        parameters.

    quantized_model_dirpath : str, default='./quantized_models'
        Path to output file directory, which the resulting quantized model will be stored,
        alongside any possible coproducts also generated during the quantization procedure.

    task_name : str, default='unknown_task'
        Model task name. Used to build quantized model filename, if not provided.

    intermediary_onnx_model_name : str or None, default=None
        Name to save intermediary model in ONNX format in `quantized_model_dirpath`. This
        transformation is necessary to perform all necessary optimization and quantization.
        If None, a name will be derived from `quantized_model_filename`.

    intermediary_onnx_optimized_model_name : str or None, default=None
        Name to save intermediary optimized model in ONNX format in `quantized_model_dirpath`.
        This transformation is necessary to perform quantization. If None, a name will be
        derived from `quantized_model_filename`.

    optimization_level : {0, 1, 2, 99}, default=99
        Optimization level for ONNX models. From the ONNX Runtime specification:
        - 0: disable all optimizations;
        - 1: enable only basic optimizations;
        - 2: enable basic and extended optimizations; or
        - 99: enable all optimizations (incluing layer and hardware-specific optimizations).
        See [1]_ for more information.

    onnx_opset_version: int, default=15
        ONNX operator set version. Used only if `model_output_format='onnx'`. Check [2]_ for
        more information.

    check_cached : bool, default=True
        If True, check whether a model with the same model exists before quantization.
        If this happens to be the case, this function will not produce any new models.

    clean_intermediary_files : bool, default=False
        If True, remove ONNX optimized and base models after building quantized model.

    Returns
    -------
    paths : t.Tuple[str, ...]
        File URIs related from generated files during the quantization procedure. The
        final model URI can be accessed from the `output_uri` attribute.

    References
    ----------
    .. [1] Graph Optimizations in ONNX Runtime. Available at:
       https://onnxruntime.ai/docs/performance/graph-optimizations.html

    .. [2] ONNX Operator Schemas. Available at:
       https://github.com/onnx/onnx/blob/main/docs/Operators.md
    """
    optional_import_utils.load_required_module("onnxruntime")

    import onnxruntime.quantization

    quantized_model_dirpath = utils.expand_path(quantized_model_dirpath)
    model_config: transformers.BertConfig = model.config  # type: ignore
    is_pruned = bool(model_config.pruned_heads)

    if is_pruned:
        raise RuntimeError("SBERT with pruned attention heads will not work in ONNX format.")

    model_attributes: t.Dict[str, t.Any] = collections.OrderedDict(
        (
            ("num_layers", model_config.num_hidden_layers),
            ("vocab_size", model.tokenizer.vocab_size),
            ("pruned", is_pruned),
        )
    )

    paths = utils.build_onnx_default_uris(
        task_name=task_name,
        model_name="sbert",
        model_attributes=model_attributes,
        quantized_model_dirpath=quantized_model_dirpath,
        quantized_model_filename=quantized_model_filename,
        intermediary_onnx_model_name=intermediary_onnx_model_name,
        intermediary_onnx_optimized_model_name=intermediary_onnx_optimized_model_name,
        optimization_level=optimization_level,
        include_config_uri=False,
    )

    if check_cached and os.path.isfile(paths.onnx_quantized_uri):
        return paths

    config_bert = model.get_submodule("0.auto_model").config

    pytorch_module = models.ONNXSBERTSurrogate(config=config_bert)
    pytorch_module.load_state_dict(model.get_submodule("0.auto_model").state_dict())
    pytorch_module.eval()

    if not check_cached or not os.path.isfile(paths.onnx_base_uri):
        torch_sample_input: t.Tuple[torch.Tensor, ...] = utils.gen_dummy_inputs_for_tracing(
            batch_size=1,
            vocab_size=model_config.vocab_size,
            seq_length=256,
        )
        torch_sample_input = tuple(item.to(model.device) for item in torch_sample_input)

        torch.onnx.export(
            model=pytorch_module,
            args=torch_sample_input,
            f=paths.onnx_base_uri,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["sentence_embedding"],
            opset_version=onnx_opset_version,
            export_params=True,
            do_constant_folding=True,
            dynamic_axes=dict(
                input_ids={0: "batch_axis", 1: "sentence_length"},
                attention_mask={0: "batch_axis", 1: "sentence_length"},
                token_type_ids={0: "batch_axis", 1: "sentence_length"},
                sentence_embedding={0: "batch_axis"},
            ),
        )

    onnxruntime.quantization.quantize_dynamic(
        model_input=paths.onnx_base_uri,
        model_output=paths.onnx_quantized_uri,
        weight_type=onnxruntime.quantization.QuantType.QUInt8,
        optimize_model=True,
        per_channel=False,
        extra_options=dict(
            EnableSubgraph=True,
            MatMulConstBOnly=False,
            ForceQuantizeNoInputCheck=True,
        ),
    )

    os.rename(f"{paths.onnx_base_uri}-opt.onnx", paths.onnx_optimized_uri)

    if clean_intermediary_files:
        try:
            os.remove(paths.onnx_base_uri)

        except (FileNotFoundError, OSError):
            warnings.warn(
                message=(
                    "Could not delete base ONNX model. There will be a residual file in "
                    f"'{quantized_model_dirpath}' directory."
                ),
                category=RuntimeWarning,
            )

        try:
            os.remove(paths.onnx_optimized_uri)

        except (FileNotFoundError, OSError):
            warnings.warn(
                message=(
                    "Could not delete intermediary optimized ONNX model. There will be a residual "
                    f"file in '{quantized_model_dirpath}' directory."
                ),
                category=RuntimeWarning,
            )

    return paths
