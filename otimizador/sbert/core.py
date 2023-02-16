"""Resources to quantize SBERT models."""
import typing as t
import os
import datetime
import random
import shutil
import glob
import re

import optimum.onnxruntime

from .. import utils


__all__ = [
    "quantize_as_onnx",
]


def copy_aditional_submodules(source_dir: str, target_dir: str, submodule_pattern: str = r"[0-9]+_[A-Z][a-z]*") -> None:
    """TODO"""
    source_dir = utils.expand_path(source_dir)
    submodules = glob.glob(os.path.join(source_dir, "*"))
    submodules = [submodule for submodule in submodules if re.match(submodule_pattern, os.path.basename(submodule)))]
    shutfil.copytree(submodule, target_dir, dirs_exist_ok=True)


def quantize_as_onnx(
    model_uri: str,
    quantized_model_filename: t.Optional[str] = None,
    intermediary_onnx_model_name: t.Optional[str] = None,
    quantized_model_dirpath: str = "./quantized_models",
    device: str = "cpu",
    operators_to_quantize: t.Tuple[str, ...] = (
        "MatMul",
        "Add",
        "Gather",
        "EmbedLayerNormalization",
        "Attention",
    ),
    check_cached: bool = True,
    save_intermediary_onnx_model: bool = False,
    verbose: bool = False,
) -> utils.QuantizationOutputONNX:
    """Quantize SBERT as ONNX format.

    Parameters
    ----------
    model_uri : str
        Sentence Transformer URI to be quantized.

    quantized_model_filename : str or None, default=None
        Output filename.

    intermediary_onnx_model_name : str or None, default=None
        Name to save intermediary model in ONNX format in `quantized_model_dirpath`. This
        transformation is necessary to perform all necessary optimization and quantization.
        If None, a name will be derived from `quantized_model_filename`.

    quantized_model_dirpath : str, default='./quantized_models'
        Path to output file directory, which the resulting quantized model will be stored,
        alongside any possible coproducts also generated during the quantization procedure.

    device : str, default='cpu'
        TODO

    check_cached : bool, default=True
        If True, check whether a model with the same model exists before quantization.
        If this happens to be the case, this function will not produce any new models.

    save_intermediary_onnx_model : bool, default=False
        TODO

    verbose : bool, default=False
        TODO

    Returns
    -------
    paths : t.Tuple[str, ...]
        File URIs related from generated files during the quantization procedure. The
        final model URI can be accessed from the `output_uri` attribute.
    """
    quantized_model_dirpath = utils.expand_path(quantized_model_dirpath)

    model_name = os.path.basename(model_uri)
    if not model_name:
        model_name = os.path.basename(os.path.dirname(model_uri))

    paths = utils.build_onnx_default_uris(
        model_name="sbert",
        model_attributes={"name": model_name},
        quantized_model_dirpath=quantized_model_dirpath,
        quantized_model_filename=quantized_model_filename,
        intermediary_onnx_model_name=intermediary_onnx_model_name,
    )

    quantized_model_uri = paths.output_uri.replace(".onnx", "_onnx")

    paths = utils.QuantizationOutputONNX(
        onnx_base_uri=(
            paths.onnx_base_uri if save_intermediary_onnx_model else quantized_model_uri
        ),
        onnx_quantized_uri=quantized_model_uri,
        output_uri=quantized_model_uri,
    )

    if check_cached and os.path.exists(paths.onnx_quantized_uri):
        if verbose:  # pragma: no cover
            print(
                f"Found cached model in '{paths.onnx_quantized_uri}'.",
                "Skipping model quantization.",
            )

        return paths

    optimization_config = optimum.onnxruntime.OptimizationConfig(
        optimization_level=99,
        enable_transformers_specific_optimizations=True,
        disable_gelu_fusion=False,
        disable_embed_layer_norm_fusion=False,
        disable_attention_fusion=False,
        disable_skip_layer_norm_fusion=False,
        disable_bias_skip_layer_norm_fusion=False,
        disable_bias_gelu_fusion=False,
        enable_gelu_approximation=True,
        optimize_for_gpu=device,
    )

    quantization_config = optimum.onnxruntime.configuration.QuantizationConfig(
        is_static=False,
        format=optimum.onnxruntime.quantization.QuantFormat.QOperator,
        mode=optimum.onnxruntime.quantization.QuantizationMode.IntegerOps,
        activations_dtype=optimum.onnxruntime.quantization.QuantType.QUInt8,
        weights_dtype=optimum.onnxruntime.quantization.QuantType.QInt8,
        per_channel=True,
        operators_to_quantize=list(operators_to_quantize),
    )

    ort_model = optimum.onnxruntime.ORTModelForFeatureExtraction.from_pretrained(
        model_uri,
        from_transformers=True,
        local_files_only=True,
    )

    if save_intermediary_onnx_model:
        ort_model.save_pretrained(paths.onnx_base_uri)
        copy_aditional_submodules(source_dir=model_uri, target_dir=paths.onnx_base_uri)

    optimizer = optimum.onnxruntime.ORTOptimizer.from_pretrained(ort_model)

    temp_optimized_model_uri = "_".join(
        [
            "temp_optimized_sbert",
            datetime.datetime.utcnow().strftime("%Y_%m_%d__%H_%M_%S"),
            hex(random.getrandbits(128))[2:],
        ]
    )
    temp_optimized_model_uri = os.path.join(quantized_model_dirpath, temp_optimized_model_uri)

    optimizer.optimize(
        save_dir=temp_optimized_model_uri,
        file_suffix="",
        optimization_config=optimization_config,
    )

    try:
        ort_model = optimum.onnxruntime.ORTModelForFeatureExtraction.from_pretrained(
            temp_optimized_model_uri,
            local_files_only=True,
        )

        quantizer = optimum.onnxruntime.ORTQuantizer.from_pretrained(ort_model)

        quantizer.quantize(
            save_dir=paths.onnx_quantized_uri,
            file_suffix="quantized",
            quantization_config=quantization_config,
        )

        copy_aditional_submodules(source_dir=model_uri, target_dir=paths.onnx_quantized_uri)

    finally:
        shutil.rmtree(temp_optimized_model_uri)

    return paths
