"""Resources to quantize SBERT models."""
import typing as t
import os
import datetime
import random
import shutil
import glob
import re
import functools

import optimum.onnxruntime
import optimum.onnxruntime.preprocessors.passes
import datasets
import transformers

from .. import utils


__all__ = [
    "quantize_as_onnx",
    "read_additional_submodules",
]


def read_additional_submodules(
    source_dir: str, submodule_pattern: str = r"[0-9]+_[A-Z][a-z]*"
) -> t.List[str]:
    """TODO"""
    source_dir = utils.expand_path(source_dir)
    submodules = glob.glob(os.path.join(source_dir, "*"))
    submodules = [
        submodule
        for submodule in submodules
        if re.match(submodule_pattern, os.path.basename(submodule))
    ]
    submodules = sorted(
        submodules, key=lambda submodule: int(os.path.basename(submodule).split("_")[0])
    )
    return submodules


def copy_aditional_submodules(
    source_dir: str, target_dir: str, submodule_pattern: str = r"[0-9]+_[A-Z][a-z]*"
) -> None:
    """TODO"""
    submodules = read_additional_submodules(
        source_dir=source_dir, submodule_pattern=submodule_pattern
    )
    for submodule in submodules:
        submodule_name = os.path.basename(submodule)
        shutil.copytree(
            src=submodule, dst=os.path.join(target_dir, submodule_name), dirs_exist_ok=True
        )


def preprocess_function(examples, tokenizer, data_key: str = "text"):
    return tokenizer(examples[data_key], padding="max_length", max_length=128, truncation=True)


def to_onnx(
    model_uri: str,
    quantized_model_filename: t.Optional[str] = None,
    optimized_model_filename: t.Optional[str] = None,
    onnx_model_filename: t.Optional[str] = None,
    output_dir: str = "./quantized_models",
    device: str = "cpu",
    operators_to_quantize: t.Tuple[str, ...] = (
        "MatMul",
        "Attention",
        "LayerNormalization",
        "SkipLayerNormalization",
        "Mul",
        "Div",
        "Add",
        "Gather",
    ),
    check_cached: bool = True,
    static_quantization: bool = False,
    optimize_before_quantization: bool = True,
    keep_onnx_model: bool = False,
    verbose: bool = False,
) -> utils.QuantizationOutputONNX:
    """Quantize SBERT as ONNX format.

    Parameters
    ----------
    model_uri : str
        Sentence Transformer URI to be quantized.

    quantized_model_filename : str or None, default=None
        Output filename.

    optimized_model_filename : str or None, deault=None
        TODO

    onnx_model_filename : str or None, default=None
        Name to save intermediary model in ONNX format in `output_dir`. This
        transformation is necessary to perform all necessary optimization and quantization.
        If None, a name will be derived from `quantized_model_filename`.

    output_dir : str, default='./quantized_models'
        Path to output file directory, which the resulting quantized model will be stored,
        alongside any possible coproducts also generated during the quantization procedure.

    device : str, default='cpu'
        TODO

    check_cached : bool, default=True
        If True, check whether a model with the same model exists before quantization.
        If this happens to be the case, this function will not produce any new models.

    static_quantization : bool, default=False
        TODO

    optimize_before_quantization : bool, default=True
        TODO

    keep_onnx_model : bool, default=False
        TODO

    verbose : bool, default=False
        TODO

    Returns
    -------
    paths : t.Tuple[str, ...]
        File URIs related from generated files during the quantization procedure. The
        final model URI can be accessed from the `output_uri` attribute.
    """
    output_dir = utils.expand_path(output_dir)

    if not optimize_before_quantization:
        optimized_model_filename = None

    model_name = os.path.basename(model_uri)
    if not model_name:
        model_name = os.path.basename(os.path.dirname(model_uri))

    paths = utils.build_onnx_default_uris(
        model_name="sbert",
        model_attributes={"name": model_name},
        output_dir=output_dir,
        quantized_model_filename=quantized_model_filename,
        onnx_model_filename=onnx_model_filename,
    )

    onnx_base_uri = paths.onnx_base_uri.replace(".onnx", "_onnx")
    quantized_model_uri = paths.output_uri.replace(".onnx", "_onnx")

    paths = utils.QuantizationOutputONNX(
        onnx_base_uri=(onnx_base_uri if keep_onnx_model else quantized_model_uri),
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

    ooq = optimum.onnxruntime.quantization

    quantization_config = optimum.onnxruntime.configuration.QuantizationConfig(
        is_static=static_quantization,
        format=ooq.QuantFormat.QDQ if static_quantization else ooq.QuantFormat.QOperator,
        mode=ooq.QuantizationMode.QLinearOps
        if static_quantization
        else ooq.QuantizationMode.IntegerOps,
        activations_dtype=ooq.QuantType.QInt8 if static_quantization else ooq.QuantType.QUInt8,
        weights_dtype=ooq.QuantType.QInt8,
        per_channel=not static_quantization,
        operators_to_quantize=list(operators_to_quantize),
    )
    quant_ranges = None
    quant_preprocessor = None
    onnx_augmented_model_name = None

    ort_model = optimum.onnxruntime.ORTModelForFeatureExtraction.from_pretrained(
        model_uri,
        from_transformers=True,
        local_files_only=True,
    )

    if keep_onnx_model:
        ort_model.save_pretrained(paths.onnx_base_uri)
        copy_aditional_submodules(source_dir=model_uri, target_dir=paths.onnx_base_uri)

    if optimize_before_quantization:
        optimizer = optimum.onnxruntime.ORTOptimizer.from_pretrained(ort_model)

        temp_optimized_model_uri = "_".join(
            [
                "temp_optimized_sbert",
                datetime.datetime.utcnow().strftime("%Y_%m_%d__%H_%M_%S"),
                hex(random.getrandbits(128))[2:],
            ]
        )

        if optimized_model_filename:
            optimized_model_filename = os.path.join(output_dir, optimized_model_filename)

        temp_optimized_model_uri = os.path.join(output_dir, temp_optimized_model_uri)

        optimizer.optimize(
            save_dir=optimized_model_filename or temp_optimized_model_uri,
            file_suffix="",
            optimization_config=optimization_config,
        )

        if optimized_model_filename:
            copy_aditional_submodules(source_dir=model_uri, target_dir=optimized_model_filename)

    else:
        temp_optimized_model_uri = model_uri

    try:
        ort_model = optimum.onnxruntime.ORTModelForFeatureExtraction.from_pretrained(
            optimized_model_filename or temp_optimized_model_uri,
            from_transformers=not optimize_before_quantization,
            local_files_only=True,
        )

        quantizer = optimum.onnxruntime.ORTQuantizer.from_pretrained(ort_model)

        if static_quantization:
            # Create the calibration dataset used for the calibration step
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_uri,
                local_files_only=True,
            )

            # TODO: remove dataset load from here.
            calibration_dataset_uri = (
                "/media/nvme/sentence-model/ulysses_tesemo_v2_subset_static_quantization"
            )
            calibration_dataset_uri = os.path.abspath(calibration_dataset_uri)

            calibration_dataset = datasets.load_from_disk(calibration_dataset_uri)
            calibration_dataset = calibration_dataset.map(
                functools.partial(preprocess_function, tokenizer=tokenizer),
                remove_columns="text",
            )

            calibration_config = (
                optimum.onnxruntime.configuration.AutoCalibrationConfig.percentiles(
                    calibration_dataset, percentile=99.999
                )
            )
            onnx_augmented_model_name = os.path.join(output_dir, "augmented_model.onnx")

            # Perform the calibration step: computes the activations quantization ranges
            quant_ranges = quantizer.fit(
                dataset=calibration_dataset,
                calibration_config=calibration_config,
                operators_to_quantize=quantization_config.operators_to_quantize,
                batch_size=8,
                onnx_augmented_model_name=onnx_augmented_model_name,
            )

            quant_preprocessor = optimum.onnxruntime.preprocessors.QuantizationPreprocessor()
            oopp = optimum.onnxruntime.preprocessors.passes
            quant_preprocessor.register_pass(oopp.ExcludeLayerNormNodes())
            quant_preprocessor.register_pass(oopp.ExcludeGeLUNodes())
            quant_preprocessor.register_pass(oopp.ExcludeNodeAfter("Add", "Add"))
            quant_preprocessor.register_pass(oopp.ExcludeNodeAfter("Gather", "Add"))
            quant_preprocessor.register_pass(oopp.ExcludeNodeFollowedBy("Add", "Softmax"))

        quantizer.quantize(
            save_dir=paths.onnx_quantized_uri,
            file_suffix="quantized",
            quantization_config=quantization_config,
            calibration_tensors_range=quant_ranges,
            preprocessor=quant_preprocessor,
        )

        copy_aditional_submodules(source_dir=model_uri, target_dir=paths.onnx_quantized_uri)

    finally:
        if (
            optimize_before_quantization
            and temp_optimized_model_uri != model_uri
            and os.path.exists(temp_optimized_model_uri)
        ):
            shutil.rmtree(temp_optimized_model_uri)

        if onnx_augmented_model_name is not None:
            os.remove(onnx_augmented_model_name)

    return paths
