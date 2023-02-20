"""Resources to quantize SBERT models."""
import typing as t
import os
import shutil
import glob
import re
import functools

import optimum.onnxruntime
import optimum.onnxruntime.preprocessors.passes as oopp
import datasets
import transformers

from .. import utils


__all__ = [
    "to_onnx",
    "read_additional_submodules",
]


def read_additional_submodules(
    source_dir: str, submodule_pattern: str = r"[0-9]+_[A-Z][a-z]*"
) -> t.List[str]:
    """Read SentenceTransformer additional submodules from disk.

    SentenceTransformer submodules are stored as subdirectories named in the format
    `ID_SubmoduleType`, with a config.json file within.

    The argument `submodule_pattern` is a regular expression that matches the pattern
    described above.
    """
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
    """Copy SentenceTransformer submodules from `source_dir` to `target_dir`.

    SentenceTransformer submodules are stored as subdirectories named in the format
    `ID_SubmoduleType`, with a config.json file within.

    The argument `submodule_pattern` is a regular expression that matches the pattern
    described above.
    """
    submodules = read_additional_submodules(
        source_dir=source_dir, submodule_pattern=submodule_pattern
    )
    for submodule in submodules:
        submodule_name = os.path.basename(submodule)
        shutil.copytree(
            src=submodule, dst=os.path.join(target_dir, submodule_name), dirs_exist_ok=True
        )


def preprocess_function(
    examples: t.Dict[str, t.List[str]],
    tokenizer: transformers.AutoTokenizer,
    content_column: str = "text",
) -> t.Dict[str, t.List[t.Any]]:
    """Preprocess calibration dataset for static quantization."""
    return tokenizer(  # type: ignore
        examples[content_column], padding="max_length", max_length=128, truncation=True
    )


def to_onnx(
    model_uri: str,
    output_dir: str = "./quantized_models",
    onnx_model_filename: t.Optional[str] = None,
    optimized_model_filename: t.Optional[str] = None,
    quantized_model_filename: t.Optional[str] = None,
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
    static_quantization_dataset_uri: t.Optional[str] = None,
    content_column: str = "text",
    optimize_before_quantization: bool = True,
    optimization_level: int = 99,
    quant_per_channel: bool = False,
    quant_reduce_range: bool = True,
    keep_onnx_model: bool = False,
    verbose: bool = False,
) -> utils.QuantizationOutputONNX:
    """Quantize SBERT as ONNX format.

    Parameters
    ----------
    model_uri : str
        Sentence Transformer URI to be quantized.

    output_dir : str, default='./quantized_models'
        Path to output file directory, which the resulting quantized model will be stored,
        alongside any possible coproducts also generated during the quantization procedure.

    onnx_model_filename : str or None, default=None
        Filename of base model in ONNX format. This preprocessing is necessary for optimization and
        quantization.

    optimized_model_filename : str or None, deault=None
        Optimized model filename. If None, a random temporary name will be created, and the
        optimized model is removed after the creation of the quantized model.

    quantized_model_filename : str or None, default=None
        Quantized model filename.

    device : {'cpu', 'cuda'}, default='cpu'
        Device for which the model is to be optimized.

    check_cached : bool, default=True
        If True, check whether a model with the same model exists before quantization.
        If this happens to be the case, this function will not produce any new models.

    static_quantization_dataset_uri : str or None, default=None
        Path to dataset for quantized parameter range calibration.
        If provided, will perform static quantization.
        If None, will perform dynamic quantization.

    content_column : str, default='text'
        Column name from `static_quantization_dataset_uri` where sentence textual contents are
        kept.

    optimize_before_quantization : bool, default=True
        If True, optimize model before quantization.

    optimization_level : int, default=99
        Optimization level to use when `optimize_before_quantization=True`. Check `optimum`
        documentation for more information.

    quant_per_channel : bool, default=False
        If True, quantize model parameters separated by channel. The quantized model may perform
        statistically better at the expense of being computationally heavier.
        Available only for dynamic quantization. If `static_quantization_dataset_uri` is provided,
        this parameter is automatically set to False.

    quant_reduce_range : bool, default=True
        If True, quantize using 7-bits instead of 8-bits.
        If False, dynamic quantization may perform very poorly.

    keep_onnx_model : bool, default=False
        If True, keep ONNX base model (saved as `onnx_model_filename`).
        If False, remove this model after quantization.

    verbose : bool, default=False
        If True, enable print messages.

    Returns
    -------
    paths : t.Tuple[str, ...]
        File URIs related from generated files during the quantization procedure. The final model
        URI can be accessed from the `output_uri` attribute.
    """
    output_dir = utils.expand_path(output_dir)
    model_uri = utils.expand_path(model_uri)

    if not optimize_before_quantization:
        optimized_model_filename = None

    model_name = os.path.basename(model_uri)
    if not model_name:
        model_name = os.path.basename(os.path.dirname(model_uri))

    is_static_quant = False

    if static_quantization_dataset_uri:
        static_quantization_dataset_uri = utils.expand_path(static_quantization_dataset_uri)
        is_static_quant = True

    paths = utils.build_onnx_default_uris(
        model_name="sbert",
        model_attributes={"name": model_name},
        output_dir=output_dir,
        quantized_model_filename=quantized_model_filename,
        onnx_model_filename=onnx_model_filename,
    )

    paths = utils.QuantizationOutputONNX(
        onnx_base_uri=paths.onnx_base_uri if keep_onnx_model else paths.onnx_quantized_uri,
        onnx_quantized_uri=paths.onnx_quantized_uri,
        output_uri=paths.output_uri,
    )

    if check_cached and os.path.exists(paths.onnx_quantized_uri):
        if verbose:  # pragma: no cover
            print(
                f"Found cached model in '{paths.onnx_quantized_uri}'.",
                "Skipping model quantization.",
            )

        return paths

    optimization_config = optimum.onnxruntime.OptimizationConfig(
        optimization_level=optimization_level,
        enable_transformers_specific_optimizations=True,
        disable_gelu_fusion=False,
        disable_embed_layer_norm_fusion=False,
        disable_attention_fusion=False,
        disable_skip_layer_norm_fusion=False,
        disable_bias_skip_layer_norm_fusion=False,
        disable_bias_gelu_fusion=False,
        enable_gelu_approximation=True,
        optimize_for_gpu=device.strip().lower() == "cuda",
    )

    ooq = optimum.onnxruntime.quantization

    quantization_config = optimum.onnxruntime.configuration.QuantizationConfig(
        is_static=is_static_quant,
        format=ooq.QuantFormat.QDQ if is_static_quant else ooq.QuantFormat.QOperator,
        mode=(
            ooq.QuantizationMode.QLinearOps if is_static_quant else ooq.QuantizationMode.IntegerOps
        ),
        activations_dtype=ooq.QuantType.QInt8 if is_static_quant else ooq.QuantType.QUInt8,
        weights_dtype=ooq.QuantType.QInt8,
        per_channel=quant_per_channel and not is_static_quant,
        reduce_range=quant_reduce_range,
        operators_to_quantize=list(operators_to_quantize),
    )
    quant_ranges = None
    quant_preprocessor = None
    onnx_augmented_model_name: t.Optional[str] = None

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

        temp_optimized_model_uri = utils.build_random_model_name(base_name="temp_optimized_sbert")

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

        if is_static_quant and static_quantization_dataset_uri:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_uri,
                local_files_only=True,
            )

            calibration_dataset = datasets.load_from_disk(static_quantization_dataset_uri)
            calibration_dataset = calibration_dataset.map(
                functools.partial(
                    preprocess_function, tokenizer=tokenizer, content_column=content_column
                ),
                remove_columns=content_column,
            )

            calibration_config = (
                optimum.onnxruntime.configuration.AutoCalibrationConfig.percentiles(
                    calibration_dataset, percentile=99.995
                )
            )

            onnx_augmented_model_name = utils.build_random_model_name("augmented_model.onnx")
            onnx_augmented_model_name = os.path.join(output_dir, onnx_augmented_model_name)

            quant_ranges = quantizer.fit(
                dataset=calibration_dataset,
                calibration_config=calibration_config,
                operators_to_quantize=quantization_config.operators_to_quantize,
                batch_size=8,
                onnx_augmented_model_name=onnx_augmented_model_name,
            )

            quant_preprocessor = optimum.onnxruntime.preprocessors.QuantizationPreprocessor()
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
