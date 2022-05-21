"""General purpose, algorithm agnostic utility functions."""
import typing as t
import os

import torch


class QuantizationOutputONNX(t.NamedTuple):
    """Output paths for quantization as ONNX format."""

    output_uri: str
    onnx_quantized_uri: str
    onnx_base_uri: str
    onnx_optimized_uri: str
    onnx_config_uri: t.Optional[str] = None


class QuantizationOutputTorch(t.NamedTuple):
    """Quantization output paths as Torch format."""

    output_uri: str


QuantizationOutput = t.Union[QuantizationOutputONNX, QuantizationOutputTorch]


def build_onnx_default_uris(
    task_name: str,
    model_name: str,
    model_attributes: t.Dict[str, t.Any],
    quantized_model_dirpath: str,
    quantized_model_filename: t.Optional[str] = None,
    intermediary_onnx_model_name: t.Optional[str] = None,
    intermediary_onnx_optimized_model_name: t.Optional[str] = None,
    onnx_config_name: t.Optional[str] = None,
    optimization_level: t.Union[str, int] = 99,
    include_config_uri: bool = True,
) -> QuantizationOutputONNX:
    """Build default URIs for quantized output in ONNX format."""
    if not intermediary_onnx_model_name:
        attrs_to_name = "_".join("_".join(map(str, item)) for item in model_attributes.items())
        intermediary_onnx_model_name = f"{task_name}_{attrs_to_name}_{model_name}_model"

    if not intermediary_onnx_optimized_model_name:
        intermediary_onnx_optimized_model_name = (
            f"{intermediary_onnx_model_name}_{optimization_level}_opt_level"
        )

    if not quantized_model_filename:
        quantized_model_filename = f"q_{intermediary_onnx_optimized_model_name}"

    onnx_config_name = onnx_config_name or f"{quantized_model_filename}.config"

    if not intermediary_onnx_model_name.endswith(".onnx"):
        intermediary_onnx_model_name += ".onnx"

    if not intermediary_onnx_optimized_model_name.endswith(".onnx"):
        intermediary_onnx_optimized_model_name += ".onnx"

    if not quantized_model_filename.endswith(".onnx"):
        quantized_model_filename += ".onnx"

    os.makedirs(quantized_model_dirpath, exist_ok=True)

    onnx_base_uri = os.path.join(quantized_model_dirpath, intermediary_onnx_model_name)
    onnx_optimized_uri = os.path.join(
        quantized_model_dirpath, intermediary_onnx_optimized_model_name
    )
    onnx_quantized_uri = os.path.join(quantized_model_dirpath, quantized_model_filename)

    paths_dict: t.Dict[str, str] = dict(
        onnx_base_uri=onnx_base_uri,
        onnx_optimized_uri=onnx_optimized_uri,
        onnx_quantized_uri=onnx_quantized_uri,
        output_uri=onnx_quantized_uri,
    )

    if include_config_uri:
        onnx_config_uri = os.path.join(quantized_model_dirpath, onnx_config_name)
        paths_dict["onnx_config_uri"] = onnx_config_uri

    paths = QuantizationOutputONNX(**paths_dict)

    all_path_set = {paths.onnx_base_uri, paths.onnx_optimized_uri, paths.onnx_quantized_uri}
    num_distinct_paths = len(all_path_set)

    if num_distinct_paths < 3:
        raise ValueError(
            f"{3 - num_distinct_paths} URI for ONNX models (including intermediary models) "
            "are the same, which will cause undefined behaviour while quantizing the model. "
            "Please provide distinct filenames for ONNX files."
        )

    return paths


def build_torch_default_uris(
    task_name: str,
    model_name: str,
    model_attributes: t.Dict[str, t.Any],
    quantized_model_dirpath: str,
    quantized_model_filename: t.Optional[str] = None,
) -> QuantizationOutputTorch:
    """Build default URIs for quantized output in Torch format."""
    if not quantized_model_filename:
        attrs_to_name = "_".join("_".join(map(str, item)) for item in model_attributes.items())
        quantized_model_filename = f"q_{task_name}_{attrs_to_name}_{model_name}_model.pt"

    os.makedirs(quantized_model_dirpath, exist_ok=True)
    output_uri = os.path.join(quantized_model_dirpath, quantized_model_filename)

    paths = QuantizationOutputTorch(output_uri=output_uri)

    return paths


def expand_path(path: str) -> str:
    """Expand path user and sytem variables and remove symlinks."""
    path = os.path.expanduser(path)
    path = os.path.expandvars(path)
    path = os.path.realpath(path)
    return path


def gen_dummy_inputs_for_tracing(
    batch_size: int, vocab_size: int, seq_length: int
) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate dummy inputs for Torch JIT tracing."""
    dummy_input_ids = torch.randint(
        low=0, high=vocab_size, size=(batch_size, seq_length), dtype=torch.long
    )
    dummy_attention_mask = torch.randint(
        low=0, high=2, size=(batch_size, seq_length), dtype=torch.long
    )
    dummy_token_type_ids = torch.zeros(batch_size, seq_length, dtype=torch.long)
    return dummy_input_ids, dummy_attention_mask, dummy_token_type_ids
