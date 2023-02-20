"""General purpose, algorithm-agnostic utility functions."""
import typing as t
import os
import datetime
import random

import torch


class QuantizationOutputONNX(t.NamedTuple):
    """Output paths for quantization as ONNX format."""

    onnx_base_uri: str
    onnx_quantized_uri: str
    output_uri: str


def build_random_model_name(base_name: str) -> str:
    """Build model name using random bits and current UTC time."""
    random_name = "_".join(
        [
            base_name,
            datetime.datetime.utcnow().strftime("%Y_%m_%d__%H_%M_%S"),
            hex(random.getrandbits(128))[2:],
        ]
    )
    return random_name


def build_onnx_default_uris(
    model_name: str,
    output_dir: str,
    model_attributes: t.Optional[t.Dict[str, t.Any]] = None,
    quantized_model_filename: t.Optional[str] = None,
    onnx_model_filename: t.Optional[str] = None,
) -> QuantizationOutputONNX:
    """Build default URIs for quantized output in ONNX format."""
    model_attributes = model_attributes or {}

    if not onnx_model_filename:
        attrs_to_name = "_".join("_".join(map(str, item)) for item in model_attributes.items())
        onnx_model_filename = f"{model_name}_{attrs_to_name}_model"

    if not quantized_model_filename:
        quantized_model_filename = f"q_{onnx_model_filename}"

    os.makedirs(output_dir, exist_ok=True)

    onnx_base_uri = os.path.join(output_dir, onnx_model_filename)
    onnx_quantized_uri = os.path.join(output_dir, quantized_model_filename)

    paths_dict: t.Dict[str, str] = {
        "onnx_base_uri": onnx_base_uri,
        "onnx_quantized_uri": onnx_quantized_uri,
        "output_uri": onnx_quantized_uri,
    }

    paths = QuantizationOutputONNX(**paths_dict)

    all_path_set = {paths.onnx_base_uri, paths.onnx_quantized_uri}
    num_distinct_paths = len(all_path_set)

    if num_distinct_paths < 2:
        raise ValueError(
            f"{2 - num_distinct_paths} URI for ONNX models (including intermediary models) "
            "are the same, which will cause undefined behaviour while quantizing the model. "
            "Please provide distinct filenames for ONNX files."
        )

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
