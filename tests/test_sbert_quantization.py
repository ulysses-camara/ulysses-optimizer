"""Test SBERT quantization functions."""
import typing as t
import time
import os
import shutil

import pytest
import sentence_transformers
import numpy as np

import otimizador


def test_quantization_sbert_custom_name(
    fixture_sroberta_model: sentence_transformers.SentenceTransformer,
    fixture_quantized_model_dir: str,
    fixture_static_quantization_dataset_uri: str,
):
    model_uri = fixture_sroberta_model.get_submodule("0.auto_model").name_or_path

    custom_name = "custom_sbert_name"

    paths_a = otimizador.sbert.to_onnx(
        model_uri=model_uri,
        output_dir=fixture_quantized_model_dir,
        quantized_model_filename=custom_name,
        check_cached=False,
        keep_onnx_model=False,
        static_quantization_dataset_uri=fixture_static_quantization_dataset_uri,
    )

    t_start = time.perf_counter()

    paths_b = otimizador.sbert.to_onnx(
        model_uri=model_uri,
        output_dir=fixture_quantized_model_dir,
        quantized_model_filename=custom_name,
        check_cached=True,
        keep_onnx_model=False,
        static_quantization_dataset_uri=fixture_static_quantization_dataset_uri,
    )

    t_delta = time.perf_counter() - t_start

    assert t_delta <= 0.10, "Cache may not be working."
    assert paths_a == paths_b

    assert paths_a.output_uri == paths_a.onnx_quantized_uri
    assert os.path.basename(paths_a.output_uri) == custom_name
    assert os.path.exists(paths_a.output_uri)

    shutil.rmtree(paths_a.output_uri)


def test_quantization_sbert_default_name(
    fixture_sroberta_model: sentence_transformers.SentenceTransformer,
    fixture_static_quantization_dataset_uri: str,
    fixture_quantized_model_dir: str,
):
    model_uri = fixture_sroberta_model.get_submodule("0.auto_model").name_or_path

    paths_a = otimizador.sbert.to_onnx(
        model_uri=model_uri,
        output_dir=fixture_quantized_model_dir,
        check_cached=False,
        keep_onnx_model=False,
        static_quantization_dataset_uri=fixture_static_quantization_dataset_uri,
    )

    t_start = time.perf_counter()

    paths_b = otimizador.sbert.to_onnx(
        model_uri=model_uri,
        output_dir=fixture_quantized_model_dir,
        check_cached=True,
        keep_onnx_model=False,
        static_quantization_dataset_uri=fixture_static_quantization_dataset_uri,
    )

    t_delta = time.perf_counter() - t_start

    assert t_delta <= 0.10, "Cache may not be working."
    assert paths_a == paths_b

    assert paths_a.output_uri == paths_a.onnx_quantized_uri

    assert os.path.exists(paths_a.output_uri)

    shutil.rmtree(paths_a.output_uri)


@pytest.mark.parametrize(
    "sbert_fixture_name,batch_size",
    [
        ("fixture_anama_model", 1),
        ("fixture_anama_model", 3),
        ("fixture_labse_model", 1),
        ("fixture_labse_model", 3),
        ("fixture_sroberta_model", 1),
        ("fixture_sroberta_model", 3),
    ],
)
def test_quantization_sbert_inference(
    sbert_fixture_name: str,
    batch_size: int,
    fixture_quantized_model_dir: str,
    fixture_static_quantization_dataset_uri: str,
    request: pytest.FixtureRequest,
):
    fixture_sbert_model = request.getfixturevalue(sbert_fixture_name)

    model_uri = fixture_sbert_model.get_submodule("0.auto_model").name_or_path
    model_name = os.path.basename(model_uri.rstrip("/"))

    paths = otimizador.sbert.to_onnx(
        model_uri=model_uri,
        output_dir=fixture_quantized_model_dir,
        quantized_model_filename=f"{model_name}_inference_quant_model",
        check_cached=True,
        optimize_before_quantization=True,
        keep_onnx_model=True,
        static_quantization_dataset_uri=fixture_static_quantization_dataset_uri,
    )

    onnx_sbert_base = otimizador.sbert.ONNXSBERT(paths.onnx_base_uri)
    onnx_sbert_quantized = otimizador.sbert.ONNXSBERT(paths.output_uri)

    test_sequences: t.List[str] = [
        "Sequência de teste para inferência :)",
        "Outra sequência, em Português, para testar inferência no SBERT quantizado.",
        "Oi",
    ]

    sentence_embs_onnx = onnx_sbert_base.encode(test_sequences, batch_size=batch_size)
    sentence_embs_quant = onnx_sbert_quantized.encode(test_sequences, batch_size=batch_size)
    sentence_embs_orig = fixture_sbert_model.encode(test_sequences, batch_size=batch_size)

    assert np.allclose(sentence_embs_onnx, sentence_embs_orig, atol=0.01, rtol=0.02)
    assert np.allclose(sentence_embs_quant, sentence_embs_orig, atol=0.20, rtol=0.50)

    assert isinstance(sentence_embs_quant, np.ndarray)
    assert sentence_embs_quant.size
    assert sentence_embs_quant.ndim == 2
    assert float(np.max(np.abs(sentence_embs_quant))) > 0.0
