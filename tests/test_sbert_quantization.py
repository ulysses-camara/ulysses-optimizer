"""Test SBERT quantization functions."""
import typing as t
import time
import os
import shutil

import pytest
import sentence_transformers
import numpy as np
import torch, torch.nn.functional

import otimizador


def test_quantization_sbert_custom_name(
    fixture_sbert_model: sentence_transformers.SentenceTransformer, fixture_quantized_model_dir: str
):
    model_uri = fixture_sbert_model.get_submodule("0.auto_model").name_or_path

    paths_a = otimizador.sbert.quantize_as_onnx(
        model_uri=model_uri,
        quantized_model_dirpath=fixture_quantized_model_dir,
        quantized_model_filename="custom_sbert_name",
        check_cached=False,
        save_intermediary_onnx_model=False,
    )

    t_start = time.perf_counter()

    paths_b = otimizador.sbert.quantize_as_onnx(
        model_uri=model_uri,
        quantized_model_dirpath=fixture_quantized_model_dir,
        quantized_model_filename="custom_sbert_name",
        check_cached=True,
        save_intermediary_onnx_model=False,
    )

    t_delta = time.perf_counter() - t_start

    assert t_delta <= 0.10, "Cache may not be working."
    assert paths_a == paths_b

    assert paths_a.output_uri == paths_a.onnx_quantized_uri
    assert os.path.basename(paths_a.output_uri) == "custom_sbert_name_onnx"
    assert os.path.exists(paths_a.output_uri)

    shutil.rmtree(paths_a.output_uri)


def test_quantization_sbert_default_name(
    fixture_sbert_model: sentence_transformers.SentenceTransformer, fixture_quantized_model_dir: str
):
    model_uri = fixture_sbert_model.get_submodule("0.auto_model").name_or_path

    paths_a = otimizador.sbert.quantize_as_onnx(
        model_uri=model_uri,
        quantized_model_dirpath=fixture_quantized_model_dir,
        check_cached=False,
        save_intermediary_onnx_model=False,
    )

    t_start = time.perf_counter()

    paths_b = otimizador.sbert.quantize_as_onnx(
        model_uri=model_uri,
        quantized_model_dirpath=fixture_quantized_model_dir,
        check_cached=True,
        save_intermediary_onnx_model=False,
    )

    t_delta = time.perf_counter() - t_start

    assert t_delta <= 0.10, "Cache may not be working."
    assert paths_a == paths_b

    assert paths_a.output_uri == paths_a.onnx_quantized_uri

    assert os.path.exists(paths_a.output_uri)

    shutil.rmtree(paths_a.output_uri)


@pytest.mark.parametrize("batch_size", (1,))
def test_quantization_sbert_inference(
    fixture_sbert_model: sentence_transformers.SentenceTransformer,
    fixture_quantized_model_dir: str,
    batch_size: int,
):
    model_uri = fixture_sbert_model.get_submodule("0.auto_model").name_or_path

    paths = otimizador.sbert.quantize_as_onnx(
        model_uri=model_uri,
        quantized_model_dirpath=fixture_quantized_model_dir,
        check_cached=False,
        save_intermediary_onnx_model=True,
        apply_static_quantization=True,
    )

    # Note: quantized distilLegBert performance will severily drop if 'operators_to_quantize'
    # contains 'MalMul' or 'Attention'.

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

    assert np.allclose(sentence_embs_onnx, sentence_embs_orig, atol=1e-2, rtol=0.05)
    assert np.allclose(sentence_embs_quant, sentence_embs_orig, atol=1e-1, rtol=0.25)

    assert isinstance(sentence_embs_quant, np.ndarray)
    assert sentence_embs_quant.size
    assert sentence_embs_quant.ndim == 2
    assert float(np.max(np.abs(sentence_embs_quant))) > 0.0


@pytest.mark.parametrize("batch_size", (1,))
def test_quantization_labse_inference(
    fixture_labse_model: sentence_transformers.SentenceTransformer,
    fixture_quantized_model_dir: str,
    batch_size: int,
):
    model_uri = fixture_labse_model.get_submodule("0.auto_model").name_or_path

    paths = otimizador.sbert.quantize_as_onnx(
        model_uri=model_uri,
        quantized_model_dirpath=fixture_quantized_model_dir,
        optimized_model_filename="labse_optimized_onnx",
        check_cached=False,
        save_intermediary_onnx_model=True,
        apply_static_quantization=True,
    )

    onnx_sbert_base = otimizador.sbert.ONNXSBERT(paths.onnx_base_uri)
    onnx_sbert_quantized = otimizador.sbert.ONNXSBERT(paths.output_uri)

    assert len(onnx_sbert_base) == len(fixture_labse_model)
    assert len(onnx_sbert_quantized) == len(fixture_labse_model)

    test_sequences: t.List[str] = [
        "Sequência de teste para inferência :)",
        "Outra sequência, em Português, para testar inferência no SBERT quantizado.",
        "Oi",
    ]

    sentence_embs_onnx = onnx_sbert_base.encode(test_sequences, batch_size=batch_size)
    sentence_embs_orig = fixture_labse_model.encode(test_sequences, batch_size=batch_size)
    sentence_embs_quant = onnx_sbert_quantized.encode(test_sequences, batch_size=batch_size)

    assert np.allclose(sentence_embs_onnx, sentence_embs_orig, atol=1e-2, rtol=0.05)
    assert np.allclose(sentence_embs_quant, sentence_embs_orig, atol=1e-1, rtol=0.25)

    assert isinstance(sentence_embs_quant, np.ndarray)
    assert sentence_embs_quant.size
    assert sentence_embs_quant.ndim == 2
    assert float(np.max(np.abs(sentence_embs_quant))) > 0.0
