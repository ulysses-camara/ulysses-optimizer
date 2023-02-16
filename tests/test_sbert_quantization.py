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


@pytest.mark.parametrize("batch_size", (1, 2, 3, 4))
def test_quantization_sbert_inference(
    fixture_sbert_model: sentence_transformers.SentenceTransformer,
    fixture_quantized_model_dir: str,
    batch_size: int,
):
    model_uri = fixture_sbert_model.get_submodule("0.auto_model").name_or_path

    paths = otimizador.sbert.quantize_as_onnx(
        model_uri=model_uri,
        quantized_model_dirpath=fixture_quantized_model_dir,
        check_cached=True,
        save_intermediary_onnx_model=True,
    )

    onnx_sbert_base = otimizador.sbert.ONNXSBERT(paths.onnx_base_uri)
    onnx_sbert_quantized = otimizador.sbert.ONNXSBERT(paths.output_uri)

    test_sequences: t.List[str] = [
        "Sequência de teste para inferência :)",
        "Outra sequência, em Português, para testar inferência no SBERT quantizado.",
        "Oi",
    ]

    sentence_embs_onnx = onnx_sbert_base.encode(test_sequences, batch_size=batch_size)
    sentence_embs_orig = fixture_sbert_model.encode(test_sequences, batch_size=batch_size)
    sentence_embs_quant = onnx_sbert_quantized.encode(test_sequences, batch_size=batch_size)

    assert np.allclose(sentence_embs_onnx, sentence_embs_orig, atol=1e-6, rtol=0.005)

    assert isinstance(sentence_embs_quant, np.ndarray)
    assert sentence_embs_quant.size
    assert sentence_embs_quant.ndim == 2
    assert float(np.max(np.abs(sentence_embs_quant))) > 0.0


@pytest.mark.parametrize("batch_size", (1, 2, 3, 4))
def test_quantization_sbert_inference(
    fixture_labse_model: sentence_transformers.SentenceTransformer,
    fixture_quantized_model_dir: str,
    batch_size: int,
):
    model_uri = fixture_labse_model.get_submodule("0.auto_model").name_or_path

    paths = otimizador.sbert.quantize_as_onnx(
        model_uri=model_uri,
        quantized_model_dirpath=fixture_quantized_model_dir,
        check_cached=True,
        save_intermediary_onnx_model=True,
    )

    onnx_sbert_base = otimizador.sbert.ONNXSBERT(paths.onnx_base_uri, pooling_function="cls", normalize_embeddings=False)
    onnx_sbert_quantized = otimizador.sbert.ONNXSBERT(paths.output_uri, pooling_function="cls", normalize_embeddings=False)

    test_sequences: t.List[str] = [
        "Sequência de teste para inferência :)",
        "Outra sequência, em Português, para testar inferência no SBERT quantizado.",
        "Oi",
    ]

    sentence_embs_onnx = onnx_sbert_base.encode(test_sequences, batch_size=batch_size)
    sentence_embs_orig = fixture_labse_model.encode(test_sequences, batch_size=batch_size)
    sentence_embs_quant = onnx_sbert_quantized.encode(test_sequences, batch_size=batch_size)

    fn_aux = fixture_labse_model.get_submodule("2")

    with torch.no_grad():
        sentence_embs_onnx = fn_aux({"sentence_embedding": torch.from_numpy(sentence_embs_onnx).float()})["sentence_embedding"]
        sentence_embs_quant = fn_aux({"sentence_embedding": torch.from_numpy(sentence_embs_quant).float()})["sentence_embedding"]

        torch.nn.functional.normalize(sentence_embs_onnx, p=2, dim=1, out=sentence_embs_onnx)
        torch.nn.functional.normalize(sentence_embs_quant, p=2, dim=1, out=sentence_embs_quant)

    assert np.allclose(sentence_embs_onnx, sentence_embs_orig, atol=1e-6, rtol=0.005)

    assert isinstance(sentence_embs_quant, np.ndarray)
    assert sentence_embs_quant.size
    assert sentence_embs_quant.ndim == 2
    assert float(np.max(np.abs(sentence_embs_quant))) > 0.0
