import time
import os

import sentence_transformers

import otimizador


def test_quantization_sbert_default_name_clean_files(
    fixture_sbert_model: sentence_transformers.SentenceTransformer, fixture_quantized_model_dir: str
):
    paths_a = otimizador.sbert.quantize_as_onnx(
        model=fixture_sbert_model,
        quantized_model_dirpath=fixture_quantized_model_dir,
        task_name="test_task",
        check_cached=False,
        clean_intermediary_files=True,
    )

    t_start = time.perf_counter()

    paths_b = otimizador.sbert.quantize_as_onnx(
        model=fixture_sbert_model,
        quantized_model_dirpath=fixture_quantized_model_dir,
        task_name="test_task",
        check_cached=True,
        clean_intermediary_files=True,
    )

    t_delta = time.perf_counter() - t_start

    assert t_delta <= 0.10, "Cache may not be working."
    assert paths_a == paths_b

    assert not os.path.exists(paths_a.onnx_base_uri)
    assert not os.path.exists(paths_a.onnx_optimized_uri)

    assert paths_a.output_uri == paths_a.onnx_quantized_uri

    assert os.path.exists(paths_a.output_uri)

    os.remove(paths_a.output_uri)
