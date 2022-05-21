import os
import shutil

import pytest
import sentence_transformers
import buscador


@pytest.fixture(name="fixture_quantized_model_dir", scope="session")
def fn_fixture_quantized_model_dir():
    try:
        test_quantized_models_dir = os.path.join(os.path.dirname(__file__), "./test_quantized_models")
        os.makedirs(test_quantized_models_dir, exist_ok=False)
        yield test_quantized_models_dir

    finally:
        shutil.rmtree(test_quantized_models_dir)


@pytest.fixture(name="fixture_pretrained_model_dir", scope="session")
def fn_fixture_pretrained_model_dir():
    try:
        test_pretrained_models_dir = os.path.join(os.path.dirname(__file__), "./test_pretrained_models")
        os.makedirs(test_pretrained_models_dir, exist_ok=True)
        yield test_pretrained_models_dir

    finally:
        shutil.rmtree(test_pretrained_models_dir)


@pytest.fixture(name="fixture_sbert_model", scope="session")
def fn_fixture_sbert_model(fixture_pretrained_model_dir: str):
    model_name = "distil_sbert_br_ctimproved_12_epochs_v1"

    buscador.download_model(
        task_name="sentence_similarity",
        model_name=model_name,
        output_dir=fixture_pretrained_model_dir,
        check_cached=True,
        check_model_hash=True,
    )

    sbert = sentence_transformers.SentenceTransformer(
        os.path.join(fixture_pretrained_model_dir, model_name),
        device="cpu",
    )

    yield sbert
