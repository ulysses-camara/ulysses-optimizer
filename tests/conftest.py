"""Pretrained model setup for quantization."""
import os
import shutil

import pytest
import sentence_transformers
import buscador


@pytest.fixture(name="fixture_quantized_model_dir", scope="session")
def fn_fixture_quantized_model_dir():
    try:
        test_quantized_models_dir = os.path.join(os.path.dirname(__file__), "./test_quantized_models")
        os.makedirs(test_quantized_models_dir, exist_ok=True)
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
        # shutil.rmtree(test_pretrained_models_dir)
        pass


@pytest.fixture(name="fixture_labse_model", scope="session")
def fn_fixture_labse_model(fixture_pretrained_model_dir: str):
    model_name = "ulysses_LaBSE_30000"

    buscador.download_resource(
        task_name="sentence_similarity",
        resource_name=model_name,
        output_dir=fixture_pretrained_model_dir,
        check_cached=True,
        check_resource_hash=True,
    )

    sbert = sentence_transformers.SentenceTransformer(
        os.path.join(fixture_pretrained_model_dir, model_name),
        device="cpu",
    )

    yield sbert


@pytest.fixture(name="fixture_anama_model", scope="session")
def fn_fixture_anama_model(fixture_pretrained_model_dir: str):
    model_name = "sbert_1mil_anama"

    buscador.download_resource(
        task_name="sentence_similarity",
        resource_name=model_name,
        output_dir=fixture_pretrained_model_dir,
        check_cached=True,
        check_resource_hash=True,
    )

    sbert = sentence_transformers.SentenceTransformer(
        os.path.join(fixture_pretrained_model_dir, model_name),
        device="cpu",
    )

    yield sbert


@pytest.fixture(name="fixture_static_quantization_dataset_uri", scope="session")
def fn_fixture_static_quantization_dataset_uri(fixture_pretrained_model_dir: str) -> str:
    dataset_name = "ulysses_tesemo_v2_subset_static_quantization"

    buscador.download_resource(
        task_name="quantization",
        resource_name=dataset_name,
        output_dir=fixture_pretrained_model_dir,
        check_cached=True,
        check_resource_hash=True,
    )

    yield os.path.join(fixture_pretrained_model_dir, dataset_name)
