[![Tests](https://github.com/ulysses-camara/ulysses-optimizer/actions/workflows/tests.yml/badge.svg)](https://github.com/ulysses-camara/ulysses-optimizer/actions/workflows/tests.yml)

# Ulysses Optimizer

Optimization and quantization methods for pretrained models. While the resources presented here may be generalized to any supported pretrained model, this package API is built targeting conveniet use in the Ulysses project modules.

---

## Table of Contents
1. [Supported algorithms](#supported-algorithms)
2. [Installation](#installation)
3. [Usage examples](#usage-examples)
4. [License](#license)

---

## Supported algorithms
All models must be in PyTorch format (tensorflow is not supported). Models in ONNX format are executed with `onnxruntime` package, while models in Torch JIT format are executed with `torch`.

| Algorithm | Package | *ONNX* | *Torch JIT* |
| --------- | ------- | ------ | ----------- |
| (Bi-)LSTM | `torch` | Soon  | Soon |
| BERT      | `transformers` | Soon  | Soon |
| SBERT     | `sentence_transformers` | :heavy_check_mark: | Later |

---

## Installation
For BERT models:
```bash
pip install "otimizador @ git+https://github.com/ulysses-camara/ulysses-optimizer"
```

For SBERT models:
```bash
pip install "otimizador[sentence] @ git+https://github.com/ulysses-camara/ulysses-optimizer"
```

Developer dependencies can be installed as:
```bash
pip install "otimizador[dev] @ git+https://github.com/ulysses-camara/ulysses-optimizer"
```

---

## Usage examples

### Sentence BERT (SBERT) example
```python
import otimizador
import sentence_transformers

pretrained_sbert = sentence_transformers.SentenceTransformer(
    "<path_to_my_pretrained_model>",
    device="cpu",
)

# Quantize SBERT model
quantized_model_paths = otimizador.sbert.quantize_as_onnx(
    model=pretrained_sbert,
    task_name="quantization_test",
    quantized_model_dirpath="./quantized_models",
)

# Load quantized model
quantized_model = otimizador.sbert.ONNXSBERT(
    uri_model=quantized_model_paths.output_uri,
    uri_tokenizer=pretrained_sbert.tokenizer.name_or_path,
)

# Use quantized model for inference
logits = quantized_model(
    ["Exemplo de inferência", "Olá"],
    batch_size=1,
    show_progress_bar=True,
)

print(logits.shape)
# >>> (2, 768)

print(logits)
# >>> [[ 0.21573983  0.09294462  0.81110716 ...  0.1845829   0.44957376
# ...   -0.8655164 ]
# ...  [ 0.14329034  0.39949742  0.62624204 ...  0.34124994  0.5183566
# ...   -0.4494257 ]]
```

Check [example notebooks](./notebooks) for more information.

---

## License
[MIT.](./LICENSE)
