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
- [SentenceTransformers](https://github.com/UKPLab/sentence-transformers): static and dynamic quantization.

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

model_uri = "<path_to_my_sbert_pretrained_model>"

# Quantize SBERT model
quantized_model_paths = otimizador.sbert.to_onnx(
    model_uri=model_uri,
    output_dir="./onnx_models",
    device="cpu",
)

# Load quantized model
quantized_model = otimizador.sbert.ONNXSBERT(quantized_model_paths.output_uri)

# Use quantized model for inference
embeddings = quantized_model(
    ["Exemplo de inferência", "Olá"],
    batch_size=2,
    show_progress_bar=True,
)

print(embeddings.shape)
# >>> (2, 768)

print(embeddings)
# >>> [[ 0.21573983  0.09294462  0.81110716 ...  0.1845829   0.44957376
# ...   -0.8655164 ]
# ...  [ 0.14329034  0.39949742  0.62624204 ...  0.34124994  0.5183566
# ...   -0.4494257 ]]
```

---

## License
[MIT.](./LICENSE)
