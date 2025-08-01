# {{project_name}}

## {{project_description}}

![PyPI - License](https://img.shields.io/pypi/l/spatial-attention)
[![CI/CD Pipeline](https://github.com/rhoadesScholar/spatial-attention/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/rhoadesScholar/spatial-attention/actions/workflows/ci-cd.yml)
[![codecov](https://codecov.io/github/rhoadesScholar/spatial-attention/graph/badge.svg?token=)](https://codecov.io/github/rhoadesScholar/spatial-attention)
![PyPI - Version](https://img.shields.io/pypi/v/spatial-attention)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/spatial-attention)

Inspired by the spatial grouping layer in Native Segmentation Vision Transformers (https://arxiv.org/abs/2505.16993), implemented in PyTorch with a modified rotary position embedding generalized to N-dimensions and incorporating real-world pixel spacing.

## Installation

### From PyPI

```bash
pip install spatial-attention
```

### From source

```bash
pip install git+https://github.com/rhoadesScholar/spatial-attention.git
```

## Usage

```python
import spatial_attention

# Example usage
# TODO: Add your usage examples here
```

## Development

### Setting up development environment

```bash
# Clone the repository
git clone https://github.com/rhoadesScholar/spatial-attention.git
cd spatial-attention

# Install in development mode with all dependencies
make dev-setup
```

### Running tests

```bash
# Run all tests
make test

# Run tests with coverage
make test-cov

# Run fast tests (stop on first failure)
make test-fast
```

### Code quality

```bash
# Format code
make format

# Lint code
make lint

# Type check
make type-check

# Run all checks
make check-all
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the test suite (`make test`)
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.

## Citation

If you use this software in your research, please cite it using the information in [CITATION.cff](CITATION.cff).
