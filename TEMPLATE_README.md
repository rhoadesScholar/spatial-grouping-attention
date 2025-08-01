# Python Package Template

This is a comprehensive template for creating Python packages with modern best practices, automated testing, CI/CD, and proper documentation.

## 🚀 Features

- **Modern Python packaging** with `pyproject.toml` and `hatchling`
- **Automated testing** with pytest, coverage, and multiple Python versions
- **Code quality tools** (black, isort, flake8, mypy)
- **Pre-commit hooks** for code quality and automatic versioning
- **GitHub Actions CI/CD** with automated PyPI publishing
- **Documentation structure** ready for Sphinx or mkdocs
- **Date-based semantic versioning** with automatic CITATION.cff updates
- **Multiple platform testing** (Linux, macOS, Windows)
- **Comprehensive Makefile** for development tasks

## 📁 Template Structure

```
spatial-attention/
├── .github/
│   └── workflows/
│       └── ci-cd.yml              # GitHub Actions workflow
├── .pre-commit-hooks/
│   ├── update-version.py          # Auto-version update hook
│   └── README.md                  # Pre-commit hooks documentation
├── src/
│   └── spatial_attention/
│       ├── __init__.py            # Package initialization
│       └── core.py                # Main package code
├── tests/
│   └── test_spatial_attention.py   # Test suite
├── docs/
│   └── README.md                  # Documentation
├── .gitignore                     # Git ignore patterns
├── .pre-commit-config.yaml        # Pre-commit configuration
├── pyproject.toml                 # Package configuration
├── CITATION.cff                   # Citation metadata
├── LICENSE                        # License file
├── Makefile                       # Development commands
└── README.md                      # This file
```

## 🛠️ How to Use This Template

### 1. Create a new repository from this template

1. Click "Use this template" on GitHub
2. Or clone this repository and use it as a starting point

### 2. Replace template placeholders

Search and replace the following placeholders throughout the codebase:

#### Required Placeholders

| Placeholder | Description | Example |
|-------------|-------------|---------|
| `spatial-attention` | Repository name | `my-awesome-package` |
| `spatial_attention` | Python package name (src/ folder) | `my_awesome_package` |
| `spatial-attention` | PyPI package name | `my-awesome-package` |
| `Spatial grouping attention using Rotary Spatial Embeddings (RoSE)` | Short project description | `A awesome Python package` |
| `Inspired by the spatial grouping layer in Native Segmentation Vision Transformers (https://arxiv.org/abs/2505.16993), implemented in PyTorch with a modified rotary position embedding generalized to N-dimensions and incorporating real-world pixel spacing.` | Detailed project description | `This package provides...` |
| `Jeff Rhoades` | Your full name | `John Doe` |
| `Jeff` | Your first name | `John` |
| `Rhoades` | Your last name | `Doe` |
| `rhoadesj@hhmi.org` | Your email | `john@example.com` |
| `rhoadesScholar` | Your GitHub username | `johndoe` |
| `BSD 3-Clause License` | License name | `BSD 3-Clause License` |
| `BSD-3-Clause` | SPDX license identifier | `BSD-3-Clause` |
| `BSD License` | PyPI license classifier | `BSD License` |
| `SpatialAttention` | Main class name | `MyAwesomeClass` |
| `2025` | Current year | `2025` |

#### Optional Placeholders

| Placeholder | Description | Default/Example |
|-------------|-------------|-----------------|
| `"attention", "transformer", "rotary spatial embedding", "segmentation"` | PyPI keywords | `["python", "package"]` |
| `0000-0001-5077-2533` | Your ORCID ID | `https://orcid.org/0000-0000-0000-0000` |
| `` | Codecov token | `YOUR_TOKEN_HERE` |
| `rhoadesScholar2025spatial_attention` | BibTeX citation key | `doe2025awesome` |
| `0.1.0` | Initial version | `0.1.0` |
| `2025-07-31` | Release date | `2025-01-01` |

### 3. Rename directories and files

1. Rename `src/spatial_attention/` to `src/your_actual_package_name/`
2. Rename `tests/test_spatial_attention.py` to `tests/test_your_package.py`

### 4. Update dependencies

Edit `pyproject.toml` to add your specific dependencies:

```toml
dependencies = [
    "numpy>=1.20.0",
    "requests>=2.25.0",
    # your dependencies here
]
```

### 5. Set up development environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
make dev-setup
```

### 6. Configure GitHub repository

#### Required Secrets

Set these in your GitHub repository settings under **Settings > Secrets and variables > Actions**:

- `CODECOV_TOKEN`: Your Codecov token (get from [codecov.io](https://codecov.io))
- `PYPI_API_TOKEN`: Your PyPI API token for publishing (optional, for automated releases)

#### Branch Protection

Consider setting up branch protection rules for `main`:

1. Go to **Settings > Branches**
2. Add rule for `main` branch
3. Enable "Require status checks to pass before merging"
4. Select the CI checks you want to require

### 7. Start developing

```bash
# Run tests
make test

# Format code
make format

# Run all checks
make check-all

# View all available commands
make help
```

## 🔧 Development Workflow

This template includes a full development workflow:

### Code Quality

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pre-commit**: Automated checks

### Testing

- **pytest**: Testing framework
- **pytest-cov**: Coverage reporting
- **pytest-xdist**: Parallel testing
- Automatic testing on multiple Python versions (3.10, 3.11, 3.12)
- Cross-platform testing (Linux, macOS, Windows)

### Versioning

- **Date-based versioning**: `YYYY.M.D.HHMM` format
- **Automatic updates**: Version and release date updated on main branch commits
- **CITATION.cff**: Academic citation metadata automatically maintained

### Publishing

- **Automated PyPI publishing**: Triggered by version tags
- **GitHub Releases**: Create releases with `git tag v1.0.0 && git push origin v1.0.0`
- **Build artifacts**: Stored for every successful build

## 📚 Additional Setup Options

### Documentation with Sphinx

```bash
pip install sphinx sphinx-rtd-theme
sphinx-quickstart docs
```

### Docker Support

Add a `Dockerfile`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e .
CMD ["python", "-m", "spatial_attention"]
```

### Advanced Testing

```bash
# Add coverage configuration to pyproject.toml
[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
]
```

## 🤝 Contributing to This Template

If you have suggestions for improving this template:

1. Fork this repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This template is released under the MIT License. Projects created from this template can use any license you choose.

---

**Happy coding! 🎉**
