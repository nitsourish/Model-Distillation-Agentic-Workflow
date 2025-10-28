# Contributing to Model Distillation Agentic Workflow

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/Model-Distillation-Agentic-Workflow.git`
3. Install dependencies: `pip install -r requirements.txt`
4. Create a branch: `git checkout -b feature/your-feature-name`

## Development Setup

```bash
# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to all public functions and classes
- Keep functions focused and modular

## Testing

Run tests before submitting:

```bash
# Run basic tests (no heavy dependencies needed)
python tests/test_basic.py

# If you have all dependencies installed
pytest tests/
```

## Pull Request Process

1. Update documentation if needed
2. Add tests for new features
3. Ensure all tests pass
4. Update CHANGELOG.md if applicable
5. Submit PR with clear description

## Areas for Contribution

- **New data sources**: Add support for databases, APIs, etc.
- **Additional models**: Support for BERT, T5, LLaMA, etc.
- **Workflow enhancements**: New agent types, better routing
- **Documentation**: Examples, tutorials, translations
- **Performance**: Optimization, quantization, caching
- **Testing**: More comprehensive test coverage

## Questions?

Open an issue for discussion before major changes.
