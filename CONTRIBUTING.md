# Contributing to Audar-ASR

We welcome contributions to Audar-ASR! This document provides guidelines for contributing.

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/AudarAI/Audar-ASR-V1.git
   cd Audar-ASR-V1
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install in development mode**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Run tests**
   ```bash
   pytest tests/ -v
   ```

## Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Write docstrings for public APIs
- Keep functions focused and concise

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear commit messages
3. Add tests for new functionality
4. Ensure all tests pass
5. Update documentation if needed
6. Submit a pull request

## Reporting Issues

When reporting issues, please include:

- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Relevant error messages

## Code of Conduct

Be respectful and constructive in all interactions. We're building something great together.

---

Thank you for contributing to Audar-ASR!
