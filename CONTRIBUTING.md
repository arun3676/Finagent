# Contributing to FinAgent

Thank you for your interest in contributing to FinAgent! This document provides guidelines and instructions for contributing.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/finagent.git
   cd finagent
   ```
3. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 💻 Development Setup

### Prerequisites
- Python 3.11+
- Docker (for Qdrant)
- Node.js 18+ (for frontend)

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Configuration
```bash
cp backend/.env.example backend/.env
# Add your API keys to .env
```

### Running Tests
```bash
# Test API keys
python scripts/test_api_keys.py

# Test workflow
python scripts/test_workflow.py

# Validate code
python scripts/validate.py
```

## 📝 Code Standards

### Python
- **Python 3.11+** with type hints on ALL functions
- **Pydantic v2** for ALL data models
- **async/await** for ALL I/O operations
- Use **logging** module (never `print()`)
- Handle exceptions explicitly (no bare `except`)

### Cross-Platform Compatibility
❌ **NEVER** use `&&` in subprocess calls (Windows incompatible)
✅ **ALWAYS** use `pathlib.Path` for file paths
✅ **ALWAYS** use list-style subprocess args

### Pydantic Settings
✅ **ALWAYS** use `SettingsConfigDict` for environment loading:
```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    api_key: str
    
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
```

## 🧪 Testing Requirements

Before submitting a PR:
- [ ] All tests pass: `python scripts/test_workflow.py`
- [ ] Code validation passes: `python scripts/validate.py`
- [ ] Type checking passes: `mypy backend/app`
- [ ] Linting passes: `ruff check backend/app`
- [ ] No secrets in code (check with `git diff`)

## 📋 Pull Request Process

1. **Update documentation** if you change functionality
2. **Add tests** for new features
3. **Follow commit message conventions**:
   ```
   feat: add validator agent hallucination detection
   fix: resolve Pydantic v2 settings loading issue
   docs: update README with new architecture diagram
   ```
4. **Ensure CI passes** (when available)
5. **Request review** from maintainers

## 🎯 Areas for Contribution

### High Priority
- [ ] Frontend-backend integration testing
- [ ] Additional SEC filing types (8-K, DEF 14A)
- [ ] Performance optimization for large documents
- [ ] Enhanced error handling and logging

### Medium Priority
- [ ] Additional evaluation metrics
- [ ] Support for more LLM providers
- [ ] Caching layer for repeated queries
- [ ] Rate limiting improvements

### Documentation
- [ ] API documentation improvements
- [ ] Architecture diagrams
- [ ] Tutorial videos
- [ ] Example use cases

## 🐛 Bug Reports

When reporting bugs, please include:
- **Environment**: OS, Python version, package versions
- **Steps to reproduce**: Minimal code example
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Error messages**: Full stack trace if applicable

## 💡 Feature Requests

For feature requests, please describe:
- **Use case**: What problem does this solve?
- **Proposed solution**: How would you implement it?
- **Alternatives**: Other approaches you considered
- **Impact**: Who would benefit from this feature?

## 📞 Getting Help

- **Issues**: Open a GitHub issue for bugs or features
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact maintainers for security issues

## 🙏 Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help others learn and grow

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to FinAgent! 🚀
