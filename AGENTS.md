# Agent Instructions for LeRobot Repository

This file contains guidelines for AI agents working in the LeRobot codebase.

## Development Commands

### Setup
```bash
# Install with development dependencies
pip install -e ".[dev,test]"  # or use uv

# Or using uv (if available)
uv pip install -e ".[dev,test]"
```

### Code Quality
```bash
# Format code with ruff
ruff format .

# Check for lint issues
ruff check .

# Fix lint issues automatically
ruff check --fix .

# Run type checking (gradual typing enabled)
mypy src

# Run security checks
bandit -c pyproject.toml -r src

# Check spelling
typos --write-changes .
```

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=lerobot --cov-report=term-missing

# Run specific test file
pytest tests/path/to/test_file.py

# Run specific test function
pytest tests/path/to/test_file.py::test_function_name

# Run tests matching pattern
pytest -k "test_pattern"

# Run end-to-end tests (requires specific hardware/setup)
make test-end-to-end  # see Makefile for individual test targets

# Run tests with timeout
pytest --timeout=30
```

### Pre-commit
```bash
# Install pre-commit hooks
pre-commit install

# Run pre-commit on all files
pre-commit run --all-files

# Run specific hook
pre-commit run ruff --all-files
```

## Code Style Guidelines

### Python Version
- Target Python 3.12+ (configured in `pyproject.toml`)

### Formatting
- **Line length**: 110 characters (configured in Ruff)
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Double quotes for strings
- **Trailing commas**: Use when appropriate (skip-magic-trailing-comma=false)

### Imports
- Use absolute imports
- Group imports: stdlib, third-party, local
- Combine `as` imports when appropriate (configured in isort)
- Known first-party module: `lerobot`

### Naming Conventions
- **Classes**: `CamelCase`
- **Functions/Methods**: `snake_case`
- **Variables**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private**: Prefix with `_` for non-public methods/variables

### Type Annotations
- Use type hints throughout
- Strict typing enforced for configs modules (`lerobot.configs.*`)
- Other modules have gradual typing enabled (ignore_errors=true)
- Required for: `configs`, `envs`, `optim`, `model`, `cameras`, `motors`, `transport`

### Error Handling
- Use appropriate exception types
- Avoid bare `except:` clauses
- Document exceptions in docstrings when non-obvious

### Documentation
- Google-style docstrings (configured in pydocstyle)
- Include Args, Returns, Raises sections
- Document public API thoroughly

### File Headers
- Include Apache 2.0 license header in all source files
- Format: Copyright notice followed by license boilerplate

### Lint Rules (Selected)
Enabled: E, W (pycodestyle errors/warnings), F (PyFlakes), I (isort), B (flake8-bugbear), 
         C4 (flake8-comprehensions), T20 (flake8-print), N (pep8-naming), UP (pyupgrade), SIM (flake8-simplify)

Ignored: E501 (line too long), T201/T203 (print statements), B008 (function call in defaults),
         F401/F403 in `__init__.py`

### Module-Specific Rules
- `src/lerobot/policies/wall_x/`: Suppressed rules for original Qwen2_5_vl code

## Testing Guidelines

### Test Structure
- Tests located in `tests/` directory
- Use pytest fixtures defined in `tests/fixtures/`
- Mark hardware-dependent tests with `@require_env` or appropriate pytest markers
- Test names should be descriptive: `test_<functionality>_<scenario>`

### Hardware Integration
- Tests requiring physical hardware should gracefully skip when unavailable
- Use `_check_component_availability()` pattern from `conftest.py`
- Provide clear error messages when hardware is missing

### End-to-End Tests
- Defined in Makefile with `test-*` targets
- Test training/evaluation pipelines for different policies
- Use minimal configurations for speed
- Output to `tests/outputs/` directory

## Repository Structure

```
lerobot/
├── src/lerobot/           # Main source code
│   ├── async_inference/   # Async inference server
│   ├── cameras/          # Camera interfaces
│   ├── configs/          # Configuration schemas (strict typing)
│   ├── datasets/         # Dataset loading and processing
│   ├── envs/            # Environment wrappers
│   ├── model/           # Model architectures
│   ├── motors/          # Motor controllers (strict typing)
│   ├── optim/           # Optimizers and schedulers (strict typing)
│   ├── policies/        # Policy implementations
│   ├── processor/       # Data processors
│   ├── rl/              # Reinforcement learning components
│   ├── robots/          # Robot interfaces
│   ├── scripts/         # CLI entry points
│   ├── teleoperators/   # Teleoperation devices
│   ├── transport/       # Transport layer (strict typing)
│   └── utils/           # Utility functions
├── tests/               # Test suite
└── examples/           # Example scripts
```

## Workflow Tips

1. **Before committing**: Run `pre-commit run --all-files`
2. **Type checking**: Run `mypy src` for affected modules
3. **Test coverage**: Aim to maintain or improve coverage
4. **Hardware tests**: Mark appropriately and handle missing hardware gracefully
5. **New features**: Update `lerobot/__init__.py` with availability lists
6. **Policy/Env additions**: Follow update instructions in `__init__.py` docstring

## Notes
- The project uses gradual typing; new code should include type hints
- Some policy directories have suppressed lint rules due to imported code
- End-to-end tests require specific dataset availability
- Pre-commit hooks enforce code quality standards

## Resources
- [Ruff documentation](https://docs.astral.sh/ruff/)
- [pytest documentation](https://docs.pytest.org/)
- [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)
- [LeRobot Documentation](https://huggingface.co/docs/lerobot/index)