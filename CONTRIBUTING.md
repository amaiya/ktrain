# Contributing to ktrain

We are happy to accept your contributions to make `ktrain` better! To avoid unnecessary work, please stick to the following process:

1. Check if there is already [an issue](https://github.com/amaiya/ktrain/issues) for your concern.
2. If there is not, open a new one to start a discussion. We hate to close finished PRs.
3. We would be happy to accept a pull request, if it is decided that your concern requires a code change.


## Developing locally

We suggest cloning the repository and then checking out tutorials and examples for information on how to call various methods.
Most relevant classes and methods should be documented. If not, you might consider helping to improve the docstrings.

### setup

See the [installation instructions](https://github.com/amaiya/ktrain#installation) for setting things up.

### tests

To run all tests, execute:
```bash
cd ktrain/tests
python3 -m unittest
```

To run a specific test (e.g., `test_dataloading.py`)
```bash
python3 test_dataloading.py
```

### PR Guidelines

- Keep each PR focused. While it's more convenient, please try to avoid combining several unrelated fixes together.
- Try to maintain backwards compatibility.  If this is not possible, please discuss with maintainer(s).
- Use four spaces for indentation. 
