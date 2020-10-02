# Contributing

If you'd like to contribute to the library here is a guide to setting up the development environment.

1. Clone

```
git clone https://github.com/pyannote/pyannote-audio
```

2. Virtual Environment

Setup some sort of virtaul environment to prevent clashes with other projects libraries.

If you're using conda we've got an environment.yaml file you can use:

```
conda env create -f environment.yaml
```

3. Setup Pre-Commit Hooks

We use pre-commit enforce import sorting and code formatting.

These commands below will make sure that you have pre-commit set up and pytest as well

```
pip install -e .[dev,testing]
pre-commit install
```
