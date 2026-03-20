# 02460 Advanced Machine Learning

Assignments for the DTU course 02460 Advanced Machine Learning.

## Repository structure

```
assignments/
├── pyproject.toml        # uv workspace root
├── uv.lock               # shared lockfile
├── assignment-1/         
│   ├── pyproject.toml    # assignment-specific dependencies
│   ├── src/              # assignment-specific source code
│   └── report/           # assignment-specific LaTeX report
├── assignment-2/         
│   ├── pyproject.toml
│   ├── src/
│   └── report/
└── assignment-3/         
    ├── pyproject.toml
    ├── src/
    └── report/
```

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-packages
```

To add a dependency to a specific assignment:

```bash
uv add <package> --package assignment-1
```
