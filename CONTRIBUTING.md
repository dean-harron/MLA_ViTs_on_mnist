# Contributing

Keep experiments reproducible and keep the reusable implementation in `src/mla_mnist/`.

Before opening a pull request:

```bash
pytest
```

For style checks, install the development dependencies and run:

```bash
ruff check src scripts tests
```

Please keep generated datasets, checkpoints, and figures out of version control. The repository `.gitignore` already excludes the standard generated directories.
