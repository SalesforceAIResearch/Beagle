# XRLEnv documentation

## Building the docs

Install doc dependencies into the project venv:

```bash
uv pip install -e ".[docs]"
```

Build HTML:

```bash
.venv/bin/sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser.

For a stricter build that treats warnings as errors:

```bash
.venv/bin/sphinx-build -W -b html docs docs/_build/html
```

## Source layout

```
docs/
├── conf.py                # Sphinx configuration
├── index.rst              # Role-based landing page + flattened toctrees
├── installation.md        # Runtime install for local use
├── quickstart.md          # First local rollout
├── architecture.md        # User-facing architecture concepts
├── consumer/              # Client usage, run-configs, timeouts
├── operations/            # Operator CLI reference
├── deployment/            # Local and multi-node rollout service deployment
├── observability/         # Admin panel, artifacts, metrics, logs
├── integration/           # Benchmark plug-in authoring and reference integrations
├── developer/             # Contributor setup and design rationale
└── api/                   # Autodoc reference
```
