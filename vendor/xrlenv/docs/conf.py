# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re
import sys
from pathlib import Path

from docutils import nodes
from sphinx.util.docutils import SphinxDirective

# Make the xrlenv package importable during autodoc.
sys.path.insert(0, str(Path(__file__).parent.parent))

from xrlenv import __version__ as _xrlenv_version

# -- Project information -------------------------------------------------------
project = "XRLEnv"
copyright = "XRLEnv contributors"
author = "XRLEnv contributors"
release = _xrlenv_version
version = _xrlenv_version

# -- General configuration -----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinxcontrib.mermaid",
]

# Mermaid: render client-side (default). No mmdc/Node toolchain needed.
mermaid_output_format = "raw"

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # README.md is a build-instructions doc, not a Sphinx page.
    "README.md",
]

# MyST: enable colon-fence directives and header anchors.
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

# Napoleon: Google-style docstrings are the convention in this codebase.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_attr_annotations = True

# Autodoc: show both class docstring and __init__ docstring.
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"

# Intersphinx: link to Python stdlib docs.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# -- HTML output ---------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 4,
    # ``titles_only=True`` keeps the sidebar to toctree entries
    # only — H2 / H3 section headings of the active page don't
    # bleed into the nav (no "Cross-platform notes" / "The two
    # adapter shapes" pollution under the section landing).
    "titles_only": True,
}
html_title = f"XRLEnv {_xrlenv_version}"
html_css_files = ["custom.css"]

# "Edit on GitHub" link in the sphinx_rtd_theme header.
html_context = {
    "display_github": True,
    "github_user": "<your-org>",
    "github_repo": "xrlenv",
    "github_version": "main",
    "conf_py_path": "/docs/",
}


_CSS_LENGTH_RE = re.compile(r"^\d+(?:\.\d+)?(?:px|rem|em|vh|vw|%)$")


def _css_length(value: str) -> str:
    value = value.strip()
    if not _CSS_LENGTH_RE.fullmatch(value):
        raise ValueError("expected a CSS length such as 150px, 20rem, or 50vh")
    return value


def _overflow(value: str) -> str:
    value = value.strip()
    allowed = {"auto", "clip", "hidden", "scroll", "visible"}
    if value not in allowed:
        raise ValueError(f"expected one of: {', '.join(sorted(allowed))}")
    return value


def _fit(value: str) -> str:
    value = value.strip()
    allowed = {"contain", "height"}
    if value not in allowed:
        raise ValueError(f"expected one of: {', '.join(sorted(allowed))}")
    return value


class HeightLimitDirective(SphinxDirective):
    """Wrap arbitrary documentation content in a per-instance height limit."""

    has_content = True
    # docutils declares option_spec as a class attribute (its public directive
    # API); the "mutable class default" here is intentional and expected.
    option_spec = {  # noqa: RUF012
        "height": _css_length,
        "overflow": _overflow,
        "fit": _fit,
    }

    def run(self) -> list[nodes.Node]:
        height = self.options["height"]
        fit = self.options.get("fit", "contain")
        node = nodes.container(classes=["height-limit", f"height-limit-fit-{fit}"])
        styles = [
            f"--height-limit: {height};",
            f"--height-limit-overflow: {self.options.get('overflow', 'hidden')};",
        ]
        node["style"] = " ".join(styles)
        self.state.nested_parse(self.content, self.content_offset, node)
        return [node]


def setup(app):
    app.add_directive("height-limit", HeightLimitDirective)
