"""Sphinx configuration for the elastic scaling documentation site."""

import sys
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DOCS_DIR))

project = "Elastic Scaling PoC"
author = "Dmytro Hryshchenko"

html_show_copyright = False

extensions = ["myst_parser", "sphinxcontrib.mermaid"]

myst_enable_extensions = ["colon_fence"]

html_theme = "sphinx_rtd_theme"
html_title = "Elastic Scaling PoC"

exclude_patterns = ["_build", "_generated", "_figures.py"]


def _render_figures(app):
    """Regenerate the charts from results/ before the build reads any page."""
    import _figures

    _figures.render_all(DOCS_DIR / "_generated")


def setup(app):
    app.connect("builder-inited", _render_figures)
