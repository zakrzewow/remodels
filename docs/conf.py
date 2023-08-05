"""Sphinx configuration."""
project = "remodels"
author = "Grzegorz Zakrzewski"
copyright = "2023, Grzegorz Zakrzewski"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
