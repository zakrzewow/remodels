"""Sphinx configuration."""
project = "remodels"
author = " Kacper Skonieczka & Grzegorz Zakrzewski"
copyright = "2023,  Kacper Skonieczka & Grzegorz Zakrzewski"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
    "nbsphinx",
]
autodoc_typehints = "description"
html_theme = "furo"
autoclass_content = "both"
