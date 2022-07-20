# Configuration file for the Sphinx documentation builder.

# path setup
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information
project = "test rtd"
copyright = "2022, test rtd"
author = "test rtd"


# -- General configuration
extensions = [
    "sphinx_rtd_theme",
    "recommonmark",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_markdown_tables",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx.ext.extlinks",
]


autodoc_default_options = {
    "member-order": "bysource",
    "undoc-members": False,
}


# sphinx supported file extensions.
source_suffix = [".rst", ".md"]


# -- Localization of Documentation.
# https://docs.readthedocs.io/en/stable/localization.html
locale_dirs = ['locale/']
gettext_compact = False
language = 'en'
add_module_names = False
gettext_uuid = True


# -- Options for HTML output
html_theme = "sphinx_rtd_theme"


# -- Options for EPUB output
# epub_show_urls = "footnote"
