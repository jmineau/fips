"""Configuration file for the Sphinx documentation builder."""

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from importlib.metadata import version

sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "fips"
copyright = "2025, James Mineau"
author = "James Mineau"
release = version("fips")


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# MyST settings
myst_enable_extensions = ["html_image", "colon_fence"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "github_url": "https://github.com/jmineau/fips",
    "show_toc_level": 2,
    "navbar_align": "left",
    "logo": {
        "image_light": "_static/logo.png",
        "image_dark": "_static/logo_dark.png",
    },
}

# Hide primary (left) sidebar on specific pages
html_sidebars = {
    "installation": [],
    "quickstart": [],
    "usage": [],
}

# -- Extension configuration -------------------------------------------------

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

autoclass_content = "class"

# Mock imports for dependencies not available during doc builds
autodoc_mock_imports = ["stilt", "cartopy", "matplotlib"]

# Autosummary settings
autosummary_generate = True

# Intersphinx settings
intersphinx_mapping = {
    "cartopy": ("https://scitools.org.uk/cartopy/docs/latest/", None),
    "python": ("https://docs.python.org/3", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "xarray": ("https://xarray.pydata.org/en/stable/", None),
}


def process_class_docstrings(app, what, name, obj, options, lines) -> None:
    """
    Process class docstrings to remove empty autosummary sections.

    For classes using custom autosummary templates, this removes any
    Attributes or Methods sections that contain only 'None' as placeholder,
    preventing Sphinx warnings and ugly HTML output.

    This is called during the "autodoc-process-docstring" event for each
    docstring being processed.
    """
    if what == "class":
        joined = "\n".join(lines)

        templates = [
            """.. rubric:: Attributes

.. autosummary::
   :toctree:

   None
""",
            """.. rubric:: Methods

.. autosummary::
   :toctree:

   None
""",
        ]

        for template in templates:
            if template in joined:
                joined = joined.replace(template, "")
        lines[:] = joined.split("\n")


def setup(app):
    """
    Set up Sphinx extension hooks.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        The Sphinx application object.
    """
    app.connect("autodoc-process-docstring", process_class_docstrings)
