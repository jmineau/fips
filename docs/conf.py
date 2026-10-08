"""Configuration file for the Sphinx documentation builder."""

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import importlib.metadata
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "_ext"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "fips"
copyright = f"2025-{datetime.date.today().year}, James Mineau"
author = "James Mineau"
release = importlib.metadata.version("fips")
version = ".".join(release.split(".")[:2])
# Builds from main (and local builds) are "dev"; release builds are their version.
version_match = "dev" if (".dev" in release or "+" in release) else release


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "api_pages",  # _ext/api_pages.py: class pages with member tables
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
# The docs build treats warnings as errors; these two are known and harmless:
# - BlockLike is a TYPE_CHECKING-only alias, so typehints cannot resolve it
# - the README (included in index.rst) starts its headings at H2
suppress_warnings = ["sphinx_autodoc_typehints.forward_reference", "myst.header"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# MyST settings
myst_enable_extensions = ["html_image", "colon_fence", "dollarmath"]

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
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    # The version dropdown. The Documentation workflow publishes dev/ (main),
    # one folder per release and stable/, and writes switcher.json listing them.
    "switcher": {
        "json_url": "https://jmineau.github.io/fips/switcher.json",
        "version_match": version_match,
    },
    "check_switcher": False,  # switcher.json exists only on the deployed site
    "show_version_warning_banner": True,  # point old versions at stable
}

# Hide primary (left) sidebar on specific pages
html_sidebars = {
    "installation": [],
    "getting_started": [],
    "usage": [],
    "terminology": [],
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
napoleon_use_ivar = True  # what a class page's tables leave in "Attributes"
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

autoclass_content = "class"

# Autosummary settings
autosummary_generate = True

# Nbsphinx settings
nbsphinx_execute = (
    "never"  # Don't execute notebooks during build (they should be pre-run)
)
nbsphinx_allow_errors = False  # Fail if a notebook has errors

# Intersphinx settings
intersphinx_mapping = {
    "cartopy": ("https://cartopy.readthedocs.io/stable/", None),
    "python": ("https://docs.python.org/3", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}
