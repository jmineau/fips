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
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "matplotlib.sphinxext.plot_directive",  # `.. plot::` runs code, shows the figure
    "IPython.sphinxext.ipython_directive",  # `.. ipython::` runs code in an .rst page
    "IPython.sphinxext.ipython_console_highlighting",  # colors its In/Out prompts
    "myst_nb",  # notebooks (.ipynb, or Markdown with code cells), run at build time
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "api_pages",  # _ext/api_pages.py: class pages with member tables
]

templates_path = ["_templates"]
# The docs build treats warnings as errors; these two are known and harmless:
# - BlockLike is a TYPE_CHECKING-only alias, so typehints cannot resolve it
# - the README (included in index.rst) starts its headings at H2
suppress_warnings = ["sphinx_autodoc_typehints.forward_reference", "myst.header"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

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

# Examples run when the docs build, so they show real output and fail the build
# when they break. A `.. plot::` directive (in a docstring's Examples section or
# on any page) shows its code and the figure it draws.
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = [("png", 100)]

# In an .rst page, `.. ipython:: python` runs its code and shows each line with
# its output; a block that raises or warns fails the build. Its figures (a line
# `@savefig name.png` above the plotting call) are saved under docs/_build.
ipython_savefig_dir = "_build/savefig"

# Code cells run too, in notebooks and in Markdown pages whose header names a
# kernel, and show every output; a cell that raises fails the build. Outputs are
# cached in docs/_build/.jupyter_cache, so `just build-docs` (which starts clean,
# as CI does) runs them all, and `just docs-serve` reruns a page only when its
# code changes. A notebook that cannot run in CI keeps the outputs it was
# committed with if its metadata has "mystnb": {"execution_mode": "off"}.
nb_execution_mode = "cache"
nb_execution_show_tb = True  # print the traceback of a failing cell in the log

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
