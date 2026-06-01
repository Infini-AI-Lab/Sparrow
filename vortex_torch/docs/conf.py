# -- Path setup --------------------------------------------------------------
import os
import sys
from datetime import datetime
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "Vortex"
author = "Zhuoming Chen"
copyright = f"{datetime.now():%Y}, {author}"

# Read package version safely (avoid clashing with Sphinx's `version` config)
try:
    from importlib.metadata import version as pkg_version
    release = pkg_version("Vortex")   # <-- change to your real distribution name if needed (e.g. "vortex")
except Exception:
    release = "1.0.0"

# Short X.Y version for the sidebar/footer
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx_copybutton",
]


autodoc_mock_imports = [
    "torch",
    "triton",
    "numpy",
    "vortex_torch_C",
    "flash-attn",
    "flashinfer"
]


autosummary_generate = True
autosummary_generate_overwrite = False
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    #"members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_enable_extensions = ["deflist", "linkify", "substitution", "tasklist"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
language = "en"

# -- HTML --------------------------------------------------------------------
html_theme = "furo"
html_title = f"{project} Documentation"
html_static_path = ["_static"]

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),  # <- None instead of {}
}

todo_include_todos = True

def setup(app):
    app.add_css_file("custom.css")