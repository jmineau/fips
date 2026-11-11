"""Test package metadata."""

import difflib
import functools
import inspect
import os
import types
import typing
from importlib.metadata import version
from pathlib import Path

import pytest

import fips


def test_version():
    """`__version__` is the installed distribution's version."""
    assert isinstance(fips.__version__, str)
    assert fips.__version__ == version("fips")


def test_no_warning_filter_imports_fips_at_startup(pytestconfig):
    """
    No ``filterwarnings`` line of pyproject.toml names a fips class.

    pytest resolves those lines at startup, which imports fips before
    pytest-cov measures: every module-level line (each import, ``def`` and
    ``class``) is then reported as never run, and coverage reads about twenty
    points low. ``tests/conftest.py`` ignores ``fips.FipsWarning`` with a mark,
    which is read when a test runs.
    """
    named = [f for f in pytestconfig.getini("filterwarnings") if "fips." in f]
    assert not named, named


def test_no_blanket_type_ignore():
    """The source silences pyrefly only with ``# pyrefly: ignore[<code>]`` and a reason, never ``# type: ignore``."""
    src = Path(fips.__file__).parent
    found = [
        f"{path.relative_to(src)}:{number}"
        for path in sorted(src.rglob("*.py"))
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if "# type: ignore" in line
    ]
    assert not found, found


def _public_callables():
    """Yield ``(name, callable)`` for each public function, class, ``__init__`` and public method of ``fips.__all__`` and ``fips.distributions``."""
    modules = [fips, fips.distributions, fips.aggregate]
    seen = set()
    for module in modules:
        for name in getattr(module, "__all__", None) or [
            n for n in vars(module) if not n.startswith("_")
        ]:
            obj = getattr(module, name)
            if isinstance(obj, types.ModuleType):
                continue
            if getattr(obj, "__module__", "").split(".")[0] != "fips":
                continue
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            if inspect.isclass(obj):
                yield f"{name}", obj
                for member_name, member in vars(obj).items():
                    if member_name.startswith("_") and member_name != "__init__":
                        continue
                    if isinstance(member, property):
                        member = member.fget
                    elif isinstance(member, classmethod | staticmethod):
                        member = member.__func__
                    elif isinstance(member, functools.cached_property):
                        member = member.func
                    if callable(member):
                        yield f"{name}.{member_name}", member
            elif callable(obj):
                yield name, obj


@pytest.mark.parametrize(
    "name, obj", list(_public_callables()), ids=[n for n, _ in _public_callables()]
)
def test_type_hints_resolve(name, obj):
    """`typing.get_type_hints` resolves every annotation of the public API (a name only the defining module imports, or a method named like a builtin, would raise or resolve wrongly)."""
    hints = typing.get_type_hints(obj)
    for hint in hints.values():
        assert not inspect.isfunction(hint), (name, hint)


def test_distributions_all_names_the_public_classes():
    """`fips.distributions.__all__` lists the three priors of a setting, so ``import *`` brings no imported helper (``np``, ``ABC``, ``ndtri``)."""
    namespace = {}
    exec("from fips.distributions import *", namespace)
    public = {k for k in namespace if k != "__builtins__"}
    assert public == {"HalfNormal", "LogNormal", "Uniform"}


SIGNATURES = Path(__file__).parent / "data" / "public_signatures.txt"
"""The signature of every public callable, as the API promises it (names, order, keyword-only marks and defaults; no annotations)."""


def _signature_table():
    """Return one line a public callable: its name, and its parameters without annotations."""
    lines = []
    for name, obj in _public_callables():
        if inspect.isclass(obj) and not inspect.isfunction(obj.__init__):
            lines.append(f"{name}(...)")  # takes a builtin's: no signature to store
            continue
        signature = inspect.signature(obj)
        bare = signature.replace(
            parameters=[
                parameter.replace(annotation=inspect.Parameter.empty)
                for parameter in signature.parameters.values()
            ],
            return_annotation=inspect.Signature.empty,
        )
        lines.append(f"{name}{bare}")
    return sorted(lines)


def test_the_public_signatures_are_the_stored_ones():
    """
    No argument of the public API is renamed, reordered, made keyword-only or given another default without this table changing in the same pull request.

    To accept a change, run ``FIPS_WRITE_SIGNATURES=1 pytest tests/test_pkg.py`` and
    commit ``tests/data/public_signatures.txt``.
    """
    table = _signature_table()
    if os.environ.get("FIPS_WRITE_SIGNATURES"):
        SIGNATURES.write_text("\n".join(table) + "\n")
    stored = SIGNATURES.read_text().splitlines()
    changes = [
        line
        for line in difflib.unified_diff(
            stored, table, "stored", "now", lineterm="", n=0
        )
        if not line.startswith(("---", "+++", "@@"))
    ]
    assert not changes, "\n".join(changes)
