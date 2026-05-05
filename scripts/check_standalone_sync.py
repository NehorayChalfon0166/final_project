"""Check the standalone CLI for structural drift from src/.

`standalone/wallet_analyzer.py` is shipped as a single self-contained file: it
inlines copies of the model (OptimalBitcoinGNN), the graph builder
(EgoGraphBuilder), and a handful of constants from src/graph/config.py. When
src/ changes, those inlined copies must be updated by hand.

This script catches the common ways the two can drift:
- Constants from config.py that the standalone uses get reassigned upstream.
- The model class gains/loses an __init__ parameter or a named layer.
- The graph builder gains/loses a method or changes a public method signature.

It is a structural check, not a textual one — comments, docstrings, and
formatting are ignored on purpose so cosmetic edits to either file don't
trigger false alarms.

Exits 0 on PASS, 1 on drift. Print actionable hints either way.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_CONFIG = PROJECT_ROOT / "src" / "graph" / "config.py"
SRC_MODEL = PROJECT_ROOT / "src" / "models" / "optimal_gnn.py"
SRC_BUILDER = PROJECT_ROOT / "src" / "graph" / "graph_builder.py"
STANDALONE = PROJECT_ROOT / "standalone" / "wallet_analyzer.py"

# Constants that must agree (name in standalone -> name in src/graph/config.py).
TRACKED_CONSTANTS = {
    "FEATURE_COLUMNS": "FEATURE_COLUMNS",
    "NUM_NODE_FEATURES": "NUM_NODE_FEATURES",
    "NUM_EDGE_FEATURES": "NUM_EDGE_FEATURES",
    "TIMESTAMP_MIN": "TIMESTAMP_MIN",
    "TIMESTAMP_MAX": "TIMESTAMP_MAX",
}


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def top_level_assigns(module: ast.Module) -> dict:
    """Return {name: value} for top-level constant assignments.

    Handles literal RHS values plus a few simple computed forms: ``len(NAME)``
    where NAME has already been resolved earlier in the module.
    """
    out: dict = {}
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            try:
                out[target.id] = ast.literal_eval(node.value)
                continue
            except ValueError:
                pass
            # `len(NAME)` where NAME is a previously-resolved list/tuple.
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "len"
                and len(node.value.args) == 1
                and isinstance(node.value.args[0], ast.Name)
                and node.value.args[0].id in out
            ):
                out[target.id] = len(out[node.value.args[0].id])
    return out


def find_class(module: ast.Module, name: str) -> ast.ClassDef | None:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def init_signature(cls: ast.ClassDef) -> list[tuple[str, str | None]]:
    """List of (param_name, default_repr) for the class's __init__, skipping self."""
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            args = node.args.args[1:]  # drop self
            defaults = [None] * (len(args) - len(node.args.defaults)) + [
                ast.unparse(d) for d in node.args.defaults
            ]
            return [(a.arg, d) for a, d in zip(args, defaults)]
    return []


def class_attribute_names(cls: ast.ClassDef) -> set[str]:
    """All `self.<name> = ...` assignments inside the class body."""
    names: set[str] = set()
    for fn in cls.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        names.add(target.attr)
    return names


def public_method_signatures(cls: ast.ClassDef) -> dict[str, list[tuple[str, str | None]]]:
    """All public methods (no underscore prefix) and their full signature."""
    out: dict[str, list[tuple[str, str | None]]] = {}
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            args = node.args.args[1:]
            defaults = [None] * (len(args) - len(node.args.defaults)) + [
                ast.unparse(d) for d in node.args.defaults
            ]
            out[node.name] = [(a.arg, d) for a, d in zip(args, defaults)]
    return out


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_constants(src_assigns: dict, sa_assigns: dict, errors: list[str]) -> None:
    for sa_name, src_name in TRACKED_CONSTANTS.items():
        if src_name not in src_assigns:
            errors.append(f"[constants] missing {src_name} in src/graph/config.py")
            continue
        if sa_name not in sa_assigns:
            errors.append(f"[constants] missing {sa_name} in standalone")
            continue
        if src_assigns[src_name] != sa_assigns[sa_name]:
            errors.append(
                f"[constants] {sa_name}: standalone={sa_assigns[sa_name]!r} "
                f"src={src_assigns[src_name]!r}"
            )


def check_class_init(src_cls: ast.ClassDef, sa_cls: ast.ClassDef, label: str, errors: list[str]) -> None:
    src_sig = init_signature(src_cls)
    sa_sig = init_signature(sa_cls)
    if src_sig != sa_sig:
        errors.append(
            f"[{label}.__init__] signature drift\n"
            f"    src:  {src_sig}\n"
            f"    sa:   {sa_sig}"
        )


def check_class_attrs(src_cls: ast.ClassDef, sa_cls: ast.ClassDef, label: str, errors: list[str]) -> None:
    src_attrs = class_attribute_names(src_cls)
    sa_attrs = class_attribute_names(sa_cls)
    missing_in_sa = src_attrs - sa_attrs
    extra_in_sa = sa_attrs - src_attrs
    if missing_in_sa:
        errors.append(f"[{label}] attributes missing in standalone: {sorted(missing_in_sa)}")
    if extra_in_sa:
        errors.append(f"[{label}] attributes only in standalone (verify intentional): {sorted(extra_in_sa)}")


def check_public_methods(
    src_cls: ast.ClassDef, sa_cls: ast.ClassDef, label: str, errors: list[str]
) -> None:
    src_methods = public_method_signatures(src_cls)
    sa_methods = public_method_signatures(sa_cls)
    for name, src_sig in src_methods.items():
        if name not in sa_methods:
            errors.append(f"[{label}.{name}] missing in standalone")
            continue
        if sa_methods[name] != src_sig:
            errors.append(
                f"[{label}.{name}] signature drift\n"
                f"    src:  {src_sig}\n"
                f"    sa:   {sa_methods[name]}"
            )


def main() -> int:
    errors: list[str] = []

    src_config = parse(SRC_CONFIG)
    src_model = parse(SRC_MODEL)
    src_builder = parse(SRC_BUILDER)
    sa = parse(STANDALONE)

    # Constants
    check_constants(top_level_assigns(src_config), top_level_assigns(sa), errors)

    # OptimalBitcoinGNN
    src_gnn = find_class(src_model, "OptimalBitcoinGNN")
    sa_gnn = find_class(sa, "OptimalBitcoinGNN")
    if not src_gnn or not sa_gnn:
        errors.append("[OptimalBitcoinGNN] class not found in one of the files")
    else:
        check_class_init(src_gnn, sa_gnn, "OptimalBitcoinGNN", errors)
        check_class_attrs(src_gnn, sa_gnn, "OptimalBitcoinGNN", errors)

    # EgoGraphBuilder
    src_geb = find_class(src_builder, "EgoGraphBuilder")
    sa_geb = find_class(sa, "EgoGraphBuilder")
    if not src_geb or not sa_geb:
        errors.append("[EgoGraphBuilder] class not found in one of the files")
    else:
        check_class_init(src_geb, sa_geb, "EgoGraphBuilder", errors)
        check_public_methods(src_geb, sa_geb, "EgoGraphBuilder", errors)

    if errors:
        print("DRIFT detected between standalone/wallet_analyzer.py and src/:\n")
        for e in errors:
            print(f"  - {e}")
        print(
            "\nFix: open standalone/wallet_analyzer.py and update the inlined "
            "block(s) between the BEGIN/END INLINED markers to match src/. "
            "Re-run this script until it prints PASS."
        )
        return 1

    print("PASS — standalone/wallet_analyzer.py is structurally in sync with src/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
