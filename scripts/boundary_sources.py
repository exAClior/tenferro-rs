"""Source interpretation shared by the existing boundary checks."""

from pathlib import Path
import runpy
import tomllib


# Reuse the audit system's Rust lexer rather than maintain another parser.
_rust_code_lines = runpy.run_path(
    str(Path(__file__).with_name("repository-rules-review.py"))
)["rust_code_lines"]


def code_lines(text: str) -> list[str]:
    """Mask Rust comments/literals while preserving diagnostic line numbers."""
    return _rust_code_lines(text)[1]


def dependency_names(manifest: dict):
    """Yield dependency aliases and package names, including target tables."""
    for key, value in manifest.items():
        if key in {"dependencies", "dev-dependencies", "build-dependencies"}:
            for alias, spec in value.items():
                yield alias
                if isinstance(spec, dict) and "package" in spec:
                    yield spec["package"]
        elif key in {"target", "workspace"} and isinstance(value, dict):
            if key == "target":
                for target in value.values():
                    yield from dependency_names(target)
            else:
                yield from dependency_names(value)


def boundary_lines(path: Path):
    """Yield code lines or parsed manifest dependency names for a boundary."""
    text = path.read_text(encoding="utf-8")
    if path.name == "Cargo.toml":
        for name in dependency_names(tomllib.loads(text)):
            yield 1, name
    elif path.suffix == ".rs":
        yield from enumerate(code_lines(text), start=1)
