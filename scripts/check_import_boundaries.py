#!/usr/bin/env python3
"""Static import-boundary checks for architecture refactor.

Current hard rules:
1) `apps/` and `packages/` must not import `src.*`.
2) `packages/` must not import `apps.*`.
3) cross-app imports inside `apps/` are disallowed except explicit allowlist.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([a-zA-Z_][\w\.]*)")

_ALLOWED_CROSS_APP_IMPORTS: set[tuple[str, str]] = set()


@dataclass(frozen=True, slots=True)
class Violation:
    rule: str
    file: str
    line: int
    module: str
    source: str


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*.py")
        if "__pycache__" not in p.parts and ".venv" not in p.parts
    )


def _extract_imports(line: str) -> str | None:
    match = _IMPORT_RE.match(line)
    if not match:
        return None
    return match.group(1)


def _scan_root(root: Path) -> list[Violation]:
    violations: list[Violation] = []
    for file_path in _iter_python_files(root / "apps"):
        rel = file_path.relative_to(root).as_posix()
        app_name = file_path.parts[file_path.parts.index("apps") + 1]
        for idx, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
            module = _extract_imports(line)
            if module is None:
                continue
            if module.startswith("src."):
                violations.append(Violation("no-src-import", rel, idx, module, line.strip()))
            if module.startswith("apps."):
                parts = module.split(".")
                if len(parts) >= 2 and parts[1] != app_name:
                    key = (rel, module)
                    if key not in _ALLOWED_CROSS_APP_IMPORTS:
                        violations.append(Violation("cross-app-import", rel, idx, module, line.strip()))

    for file_path in _iter_python_files(root / "packages"):
        rel = file_path.relative_to(root).as_posix()
        for idx, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
            module = _extract_imports(line)
            if module is None:
                continue
            if module.startswith("src."):
                violations.append(Violation("no-src-import", rel, idx, module, line.strip()))
            if module.startswith("apps."):
                violations.append(Violation("packages-no-apps-import", rel, idx, module, line.strip()))

    return violations


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    violations = _scan_root(root)
    if not violations:
        print("Import boundary check passed.")
        return 0

    print("Import boundary violations:")
    for item in violations:
        print(
            f"- [{item.rule}] {item.file}:{item.line} -> {item.module} | {item.source}"
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
