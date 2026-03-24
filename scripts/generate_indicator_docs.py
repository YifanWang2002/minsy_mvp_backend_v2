"""Generate indicator documentation from registry (single source of truth).

Usage:
  PYTHONPATH=. uv run python scripts/generate_indicator_docs.py
  PYTHONPATH=. uv run python scripts/generate_indicator_docs.py --check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from packages.domain.strategy.feature.indicators import IndicatorCategory, IndicatorRegistry


README_PATH = Path("packages/domain/strategy/feature/README.md")
JSON_PATH = Path("packages/domain/strategy/feature/indicator_catalog.generated.json")


def _build_catalog() -> dict[str, Any]:
    categories: list[dict[str, Any]] = []
    total = 0
    for category in IndicatorCategory:
        names = IndicatorRegistry.list_by_category(category)
        items: list[dict[str, Any]] = []
        for name in names:
            metadata = IndicatorRegistry.get(name)
            if metadata is None:
                continue
            items.append(
                {
                    "name": metadata.name,
                    "full_name": metadata.full_name,
                    "description": metadata.description,
                    "params": [
                        {
                            "name": param.name,
                            "type": param.type,
                            "default": param.default,
                            "min": param.min_value,
                            "max": param.max_value,
                            "choices": param.choices,
                            "description": param.description,
                        }
                        for param in metadata.params
                    ],
                    "outputs": [
                        {"name": output.name, "description": output.description}
                        for output in metadata.outputs
                    ],
                    "required_columns": list(metadata.required_columns),
                    "version": str(getattr(metadata, "version", "1.0.0")),
                    "status": str(getattr(metadata, "status", "active")),
                    "deprecated_since": getattr(metadata, "deprecated_since", None),
                    "replacement": getattr(metadata, "replacement", None),
                    "remove_after": getattr(metadata, "remove_after", None),
                }
            )
        if not items:
            continue
        total += len(items)
        categories.append(
            {
                "category": category.value,
                "count": len(items),
                "indicators": items,
            }
        )

    return {
        "total_indicators": total,
        "categories": categories,
    }


def _format_outputs(value: list[dict[str, Any]]) -> str:
    names = [str(item.get("name", "")).strip() for item in value]
    names = [item for item in names if item]
    if not names:
        return "-"
    return ", ".join(names)


def _build_markdown(catalog: dict[str, Any]) -> str:
    total = int(catalog.get("total_indicators", 0) or 0)
    categories = catalog.get("categories", [])
    if not isinstance(categories, list):
        categories = []

    lines: list[str] = [
        "# 因子系统 README（自动生成）",
        "",
        "> 本文件由 `scripts/generate_indicator_docs.py` 自动生成，请勿手改。",
        "",
        f"- 指标总数：`{total}`",
        "",
        "## 分类统计",
        "",
        "| category | count |",
        "|---|---:|",
    ]
    for category in categories:
        lines.append(f"| {category['category']} | {int(category['count'])} |")

    for category in categories:
        lines.extend(
            [
                "",
                f"## {category['category']} ({int(category['count'])})",
                "",
                "| indicator | full_name | outputs | version | status | required_columns |",
                "|---|---|---|---|---|---|",
            ]
        )
        for item in category.get("indicators", []):
            name = str(item.get("name", "")).strip()
            full_name = str(item.get("full_name", "")).strip()
            outputs = _format_outputs(item.get("outputs", []))
            required_columns = ",".join(item.get("required_columns", [])) or "-"
            version = str(item.get("version", "1.0.0")).strip() or "1.0.0"
            status = str(item.get("status", "active")).strip() or "active"
            lines.append(
                f"| `{name}` | {full_name} | `{outputs}` | `{version}` | `{status}` | `{required_columns}` |"
            )

    lines.extend(
        [
            "",
            "## 生成命令",
            "",
            "```bash",
            "PYTHONPATH=. uv run python scripts/generate_indicator_docs.py",
            "PYTHONPATH=. uv run python scripts/generate_indicator_docs.py --check",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _write_if_changed(path: Path, content: str) -> bool:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if existing == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    catalog = _build_catalog()
    markdown = _build_markdown(catalog)
    json_text = json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    if args.check:
        current_readme = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else ""
        current_json = JSON_PATH.read_text(encoding="utf-8") if JSON_PATH.exists() else ""
        if current_readme != markdown or current_json != json_text:
            raise SystemExit("indicator docs are out of date. run scripts/generate_indicator_docs.py")
        print("indicator docs are up to date")
        return

    readme_changed = _write_if_changed(README_PATH, markdown)
    json_changed = _write_if_changed(JSON_PATH, json_text)
    print(
        f"generated indicator docs: readme_changed={readme_changed} json_changed={json_changed}"
    )


if __name__ == "__main__":
    main()
