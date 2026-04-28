#!/usr/bin/env python3
"""Create transformed txt files in knowlage_base using terms_map.json."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def load_mapping(mapping_file: Path) -> list[tuple[str, str]]:
    data = json.loads(mapping_file.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("JSON должен содержать массив объектов.")

    mapping: list[tuple[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        original = item.get("original")
        replaced = item.get("replaced")
        if isinstance(original, str) and isinstance(replaced, str) and original and replaced:
            mapping.append((original, replaced))

    mapping.sort(key=lambda x: len(x[0]), reverse=True)
    return mapping


def resolve_input_dir(project_root: Path, requested_dir: str) -> Path:
    explicit = project_root / requested_dir
    if explicit.exists():
        return explicit

    for fallback in ("misc/text", "misc/texts"):
        candidate = project_root / fallback
        if candidate.exists():
            return candidate
    return explicit


def sanitize_filename(name: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*]', "", name).strip()
    safe = re.sub(r"\s+", " ", safe)
    return safe or "untitled"


def replace_content(text: str, mapping: list[tuple[str, str]]) -> tuple[str, int]:
    replaced_count = 0
    output = text
    for original, replaced in mapping:
        occurrences = output.count(original)
        if occurrences > 0:
            output = output.replace(original, replaced)
            replaced_count += occurrences
    return output, replaced_count


def rename_stem(stem: str, mapping_dict: dict[str, str]) -> str:
    new_name = mapping_dict.get(stem, stem)
    return sanitize_filename(new_name)


def main() -> int:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Replace terms in txt files and save results to knowlage_base."
    )
    parser.add_argument(
        "--mapping-file",
        type=Path,
        default=project_root / "terms_map.json",
        help="Путь к JSON с полями original/replaced (default: ./terms_map.json).",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="misc/text",
        help="Каталог с исходными txt (default: misc/text; fallback: misc/texts).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "knowlage_base",
        help="Каталог для новых txt файлов (default: ./knowlage_base).",
    )
    args = parser.parse_args()

    if not args.mapping_file.exists():
        print(f"Файл маппинга не найден: {args.mapping_file}", file=sys.stderr)
        return 1

    input_dir = resolve_input_dir(project_root, args.input_dir)
    if not input_dir.exists():
        print(f"Каталог с исходными txt файлами не найден: {input_dir}", file=sys.stderr)
        return 1

    mapping = load_mapping(args.mapping_file)
    if not mapping:
        print("Нет валидных пар original/replaced для замены.", file=sys.stderr)
        return 1

    mapping_dict = {original: replaced for original, replaced in mapping}
    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        print(f"В каталоге нет txt файлов: {input_dir}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_replacements = 0
    written_files = 0
    for file_path in txt_files:
        original_text = file_path.read_text(encoding="utf-8")
        new_text, count = replace_content(original_text, mapping)
        total_replacements += count

        output_name = f"{rename_stem(file_path.stem, mapping_dict)}.txt"
        output_path = args.output_dir / output_name
        output_path.write_text(new_text, encoding="utf-8")
        written_files += 1
        print(f"WRITTEN: {output_path.name} ({count} замен)")

    print(
        f"Готово. Файлов записано: {written_files}, "
        f"всего замен в текстах: {total_replacements}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
