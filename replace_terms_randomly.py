import re
import random
from pathlib import Path


YODA_DIR = Path("./knowlage_base_update")
KNOWLAGE_DIR = Path("./knowlage_base")
YODA_REPLACEMENTS = ["", "Йогурт", "Жора"]
YODA_WORD = "Йода"
STASIS_PATTERN = re.compile(r"стазис", re.IGNORECASE)


def replace_stasis(match: re.Match[str]) -> str:
    source = match.group(0)
    if source.isupper():
        return "БАЗИС"
    if source[0].isupper():
        return "Базис"
    return "базис"


def replace_yoda(_: str) -> str:
    return random.choice(YODA_REPLACEMENTS)


def process_yoda_file(file_path: Path) -> int:
    text = file_path.read_text(encoding="utf-8")
    count = text.count(YODA_WORD)
    if count == 0:
        return 0

    parts = text.split(YODA_WORD)
    replaced_text = parts[0]
    for part in parts[1:]:
        replaced_text += replace_yoda(YODA_WORD) + part

    file_path.write_text(replaced_text, encoding="utf-8")
    return count


def process_stasis_file(file_path: Path) -> int:
    text = file_path.read_text(encoding="utf-8")
    replaced_text, count = STASIS_PATTERN.subn(replace_stasis, text)
    if count == 0:
        return 0

    file_path.write_text(replaced_text, encoding="utf-8")
    return count


def process_directory(
    target_dir: Path,
    processor: callable,
    label: str,
) -> tuple[int, int]:
    if not target_dir.exists():
        print(f"Папка {target_dir} не существует")
        return 0, 0

    files = sorted(list(target_dir.rglob("*.txt")) + list(target_dir.rglob("*.md")))
    if not files:
        print(f"Файлы не найдены в {target_dir}")
        return 0, 0

    total_files = 0
    total_replacements = 0
    for file_path in files:
        replacements = processor(file_path)
        if replacements > 0:
            total_files += 1
            total_replacements += replacements
            print(f"{label}: {file_path}: заменено {replacements}")

    return total_files, total_replacements


def main() -> None:
    yoda_files, yoda_replacements = process_directory(
        YODA_DIR,
        process_yoda_file,
        "Йода",
    )
    stasis_files, stasis_replacements = process_directory(
        KNOWLAGE_DIR,
        process_stasis_file,
        "Стазис",
    )

    total_files = yoda_files + stasis_files
    total_replacements = yoda_replacements + stasis_replacements
    if total_files == 0:
        print("Изменений не найдено")
        return

    print(f"Готово. Обработано файлов: {total_files}, замен: {total_replacements}")


if __name__ == "__main__":
    main()
