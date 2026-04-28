import random
from pathlib import Path


TARGET_DIR = Path("./knowlage_base_update")
REPLACEMENTS = ["", "Йогурт", "Жора"]
TARGET_WORD = "Йода"


def replace_match(_: str) -> str:
    return random.choice(REPLACEMENTS)


def process_file(file_path: Path) -> int:
    text = file_path.read_text(encoding="utf-8")
    count = text.count(TARGET_WORD)
    if count == 0:
        return 0

    parts = text.split(TARGET_WORD)
    replaced_text = parts[0]
    for part in parts[1:]:
        replaced_text += replace_match(TARGET_WORD) + part

    file_path.write_text(replaced_text, encoding="utf-8")
    return count


def main() -> None:
    if not TARGET_DIR.exists():
        print(f"Папка {TARGET_DIR} не существует")
        return

    files = sorted(list(TARGET_DIR.rglob("*.txt")) + list(TARGET_DIR.rglob("*.md")))
    if not files:
        print("Файлы не найдены")
        return

    total_files = 0
    total_replacements = 0

    for file_path in files:
        replacements = process_file(file_path)
        if replacements > 0:
            total_files += 1
            total_replacements += replacements
            print(f"{file_path}: заменено {replacements}")

    print(f"Готово. Обработано файлов: {total_files}, замен: {total_replacements}")


if __name__ == "__main__":
    main()
