#!/usr/bin/env python3
"""Download and extract plain text from Warhammer 40k Fandom links."""

from __future__ import annotations

import argparse
import html
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

TARGET_DIV_CLASS = "mw-content-ltr mw-parser-output"
TITLE_SUFFIX = "| Warhammer 40000 Wiki | Fandom"
DEFAULT_TIMEOUT = 20


class FandomContentParser(HTMLParser):
    """Extract title and plain text from the target content div."""

    def __init__(self) -> None:
        super().__init__()
        self._class_stack: list[str | None] = []
        self._inside_title = False
        self._inside_target_div = False
        self._ignore_depth = 0
        self._title_parts: list[str] = []
        self._text_parts: list[str] = []

    @property
    def title(self) -> str:
        return "".join(self._title_parts).strip()

    @property
    def text(self) -> str:
        raw = "".join(self._text_parts)
        normalized = raw.replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"[ \t\f\v]+", " ", normalized)
        normalized = re.sub(r"\n\s*\n\s*\n+", "\n\n", normalized)
        return normalized.strip()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        class_attr = attrs_dict.get("class")
        self._class_stack.append(class_attr)

        if tag == "title":
            self._inside_title = True

        classes = set(class_attr.split()) if class_attr else set()
        if tag == "div" and {"mw-content-ltr", "mw-parser-output"}.issubset(classes):
            self._inside_target_div = True

        if self._inside_target_div and tag in {"script", "style", "noscript"}:
            self._ignore_depth += 1

        if self._inside_target_div and self._ignore_depth == 0 and tag in {"p", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self._text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        class_attr = self._class_stack.pop() if self._class_stack else None

        if tag == "title":
            self._inside_title = False

        if self._inside_target_div and tag in {"script", "style", "noscript"} and self._ignore_depth > 0:
            self._ignore_depth -= 1

        classes = set(class_attr.split()) if class_attr else set()
        if tag == "div" and self._inside_target_div and {"mw-content-ltr", "mw-parser-output"}.issubset(classes):
            self._inside_target_div = False

    def handle_data(self, data: str) -> None:
        if self._inside_title:
            self._title_parts.append(data)
        if self._inside_target_div and self._ignore_depth == 0:
            cleaned = html.unescape(data)
            if cleaned.strip():
                self._text_parts.append(cleaned)


def extract_unique_urls(markdown_text: str) -> list[str]:
    pattern = re.compile(r"\[[^\]]+\]\((https?://[^)\s]+)\)")
    seen: set[str] = set()
    unique_urls: list[str] = []
    for match in pattern.finditer(markdown_text):
        url = match.group(1)
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    return unique_urls


def fetch_html(url: str, timeout: int) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,"
            "image/webp,image/apng,*/*;q=0.8"
        ),
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Referer": "https://www.google.com/",
        "Upgrade-Insecure-Requests": "1",
    }

    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="replace")
    except HTTPError as exc:
        if exc.code != 403:
            raise
        return fetch_html_with_curl(url, timeout)


def fetch_html_with_curl(url: str, timeout: int) -> str:
    cmd = [
        "curl",
        "--silent",
        "--show-error",
        "--location",
        "--compressed",
        "--max-time",
        str(timeout),
        "--user-agent",
        (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "--header",
        "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "--header",
        "Accept-Language: ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "--header",
        "Cache-Control: no-cache",
        "--header",
        "Pragma: no-cache",
        "--header",
        "Referer: https://www.google.com/",
        "--header",
        "Upgrade-Insecure-Requests: 1",
        url,
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise URLError(f"curl failed ({result.returncode}): {result.stderr.strip()}")
    return result.stdout


def sanitize_filename(name: str) -> str:
    name = name.strip().replace("/", "-")
    name = re.sub(r'[<>:"\\|?*]', "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name or "untitled"


def normalize_title(raw_title: str) -> str:
    title = raw_title.replace(TITLE_SUFFIX, "").strip()
    return sanitize_filename(title)


def process_url(url: str, output_dir: Path, timeout: int) -> tuple[str, str]:
    page_html = fetch_html(url, timeout=timeout)
    parser = FandomContentParser()
    parser.feed(page_html)

    if not parser.text:
        raise ValueError("Не удалось извлечь текст из целевого div.")

    title = normalize_title(parser.title or "untitled")
    output_path = output_dir / f"{title}.txt"
    output_path.write_text(parser.text + "\n", encoding="utf-8")
    return title, str(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download links from links.md and save plain text content."
    )
    parser.add_argument(
        "--links-file",
        type=Path,
        default=Path(__file__).resolve().parent / "links.md",
        help="Path to markdown file with links (default: misc/links.md).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "texts",
        help="Directory for output txt files (default: misc/texts).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"HTTP timeout in seconds (default: {DEFAULT_TIMEOUT}).",
    )
    args = parser.parse_args()

    if not args.links_file.exists():
        print(f"Файл не найден: {args.links_file}", file=sys.stderr)
        return 1

    markdown_text = args.links_file.read_text(encoding="utf-8")
    urls = extract_unique_urls(markdown_text)
    if not urls:
        print("Ссылки не найдены.", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Найдено уникальных ссылок: {len(urls)}")
    success_count = 0
    for index, url in enumerate(urls, start=1):
        try:
            title, file_path = process_url(url, args.output_dir, args.timeout)
            success_count += 1
            print(f"[{index}/{len(urls)}] OK: {title} -> {file_path}")
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            print(f"[{index}/{len(urls)}] ERROR: {url} ({exc})", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"[{index}/{len(urls)}] UNEXPECTED: {url} ({exc})", file=sys.stderr)

    print(f"Готово. Успешно: {success_count}/{len(urls)}")
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
