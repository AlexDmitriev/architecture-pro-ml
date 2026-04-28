import hashlib
import os
from datetime import datetime
import uuid
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

load_dotenv()

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
COLLECTION_NAME = "document_chunks"
MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
DATA_DIR = Path("./knowlage_base_update")
LOG_PATH = Path("./log.txt")

print(f"Загрузка модели {MODEL_NAME}...")
embedder = SentenceTransformer(MODEL_NAME)
print("Модель загружена")

client = chromadb.HttpClient(
    host=CHROMA_HOST,
    port=CHROMA_PORT,
    settings=Settings(anonymized_telemetry=False),
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", " ", ""],
)


def compute_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def append_log(message: str) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(f"{message}\n")


def get_instruction_for_chunk(chunk_text: str, filename: str, chunk_index: int) -> str:
    return (
        f"Instruct: Given the following excerpt from document '{filename}' "
        f"(chunk {chunk_index}), retrieve relevant information for answering questions.\n\n"
        f"Query: {chunk_text}"
    )


def process_file(file_path: Path) -> list[dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    content_hash = compute_content_hash(text)
    raw_chunks = text_splitter.split_text(text)

    chunks: list[dict[str, Any]] = []
    for i, chunk in enumerate(raw_chunks):
        if len(chunk.strip()) < 50:
            continue
        chunks.append(
            {
                "text": chunk,
                "source": file_path.name,
                "chunk_index": i,
                "total_chunks": len(raw_chunks),
                "content_hash": content_hash,
            }
        )

    return chunks


def add_to_chroma(chunks: list[dict[str, Any]], batch_size: int = 100) -> None:
    total = len(chunks)
    print(f"Добавление {total} чанков в Chroma...")

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]

        ids = []
        texts = []
        metadatas = []
        embeddings = []

        for chunk in batch:
            chunk_id = f"{chunk['source']}_chunk_{chunk['chunk_index']}_{uuid.uuid4().hex[:8]}"
            ids.append(chunk_id)
            texts.append(chunk["text"])
            metadatas.append(
                {
                    "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"],
                    "total_chunks": chunk["total_chunks"],
                    "content_hash": chunk["content_hash"],
                }
            )

            instruction = get_instruction_for_chunk(
                chunk["text"],
                chunk["source"],
                chunk["chunk_index"],
            )
            embedding = embedder.encode(instruction).tolist()
            embeddings.append(embedding)

        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        print(f"   Добавлено {min(i + batch_size, total)}/{total} чанков")


def get_indexed_file_hashes() -> dict[str, str | None]:
    count = collection.count()
    if count == 0:
        return {}

    data = collection.get(include=["metadatas"])
    metadatas = data.get("metadatas") or []

    indexed: dict[str, str | None] = {}
    for meta in metadatas:
        if not meta:
            continue
        source = meta.get("source")
        if not source:
            continue
        indexed[source] = meta.get("content_hash")

    return indexed


def delete_chunks_for_source(source: str) -> None:
    existing = collection.get(where={"source": source}, include=[])
    ids = existing.get("ids") or []
    if ids:
        collection.delete(ids=ids)


def main() -> None:
    started_at = datetime.now()
    errors: list[str] = []
    added_chunks = 0

    append_log(f"index update started at {started_at.isoformat(timespec='seconds')}")

    data_dir = DATA_DIR
    if not data_dir.exists():
        message = f"Папка {data_dir} не существует."
        print(message)
        append_log(message)
        append_log("index update finished with 0 chunks added, 0 index size, 1 errors")
        return

    files = sorted(list(data_dir.glob("*.txt")) + list(data_dir.glob("*.md")))
    if not files:
        message = f"Нет .txt или .md файлов в {data_dir}"
        print(message)
        append_log(message)
        append_log("index update finished with 0 chunks added, 0 index size, 0 errors")
        return

    print(f"Найдено файлов: {len(files)}")
    indexed_hashes = get_indexed_file_hashes()

    new_files: list[Path] = []
    changed_files: list[Path] = []
    skipped_files: list[Path] = []

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            file_text = f.read()

        current_hash = compute_content_hash(file_text)
        stored_hash = indexed_hashes.get(file_path.name)

        if file_path.name not in indexed_hashes:
            new_files.append(file_path)
        elif stored_hash != current_hash:
            changed_files.append(file_path)
        else:
            skipped_files.append(file_path)

    print(f"Новых файлов: {len(new_files)}")
    print(f"Изменённых файлов: {len(changed_files)}")
    print(f"Без изменений: {len(skipped_files)}")

    for file_path in changed_files:
        print(f"Удаление старых чанков для {file_path.name}")
        try:
            delete_chunks_for_source(file_path.name)
        except Exception as exc:  # noqa: BLE001
            error_message = f"Ошибка удаления чанков для {file_path.name}: {exc}"
            print(error_message)
            append_log(error_message)
            errors.append(error_message)

    files_to_index = new_files + changed_files
    if not files_to_index:
        print("Новых или изменённых файлов нет.")
        finished_at = datetime.now()
        final_size = collection.count()
        append_log(
            f"index updated at {finished_at.date()}, 0 files added, 0 chunks added, "
            f"index size {final_size}, {len(errors)} errors"
        )
        return

    all_chunks: list[dict[str, Any]] = []
    for file_path in files_to_index:
        try:
            print(f"Обработка {file_path.name}...")
            chunks = process_file(file_path)
            print(f"   Подготовлено чанков: {len(chunks)}")
            all_chunks.extend(chunks)
        except Exception as exc:  # noqa: BLE001
            error_message = f"Ошибка обработки {file_path.name}: {exc}"
            print(error_message)
            append_log(error_message)
            errors.append(error_message)

    if not all_chunks:
        print("Нет чанков для обновления индекса.")
        finished_at = datetime.now()
        final_size = collection.count()
        append_log(
            f"index updated at {finished_at.date()}, 0 files added, 0 chunks added, "
            f"index size {final_size}, {len(errors)} errors"
        )
        return

    try:
        add_to_chroma(all_chunks)
        added_chunks = len(all_chunks)
    except Exception as exc:  # noqa: BLE001
        error_message = f"Ошибка добавления чанков в индекс: {exc}"
        print(error_message)
        append_log(error_message)
        errors.append(error_message)

    finished_at = datetime.now()
    final_size = collection.count()
    print("Обновление индекса завершено")
    print(f"Всего чанков в коллекции: {final_size}")

    append_log(f"index update finished at {finished_at.isoformat(timespec='seconds')}")
    append_log(
        f"index updated at {finished_at.date()}, {len(files_to_index)} files added, "
        f"{added_chunks} chunks added, index size {final_size}, {len(errors)} errors"
    )


if __name__ == "__main__":
    main()
