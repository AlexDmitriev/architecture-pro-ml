#!/usr/bin/env python3
"""Консольный RAG-бот поверх Chroma + SentenceTransformer."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "document_chunks")
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = "deepseek-r1:8b"


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url=url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Не удалось подключиться к {url}: {exc}") from exc


def load_embedder() -> SentenceTransformer:
    print(f"📥 Загрузка модели эмбеддингов: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("✅ Модель эмбеддингов загружена")
    return model


def get_collection() -> chromadb.Collection:
    client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        settings=Settings(anonymized_telemetry=False),
    )
    # Если коллекции еще нет, создаем ее, чтобы бот запускался без падения.
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def build_query_embedding(embedder: SentenceTransformer, user_query: str) -> list[float]:
    # Используем тот же шаблон, что и при тестовом поиске/индексации.
    query_with_instruction = f"Instruct: {user_query}"
    return embedder.encode(query_with_instruction).tolist()


def search_chunks(
    collection: chromadb.Collection,
    query_embedding: list[float],
    n_results: int,
) -> list[dict[str, Any]]:
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    return [
        {
            "text": doc,
            "source": meta.get("source", "unknown"),
            "chunk_index": meta.get("chunk_index", -1),
            "relevance": 1 - dist,
        }
        for doc, meta, dist in zip(docs, metas, dists)
    ]


def build_prompt(user_query: str, chunks: list[dict[str, Any]]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        context_blocks.append(
            (
                f"[Фрагмент {i}] source={chunk['source']}, chunk={chunk['chunk_index']}, "
                f"relevance={chunk['relevance']:.3f}\n{chunk['text']}"
            )
        )

    context_text = "\n\n".join(context_blocks) if context_blocks else "Контекст не найден."
    return (
        "Ты ассистент, отвечающий только на основе предоставленного контекста.\n"
        "Если информации недостаточно, явно скажи об этом.\n"
        "Отвечай по-русски, структурированно и кратко.\n\n"
        f"Контекст:\n{context_text}\n\n"
        f"Вопрос пользователя: {user_query}"
    )


def call_ollama(prompt: str, temperature: float, max_tokens: int) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    headers = {"Content-Type": "application/json"}
    data = _post_json(f"{OLLAMA_BASE_URL}/api/generate", payload, headers)
    return data["response"]


def ask_llm(prompt: str, temperature: float, max_tokens: int) -> str:
    return call_ollama(prompt=prompt, temperature=temperature, max_tokens=max_tokens)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Консольный RAG-бот (Chroma + LLM).")
    parser.add_argument(
        "--n-results",
        type=int,
        default=int(os.getenv("RAG_N_RESULTS", 5)),
        help="Сколько релевантных чанков искать в Chroma (default: 5).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("LLM_TEMPERATURE", 0.2)),
        help="Температура генерации (default: 0.2).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.getenv("LLM_MAX_TOKENS", 700)),
        help="Максимум токенов ответа LLM (default: 700).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"⚙️ Модель LLM: {OLLAMA_MODEL} (через Ollama)")
    print(f"⚙️ Chroma: {CHROMA_HOST}:{CHROMA_PORT}, коллекция={COLLECTION_NAME}")

    embedder = load_embedder()
    collection = get_collection()
    chunks_count = collection.count()
    if chunks_count == 0:
        print(
            f"⚠️ Коллекция '{COLLECTION_NAME}' пуста. "
            "Сначала запустите индексацию: python3 build_index.py"
        )
    else:
        print(f"✅ В коллекции найдено чанков: {chunks_count}")

    print("\nВведите вопрос (или 'exit' для выхода):")
    while True:
        user_query = input("\n> ").strip()
        if not user_query:
            print("Введите непустой запрос.")
            continue
        if user_query.lower() in {"exit", "quit", "q"}:
            print("👋 Завершение работы.")
            break

        try:
            query_embedding = build_query_embedding(embedder, user_query)
            chunks = search_chunks(
                collection=collection,
                query_embedding=query_embedding,
                n_results=args.n_results,
            )
            prompt = build_prompt(user_query=user_query, chunks=chunks)
            answer = ask_llm(
                prompt=prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"❌ Ошибка: {exc}")
            continue

        print("\n📚 Найденные фрагменты:")
        for idx, chunk in enumerate(chunks, start=1):
            preview = chunk["text"].replace("\n", " ")[:140]
            print(
                f"{idx}. {chunk['source']} (chunk {chunk['chunk_index']}), "
                f"relevance={chunk['relevance']:.3f}\n   {preview}..."
            )

        print("\n🤖 Ответ:")
        print(answer)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Прервано пользователем.")
        sys.exit(0)
