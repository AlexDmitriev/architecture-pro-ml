#!/usr/bin/env python3
"""RAG-бот: Chroma + DeepSeek."""

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

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_API_KEY_HARDCODED = "sk-4057569ed6964c6db3e86bc579bd75fc"

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "document_chunks")


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} для {url}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Не удалось подключиться к {url}: {exc}") from exc


def load_embedder() -> SentenceTransformer:
    print(f"Загрузка эмбеддинг-модели: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Эмбеддинг-модель загружена")
    return model


def build_query_embedding(embedder: SentenceTransformer, query: str) -> list[float]:
    # Используем ту же схему, что в индексации/поиске проекта.
    return embedder.encode(f"Instruct: {query}").tolist()


class ChromaRetriever:
    def __init__(self, host: str, port: int, collection_name: str) -> None:
        client = chromadb.HttpClient(
            host=host,
            port=port,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def search(self, query_embedding: list[float], n_results: int) -> list[dict[str, Any]]:
        results = self.collection.query(
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

    def count(self) -> int:
        return self.collection.count()


def build_prompt(user_query: str, chunks: list[dict[str, Any]]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        context_blocks.append(
            (
                f"[Фрагмент {i}] source={chunk['source']}, chunk={chunk['chunk_index']}, "
                f"relevance={chunk['relevance']:.3f}\n{chunk['text']}"
            )
        )

    context = "\n\n".join(context_blocks) if context_blocks else "Контекст не найден."
    return (
        "System: Ты помощник, который сначала размышляет, а потом отвечает. Всегда пиши свои шаги."
        "Отвечай только на основе контекста.\n"
        "Если данных недостаточно, прямо сообщи об этом.\n"
        "Отвечай по-русски.\n\n"        
        f"Контекст:\n{context}\n\n"
        f"Вопрос: {user_query}"
    )


def ask_deepseek(prompt: str, api_key: str, temperature: float, max_tokens: int) -> str:
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = _post_json(f"{DEEPSEEK_BASE_URL}/chat/completions", payload, headers)
    return data["choices"][0]["message"]["content"]


def make_retriever(args: argparse.Namespace) -> ChromaRetriever:
    return ChromaRetriever(
        host=args.chroma_host,
        port=args.chroma_port,
        collection_name=args.chroma_collection,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG-бот (Chroma + DeepSeek).")
    parser.add_argument("--n-results", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=700)

    parser.add_argument("--chroma-host", type=str, default=CHROMA_HOST)
    parser.add_argument("--chroma-port", type=int, default=CHROMA_PORT)
    parser.add_argument("--chroma-collection", type=str, default=CHROMA_COLLECTION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not DEEPSEEK_API_KEY_HARDCODED:
        raise RuntimeError("Не задан ключ DeepSeek в DEEPSEEK_API_KEY_HARDCODED.")

    print("Векторная БД: chroma")
    print(f"LLM: {DEEPSEEK_MODEL} ({DEEPSEEK_BASE_URL})")

    embedder = load_embedder()
    retriever = make_retriever(args)
    total = retriever.count()
    if total == 0:
        print("Векторная база пуста. Сначала проиндексируйте документы.")
    else:
        print(f"В базе найдено чанков: {total}")

    print("\nВведите вопрос (или 'exit' для выхода):")
    while True:
        user_query = input("\n> ").strip()
        if not user_query:
            print("Введите непустой запрос.")
            continue
        if user_query.lower() in {"exit", "quit", "q"}:
            print("Завершение работы.")
            break

        try:
            query_embedding = build_query_embedding(embedder, user_query)
            chunks = retriever.search(query_embedding=query_embedding, n_results=args.n_results)
            prompt = build_prompt(user_query=user_query, chunks=chunks)
            answer = ask_deepseek(
                prompt=prompt,
                api_key=DEEPSEEK_API_KEY_HARDCODED,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Ошибка: {exc}")
            continue

        print("\nНайденные фрагменты:")
        for i, chunk in enumerate(chunks, start=1):
            preview = chunk["text"].replace("\n", " ")[:140]
            print(
                f"{i}. {chunk['source']} (chunk {chunk['chunk_index']}), "
                f"relevance={chunk['relevance']:.3f}\n   {preview}..."
            )

        print("\nОтвет:")
        print(answer)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
        sys.exit(0)
