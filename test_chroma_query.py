#!/usr/bin/env python3
"""Test queries against Chroma collection."""

from __future__ import annotations

import argparse
import os
from typing import Any

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "document_chunks")
MODEL_NAME = "intfloat/multilingual-e5-large-instruct"

def search_documents(
    query: str,
    embedder: SentenceTransformer,
    collection: chromadb.Collection,
    n_results: int = 5,
) -> list[dict[str, Any]]:
    instruction = f"Instruct: {query}"
    query_embedding = embedder.encode(instruction).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    return [
        {
            "text": doc,
            "source": meta.get("source", "unknown"),
            "chunk": meta.get("chunk_index", -1),
            "relevance": 1 - dist,
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Тестовый поиск по Chroma.")
    parser.add_argument(
        "--query",
        type=str,
        default="расскажи про Империум",
        help="Текст запроса (default: 'расскажи про Империум').",
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=5,
        help="Сколько результатов вернуть (default: 5).",
    )
    args = parser.parse_args()

    print(f"📥 Загрузка модели {MODEL_NAME}...")
    embedder = SentenceTransformer(MODEL_NAME)
    print("✅ Модель загружена")

    client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(COLLECTION_NAME)

    print(f"\n🔎 Запрос: {args.query}")
    results = search_documents(
        args.query,
        embedder=embedder,
        collection=collection,
        n_results=args.n_results,
    )
    if not results:
        print("⚠️ Ничего не найдено.")
        return

    print("\n📋 Результаты:")
    for i, r in enumerate(results, 1):
        snippet = r["text"][:160].replace("\n", " ")
        print(
            f"{i}. 📄 {r['source']} (часть {r['chunk']}), "
            f"релевантность {r['relevance']:.3f}\n   {snippet}..."
        )


if __name__ == "__main__":
    main()
