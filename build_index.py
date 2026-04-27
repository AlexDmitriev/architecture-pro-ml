import os
from pathlib import Path
from typing import List, Dict, Any
import uuid
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

load_dotenv()

# Конфигурация
CHUNK_SIZE = 500  # размер чанка в токенах (примерно 350-400 слов)
CHUNK_OVERLAP = 50  # перекрытие в токенах

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
COLLECTION_NAME = "document_chunks"

# Инициализация эмбеддера
print("Загрузка модели multilingual-e5-large-instruct...")
embedder = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
print("Модель загружена")

# Подключение к Chroma в Docker
# Отключаем telemetry на клиенте, чтобы убрать лишние ошибки capture().
client = chromadb.HttpClient(
    host=CHROMA_HOST,
    port=CHROMA_PORT,
    settings=Settings(anonymized_telemetry=False),
)

# Создаем или получаем коллекцию
try:
    # Опционально: удаляем старую коллекцию для переиндексации
    # client.delete_collection(COLLECTION_NAME)
    collection = client.get_collection(COLLECTION_NAME)
    print(f"Используем существующую коллекцию: {COLLECTION_NAME}")
except:
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"Создана новая коллекция: {COLLECTION_NAME}")

# Настройка сплиттера LangChain
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,  # можно заменить на токенизатор
    separators=["\n\n", "\n", " ", ""],
)

def get_instruction_for_chunk(chunk_text: str, filename: str, chunk_index: int) -> str:
    """
    Формирует инструкцию для модели multilingual-e5-large-instruct
    Это улучшает качество эмбеддингов для поиска
    """
    return f"Instruct: Given the following excerpt from document '{filename}' (chunk {chunk_index}), retrieve relevant information for answering questions.\n\nQuery: {chunk_text}"

def process_file(file_path: Path) -> List[Dict[str, Any]]:
    """Обрабатывает один файл: читает, разбивает на чанки, готовит метаданные"""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Разбиваем на чанки с помощью LangChain
    chunks = text_splitter.split_text(text)
    
    result = []
    for i, chunk in enumerate(chunks):
        # Пропускаем слишком короткие чанки
        if len(chunk.strip()) < 50:
            continue
            
        result.append({
            "text": chunk,
            "source": file_path.name,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "char_start": None,  # можно добавить при необходимости
            "char_end": None
        })
    
    return result

def add_to_chroma(chunks: List[Dict[str, Any]], batch_size: int = 100):
    """Добавляет чанки в Chroma с эмбеддингами"""
    total = len(chunks)
    print(f"\nДобавление {total} чанков в Chroma...")
    
    # Добавляем батчами для эффективности
    for i in range(0, total, batch_size):
        batch = chunks[i:i+batch_size]
        
        ids = []
        texts = []
        metadatas = []
        embeddings = []
        
        for chunk in batch:
            chunk_id = f"{chunk['source']}_chunk_{chunk['chunk_index']}_{uuid.uuid4().hex[:8]}"
            ids.append(chunk_id)
            texts.append(chunk["text"])
            metadatas.append({
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
                "total_chunks": chunk["total_chunks"]
            })
            
            # Создаем эмбеддинг с инструкцией
            instruction = get_instruction_for_chunk(
                chunk["text"], 
                chunk["source"], 
                chunk["chunk_index"]
            )
            embedding = embedder.encode(instruction).tolist()
            embeddings.append(embedding)
        
        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )
        
        print(f"   Добавлено {min(i+batch_size, total)}/{total} чанков")
    
    print(f"Готово. Все {total} чанков индексированы")


def get_indexed_sources() -> set[str]:
    """Возвращает имена файлов (source), уже находящихся в коллекции."""
    count = collection.count()
    if count == 0:
        return set()

    # В некоторых версиях Chroma при include=[] metadatas может быть None.
    existing = collection.get(include=["metadatas"])
    sources: set[str] = set()
    metadatas = existing.get("metadatas") or []
    for meta in metadatas:
        if not meta:
            continue
        source = meta.get("source")
        if source:
            sources.add(source)
    return sources

def main():
    data_dir = Path("./knowledge_base")
    
    if not data_dir.exists():
        print(f"Папка {data_dir} не существует. Создайте её и положите туда .txt файлы")
        return
    
    txt_files = list(data_dir.glob("*.txt")) + list(data_dir.glob("*.md"))
    
    if not txt_files:
        print(f"Нет .txt или .md файлов в {data_dir}")
        return
    
    print(f"Найдено {len(txt_files)} файлов:")
    for f in txt_files:
        print(f"   - {f.name}")

    indexed_sources = get_indexed_sources()
    if indexed_sources:
        print(f"\nВ коллекции уже есть файлов: {len(indexed_sources)}")
    else:
        print("\nКоллекция пока пустая, будет выполнена полная индексация.")

    files_to_index = [f for f in txt_files if f.name not in indexed_sources]
    skipped_count = len(txt_files) - len(files_to_index)
    if skipped_count:
        print(f"Пропущено уже загруженных файлов: {skipped_count}")
    if not files_to_index:
        print("Новых файлов для индексации нет.")
        return

    print(f"Новых файлов для индексации: {len(files_to_index)}")

    all_chunks = []
    for file_path in files_to_index:
        print(f"\nОбработка {file_path.name}...")
        chunks = process_file(file_path)
        print(f"   → {len(chunks)} чанков (размер: {CHUNK_SIZE}, перекрытие: {CHUNK_OVERLAP})")
        all_chunks.extend(chunks)
    
    if all_chunks:
        add_to_chroma(all_chunks)
        
        # Показываем пример
        sample = all_chunks[0]
        print(f"\nПример чанка:")
        print(f"   Источник: {sample['source']}")
        print(f"   Позиция: чанк {sample['chunk_index']} из {sample['total_chunks']}")
        print(f"   Текст: {sample['text'][:200]}...")
        
        # Статистика по коллекции
        count = collection.count()
        print(f"\nСтатистика коллекции '{COLLECTION_NAME}':")
        print(f"   Всего чанков: {count}")
    else:
        print("Нет чанков для индексации")

def test_search(query: str = None):
    """Тестовый поиск (опционально)"""
    if query is None:
        query = input("\nВведите поисковый запрос (или Enter для пропуска): ").strip()
        if not query:
            return
    
    print(f"\nПоиск: '{query}'")
    
    # Добавляем инструкцию для поиска
    instruction = f"Instruct: {query}"
    query_embedding = embedder.encode(instruction).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )
    
    print("\nРезультаты:")
    for i, (doc, meta, dist) in enumerate(zip(
        results['documents'][0], 
        results['metadatas'][0], 
        results['distances'][0]
    ), 1):
        print(f"\n{i}. Источник: {meta['source']} (чанк {meta['chunk_index']})")
        print(f"   Релевантность: {1-dist:.3f}")
        print(f"   Текст: {doc[:150]}...")

if __name__ == "__main__":
    main()
    
    # Опционально: тестовый поиск
    if input("\nПротестировать поиск? (y/N): ").lower() == 'y':
        test_search()