import os
import re
import string
import time
from typing import Any, Dict, List, Literal, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
import torch
from dotenv import find_dotenv, load_dotenv
from loguru import logger
from langchain_chroma import Chroma
import json

load_dotenv(find_dotenv())

start_time = time.time()

PUNCTUATION_PATTERN = re.compile(f"[{re.escape(string.punctuation)}]")
WHITESPACE_PATTERN = re.compile(r"\s+")


class ChatWithAI:
    """
    Класс для создания чат-бота, который использует RAG (Retrieval-Augmented Generation)
    для ответов на вопросы. Он интегрируется с векторной базой данных ChromaDB и
    большими языковыми моделями (LLM) от провайдеров Deepseek или Qwen.

    Args:
        provider (Literal["deepseek", "qwen"]):
            Провайдер LLM. Определяет, какая модель будет использоваться для генерации
            ответов. По умолчанию "qwen".

    Raises:
        ValueError: Если указан неподдерживаемый провайдер.
    """

    def __init__(self, provider: Literal["deepseek", "qwen"] = "qwen"):
        """
        Инициализирует ChatWithAI, настраивая эмбеддинги, LLM и подключение к ChromaDB.
        """
        self.provider = provider
        self.embeddings = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL_NAME"),
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        if provider == "deepseek":
            self.llm = ChatOllama(
                model=os.getenv("DEEPSEEK_MODEL_NAME"),
                temperature=0.1,
            )
        elif provider == "qwen":
            self.llm = ChatOllama(
                model=os.getenv("QWEN_MODEL_NAME"),
                temperature=0.1,
            )
        else:
            raise ValueError(f"Неподдерживаемый провайдер: {provider}")

        self.chroma_db = Chroma(
            persist_directory=os.getenv("CHROMA_PATH"),
            embedding_function=self.embeddings,
            collection_name=os.getenv("CHROMA_COLLECTION_NAME"),
        )

    def get_relevant_context(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Выполняет семантический поиск по векторной базе данных ChromaDB для нахождения
        наиболее релевантных документов по запросу.

        Args:
            query (str): Текст запроса для поиска релевантного контекста.
            k (int): Максимальное количество возвращаемых документов. По умолчанию 3.

        Returns:
            List[Dict[str, Any]]:
                Список словарей, где каждый словарь представляет релевантный документ
                и содержит его текст и метаданные.
        """
        try:
            results = self.chroma_db.similarity_search(
                self.normalize_query(query), k=k
            )
            return [
                {
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                }
                for doc in results
            ]
        except Exception as e:
            logger.error(f"Ошибка при получении контекста: {e}")
            return []

    @staticmethod
    def normalize_query(text: str) -> str:
        """
        Предварительная обработка текста запроса для улучшения качества поиска.
        Удаляет пунктуацию, лишние пробелы и приводит текст к нижнему регистру.

        Args:
            text (str): Входной текст запроса.

        Returns:
            str: Нормализованный текст.

        Raises:
            ValueError: Если входной текст не является строкой.
        """
        if not isinstance(text, str):
            raise ValueError("Входной текст должен быть строкой")

        # Удаление знаков препинания
        text = PUNCTUATION_PATTERN.sub(" ", text)
        # Удаление переносов строк и лишних пробелов
        text = WHITESPACE_PATTERN.sub(" ", text)
        # Приведение к нижнему регистру
        return text.lower().strip()


def main():
    """
    Основная функция для демонстрации работы ChatWithAI.
    В данный момент содержит закомментированный пример использования.
    """
    logger.success("Старт")

    # Пример использования:
    # chat_bot = ChatWithAI(provider="qwen")
    # context = chat_bot.get_relevant_context("ваш вопрос")
    # print(context)

if __name__ == "__main__":
    main()