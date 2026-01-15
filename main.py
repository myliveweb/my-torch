import os
import re
import string
import time
from typing import Any, Dict, List, Literal, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
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
        
        self.retriever = self.chroma_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

    def ask(self, query: str) -> str:
        """
        Выполняет полный RAG-пайплайн: извлекает контекст, форматирует промпт и генерирует ответ.
        """
        
        # Нормализуем запрос перед использованием
        normalized_query = self.normalize_query(query)

        # Создаем шаблон промпта
        template = """
        Вы — ассистент для ответов на вопросы. Используйте следующие фрагменты контекста, 
        чтобы ответить на вопрос. Если вы не знаете ответа, просто скажите, что не знаете.
        Отвечайте кратко и по делу.

        Контекст:
        {context}

        Вопрос:
        {question}

        Ответ:
        """
        prompt = ChatPromptTemplate.from_template(template)

        # Форматируем документы для передачи в LLM
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Создаем RAG-цепочку
        rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        logger.info("Выполнение RAG-цепочки...")
        start_rag_time = time.time()
        
        # Запускаем цепочку с нормализованным запросом
        result = rag_chain.invoke(normalized_query)
        
        end_rag_time = time.time()
        logger.success(f"RAG-цепочка выполнена за {end_rag_time - start_rag_time:.2f} сек")
        
        return result

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
    Основная функция для интерактивного общения с ChatWithAI.
    Создает цикл, в котором пользователь может задавать вопросы.
    """
    logger.success("Запуск интерактивного чат-бота...")
    chat_bot = ChatWithAI(provider="qwen")
    logger.info("Чат-бот готов. Введите ваш вопрос. Для выхода введите 'exit' или 'quit'.")

    while True:
        question = input("\nВаш вопрос: ")
        if question.lower() in ["exit", "quit"]:
            print("Завершение работы.")
            break
        
        if not question.strip():
            continue

        answer = chat_bot.ask(question)
        
        print("\n" + "="*50)
        print(f"Ответ: {answer}")
        print("="*50)


if __name__ == "__main__":
    main()