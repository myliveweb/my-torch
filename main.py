import os
import re
import string
import time
from typing import Any, Dict, List, Literal, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
import torch
from dotenv import find_dotenv, load_dotenv
from loguru import logger
from langchain_chroma import Chroma
import json

from rerank import Reranker

load_dotenv(find_dotenv())

start_time = time.time()

PUNCTUATION_PATTERN = re.compile(f"[{re.escape(string.punctuation)}]")
WHITESPACE_PATTERN = re.compile(r"\s+")


class ChatWithAI:
    def __init__(self, provider: Literal["deepseek", "qwen"] = "qwen"):
        logger.info("Инициализация ChatWithAI...")
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
        
        # Базовый ретривер для сравнения
        self.retriever_base = self.chroma_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Ретривер для reranking, получает больше документов
        self.retriever_for_reranking = self.chroma_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 20}
        )
        
        # Инициализация reranker
        self.reranker = Reranker(top_n=3)
        logger.success("ChatWithAI полностью инициализирован.")

    def _create_rag_chain(self, retriever, custom_context_processor=None):
        """Создает RAG-цепочку с заданным ретривером и обработчиком контекста."""
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

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        context_chain = retriever
        if custom_context_processor:
            context_chain = context_chain | custom_context_processor

        rag_chain = (
            {"context": context_chain | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain

    def ask(self, query: str) -> str:
        """Стандартный RAG-пайплайн без reranking."""
        logger.info("Выполнение RAG-цепочки БЕЗ reranking...")
        normalized_query = self.normalize_query(query)
        rag_chain = self._create_rag_chain(self.retriever_base)
        
        start_rag_time = time.time()
        result = rag_chain.invoke(normalized_query)
        end_rag_time = time.time()
        logger.success(f"Цепочка БЕЗ reranking выполнена за {end_rag_time - start_rag_time:.2f} сек")
        
        return result

    def ask_with_rerank(self, query: str) -> str:
        """RAG-пайплайн с reranking (более простая реализация)."""
        logger.info("Выполнение RAG-цепочки С reranking...")
        start_rag_time = time.time()
        
        normalized_query = self.normalize_query(query)

        # 1. Получаем документы
        logger.info("Шаг 1: Извлечение документов...")
        retrieved_docs = self.retriever_for_reranking.invoke(normalized_query)
        
        # 2. Переранжируем документы
        logger.info("Шаг 2: Переранжирование документов...")
        reranked_docs = self.reranker.rerank_documents(normalized_query, retrieved_docs)
        
        # 3. Формируем контекст и финальный промпт
        logger.info("Шаг 3: Генерация ответа...")
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        context = format_docs(reranked_docs)

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

        final_chain = prompt | self.llm | StrOutputParser()

        result = final_chain.invoke({
            "context": context,
            "question": normalized_query
        })
        
        end_rag_time = time.time()
        logger.success(f"Цепочка С reranking выполнена за {end_rag_time - start_rag_time:.2f} сек")
        
        return result

    @staticmethod
    def normalize_query(text: str) -> str:
        if not isinstance(text, str):
            raise ValueError("Входной текст должен быть строкой")
        text = PUNCTUATION_PATTERN.sub(" ", text)
        text = WHITESPACE_PATTERN.sub(" ", text)
        return text.lower().strip()


def main():
    """
    Основная функция для сравнения результатов RAG с reranking и без.
    """
    logger.success("Запуск сравнения RAG-пайплайнов...")
    chat_bot = ChatWithAI(provider="qwen")
    
    question = "Каковы основные методы и инструменты OSINT?"
    logger.info(f"Тестовый вопрос: '{question}'")

    # --- Вызов без reranking ---
    answer_base = chat_bot.ask(question)
    print("\n" + "="*80)
    print("Ответ БЕЗ Reranking:")
    print(answer_base)
    print("="*80)

    # --- Вызов с reranking ---
    answer_reranked = chat_bot.ask_with_rerank(question)
    print("\n" + "="*80)
    print("Ответ С Reranking:")
    print(answer_reranked)
    print("="*80 + "\n")


if __name__ == "__main__":
    main()