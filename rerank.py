from typing import List
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from loguru import logger
import torch

class Reranker:
    """
    Класс для переранжирования документов с использованием HuggingFaceCrossEncoder.
    Улучшает релевантность контекста перед подачей в LLM.
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", top_n: int = 3):
        """
        Инициализирует reranker.

        Args:
            model_name (str): Название cross-encoder модели на Hugging Face.
            top_n (int): Количество лучших документов, которые нужно вернуть.
        """
        self.top_n = top_n
        logger.info(f"Загрузка cross-encoder модели: {model_name}...")
        
        self.model = HuggingFaceCrossEncoder(
            model_name=model_name,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )
        logger.success("Cross-encoder модель успешно загружена.")

    def rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Выполняет переранжирование списка документов на основе запроса.

        Args:
            query (str): Исходный поисковый запрос.
            documents (List[Document]): Список документов, полученных от retriever.

        Returns:
            List[Document]: Отсортированный и урезанный список документов.
        """
        if not documents:
            return []
            
        logger.info(f"Reranking {len(documents)} документов с top_n={self.top_n}...")
        
        # Создаем пары [запрос, документ] для модели
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Получаем оценки релевантности
        scores = self.model.score(pairs)
        
        # Соединяем документы с их оценками
        doc_scores = list(zip(documents, scores))
        
        # Сортируем по оценке в убывающем порядке
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Отбираем top_n документов
        reranked_docs = [doc for doc, score in doc_scores[:self.top_n]]
        
        logger.success(f"Документы переранжированы. Возвращено {len(reranked_docs)} лучших.")
        return reranked_docs
