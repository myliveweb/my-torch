# -*- coding: utf-8 -*-
import sys
from googlesearch import search

def main():
    """
    Основная функция для выполнения поиска в Google из командной строки.
    """
    # Шаг 1: Анализ запроса (получаем его из аргументов командной строки)
    if len(sys.argv) < 2:
        print("Пожалуйста, укажите поисковый запрос.")
        print("Пример: uv run google-search.py 'что такое API?'")
        sys.exit(1)
        
    term = " ".join(sys.argv[1:])
    print(f"Выполняю поиск по запросу: '{term}'\n")

    try:
        # Шаг 2 и 3: Формулирование и выполнение поискового запроса
        search_results_generator = search(term, num_results=1, lang='ru', sleep_interval=2)

        # Преобразуем генератор в список, чтобы проверить, есть ли результаты
        results_list = list(search_results_generator)

        # Шаг 4: Анализ результатов (выводим их на экран)
        if results_list:
            print("Первый найденный результат:")
            print(f"1. {results_list[0]}")
        else:
            print("По вашему запросу ничего не найдено. Возможно, Google временно заблокировал запросы.")
            
    except Exception as e:
        print(f"Произошла ошибка во время поиска: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


