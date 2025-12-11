from heap import MaxHeap, MinHeap
from heapsort import heapsort, heapsort_inplace
from performance_analysis import (
    run_heap_building_experiment,
    run_operations_experiment,
    run_sorting_experiment
)
from priority_queue import PriorityQueue
from visualization import print_heap


def show_system_specs() -> None:
    """Вывод информации о системе."""
    print('''
ХАРАКТЕРИСТИКИ ПК ДЛЯ ТЕСТИРОВАНИЯ:
- Процессор: Intel Core i7-6500U @ 2.50GHz
- Оперативная память: 8 GB 
- ОС: Windows 10 PRO
- Python: 3.12.8
''')


def demonstrate_heap_behavior() -> None:
    """Демонстрация работы кучи."""
    print('Демонстрация работы кучи')

    sample_data = [10, 5, 15, 3, 7]

    print('\n1. MIN-HEAP (последовательная вставка):')
    min_pq = MinHeap()
    for element in sample_data:
        print(f'\nВставляем {element}:')
        min_pq.insert(element)
        print_heap(min_pq)

    print('\nMIN-HEAP (извлечение):')
    while not min_pq.is_empty():
        removed = min_pq.extract()
        print(f'\nИзвлекаем {removed}:')
        if not min_pq.is_empty():
            print_heap(min_pq)

    print('\n2. MAX-HEAP (последовательная вставка):')
    max_pq = MaxHeap()
    for element in sample_data:
        print(f'\nВставляем {element}:')
        max_pq.insert(element)
        print_heap(max_pq)

    print('\nMAX-HEAP (извлечение):')
    while not max_pq.is_empty():
        removed = max_pq.extract()
        print(f'\nИзвлекаем {removed}:')
        if not max_pq.is_empty():
            print_heap(max_pq)


def demonstrate_heapsort_usage() -> None:
    """Демонстрация пирамидальной сортировки."""
    print('\nДемонстрация сортировки')

    original_array = [9, 3, 7, 1, 8, 2, 5, 6, 4]
    print(f'Исходный массив: {original_array}')

    sorted_result = heapsort(original_array)
    print(f'Отсортированный массив (heapsort): {sorted_result}')

    array_copy = original_array.copy()
    heapsort_inplace(array_copy)
    print(f'Отсортированный массив (in-place): {array_copy}')


def demonstrate_priority_queue_usage() -> None:
    """Демонстрация приоритетной очереди."""
    print('\nДемонстрация приоритетной очереди')

    task_queue = PriorityQueue()

    job_list = [
        ('Задача A', 3),
        ('Задача B', 1),
        ('Задача C', 5),
        ('Задача D', 2),
        ('Задача E', 8)
    ]

    print('Добавление задач в очередь:')
    for job, priority_level in job_list:
        task_queue.enqueue(job, priority_level)
        print(f'  Добавлено: "{job}" с приоритетом {priority_level}')

    print('\nИзвлечение задач по приоритету:')
    while not task_queue.is_empty():
        next_task = task_queue.dequeue()
        print(f'  Выполняется: "{next_task}"')


def launch_main_demo() -> None:
    """Главная функция программы."""
    show_system_specs()

    demonstrate_heap_behavior()
    demonstrate_heapsort_usage()
    demonstrate_priority_queue_usage()

    run_heap_building_experiment()
    run_sorting_experiment()
    run_operations_experiment()

    print('\nГрафики сохранены в файлах:')
    print('heap_building_comparison.png')
    print('sorting_algorithms_comparison.png')
    print('heap_operations_time.png')


if __name__ == "__main__":
    launch_main_demo()