"""Экспериментальное исследование производительности."""

import random
import time
from typing import Any, Callable, List, Tuple

import matplotlib.pyplot as plt

from heap import MinHeap
from heapsort import heapsort


def benchmark_function(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Измерение времени выполнения функции.

    Args:
        func: Функция для измерения.
        *args: Аргументы функции.
        **kwargs: Ключевые аргументы.

    Returns:
        Кортеж (результат, время выполнения).
    """
    start = time.perf_counter()
    outcome = func(*args, **kwargs)
    end = time.perf_counter()
    return outcome, end - start


def construct_heap_by_insert(array: List[int]) -> MinHeap:
    """
    Построение кучи последовательной вставкой.

    Сложность: O(n log n)
    """
    heap = MinHeap()
    for element in array:
        heap.insert(element)
    return heap


def construct_heap_by_algorithm(array: List[int]) -> MinHeap:
    """
    Построение кучи алгоритмом build_heap.

    Сложность: O(n)
    """
    heap = MinHeap()
    heap.build_heap(array)
    return heap


def quick_sort_algorithm(array: List[int]) -> List[int]:
    """
    Быстрая сортировка для сравнения.

    Сложность:
        - В худшем случае: O(n²)
        - В среднем и лучшем: O(n log n)
    """
    if len(array) <= 1:
        return array
    pivot = array[len(array) // 2]
    left_part = [x for x in array if x < pivot]
    middle_part = [x for x in array if x == pivot]
    right_part = [x for x in array if x > pivot]
    return quick_sort_algorithm(left_part) + middle_part + quick_sort_algorithm(right_part)


def merge_sort_algorithm(array: List[int]) -> List[int]:
    """
    Сортировка слиянием для сравнения.

    Сложность: O(n log n)
    """
    if len(array) <= 1:
        return array

    midpoint = len(array) // 2
    left_half = merge_sort_algorithm(array[:midpoint])
    right_half = merge_sort_algorithm(array[midpoint:])

    merged = []
    i = j = 0
    while i < len(left_half) and j < len(right_half):
        if left_half[i] < right_half[j]:
            merged.append(left_half[i])
            i += 1
        else:
            merged.append(right_half[j])
            j += 1

    merged.extend(left_half[i:])
    merged.extend(right_half[j:])
    return merged


def evaluate_heap_construction_methods() -> None:
    """Эксперимент по построению кучи разными методами."""
    print('\nПостроение кучи')

    test_sizes = [100, 500, 1000, 5000, 10000]
    sequential_insertion_durations = []
    build_heap_durations = []

    for size in test_sizes:
        input_data = [random.randint(1, 10000) for _ in range(size)]

        _, time_seq = benchmark_function(construct_heap_by_insert, input_data)
        sequential_insertion_durations.append(time_seq)

        _, time_build = benchmark_function(construct_heap_by_algorithm, input_data)
        build_heap_durations.append(time_build)

        print(f'Размер: {size:5d} | '
              f'Вставка: {time_seq:.6f} сек | '
              f'Build_Heap: {time_build:.6f} сек')

    plt.figure(figsize=(10, 6))
    plt.plot(test_sizes, sequential_insertion_durations, 'ro-',
             label='Последовательная вставка', linewidth=2)
    plt.plot(test_sizes, build_heap_durations, 'bo-',
             label='Алгоритм build_heap', linewidth=2)
    plt.xlabel('Размер массива')
    plt.ylabel('Время (секунды)')
    plt.title('Сравнение методов построения кучи')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('heap_building_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def compare_sorting_algorithms() -> None:
    """Эксперимент по сравнению алгоритмов сортировки."""
    print('\nСравнение алгоритмов сортировки')

    test_sizes = [100, 500, 1000, 2000, 5000]
    heapsort_durations = []
    quicksort_durations = []
    mergesort_durations = []

    for size in test_sizes:
        input_data = [random.randint(1, 10000) for _ in range(size)]

        _, time_heap = benchmark_function(heapsort, input_data.copy())
        heapsort_durations.append(time_heap)

        _, time_quick = benchmark_function(quick_sort_algorithm, input_data.copy())
        quicksort_durations.append(time_quick)

        _, time_merge = benchmark_function(merge_sort_algorithm, input_data.copy())
        mergesort_durations.append(time_merge)

        print(f'Размер: {size:5d} | '
              f'Heapsort: {time_heap:.6f} сек | '
              f'Quicksort: {time_quick:.6f} сек | '
              f'Mergesort: {time_merge:.6f} сек')

    plt.figure(figsize=(12, 8))
    plt.plot(test_sizes, heapsort_durations, 'ro-', label='Heapsort', linewidth=2)
    plt.plot(test_sizes, quicksort_durations, 'go-', label='Quicksort', linewidth=2)
    plt.plot(test_sizes, mergesort_durations, 'bo-', label='Mergesort', linewidth=2)
    plt.xlabel('Размер массива')
    plt.ylabel('Время (секунды)')
    plt.title('Сравнение алгоритмов сортировки')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('sorting_algorithms_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def analyze_heap_operations() -> None:
    """Эксперимент по измерению времени операций кучи."""
    print('\nВремя операций кучи')

    test_sizes = [100, 500, 1000, 5000, 10000]
    avg_insert_durations = []
    avg_extract_durations = []

    for size in test_sizes:
        heap = MinHeap()

        start = time.perf_counter()
        for _ in range(size):
            heap.insert(random.randint(1, 10000))
        total_insert = time.perf_counter() - start
        avg_insert_durations.append(total_insert / size)

        start = time.perf_counter()
        while not heap.is_empty():
            heap.extract()
        total_extract = time.perf_counter() - start
        avg_extract_durations.append(total_extract / size)

        print(f'Размер: {size:5d} | '
              f'Вставка (средн.): {avg_insert_durations[-1]:.8f} сек | '
              f'Извлечение (средн.): {avg_extract_durations[-1]:.8f} сек')

    plt.figure(figsize=(10, 6))
    plt.plot(test_sizes, avg_insert_durations, 'go-',
             label='Вставка (среднее)', linewidth=2)
    plt.plot(test_sizes, avg_extract_durations, 'ro-',
             label='Извлечение (среднее)', linewidth=2)
    plt.xlabel('Размер кучи')
    plt.ylabel('Время на операцию (секунды)')
    plt.title('Зависимость времени операций от размера кучи')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('heap_operations_time.png',
                dpi=300, bbox_inches='tight')
    plt.show()