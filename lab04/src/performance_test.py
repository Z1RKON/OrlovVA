"""
Модуль для тестирования производительности алгоритмов сортировки.
"""

import time
import copy
from typing import List, Dict, Tuple
from lab04.src.sorts import bubble_sort, selection_sort, insertion_sort, merge_sort, quick_sort, is_sorted
from lab04.src.generate_data import generate_test_data


# Словарь с алгоритмами сортировки
SORTING_METHODS = {
    'bubble_sort': bubble_sort,
    'selection_sort': selection_sort,
    'insertion_sort': insertion_sort,
    'merge_sort': merge_sort,
    'quick_sort': quick_sort
}


def benchmark_sorting(sort_algorithm, array: List[int]) -> Tuple[float, List[int]]:
    """
    Измерение времени выполнения сортировки.

    Returns:
        Кортеж (время в секундах, отсортированный массив)
    """
    input_copy = copy.deepcopy(array)
    start = time.time()
    result_array = sort_algorithm(input_copy)
    end = time.time()
    return end - start, result_array


def execute_performance_benchmark(sizes: List[int] = None, repetitions: int = 1) -> Dict:
    """
    Запуск тестов производительности для всех алгоритмов и типов данных.

    Args:
        sizes: Список размеров массивов для тестирования
        repetitions: Количество запусков для усреднения

    Returns:
        Словарь с результатами тестов
    """
    if sizes is None:
        sizes = [100, 1000, 5000, 10000]

    # Генерация тестовых данных
    datasets = generate_test_data(sizes)

    performance_results = {}

    for method_name, sorting_func in SORTING_METHODS.items():
        performance_results[method_name] = {}
        print(f"Testing {method_name}...")

        for data_kind, size_map in datasets.items():
            performance_results[method_name][data_kind] = {}

            for array_size, original_data in size_map.items():
                cumulative_time = 0.0
                correctness_ok = True

                for attempt in range(repetitions):
                    elapsed, sorted_output = benchmark_sorting(sorting_func, original_data)
                    cumulative_time += elapsed

                    # Проверка корректности сортировки (только в первом запуске)
                    if attempt == 0 and not is_sorted(sorted_output):
                        correctness_ok = False

                average_duration = cumulative_time / repetitions
                performance_results[method_name][data_kind][array_size] = {
                    'time': average_duration,
                    'correct': correctness_ok
                }

                marker = "✓" if correctness_ok else "✗"
                print(f"  {data_kind} (size {array_size}): {average_duration:.6f}s {marker}")

    return performance_results


def validate_sorting_implementations():
    """Проверка корректности всех алгоритмов сортировки."""
    sample_input = [64, 34, 25, 12, 22, 11, 90]
    expected_output = sorted(sample_input)

    print("Verifying sorting algorithms:")
    for name, func in SORTING_METHODS.items():
        outcome = func(sample_input.copy())
        valid = outcome == expected_output
        status_label = "PASS" if valid else "FAIL"
        print(f"  {name}: {status_label}")
        if not valid:
            print(f"    Expected: {expected_output}")
            print(f"    Got: {outcome}")


if __name__ == "__main__":
    # Проверка корректности сортировки
    validate_sorting_implementations()
    print()

    # Запуск тестов производительности
    print("Running performance tests...")
    benchmark_data = execute_performance_benchmark(sizes=[100, 1000, 5000], repetitions=1)