import time
from typing import List, Tuple

import matplotlib.pyplot as plt

from recursion import factorial, fibonacci_naive, fast_power
from memoization import fibonacci_memo, compare_fibonacci_performance
from recursion_tasks import binary_search_recursive, hanoi_towers
from recursion_tasks import file_system_traversal


def run_performance_study() -> None:
    """
    Эксперимент по сравнению производительности рекурсивных алгоритмов.

    Проводит замеры времени выполнения и строит сравнительные графики.
    """
    print('\n=== ЭКСПЕРИМЕНТАЛЬНОЕ ИССЛЕДОВАНИЕ ===\n')

    compare_fibonacci_performance(35)

    print('\n--- Построение графика времени выполнения ---')
    n_vals, naive_durations, memo_durations = collect_fibonacci_benchmarks()
    render_performance_graph(n_vals, naive_durations, memo_durations)

    display_complexity_summary()

    showcase_directory_walk()


def collect_fibonacci_benchmarks() -> Tuple[List[int], List[float], List[float]]:
    """
    Измеряет время выполнения наивной и мемоизированной версий Фибоначчи.

    Returns:
        Tuple с n_vals, naive_durations, memo_durations
    """
    n_range = list(range(1, 25))
    naive_results = []
    memo_results = []

    for n in n_range:
        start = time.perf_counter()
        fibonacci_naive(n)
        naive_elapsed = time.perf_counter() - start
        naive_results.append(naive_elapsed)

        start = time.perf_counter()
        fibonacci_memo(n)
        memo_elapsed = time.perf_counter() - start

        if memo_elapsed < 0.0001:
            repeat_count = 1000
            start = time.perf_counter()
            for _ in range(repeat_count):
                fibonacci_memo(n)
            memo_elapsed = (time.perf_counter() - start) / repeat_count

        memo_results.append(memo_elapsed)

    return n_range, naive_results, memo_results


def render_performance_graph(
    n_values: List[int],
    naive_times: List[float],
    memo_times: List[float]
) -> None:
    """
    Строит графики сравнения производительности.

    Args:
        n_values: Значения n для оси X
        naive_times: Времена наивной реализации
        memo_times: Времена мемоизированной реализации
    """
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(n_values, naive_times, label='Наивная рекурсия',
             marker='o', color='red')
    plt.plot(n_values, memo_times, label='С мемоизацией',
             marker='s', color='green')
    plt.xlabel('n')
    plt.ylabel('Время (секунды)')
    plt.title('Сравнение времени вычисления чисел Фибоначчи')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(n_values, naive_times, label='Наивная рекурсия',
             marker='o', color='red')
    plt.plot(n_values, memo_times, label='С мемоизацией',
             marker='s', color='green')
    plt.xlabel('n')
    plt.ylabel('Время (секунды)')
    plt.title('Логарифмическая шкала времени')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig('fibonacci_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    print('График сохранен как fibonacci_performance.png')


def display_complexity_summary() -> None:
    """Выводит анализ временной сложности алгоритмов."""
    print('\n--- Анализ сложности алгоритмов ---')
    print('Наивная рекурсия Фибоначчи: O(2^n) - экспоненциальная сложность')
    print('Фибоначчи с мемоизацией: O(n) - линейная сложность')
    print('Быстрое возведение в степень: O(log n) - логарифмическая сложность')
    print('Факториал: O(n) - линейная сложность')
    print('Бинарный поиск: O(log n) - логарифмическая сложность')


def showcase_directory_walk() -> None:
    """Демонстрирует рекурсивный обход файловой системы."""
    print('\n--- Обход файловой системы ---')
    max_level = 3
    print(f'Обход с максимальной глубиной {max_level}:')
    file_system_traversal('.', max_depth=max_level)


def demonstrate_all_implementations() -> None:
    """Демонстрация всех реализованных рекурсивных функций."""
    print('=== ДЕМОНСТРАЦИЯ ВСЕХ ФУНКЦИЙ ===\n')

    print('1. Факториал 5:', factorial(5))

    print('2. 10-е число Фибоначчи:')
    print('   Наивный метод:', fibonacci_naive(10))
    print('   С мемоизацией:', fibonacci_memo(10))

    print('3. Быстрое возведение в степень:')
    print('   2^10 =', fast_power(2, 10))
    print('   3^5 =', fast_power(3, 5))

    test_array = [1, 3, 5, 7, 9, 11, 13]
    target_val = 7
    found_index = binary_search_recursive(test_array, target_val)
    print(f'4. Бинарный поиск {target_val} в {test_array}: '
          f'индекс {found_index}')

    print('5. Ханойские башни для 3 дисков:')
    hanoi_towers(3)

    print('6. Обход файловой системы (текущая директория, глубина 1):')
    file_system_traversal('.', max_depth=1)


def print_system_details() -> None:
    """Вывод информации о системе для воспроизводимости экспериментов."""
    pc_info = """
ХАРАКТЕРИСТИКИ ПК ДЛЯ ТЕСТИРОВАНИЯ:
- Процессор: Intel Core i7-6500U @ 2.50GHz
- Оперативная память: 8 GB
- ОС: Windows 10 PRO
- Python: 3.12.8
"""
    print(pc_info)


if __name__ == '__main__':
    print_system_details()
    demonstrate_all_implementations()
    run_performance_study()