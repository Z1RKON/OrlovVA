"""
Модуль для визуализации результатов тестирования.
"""

import matplotlib.pyplot as plt
import numpy as np
from performance_test import run_performance_tests


def visualize_time_by_size(results: dict, input_type: str = 'random'):
    """
    Построение графика зависимости времени от размера массива.

    Args:
        results: Результаты тестирования
        input_type: Тип данных для построения графика
    """
    plt.figure(figsize=(12, 8))

    algorithms = list(results.keys())
    array_sizes = sorted(list(next(iter(results.values()))[input_type].keys()))

    for algorithm in algorithms:
        durations = []
        for size in array_sizes:
            time_value = results[algorithm][input_type][size]['time']
            durations.append(time_value)

        plt.plot(array_sizes, durations, marker='o', label=algorithm, linewidth=2)

    plt.xlabel('Размер массива')
    plt.ylabel('Время выполнения (секунды)')
    plt.title(f'Зависимость времени сортировки от размера массива ({input_type} данные)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(f'time_vs_size_{input_type}.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_time_by_data_kind(results: dict, fixed_size: int = 5000):
    """
    Построение графика зависимости времени от типа данных.

    Args:
        results: Результаты тестирования
        fixed_size: Фиксированный размер массива
    """
    plt.figure(figsize=(12, 8))

    algorithm_names = list(results.keys())
    data_categories = ['random', 'sorted', 'reversed', 'almost_sorted']

    x_positions = np.arange(len(data_categories))
    bar_width = 0.15

    for idx, algo in enumerate(algorithm_names):
        execution_times = []
        for category in data_categories:
            if fixed_size in results[algo][category]:
                t = results[algo][category][fixed_size]['time']
                execution_times.append(t)
            else:
                execution_times.append(0)

        offset = x_positions + idx * bar_width - bar_width * (len(algorithm_names) - 1) / 2
        plt.bar(offset, execution_times, bar_width, label=algo)

    plt.xlabel('Тип данных')
    plt.ylabel('Время выполнения (секунды)')
    plt.title(f'Сравнение времени сортировки для разных типов данных (размер {fixed_size})')
    plt.xticks(x_positions, data_categories)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'time_vs_datatype_size_{fixed_size}.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def generate_summary_report(results: dict):
    """Создание сводной таблицы результатов."""
    print("\n" + "="*80)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("="*80)

    test_sizes = sorted(list(next(iter(results.values()))['random'].keys()))
    data_types = ['random', 'sorted', 'reversed', 'almost_sorted']

    for dtype in data_types:
        print(f"\n{dtype.upper()} ДАННЫЕ:")
        print("-" * 60)
        header = f"{'Алгоритм':<15} " + "".join([f"{size:>12}" for size in test_sizes])
        print(header)
        print("-" * 60)

        for algo in results.keys():
            row = f"{algo:<15}"
            for size in test_sizes:
                if size in results[algo][dtype]:
                    time_val = results[algo][dtype][size]['time']
                    row += f"{time_val:>12.6f}"
                else:
                    row += f"{'N/A':>12}"
            print(row)


if __name__ == "__main__":
    # Запуск тестов и построение графиков
    print("Running performance tests for visualization...")
    benchmark_outcomes = run_performance_tests(sizes=[100, 1000, 5000, 10000], num_runs=1)

    # Построение графиков
    visualize_time_by_size(benchmark_outcomes, 'random')
    visualize_time_by_data_kind(benchmark_outcomes, 5000)

    # Создание сводной таблицы
    generate_summary_report(benchmark_outcomes)