"""
Модуль с оптимизированными рекурсивными функциями с использованием мемоизации.
"""
import timeit
from typing import Dict

from recursion import fibonacci_naive


def fibonacci_cached(
    n: int,
    cache: Dict[int, int] = None
) -> int:
    """
    Вычисление n-го числа Фибоначчи с мемоизацией.

    Args:
        n: Порядковый номер числа Фибоначчи
        cache: Словарь для хранения вычисленных значений

    Returns:
        n-е число Фибоначчи
    """
    if cache is None:
        cache = {}

    if n in cache:
        return cache[n]

    if n == 0:
        return 0
    if n == 1:
        return 1

    cache[n] = (
        fibonacci_cached(n - 1, cache) +
        fibonacci_cached(n - 2, cache)
    )
    return cache[n]


# Временная сложность: O(n).  Глубина рекурсии: O(n).


def evaluate_fibonacci_speed(fib_index: int = 35) -> None:
    """
    Сравнение производительности наивной и мемоизированной версий.

    Args:
        fib_index: Номер числа Фибоначчи для тестирования
    """
    naive_duration = timeit.timeit(lambda: fibonacci_naive(fib_index), number=1)
    cached_duration = timeit.timeit(lambda: fibonacci_cached(fib_index), number=1)

    naive_value = fibonacci_naive(fib_index)
    cached_value = fibonacci_cached(fib_index)

    print(f'Результат для n={fib_index}:')
    print(f'Наивная версия: {naive_value}, время: {naive_duration:.6f} сек')
    print(f'Мемоизированная версия: {cached_value}, время: {cached_duration:.6f} сек')

    if cached_duration > 0:
        acceleration = naive_duration / cached_duration
        print(f'Ускорение: {acceleration:.2f} раз')
    else:
        print('Ускорение: > 1000 раз')


def benchmark_various_n() -> None:
    """Измерение времени для разных значений n."""
    input_values = [10, 20, 30, 35]

    print('\nСравнение времени выполнения для разных n:')
    print('n\tНаивная (сек)\tМемоизированная (сек)\tУскорение')
    print('-' * 60)

    for n_val in input_values:
        naive_t = timeit.timeit(
            lambda: fibonacci_naive(n_val), number=1
        )
        cached_t = timeit.timeit(lambda: fibonacci_cached(n_val), number=1)

        if cached_t > 0:
            speed_factor = naive_t / cached_t
        else:
            speed_factor = float('inf')

        if naive_t > 1:
            naive_str = f'{naive_t:.3f}'
        else:
            naive_str = f'{naive_t:.6f}'

        print(
            f'{n_val}\t{naive_str}\t\t{cached_t:.6f}\t\t\t{speed_factor:.0f}x'
        )


if __name__ == '__main__':
    evaluate_fibonacci_speed(35)
    benchmark_various_n()