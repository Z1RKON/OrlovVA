"""
Модуль с базовыми рекурсивными функциями.
"""


def compute_factorial(n: int) -> int:
    """
    Вычисление факториала числа n рекурсивным методом.

    Args:
        n: Неотрицательное целое число

    Returns:
        Факториал числа n

    Raises:
        ValueError: Если n < 0
    """
    if n < 0:
        raise ValueError(
            'Факториал определен только для неотрицательных чисел'
        )
    if n == 0 or n == 1:
        return 1
    return n * compute_factorial(n - 1)


# Временная сложность: O(n).  Глубина рекурсии: O(n).


def fibonacci_simple(n: int) -> int:
    """
    Наивное вычисление n-го числа Фибоначчи.

    Args:
        n: Порядковый номер числа Фибоначчи

    Returns:
        n-е число Фибоначчи

    Raises:
        ValueError: Если n < 0
    """
    if n < 0:
        raise ValueError('Номер числа Фибоначчи должен быть неотрицательным')
    if n == 0:
        return 0
    if n == 1:
        return 1
    return fibonacci_simple(n - 1) + fibonacci_simple(n - 2)


# Временная сложность: O(2^n).  Глубина рекурсии: O(n).


def power_fast(a: float, n: int) -> float:
    """
    Быстрое возведение числа a в степень n через степень двойки.

    Args:
        a: Основание
        n: Показатель степени (неотрицательный)

    Returns:
        a в степени n

    Raises:
        ValueError: Если n < 0
    """
    if n < 0:
        raise ValueError('Показатель степени должен быть неотрицательным')
    if n == 0:
        return 1
    if n == 1:
        return a

    half_result = power_fast(a, n // 2)
    if n % 2 == 0:
        return half_result * half_result
    else:
        return a * half_result * half_result


# Временная сложность: O(log n).  Глубина рекурсии: O(log n).


if __name__ == '__main__':
    print('Факториал 5:', compute_factorial(5))
    print('10-е число Фибоначчи:', fibonacci_simple(10))
    print('2^10:', power_fast(2, 10))