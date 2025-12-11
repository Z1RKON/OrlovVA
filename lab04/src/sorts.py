"""
Модуль с реализацией алгоритмов сортировки.
"""

from typing import List


def sort_bubble(arr: List[int]) -> List[int]:
    """
    Сортировка пузырьком.

    Временная сложность:
    - Худший случай: O(n²)
    - Средний случай: O(n²) 
    - Лучший случай: O(n)

    Пространственная сложность: O(1)
    """
    length = len(arr)
    for i in range(length):
        exchanged = False
        for j in range(0, length - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                exchanged = True
        if not exchanged:
            break
    return arr


def sort_selection(arr: List[int]) -> List[int]:
    """
    Сортировка выбором.

    Временная сложность:
    - Худший случай: O(n²)
    - Средний случай: O(n²)
    - Лучший случай: O(n²)

    Пространственная сложность: O(1)
    """
    length = len(arr)
    for i in range(length):
        min_index = i
        for j in range(i + 1, length):
            if arr[j] < arr[min_index]:
                min_index = j
        arr[i], arr[min_index] = arr[min_index], arr[i]
    return arr


def sort_insertion(arr: List[int]) -> List[int]:
    """
    Сортировка вставками.

    Временная сложность:
    - Худший случай: O(n²)
    - Средний случай: O(n²)
    - Лучший случай: O(n)

    Пространственная сложность: O(1)
    """
    for i in range(1, len(arr)):
        current = arr[i]
        j = i - 1
        while j >= 0 and current < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = current
    return arr


def sort_merge(arr: List[int]) -> List[int]:
    """
    Сортировка слиянием.

    Временная сложность:
    - Худший случай: O(n log n)
    - Средний случай: O(n log n)
    - Лучший случай: O(n log n)

    Пространственная сложность: O(n)
    """
    if len(arr) <= 1:
        return arr

    middle = len(arr) // 2
    left_part = sort_merge(arr[:middle])
    right_part = sort_merge(arr[middle:])

    return _combine_sorted(left_part, right_part)


def _combine_sorted(left: List[int], right: List[int]) -> List[int]:
    """Вспомогательная функция для слияния двух отсортированных массивов."""
    merged = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


def sort_quick(arr: List[int]) -> List[int]:
    """
    Быстрая сортировка.

    Временная сложность:
    - Худший случай: O(n²)
    - Средний случай: O(n log n)
    - Лучший случай: O(n log n)

    Пространственная сложность: O(log n)
    """
    if len(arr) <= 1:
        return arr

    pivot_value = arr[len(arr) // 2]
    less = [x for x in arr if x < pivot_value]
    equal = [x for x in arr if x == pivot_value]
    greater = [x for x in arr if x > pivot_value]

    return sort_quick(less) + equal + sort_quick(greater)


def check_if_sorted(arr: List[int]) -> bool:
    """Проверка, отсортирован ли массив."""
    return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))