"""Реализация пирамидальной сортировки (Heapsort)."""

from typing import Any, List
from heap import Heap


def heap_sort(input_list: List[Any], increasing_order: bool = True) -> List[Any]:
    """
    Сортировка кучей (heapsort).

    Args:
        input_list: Массив для сортировки.
        increasing_order: True для сортировки по возрастанию.

    Returns:
        Отсортированный массив.

    Сложность: O(n log n)
    """
    priority_queue = Heap(is_min=increasing_order)
    priority_queue.build_heap(input_list)

    result = []
    while not priority_queue.is_empty():
        result.append(priority_queue.extract())

    return result


def heap_sort_inplace(data: List[Any]) -> None:
    """
    Сортировка кучей in-place.

    Args:
        data: Массив для сортировки.

    Сложность: O(n log n)
    """
    def _percolate_down(arr: List[Any], heap_size: int, index: int) -> None:
        """Вспомогательная функция для погружения."""
        root_index = index
        left_child = 2 * index + 1
        right_child = 2 * index + 2

        if left_child < heap_size and arr[left_child] > arr[root_index]:
            root_index = left_child

        if right_child < heap_size and arr[right_child] > arr[root_index]:
            root_index = right_child

        if root_index != index:
            arr[index], arr[root_index] = arr[root_index], arr[index]
            _percolate_down(arr, heap_size, root_index)

    length = len(data)

    # Построение max-heap
    for i in range(length // 2 - 1, -1, -1):
        _percolate_down(data, length, i)

    # Извлечение элементов из кучи
    for i in range(length - 1, 0, -1):
        data[i], data[0] = data[0], data[i]
        _percolate_down(data, i, 0)