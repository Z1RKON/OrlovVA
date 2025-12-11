"""Реализация структуры данных Куча."""

from typing import Any, List, Optional


class PriorityHeap:
    """Универсальная куча (min-heap или max-heap)."""

    def __init__(self, min_heap: bool = True) -> None:
        """
        Инициализация кучи.

        Args:
            min_heap: True для min-heap, False для max-heap.
        """
        self._elements: List[Any] = []
        self._is_min_heap = min_heap

    def retrieve_array(self) -> List[Any]:
        """
        Получение массива кучи.

        Returns:
            Внутренний массив кучи.
        """
        return self._elements.copy()

    def _has_priority(self, a: Any, b: Any) -> bool:
        """
        Сравнение элементов в зависимости от типа кучи.

        Args:
            a: Первый элемент.
            b: Второй элемент.

        Returns:
            True, если a имеет приоритет над b.
        """
        if self._is_min_heap:
            return a < b
        else:
            return a > b

    def _bubble_up(self, idx: int) -> None:
        """
        Всплытие элемента (sift-up). Сложность: O(log n).

        Args:
            idx: Индекс элемента для всплытия.
        """
        if idx == 0:
            return

        parent_idx = (idx - 1) // 2
        if self._has_priority(self._elements[idx], self._elements[parent_idx]):
            self._elements[idx], self._elements[parent_idx] = (
                self._elements[parent_idx], self._elements[idx]
            )
            self._bubble_up(parent_idx)

    def _sink_down(self, idx: int) -> None:
        """
        Погружение элемента (sift-down). Сложность: O(log n).

        Args:
            idx: Индекс элемента для погружения.
        """
        left_child = 2 * idx + 1
        right_child = 2 * idx + 2
        target_idx = idx

        if (left_child < len(self._elements) and
                self._has_priority(self._elements[left_child],
                                   self._elements[target_idx])):
            target_idx = left_child

        if (right_child < len(self._elements) and
                self._has_priority(self._elements[right_child],
                                   self._elements[target_idx])):
            target_idx = right_child

        if target_idx != idx:
            self._elements[idx], self._elements[target_idx] = (
                self._elements[target_idx], self._elements[idx]
            )
            self._sink_down(target_idx)

    def push(self, item: Any) -> None:
        """
        Вставка элемента в кучу. Сложность: O(log n).

        Args:
            item: Значение для вставки.
        """
        self._elements.append(item)
        self._bubble_up(len(self._elements) - 1)

    def pop(self) -> Optional[Any]:
        """
        Извлечение корневого элемента. Сложность: O(log n).

        Returns:
            Корневой элемент или None, если куча пуста.
        """
        if not self._elements:
            return None

        if len(self._elements) == 1:
            return self._elements.pop()

        root_value = self._elements[0]
        self._elements[0] = self._elements.pop()
        self._sink_down(0)
        return root_value

    def top(self) -> Optional[Any]:
        """
        Просмотр корневого элемента. Сложность: O(1).

        Returns:
            Корневой элемент или None, если куча пуста.
        """
        return self._elements[0] if self._elements else None

    def construct_from(self, input_list: List[Any]) -> None:
        """
        Построение кучи из массива. Сложность: O(n).

        Args:
            input_list: Массив для построения кучи.
        """
        self._elements = input_list.copy()
        for i in range(len(self._elements) // 2 - 1, -1, -1):
            self._sink_down(i)

    def length(self) -> int:
        """
        Получение размера кучи. Сложность: O(1).

        Returns:
            Количество элементов в куче.
        """
        return len(self._elements)

    def empty(self) -> bool:
        """
        Проверка пустоты кучи. Сложность: O(1).

        Returns:
            True, если куча пуста.
        """
        return len(self._elements) == 0

    def __str__(self) -> str:
        """Строковое представление кучи."""
        return str(self._elements)


class MinPriorityQueue(PriorityHeap):
    """Min-Heap специализация."""

    def __init__(self) -> None:
        """Инициализация min-heap."""
        super().__init__(min_heap=True)


class MaxPriorityQueue(PriorityHeap):
    """Max-Heap специализация."""

    def __init__(self) -> None:
        """Инициализация max-heap."""
        super().__init__(min_heap=False)