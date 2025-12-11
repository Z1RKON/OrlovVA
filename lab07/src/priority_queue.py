"""Реализация приоритетной очереди на основе кучи."""

from typing import Any, Optional
from heap import Heap


class QueueEntry:
    """Элемент приоритетной очереди."""

    def __init__(self, value: Any, priority_level: float) -> None:
        """
        Инициализация элемента.

        Args:
            value: Объект.
            priority_level: Приоритет.
        """
        self.payload = value
        self.priority = priority_level

    def __lt__(self, other: 'QueueEntry') -> bool:
        """Сравнение для min-heap (меньший приоритет - выше)."""
        return self.priority < other.priority

    def __gt__(self, other: 'QueueEntry') -> bool:
        """Сравнение для max-heap."""
        return self.priority > other.priority

    def __repr__(self) -> str:
        """Строковое представление."""
        return f'QueueEntry(payload={self.payload}, priority={self.priority})'


class TaskPriorityQueue:
    """Приоритетная очередь на основе min-heap."""

    def __init__(self) -> None:
        """Инициализация приоритетной очереди."""
        self._storage = Heap(is_min=True)

    def enqueue(self, task: Any, priority: float) -> None:
        """
        Добавление элемента в очередь. Сложность: O(log n).

        Args:
            task: Объект для добавления.
            priority: Приоритет объекта.
        """
        self._storage.insert(QueueEntry(task, priority))

    def dequeue(self) -> Optional[Any]:
        """
        Извлечение элемента с наивысшим приоритетом. Сложность: O(log n).

        Returns:
            Элемент или None, если очередь пуста.
        """
        queued_item = self._storage.extract()
        return queued_item.payload if queued_item else None

    def peek(self) -> Optional[Any]:
        """
        Просмотр элемента с наивысшим приоритетом. Сложность: O(1).

        Returns:
            Элемент или None, если очередь пуста.
        """
        queued_item = self._storage.peek()
        return queued_item.payload if queued_item else None

    def is_empty(self) -> bool:
        """
        Проверка пустоты очереди. Сложность: O(1).

        Returns:
            True, если очередь пуста.
        """
        return self._storage.is_empty()

    def size(self) -> int:
        """
        Размер очереди. Сложность: O(1).

        Returns:
            Количество элементов в очереди.
        """
        return self._storage.size()