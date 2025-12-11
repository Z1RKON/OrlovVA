"""Реализация хеш-таблицы с методом цепочек."""

from typing import Any, Optional, Tuple
from hash_functions import simple_hash, polynomial_hash, djb2_hash


class ChainingHashTable:
    """Хеш-таблица с методом цепочек для разрешения коллизий."""

    def __init__(self, initial_capacity: int = 101, hash_method: str = 'simple', 
                 max_load_factor: float = 0.7):
        """
        Инициализация хеш-таблицы.

        Args:
            initial_capacity: Начальный размер таблицы (простое число)
            hash_method: Используемая хеш-функция ('simple', 'polynomial', 'djb2')
            max_load_factor: Порог для рехеширования
        """
        self.capacity = initial_capacity
        self.elements_count = 0
        self.load_threshold = max_load_factor
        self.buckets = [[] for _ in range(initial_capacity)]

        # Выбор хеш-функции
        hash_algorithms = {
            'simple': simple_hash,
            'polynomial': polynomial_hash,
            'djb2': djb2_hash
        }
        self._hash_algorithm = hash_algorithms[hash_method]

    def _compute_hash(self, key: str) -> int:
        """Вычисление хеша для ключа."""
        return self._hash_algorithm(key, self.capacity)

    def _rehash(self, new_capacity: int) -> None:
        """Изменение размера таблицы и перехеширование всех элементов."""
        old_buckets = self.buckets
        self.capacity = new_capacity
        self.buckets = [[] for _ in range(new_capacity)]
        self.elements_count = 0

        for chain in old_buckets:
            for key, value in chain:
                self.add(key, value)

    def add(self, key: str, value: Any) -> None:
        """
        Вставка элемента в хеш-таблицу.

        Args:
            key: Ключ
            value: Значение

        Time Complexity: O(1) в среднем, O(n) в худшем случае
        """
        # Проверка необходимости рехеширования
        if self.fill_ratio > self.load_threshold:
            self._rehash(self.capacity * 2)

        slot_index = self._compute_hash(key)
        chain = self.buckets[slot_index]

        # Проверка на существование ключа
        for i, (k, v) in enumerate(chain):
            if k == key:
                chain[i] = (key, value)
                return

        # Вставка нового элемента
        chain.append((key, value))
        self.elements_count += 1

    def find(self, key: str) -> Optional[Any]:
        """
        Поиск элемента по ключу.

        Args:
            key: Ключ для поиска

        Returns:
            Найденное значение или None

        Time Complexity: O(1) в среднем, O(n) в худшем случае
        """
        slot_index = self._compute_hash(key)
        chain = self.buckets[slot_index]

        for k, v in chain:
            if k == key:
                return v
        return None

    def remove(self, key: str) -> bool:
        """
        Удаление элемента по ключу.

        Args:
            key: Ключ для удаления

        Returns:
            True если элемент удален, False если не найден

        Time Complexity: O(1) в среднем, O(n) в худшем случае
        """
        slot_index = self._compute_hash(key)
        chain = self.buckets[slot_index]

        for i, (k, v) in enumerate(chain):
            if k == key:
                del chain[i]
                self.elements_count -= 1
                return True
        return False

    @property
    def fill_ratio(self) -> float:
        """Коэффициент заполнения таблицы."""
        return self.elements_count / self.capacity

    def analyze_collisions(self) -> Tuple[int, float]:
        """
        Статистика коллизий.

        Returns:
            (количество коллизий, средняя длина цепочки)
        """
        total_collisions = 0
        total_items = 0

        for chain in self.buckets:
            if len(chain) > 1:
                total_collisions += len(chain) - 1
            total_items += len(chain)

        average_chain_len = total_items / self.capacity if self.capacity else 0
        return total_collisions, average_chain_len