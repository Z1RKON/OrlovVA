"""Реализация хеш-таблицы с открытой адресацией."""

from typing import Any, Optional, Tuple
from hash_functions import simple_hash, polynomial_hash, djb2_hash


class OpenAddressingEntry:
    """Элемент хеш-таблицы для открытой адресации."""

    def __init__(self, key: Optional[str] = None, value: Any = None):
        self.key = key
        self.value = value
        self.marked_as_deleted = False


class OpenAddressingHashTable:
    """Хеш-таблица с открытой адресацией."""

    def __init__(self, initial_size: int = 101, hash_method: str = 'simple',
                 probing: str = 'linear', load_factor: float = 0.7):
        """
        Инициализация хеш-таблицы.

        Args:
            initial_size: Начальный размер таблицы (простое число)
            hash_method: Используемая хеш-функция
            probing: Метод пробирования ('linear', 'double')
            load_factor: Порог коэффициента заполнения
        """
        self.capacity = initial_size
        self.active_items = 0
        self.removed_items = 0
        self.load_threshold = load_factor
        self.probe_strategy = probing
        self.slots = [OpenAddressingEntry() for _ in range(initial_size)]

        # Выбор хеш-функции
        hash_algorithms = {
            'simple': simple_hash,
            'polynomial': polynomial_hash,
            'djb2': djb2_hash
        }
        self._hash_algorithm = hash_algorithms[hash_method]

    def _calculate_index(self, key: str, probe_attempt: int = 0) -> int:
        """Вычисление индекса с учетом номера попытки пробирования."""
        if self.probe_strategy == 'linear':
            return (self._hash_algorithm(key, self.capacity) + probe_attempt) % self.capacity
        elif self.probe_strategy == 'double':
            primary_hash = self._hash_algorithm(key, self.capacity)
            secondary_hash = 1 + (self._hash_algorithm(key, self.capacity - 1))
            return (primary_hash + probe_attempt * secondary_hash) % self.capacity
        else:
            raise ValueError("Неизвестный метод пробирования")

    def _rehash_table(self, new_capacity: int) -> None:
        """Изменение размера таблицы и перехеширование."""
        old_slots = self.slots
        self.capacity = new_capacity
        self.slots = [OpenAddressingEntry() for _ in range(new_capacity)]
        self.active_items = 0
        self.removed_items = 0

        for entry in old_slots:
            if entry.key is not None and not entry.marked_as_deleted:
                self.put(entry.key, entry.value)

    def put(self, key: str, value: Any) -> None:
        """
        Вставка элемента в хеш-таблицу.

        Args:
            key: Ключ
            value: Значение

        Time Complexity: O(1) в среднем, O(n) в худшем случае
        """
        # Проверка необходимости рехеширования
        if self.effective_fill_ratio > self.load_threshold:
            self._rehash_table(self.capacity * 2)

        attempt = 0
        while attempt < self.capacity:
            index = self._calculate_index(key, attempt)
            entry = self.slots[index]

            if entry.key is None or entry.marked_as_deleted:
                # Нашли свободную ячейку
                entry.key = key
                entry.value = value
                entry.marked_as_deleted = False
                self.active_items += 1
                return
            elif entry.key == key:
                # Обновление существующего ключа
                entry.value = value
                entry.marked_as_deleted = False
                return

            attempt += 1

        # Если не нашли место — рехеширование и повторная вставка
        self._rehash_table(self.capacity * 2)
        self.put(key, value)

    def get(self, key: str) -> Optional[Any]:
        """
        Поиск элемента по ключу.

        Args:
            key: Ключ для поиска

        Returns:
            Найденное значение или None

        Time Complexity: O(1) в среднем, O(n) в худшем случае
        """
        attempt = 0
        while attempt < self.capacity:
            index = self._calculate_index(key, attempt)
            entry = self.slots[index]

            if entry.key is None and not entry.marked_as_deleted:
                # Достигли пустой ячейки — ключ отсутствует
                return None
            elif entry.key == key and not entry.marked_as_deleted:
                # Нашли ключ
                return entry.value

            attempt += 1

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
        attempt = 0
        while attempt < self.capacity:
            index = self._calculate_index(key, attempt)
            entry = self.slots[index]

            if entry.key is None and not entry.marked_as_deleted:
                return False
            elif entry.key == key and not entry.marked_as_deleted:
                entry.marked_as_deleted = True
                self.active_items -= 1
                self.removed_items += 1

                # Периодическая очистка удаленных элементов
                if self.removed_items > self.active_items:
                    self._rehash_table(self.capacity)

                return True

            attempt += 1

        return False

    @property
    def fill_ratio(self) -> float:
        """Общий коэффициент заполнения (включая удалённые элементы)."""
        return (self.active_items + self.removed_items) / self.capacity

    @property
    def effective_fill_ratio(self) -> float:
        """Эффективный коэффициент заполнения (только активные элементы)."""
        return self.active_items / self.capacity

    def analyze_probing_stats(self) -> Tuple[int, int]:
        """
        Статистика коллизий.

        Returns:
            (количество коллизий, максимальная длина пробирования)
        """
        collision_count = 0
        max_probe_steps = 0

        for i, entry in enumerate(self.slots):
            if entry.key is not None and not entry.marked_as_deleted:
                initial_pos = self._calculate_index(entry.key, 0)
                if initial_pos != i:
                    collision_count += 1
                    distance = abs(i - initial_pos) % self.capacity
                    max_probe_steps = max(max_probe_steps, distance)

        return collision_count, max_probe_steps