"""Unit-тесты для хеш-таблиц."""

import unittest
from hash_table_chaining import HashTableChaining
from hash_table_open_addressing import HashTableOpenAddressing


class HashTableTestSuite(unittest.TestCase):
    """Тесты для всех реализаций хеш-таблиц."""

    def test_chaining_crud_operations(self):
        """Тест базовых операций для метода цепочек."""
        table = HashTableChaining(size=10)

        # Тест вставки и поиска
        table.insert("key1", "value1")
        table.insert("key2", "value2")
        self.assertEqual(table.search("key1"), "value1")
        self.assertEqual(table.search("key2"), "value2")
        self.assertIsNone(table.search("key3"))

        # Тест обновления
        table.insert("key1", "new_value1")
        self.assertEqual(table.search("key1"), "new_value1")

        # Тест удаления
        self.assertTrue(table.delete("key1"))
        self.assertIsNone(table.search("key1"))
        self.assertFalse(table.delete("key1"))

    def test_linear_probing_crud_operations(self):
        """Тест базовых операций для открытой адресации с линейным пробированием."""
        table = HashTableOpenAddressing(size=10, probing='linear')

        # Тест вставки и поиска
        table.insert("key1", "value1")
        table.insert("key2", "value2")
        self.assertEqual(table.search("key1"), "value1")
        self.assertEqual(table.search("key2"), "value2")
        self.assertIsNone(table.search("key3"))

        # Тест обновления
        table.insert("key1", "new_value1")
        self.assertEqual(table.search("key1"), "new_value1")

        # Тест удаления
        self.assertTrue(table.delete("key1"))
        self.assertIsNone(table.search("key1"))
        self.assertFalse(table.delete("key1"))

    def test_double_hashing_crud_operations(self):
        """Тест базовых операций для открытой адресации с двойным хешированием."""
        table = HashTableOpenAddressing(size=10, probing='double')

        # Тест вставки и поиска
        table.insert("key1", "value1")
        table.insert("key2", "value2")
        self.assertEqual(table.search("key1"), "value1")
        self.assertEqual(table.search("key2"), "value2")

        # Тест удаления
        self.assertTrue(table.delete("key1"))
        self.assertIsNone(table.search("key1"))

    def test_collision_resolution(self):
        """Тест обработки коллизий."""
        # Используем маленькую таблицу для гарантии коллизий
        chaining_table = HashTableChaining(size=3)
        linear_table = HashTableOpenAddressing(size=3, probing='linear')
        double_table = HashTableOpenAddressing(size=3, probing='double')

        test_keys = ["a", "b", "c", "d"]  # Должны быть коллизии

        for ht in [chaining_table, linear_table, double_table]:
            for idx, key in enumerate(test_keys):
                ht.insert(key, f"value{idx}")

            # Проверяем, что все значения доступны
            for idx, key in enumerate(test_keys):
                self.assertEqual(ht.search(key), f"value{idx}")

    def test_dynamic_resizing(self):
        """Тест операции изменения размера."""
        table = HashTableChaining(size=5, load_factor_threshold=0.6)

        # Вставляем элементы для trigger рехеширования
        for i in range(10):
            table.insert(f"key{i}", f"value{i}")

        # Проверяем, что все значения доступны после рехеширования
        for i in range(10):
            self.assertEqual(table.search(f"key{i}"), f"value{i}")


if __name__ == '__main__':
    unittest.main()