"""Модуль реализации бинарного дерева поиска."""

from __future__ import annotations

from collections import deque
from typing import Optional


class BSTNode:
    """Узел бинарного дерева поиска."""

    def __init__(self, key: int) -> None:
        """
        Инициализация узла.

        Args:
            key: Значение узла
        """
        self.value: int = key
        self.left: Optional[BSTNode] = None
        self.right: Optional[BSTNode] = None


class SearchBinaryTree:
    """Бинарное дерево поиска."""

    def __init__(self) -> None:
        """Инициализация пустого дерева."""
        self.root: Optional[BSTNode] = None

    def add(self, key: int) -> None:
        """
        Вставка значения в дерево.

        Сложность:
            В среднем: O(log n)
            В худшем случае: O(n) - для вырожденного дерева

        Args:
            key: Значение для вставки
        """
        new_vertex = BSTNode(key)

        if self.root is None:
            self.root = new_vertex
            return

        current = self.root
        while True:
            if key < current.value:
                if current.left is None:
                    current.left = new_vertex
                    return
                current = current.left
            elif key > current.value:
                if current.right is None:
                    current.right = new_vertex
                    return
                current = current.right
            else:
                return

    def contains(self, key: int) -> bool:
        """
        Поиск значения в дереве.

        Сложность:
            В среднем: O(log n)
            В худшем случае: O(n) - для вырожденного дерева

        Args:
            key: Значение для поиска

        Returns:
            True, если значение найдено, иначе False
        """
        current = self.root

        while current is not None:
            if key == current.value:
                return True
            elif key < current.value:
                current = current.left
            else:
                current = current.right

        return False

    def remove(self, key: int) -> None:
        """
        Удаление значения из дерева.

        Сложность:
            В среднем: O(log n)
            В худшем случае: O(n) - для вырожденного дерева

        Args:
            key: Значение для удаления
        """
        self.root = self._recursive_remove(self.root, key)

    def _recursive_remove(
        self, node: Optional[BSTNode], key: int
    ) -> Optional[BSTNode]:
        """
        Рекурсивное удаление значения.

        Args:
            node: Текущий узел
            key: Значение для удаления

        Returns:
            Обновленный узел
        """
        if node is None:
            return None

        if key < node.value:
            node.left = self._recursive_remove(node.left, key)
        elif key > node.value:
            node.right = self._recursive_remove(node.right, key)
        else:
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left

            successor = self._find_smallest(node.right)
            node.value = successor.value
            node.right = self._recursive_remove(node.right, successor.value)

        return node

    @staticmethod
    def _find_smallest(node: BSTNode) -> BSTNode:
        """
        Поиск минимального узла в поддереве.

        Args:
            node: Корень поддерева

        Returns:
            Узел с минимальным значением
        """
        while node.left is not None:
            node = node.left
        return node

    def get_minimum(self, subtree_root: Optional[BSTNode] = None) -> Optional[BSTNode]:
        """
        Поиск узла с минимальным значением в поддереве.

        Сложность:
            В среднем: O(log n)
            В худшем случае: O(n) - для вырожденного дерева

        Args:
            subtree_root: Узел для начала поиска (корень поддерева)

        Returns:
            Узел с минимальным значением или None
        """
        if subtree_root is None:
            if self.root is None:
                return None
            subtree_root = self.root

        while subtree_root.left is not None:
            subtree_root = subtree_root.left
        return subtree_root

    def get_maximum(self, subtree_root: Optional[BSTNode] = None) -> Optional[BSTNode]:
        """
        Поиск узла с максимальным значением в поддереве.

        Сложность:
            В среднем: O(log n)
            В худшем случае: O(n) - для вырожденного дерева

        Args:
            subtree_root: Узел для начала поиска (корень поддерева)

        Returns:
            Узел с максимальным значением или None
        """
        if subtree_root is None:
            if self.root is None:
                return None
            subtree_root = self.root

        while subtree_root.right is not None:
            subtree_root = subtree_root.right
        return subtree_root

    def compute_height(self, node: Optional[BSTNode] = None) -> int:
        """
        Вычисление высоты дерева/поддерева.

        Сложность: O(n) - необходимо посетить все узлы

        Args:
            node: Узел для вычисления высоты (корень поддерева)

        Returns:
            Высота дерева/поддерева
        """
        if node is None:
            if self.root is None:
                return 0
            node = self.root

        level_queue = deque()
        level_queue.append((node, 1))
        max_depth = 0

        while level_queue:
            current_vertex, depth = level_queue.popleft()
            max_depth = max(max_depth, depth)

            if current_vertex.left is not None:
                level_queue.append((current_vertex.left, depth + 1))
            if current_vertex.right is not None:
                level_queue.append((current_vertex.right, depth + 1))

        return max_depth

    def validate_bst_structure(self) -> bool:
        """
        Проверка, является ли дерево корректным BST.

        Сложность: O(n) - необходимо посетить все узлы

        Returns:
            True, если дерево корректно, иначе False
        """
        if self.root is None:
            return True

        traversal_stack = []
        last_seen = float('-inf')
        current = self.root

        while current is not None or traversal_stack:
            while current is not None:
                traversal_stack.append(current)
                current = current.left

            current = traversal_stack.pop()

            if current.value <= last_seen:
                return False
            last_seen = current.value

            current = current.right

        return True