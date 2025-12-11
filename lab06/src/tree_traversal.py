"""Модуль реализации методов обхода дерева."""

from typing import List, Optional

from binary_search_tree import TreeNode


def traverse_inorder_recursive(node: Optional[TreeNode], output: List[int]) -> None:
    """
    Рекурсивный in-order обход (левый-корень-правый).

    Сложность: O(n)

    Args:
        node: Текущий узел
        output: Список для сохранения результатов
    """
    if node is not None:
        traverse_inorder_recursive(node.left, output)
        output.append(node.value)
        traverse_inorder_recursive(node.right, output)


def traverse_preorder_recursive(node: Optional[TreeNode], output: List[int]) -> None:
    """
    Рекурсивный pre-order обход (корень-левый-правый).

    Сложность: O(n)

    Args:
        node: Текущий узел
        output: Список для сохранения результатов
    """
    if node is not None:
        output.append(node.value)
        traverse_preorder_recursive(node.left, output)
        traverse_preorder_recursive(node.right, output)


def traverse_postorder_recursive(node: Optional[TreeNode], output: List[int]) -> None:
    """
    Рекурсивный post-order обход (левый-правый-корень).

    Сложность: O(n)

    Args:
        node: Текущий узел
        output: Список для сохранения результатов
    """
    if node is not None:
        traverse_postorder_recursive(node.left, output)
        traverse_postorder_recursive(node.right, output)
        output.append(node.value)


def traverse_inorder_iterative(root: Optional[TreeNode]) -> List[int]:
    """
    Итеративный in-order обход.

    Сложность: O(n)

    Args:
        root: Корень дерева

    Returns:
        Список значений в порядке in-order
    """
    traversal_result: List[int] = []
    node_stack: List[TreeNode] = []
    current_node: Optional[TreeNode] = root

    while current_node is not None or node_stack:
        while current_node is not None:
            node_stack.append(current_node)
            current_node = current_node.left

        current_node = node_stack.pop()
        traversal_result.append(current_node.value)
        current_node = current_node.right

    return traversal_result


def collect_all_traversals(root: Optional[TreeNode]) -> dict:
    """
    Получение результатов всех видов обходов.

    Args:
        root: Корень дерева

    Returns:
        Словарь с результатами всех обходов
    """
    all_results: dict = {
        'inorder_recursive': [],
        'preorder_recursive': [],
        'postorder_recursive': [],
        'inorder_iterative': []
    }

    if root is not None:
        traverse_inorder_recursive(root, all_results['inorder_recursive'])
        traverse_preorder_recursive(root, all_results['preorder_recursive'])
        traverse_postorder_recursive(root, all_results['postorder_recursive'])
        all_results['inorder_iterative'] = traverse_inorder_iterative(root)

    return all_results