"""Модуль для визуализации структуры дерева."""

from collections import deque
from typing import Optional

from binary_search_tree import TreeNode


def serialize_tree_to_brackets(root: Optional[TreeNode]) -> str:
    """
    Представление дерева в виде скобочной последовательности.

    Args:
        root: Корень дерева

    Returns:
        Строка с скобочным представление
    """
    if root is None:
        return '()'

    output = f'({root.value}'

    if root.left is not None or root.right is not None:
        output += serialize_tree_to_brackets(root.left)
        output += serialize_tree_to_brackets(root.right)

    output += ')'
    return output


def check_if_degenerate(root: Optional[TreeNode]) -> bool:
    """
    Проверяет, является ли дерево вырожденным.

    Args:
        root: Корень дерева

    Returns:
        True, если дерево вырожденное
    """
    if root is None:
        return False

    def is_degenerate(node: Optional[TreeNode]) -> bool:
        if node is None:
            return True
        if node.left is not None and node.right is not None:
            return False
        return is_degenerate(node.left) and is_degenerate(node.right)

    return is_degenerate(root)


def render_degenerate_tree(root: Optional[TreeNode]) -> None:
    """
    Специальная визуализация для вырожденных деревьев.

    Args:
        root: Корень вырожденного дерева
    """
    path = []
    current = root

    while current is not None:
        path.append(str(current.value))
        if current.left is not None:
            current = current.left
        else:
            current = current.right

    if root.left is not None:
        orientation = '←'
    else:
        orientation = '→'

    tree_sequence = ' → '.join(path) if orientation == '→' else ' ← '.join(path)
    print(tree_sequence)

    if orientation == '←':
        print('Вырожденное дерево (левая ветвь)')
    else:
        print('Вырожденное дерево (правая ветвь)')


def render_tree_structure(root: Optional[TreeNode]) -> None:
    """
    Визуализация дерева.

    Args:
        root: Корень дерева
    """
    if root is None:
        print('(empty)')
        return

    if check_if_degenerate(root):
        render_degenerate_tree(root)
        return

    tree_levels = []
    node_queue: deque[tuple[Optional[TreeNode], int]] = deque([(root, 0)])
    max_depth = 0

    while node_queue:
        node, level = node_queue.popleft()

        if level >= len(tree_levels):
            tree_levels.append([])
            max_depth = level

        if node is None:
            tree_levels[level].append('  ')
            if level < max_depth:
                node_queue.append((None, level + 1))
                node_queue.append((None, level + 1))
            continue
        else:
            tree_levels[level].append(f'{node.value:2d}')

        node_queue.append((node.left, level + 1))
        node_queue.append((node.right, level + 1))

    while tree_levels and all(cell == '  ' for cell in tree_levels[-1]):
        tree_levels.pop()

    for i, level in enumerate(tree_levels):
        indent_before = 2 ** (len(tree_levels) - i - 1) - 1
        gap_between = 2 ** (len(tree_levels) - i) - 1

        line = ' ' * indent_before
        for j, cell in enumerate(level):
            line += cell
            if j < len(level) - 1:
                line += ' ' * gap_between

        print(line)

        if i < len(tree_levels) - 1:
            connector_line = ' ' * (indent_before - 1)
            next_level = tree_levels[i + 1] if i + 1 < len(tree_levels) else []

            for j in range(len(level)):
                left_idx = j * 2
                right_idx = j * 2 + 1

                left_exists = (
                    left_idx < len(next_level) and
                    next_level[left_idx] != '  '
                )
                right_exists = (
                    right_idx < len(next_level) and
                    next_level[right_idx] != '  '
                )

                if left_exists and right_exists:
                    connector_line += '╻━┻━╻'
                elif left_exists:
                    connector_line += '╻━╹  '
                elif right_exists:
                    connector_line += '  ╹━╻'
                else:
                    connector_line += '     '

                if j < len(level) - 1:
                    connector_line += ' ' * (gap_between - 3)

            print(connector_line)


def show_tree_characteristics(bst) -> None:
    """
    Отображение свойств дерева.

    Args:
        bst: Бинарное дерево поиска
    """
    min_node = bst.find_min()
    max_node = bst.find_max()
    min_val = min_node.value if min_node else 'None'
    max_val = max_node.value if max_node else 'None'

    print(f'Минимальное значение: {min_val}')
    print(f'Максимальное значение: {max_val}')
    print(f'Высота дерева: {bst.height()}')
    print(f'Корректное BST: {bst.is_valid_bst()}')

    print('\nСкобочное представление:')
    print(serialize_tree_to_brackets(bst.root))

    print('\nСтруктура дерева:')
    render_tree_structure(bst.root)