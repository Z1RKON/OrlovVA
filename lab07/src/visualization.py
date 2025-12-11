"""Визуализация кучи в виде дерева."""

from collections import deque
from typing import Deque, List, Tuple
from heap import Heap


def render_heap_as_tree(heap: Heap) -> None:
    """
    Визуализация кучи.

    Args:
        heap: Куча для визуализации
    """
    elements = heap.retrieve_array()
    if not elements:
        print('(empty)')
        return

    depth = 0
    total_nodes = len(elements)
    while (1 << depth) - 1 < total_nodes:
        depth += 1

    tree_levels: List[List[str]] = []
    node_queue: Deque[Tuple[int, int]] = deque()
    node_queue.append((0, 0))
    max_depth = 0

    while node_queue:
        index, level = node_queue.popleft()

        if level >= len(tree_levels):
            tree_levels.append([])
            max_depth = level

        if index >= len(elements):
            tree_levels[level].append('  ')
            if level < max_depth:
                node_queue.append((2 * index + 1, level + 1))
                node_queue.append((2 * index + 2, level + 1))
            continue
        else:
            tree_levels[level].append(f'{elements[index]:2d}')

        node_queue.append((2 * index + 1, level + 1))
        node_queue.append((2 * index + 2, level + 1))

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
                left_pos = j * 2
                right_pos = j * 2 + 1

                left_exists = (
                    left_pos < len(next_level) and
                    next_level[left_pos] != '  '
                )
                right_exists = (
                    right_pos < len(next_level) and
                    next_level[right_pos] != '  '
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


def display_heap_structure(heap: Heap) -> None:
    """Печать кучи в виде дерева."""
    heap_type = "min" if heap._is_min_heap else "max"
    print(f'Куча ({heap_type}):')
    render_heap_as_tree(heap)


def display_heap_array(heap: Heap) -> None:
    """Простая печать кучи в виде массива."""
    heap_type = "min" if heap._is_min_heap else "max"
    print(f'Куча ({heap_type}): {heap.retrieve_array()}')