"""Модуль для анализа производительности BST."""

import random
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from binary_search_tree import BinarySearchTree
from tree_traversal import (
    inorder_iterative,
    inorder_recursive,
    postorder_recursive,
    preorder_recursive,
)


def print_system_specs() -> None:
    """Вывод информации о системе."""
    print('''
ХАРАКТЕРИСТИКИ ПК ДЛЯ ТЕСТИРОВАНИЯ:
- Процессор: Intel Core i7-6500U @ 2.50GHz
- Оперативная память: 8 GB 
- ОС: Windows 10 PRO
- Python: 3.12.8
''')


def create_balanced_bst(size: int) -> BinarySearchTree:
    """
    Генерация сбалансированного дерева.

    Args:
        size: Количество элементов

    Returns:
        Сбалансированное BST
    """
    tree = BinarySearchTree()
    elements = list(range(size))
    random.shuffle(elements)

    for item in elements:
        tree.insert(item)

    return tree


def create_degenerate_bst(size: int) -> BinarySearchTree:
    """
    Генерация вырожденного дерева.

    Args:
        size: Количество элементов

    Returns:
        Вырожденное BST
    """
    tree = BinarySearchTree()
    for item in range(size):
        tree.insert(item)

    return tree


def benchmark_search_operations(
    bst: BinarySearchTree,
    queries: int = 100
) -> float:
    """
    Измерение времени выполнения операций поиска.

    Args:
        bst: Дерево для тестирования
        queries: Количество операций поиска

    Returns:
        Среднее время поиска в секундах
    """
    max_key = bst.height() * 2
    search_keys = [
        random.randint(0, max_key) for _ in range(queries)
    ]

    start = time.perf_counter()

    for key in search_keys:
        bst.search(key)

    end = time.perf_counter()

    return (end - start) / queries


def benchmark_delete_operations(
    bst: BinarySearchTree,
    removals: int = 50
) -> float:
    """
    Измерение времени выполнения операций удаления.

    Args:
        bst: Дерево для тестирования
        removals: Количество операций удаления

    Returns:
        Среднее время удаления в секундах
    """
    max_key = bst.height() * 2
    keys_to_remove = [
        random.randint(0, max_key) for _ in range(removals)
    ]

    total_duration = 0

    for key in keys_to_remove:
        test_copy = BinarySearchTree()

        def duplicate_tree(node):
            if node:
                test_copy.insert(node.value)
                duplicate_tree(node.left)
                duplicate_tree(node.right)

        duplicate_tree(bst.root)

        start = time.perf_counter()
        test_copy.delete(key)
        end = time.perf_counter()

        total_duration += (end - start)

    return total_duration / removals


def benchmark_traversal_methods(
    bst: BinarySearchTree,
    repetitions: int = 10
) -> Dict[str, float]:
    """
    Измерение времени выполнения различных обходов дерева.

    Args:
        bst: Дерево для тестирования
        repetitions: Количество повторений для усреднения

    Returns:
        Словарь с временами выполнения обходов
    """
    timings = {}

    cumulative_time = 0
    for _ in range(repetitions):
        output = []
        start = time.perf_counter()
        inorder_recursive(bst.root, output)
        end = time.perf_counter()
        cumulative_time += (end - start)
    timings['inorder_recursive'] = cumulative_time / repetitions

    cumulative_time = 0
    for _ in range(repetitions):
        output = []
        start = time.perf_counter()
        preorder_recursive(bst.root, output)
        end = time.perf_counter()
        cumulative_time += (end - start)
    timings['preorder_recursive'] = cumulative_time / repetitions

    cumulative_time = 0
    for _ in range(repetitions):
        output = []
        start = time.perf_counter()
        postorder_recursive(bst.root, output)
        end = time.perf_counter()
        cumulative_time += (end - start)
    timings['postorder_recursive'] = cumulative_time / repetitions

    cumulative_time = 0
    for _ in range(repetitions):
        start = time.perf_counter()
        inorder_iterative(bst.root)
        end = time.perf_counter()
        cumulative_time += (end - start)
    timings['inorder_iterative'] = cumulative_time / repetitions

    return timings


def evaluate_bst_performance() -> Dict[str, Dict[str, List[Tuple[int, float]]]]:
    """
    Анализ производительности для деревьев разного размера и структуры.

    Returns:
        Словарь с результатами анализа
    """
    test_sizes = [50, 100, 150, 200, 250]
    performance_data = {
        'balanced': {
            'search': [],
            'delete': [],
            'inorder_recursive': [],
            'preorder_recursive': [],
            'postorder_recursive': [],
            'inorder_iterative': []
        },
        'degenerate': {
            'search': [],
            'delete': [],
            'inorder_recursive': [],
            'preorder_recursive': [],
            'postorder_recursive': [],
            'inorder_iterative': []
        }
    }

    for size in test_sizes:
        print(f'Анализ для размера {size}')

        try:
            balanced_bst = create_balanced_bst(size)
            balanced_depth = balanced_bst.height()
            search_time = benchmark_search_operations(balanced_bst, 100)
            delete_time = benchmark_delete_operations(balanced_bst, 20)
            traversal_durations = benchmark_traversal_methods(balanced_bst, 5)

            performance_data['balanced']['search'].append((size, search_time))
            performance_data['balanced']['delete'].append((size, delete_time))
            performance_data['balanced']['inorder_recursive'].append(
                (size, traversal_durations['inorder_recursive'])
            )
            performance_data['balanced']['preorder_recursive'].append(
                (size, traversal_durations['preorder_recursive'])
            )
            performance_data['balanced']['postorder_recursive'].append(
                (size, traversal_durations['postorder_recursive'])
            )
            performance_data['balanced']['inorder_iterative'].append(
                (size, traversal_durations['inorder_iterative'])
            )

            print(f'  Сбалансированное: время={search_time:.6f}с, '
                  f'высота={balanced_depth}')

            degenerate_bst = create_degenerate_bst(size)
            degenerate_depth = degenerate_bst.height()
            search_time = benchmark_search_operations(degenerate_bst, 100)
            delete_time = benchmark_delete_operations(degenerate_bst, 20)
            traversal_durations = benchmark_traversal_methods(degenerate_bst, 5)

            performance_data['degenerate']['search'].append((size, search_time))
            performance_data['degenerate']['delete'].append((size, delete_time))
            performance_data['degenerate']['inorder_recursive'].append(
                (size, traversal_durations['inorder_recursive'])
            )
            performance_data['degenerate']['preorder_recursive'].append(
                (size, traversal_durations['preorder_recursive'])
            )
            performance_data['degenerate']['postorder_recursive'].append(
                (size, traversal_durations['postorder_recursive'])
            )
            performance_data['degenerate']['inorder_iterative'].append(
                (size, traversal_durations['inorder_iterative'])
            )

            print(f'  Вырожденное: время={search_time:.6f}с, '
                  f'высота={degenerate_depth}')

        except Exception as error:
            print(f'  Ошибка при размере {size}: {error}')
            continue

    return performance_data


def visualize_performance_metrics(
    results: Dict[str, Dict[str, List[Tuple[int, float]]]]
) -> None:
    """
    Построение графиков зависимости времени операций от количества элементов.

    Args:
        results: Результаты анализа производительности
    """
    operation_keys = [
        'search', 'delete', 'inorder_recursive', 'preorder_recursive',
        'postorder_recursive', 'inorder_iterative'
    ]
    operation_labels = {
        'search': 'Поиск элемента',
        'delete': 'Удаление элемента',
        'inorder_recursive': 'In-order (рекурсивный)',
        'preorder_recursive': 'Pre-order (рекурсивный)',
        'postorder_recursive': 'Post-order (рекурсивный)',
        'inorder_iterative': 'In-order (итеративный)'
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, op in enumerate(operation_keys):
        ax = axes[i]

        balanced_metrics = results['balanced'][op]
        if balanced_metrics:
            sizes_bal = [x[0] for x in balanced_metrics]
            times_bal = [x[1] * 1000000 for x in balanced_metrics]
            ax.plot(
                sizes_bal, times_bal, 'o-',
                label='Сбалансированное дерево',
                linewidth=2, markersize=6, color='blue'
            )

        degenerate_metrics = results['degenerate'][op]
        if degenerate_metrics:
            sizes_deg = [x[0] for x in degenerate_metrics]
            times_deg = [x[1] * 1000000 for x in degenerate_metrics]
            ax.plot(
                sizes_deg, times_deg, 's-',
                label='Вырожденное дерево',
                linewidth=2, markersize=6, color='red'
            )

        ax.set_xlabel('Количество элементов')
        ax.set_ylabel('Время (микросекунды)')
        ax.set_title(operation_labels[op])
        ax.legend()
        ax.grid(True, alpha=0.3)

        if op in ['search', 'delete']:
            ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def demonstrate_tree_structures() -> None:
    """Примеры деревьев разной структуры."""
    print('Примеры деревьев разной структуры')

    print('\n1. Сбалансированное дерево:')
    balanced_example = BinarySearchTree()
    sample_balanced_keys = [50, 30, 70, 20, 40, 60, 80, 10, 25, 35]
    for key in sample_balanced_keys:
        balanced_example.insert(key)

    from visualization import display_tree_properties
    display_tree_properties(balanced_example)

    print('\n2. Вырожденное дерево:')
    degenerate_example = BinarySearchTree()
    sample_degenerate_keys = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for key in sample_degenerate_keys:
        degenerate_example.insert(key)
    display_tree_properties(degenerate_example)