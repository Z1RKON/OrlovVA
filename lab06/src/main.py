from analysis import (
    analyze_performance,
    plot_results,
    system_info,
)
from binary_search_tree import BinarySearchTree
from tree_traversal import get_traversal_results
from visualization import display_tree_properties


def showcase_core_operations():
    """Демонстрация основных операций BST."""
    tree = BinarySearchTree()

    elements = [50, 30, 70, 20, 40, 60, 80, 10, 25, 35, 45]
    print(f'Значения: {elements}')
    for item in elements:
        tree.insert(item)

    display_tree_properties(tree)

    traversal_data = get_traversal_results(tree.root)
    print('\nРезультаты обходов:')
    print(f"In-order: {traversal_data['inorder_recursive']}")
    print(f"Pre-order: {traversal_data['preorder_recursive']}")
    print(f"Post-order: {traversal_data['postorder_recursive']}")
    print(f"In-order (итеративный): {traversal_data['inorder_iterative']}")

    print('\nПоиск элементов:')
    search_targets = [25, 55, 70, 100]
    for val in search_targets:
        is_found = tree.search(val)
        print(f'Поиск {val}: {"найден" if is_found else "не найден"}')

    print('\nУдаление элемента:')
    items_to_remove = [25, 70]
    for val in items_to_remove:
        print(f'Удаляемый элемент {val}')
        tree.delete(val)
        display_tree_properties(tree)


def showcase_tree_variants():
    """Демонстрация разных типов деревьев."""
    print('\nДемонстрация разных типов деревьев')

    print('\n1. Сбалансированное дерево:')
    balanced_tree = BinarySearchTree()
    balanced_keys = [40, 20, 60, 10, 30, 50, 70, 5, 15, 25, 35]
    for key in balanced_keys:
        balanced_tree.insert(key)
    display_tree_properties(balanced_tree)

    print('\n2. Вырожденное дерево:')
    skewed_tree = BinarySearchTree()
    sequential_keys = [10, 20, 30, 40, 50, 60, 70]
    for key in sequential_keys:
        skewed_tree.insert(key)
    display_tree_properties(skewed_tree)


def run_main_demo():
    """Основная функция."""
    system_info()

    showcase_core_operations()

    showcase_tree_variants()

    print('\nАнализ производительности')
    performance_data = analyze_performance()
    plot_results(performance_data)

    print('\nВыводы:')
    print('1. Сбалансированные деревья показывают производительность '
          'O(log n) для поиска и удаления')
    print('2. Вырожденные деревья деградируют до O(n) для поиска '
          'и удаления')
    print('3. Все обходы имеют сложность O(n) независимо от структуры '
          'дерева')
    print('4. Итеративный обход обычно быстрее рекурсивных из-за '
          'отсутствия накладных расходов на вызовы функций')


if __name__ == '__main__':
    run_main_demo()