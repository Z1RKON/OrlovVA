import os
from typing import List, Optional


def recursive_binary_search(
    sorted_list: List[int],
    target: int,
    low: int = 0,
    high: Optional[int] = None
) -> Optional[int]:
    """
    Рекурсивная реализация бинарного поиска.

    Args:
        sorted_list: Отсортированный массив
        target: Искомый элемент
        low: Левая граница поиска
        high: Правая граница поиска

    Returns:
        Индекс элемента или None если не найден
    """
    if high is None:
        high = len(sorted_list) - 1

    if low > high:
        return None

    mid_index = (low + high) // 2

    if sorted_list[mid_index] == target:
        return mid_index
    elif sorted_list[mid_index] < target:
        return recursive_binary_search(sorted_list, target, mid_index + 1, high)
    else:
        return recursive_binary_search(sorted_list, target, low, mid_index - 1)


# Временная сложность: O(log n).  Глубина рекурсии: O(log n).


def traverse_directory_tree(
    root_path: str,
    depth_level: int = 0,
    limit_depth: Optional[int] = None
) -> None:
    """
    Рекурсивный обход файловой системы с выводом дерева каталогов.

    Args:
        root_path: Начальный путь для обхода
        depth_level: Текущий уровень вложенности
        limit_depth: Максимальная глубина рекурсии
    """
    if limit_depth is not None and depth_level > limit_depth:
        return

    try:
        items = os.listdir(root_path)
    except PermissionError:
        print('  ' * depth_level + f'[Доступ запрещен: {root_path}]')
        return
    except FileNotFoundError:
        print('  ' * depth_level + f'[Путь не найден: {root_path}]')
        return

    for item in sorted(items):
        full_item_path = os.path.join(root_path, item)

        if os.path.isdir(full_item_path):
            print('  ' * depth_level + f'-- {item}/')
            traverse_directory_tree(full_item_path, depth_level + 1, limit_depth)
        else:
            print('  ' * depth_level + f'- {item}')


# Временная сложность: O(n).  Глубина рекурсии: O(d).


def solve_hanoi_puzzle(
    disk_count: int,
    from_rod: str = 'A',
    helper_rod: str = 'B',
    to_rod: str = 'C'
) -> None:
    """
    Решение задачи о Ханойских башнях для n дисков.

    Args:
        disk_count: Количество дисков
        from_rod: Исходный стержень
        helper_rod: Вспомогательный стержень
        to_rod: Целевой стержень
    """
    if disk_count == 1:
        print(f'Переместить диск 1 с {from_rod} на {to_rod}')
        return

    solve_hanoi_puzzle(disk_count - 1, from_rod, to_rod, helper_rod)
    print(f'Переместить диск {disk_count} с {from_rod} на {to_rod}')
    solve_hanoi_puzzle(disk_count - 1, helper_rod, from_rod, to_rod)


# Временная сложность: O(2^n).  Глубина рекурсии: O(n).


if __name__ == '__main__':
    sample_array = [1, 3, 5, 7, 9, 11, 13, 15]
    search_value = 7
    index_found = recursive_binary_search(sample_array, search_value)
    print(f'Бинарный поиск {search_value} в {sample_array}: индекс {index_found}')

    print('\nХанойские башни для 3 дисков:')
    solve_hanoi_puzzle(3)

    print('\nОбход файловой системы (текущая директория, глубина 2):')
    traverse_directory_tree('.', limit_depth=2)