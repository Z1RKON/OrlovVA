"""
Модуль для генерации тестовых данных.
"""

import random
from typing import List, Dict


def create_random_list(size: int) -> List[int]:
    """Генерация случайного массива."""
    return [random.randint(0, 10000) for _ in range(size)]


def create_sorted_list(size: int) -> List[int]:
    """Генерация отсортированного массива."""
    return list(range(size))


def create_reverse_ordered_list(size: int) -> List[int]:
    """Генерация массива, отсортированного в обратном порядке."""
    return list(range(size, 0, -1))


def create_nearly_sorted_list(size: int) -> List[int]:
    """Генерация почти отсортированного массива (95% упорядочено)."""
    arr = list(range(size))
    # Перемешиваем 5% элементов
    num_swapped = max(1, size // 20)
    for _ in range(num_swapped):
        i = random.randint(0, size - 1)
        j = random.randint(0, size - 1)
        arr[i], arr[j] = arr[j], arr[i]
    return arr


def build_test_datasets(sizes: List[int]) -> Dict[str, Dict[int, List[int]]]:
    """
    Генерация всех типов тестовых данных для заданных размеров.

    Returns:
        Словарь с данными в формате:
        {
            'random': {100: [...], 1000: [...], ...},
            'sorted': {100: [...], 1000: [...], ...},
            'reversed': {100: [...], 1000: [...], ...},
            'almost_sorted': {100: [...], 1000: [...], ...}
        }
    """
    dataset_generators = {
        'random': create_random_list,
        'sorted': create_sorted_list,
        'reversed': create_reverse_ordered_list,
        'almost_sorted': create_nearly_sorted_list
    }

    all_datasets = {}
    for name, generator in dataset_generators.items():
        all_datasets[name] = {}
        for size_val in sizes:
            all_datasets[name][size_val] = generator(size_val)

    return all_datasets


if __name__ == "__main__":
    # Тест генерации данных
    sample_sizes = [100, 500]
    generated_data = build_test_datasets(sample_sizes)

    for data_category, size_map in generated_data.items():
        print(f"{data_category}:")
        for size, array in size_map.items():
            print(f"  Size {size}: first 10 elements - {array[:10]}")
        print()