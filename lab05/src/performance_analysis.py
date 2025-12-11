"""Анализ производительности хеш-таблиц."""

import time
import random
import string
import matplotlib.pyplot as plt
from hash_table_chaining import HashTableChaining
from hash_table_open_addressing import HashTableOpenAddressing


def create_random_key(length: int = 10) -> str:
    """Генерация случайной строки."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def run_performance_benchmark():
    """Измерение производительности разных реализаций."""
    # Параметры тестирования
    table_capacities = [100, 500, 1000]
    fill_ratios = [0.1, 0.5, 0.7, 0.9]
    hash_methods = ['simple', 'polynomial', 'djb2']
    
    benchmark_data = {}
    
    for capacity in table_capacities:
        for load_ratio in fill_ratios:
            num_items = int(capacity * load_ratio)
            
            # Генерация тестовых данных
            dataset = []
            for _ in range(num_items):
                k = create_random_key()
                v = create_random_key()
                dataset.append((k, v))
            
            # Тестирование разных реализаций
            variants = [
                ('Chaining', HashTableChaining(size=capacity)),
                ('Linear Probing', HashTableOpenAddressing(size=capacity, probing='linear')),
                ('Double Hashing', HashTableOpenAddressing(size=capacity, probing='double'))
            ]
            
            for variant_name, hash_table in variants:
                # Измерение времени вставки
                start = time.time()
                for k, v in dataset:
                    hash_table.insert(k, v)
                insertion_duration = time.time() - start
                
                # Измерение времени поиска
                start = time.time()
                for k, v in dataset:
                    hash_table.search(k)
                lookup_duration = time.time() - start
                
                # Статистика коллизий
                if hasattr(hash_table, 'get_collision_stats'):
                    num_collisions, _ = hash_table.get_collision_stats()
                else:
                    num_collisions = 0
                
                config_key = (capacity, load_ratio, variant_name)
                benchmark_data[config_key] = {
                    'insert_time': insertion_duration,
                    'search_time': lookup_duration,
                    'collisions': num_collisions,
                    'load_factor': hash_table.load_factor
                }
                
                print(f"Size: {capacity}, Load: {load_ratio}, Impl: {variant_name}")
                print(f"  Insert: {insertion_duration:.6f}s, Search: {lookup_duration:.6f}s")
                print(f"  Collisions: {num_collisions}")
    
    return benchmark_data


def render_performance_graphs(results):
    """Построение графиков результатов."""
    # Группировка результатов по реализации
    implementations = ['Chaining', 'Linear Probing', 'Double Hashing']
    load_factors = [0.1, 0.5, 0.7, 0.9]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Производительность хеш-таблиц')
    
    # Время вставки
    ax = axes[0, 0]
    for impl in implementations:
        times = []
        for lf in load_factors:
            key = (100, lf, impl)  # Для размера 100
            times.append(results[key]['insert_time'])
        ax.plot(load_factors, times, marker='o', label=impl)
    ax.set_title('Время вставки (размер=100)')
    ax.set_xlabel('Коэффициент заполнения')
    ax.set_ylabel('Время (сек)')
    ax.legend()
    ax.grid(True)
    
    # Время поиска
    ax = axes[0, 1]
    for impl in implementations:
        times = []
        for lf in load_factors:
            key = (100, lf, impl)
            times.append(results[key]['search_time'])
        ax.plot(load_factors, times, marker='o', label=impl)
    ax.set_title('Время поиска (размер=100)')
    ax.set_xlabel('Коэффициент заполнения')
    ax.set_ylabel('Время (сек)')
    ax.legend()
    ax.grid(True)
    
    # Коллизии
    ax = axes[1, 0]
    for impl in implementations:
        collision_counts = []
        for lf in load_factors:
            key = (100, lf, impl)
            collision_counts.append(results[key]['collisions'])
        ax.plot(load_factors, collision_counts, marker='o', label=impl)
    ax.set_title('Количество коллизий (размер=100)')
    ax.set_xlabel('Коэффициент заполнения')
    ax.set_ylabel('Коллизии')
    ax.legend()
    ax.grid(True)
    
    # Сравнение хеш-функций
    ax = axes[1, 1]
    hash_funcs = ['simple', 'polynomial', 'djb2']
    collisions_per_function = []
    
    for hf in hash_funcs:
        ht = HashTableChaining(size=100, hash_func=hf)
        test_entries = [(create_random_key(), create_random_key()) 
                       for _ in range(70)]  # load_factor = 0.7
        for k, v in test_entries:
            ht.insert(k, v)
        coll, _ = ht.get_collision_stats()
        collisions_per_function.append(coll)
    
    ax.bar(hash_funcs, collisions_per_function)
    ax.set_title('Коллизии по хеш-функциям')
    ax.set_xlabel('Хеш-функция')
    ax.set_ylabel('Количество коллизий')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('performance_results.png')
    plt.show()


if __name__ == '__main__':
    print("Запуск анализа производительности...")
    performance_data = run_performance_benchmark()
    print("\nПостроение графиков...")
    render_performance_graphs(performance_data)
    print("Анализ завершен. Результаты сохранены в performance_results.png")