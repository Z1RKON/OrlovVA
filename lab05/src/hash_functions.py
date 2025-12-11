"""Реализация различных хеш-функций для строковых ключей."""


def hash_by_sum(key: str, table_size: int) -> int:
    """
    Простая хеш-функция - сумма кодов символов.

    Args:
        key: Строковый ключ
        table_size: Размер хеш-таблицы

    Returns:
        Хеш-значение в диапазоне [0, table_size-1]
    """
    char_sum = 0
    for symbol in key:
        char_sum += ord(symbol)
    return char_sum % table_size


def hash_polynomial(key: str, table_size: int, base: int = 31) -> int:
    """
    Полиномиальная хеш-функция.

    Args:
        key: Строковый ключ
        table_size: Размер хеш-таблицы
        base: Основание полинома

    Returns:
        Хеш-значение в диапазоне [0, table_size-1]
    """
    hash_code = 0
    for symbol in key:
        hash_code = (hash_code * base + ord(symbol)) % table_size
    return hash_code


def hash_djb2(key: str, table_size: int) -> int:
    """
    Хеш-функция DJB2.

    Args:
        key: Строковый ключ
        table_size: Размер хеш-таблицы

    Returns:
        Хеш-значение в диапазоне [0, table_size-1]
    """
    hash_code = 5381
    for symbol in key:
        hash_code = ((hash_code << 5) + hash_code) + ord(symbol)
    return hash_code % table_size