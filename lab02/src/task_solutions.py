from collections import deque


def validate_brackets(expression):
    """
    Проверка сбалансированности скобок с использованием стека.
    
    Args:
        expression: Строка со скобками
        
    Returns:
        True если скобки сбалансированы, иначе False
        
    Complexity: O(n)
    """
    stack = []  # O(1) - создание стека
    bracket_pairs = {')': '(', ']': '[', '}': '{'}  # O(1) - создание словаря
    
    for char in expression:  # O(n) - цикл по символам
        if char in '([{':  # O(1) - проверка
            stack.append(char)  # O(1) - добавление в стек
        elif char in bracket_pairs:  # O(1) - проверка
            if not stack or stack[-1] != bracket_pairs[char]:  # O(1) - проверка
                return False  # O(1) - возврат
            stack.pop()  # O(1) - удаление из стека
    
    return len(stack) == 0  # O(1) - проверка и возврат


def check_palindrome_with_deque(text):
    """
    Проверка строки на палиндром с использованием дека.
    
    Args:
        text: Строка для проверки
        
    Returns:
        True если строка - палиндром, иначе False
        
    Complexity: O(n)
    """
    # Очистка строки от пробелов и приведение к нижнему регистру
    cleaned = ''.join(char.lower() for char in text if char.isalnum())  # O(n)
    
    dq = deque(cleaned)  # O(n) - создание дека
    
    while len(dq) > 1:  # O(n) - цикл
        if dq.popleft() != dq.pop():  # O(1) - сравнение
            return False  # O(1) - возврат
    
    return True  # O(1) - возврат


class DocumentPrintQueue:
    """Очередь печати на основе deque."""
    
    def __init__(self):
        """Инициализация очереди."""  # O(1) - инициализация
        self.print_jobs = deque()  # O(1) - создание дека
    
    def submit_job(self, document):
        """
        Добавление задачи в очередь.
        
        Args:
            document: Документ для печати
            
        Complexity: O(1)
        """
        self.print_jobs.append(document)  # O(1) - добавление в конец
    
    def next_job(self):
        """
        Извлечение задачи из очереди.
        
        Returns:
            Документ или None если очередь пуста
            
        Complexity: O(1)
        """
        if self.print_jobs:  # O(1) - проверка
            return self.print_jobs.popleft()  # O(1) - удаление из начала
        return None  # O(1) - возврат
    
    def execute_all_jobs(self):
        """
        Обработка всех задач в очереди.
        
        Complexity: O(n)
        """
        print('Обработка очереди печати:')
        while self.print_jobs:  # O(n) - цикл
            job = self.next_job()  # O(1) - извлечение
            print(f'Печатается: {job}')  # O(1) - вывод
        print('Все задачи обработаны.')  # O(1) - вывод
    
    def has_no_jobs(self):
        """
        Проверка пустоты очереди.
        
        Returns:
            True если очередь пуста, иначе False
            
        Complexity: O(1)
        """
        return len(self.print_jobs) == 0  # O(1) - проверка


def run_practical_examples():
    """Тестирование практических задач."""
    # Тест проверки скобок
    sample_expressions = [
        '()',
        '()[]{}',
        '([{}])',
        '(]',
        '([)]',
        '((()'
    ]
    
    print('Проверка сбалансированности скобок:')
    for expr in sample_expressions:
        result = validate_brackets(expr)
        print(f'"{expr}" -> {"Сбалансированы" if result else "Не сбалансированы"}')
    
    print('\n' + '='*50 + '\n')
    
    # Тест проверки палиндромов
    sample_texts = [
        'А роза упала на лапу Азора',
        'racecar',
        'hello',
        'Madam, I\'m Adam',
        'not a palindrome'
    ]
    
    print('Проверка палиндромов:')
    for text in sample_texts:
        result = check_palindrome_with_deque(text)
        print(f'"{text}" -> {"Палиндром" if result else "Не палиндром"}')
    
    print('\n' + '='*50 + '\n')
    
    # Тест очереди печати
    print('Симуляция очереди печати:')
    job_queue = DocumentPrintQueue()
    
    # Добавляем задачи в очередь
    documents = ['Документ1.pdf', 'Отчет.docx', 'Презентация.pptx', 'Изображение.jpg']
    for doc in documents:
        job_queue.submit_job(doc)
        print(f'Добавлено в очередь: {doc}')
    
    print()
    job_queue.execute_all_jobs()


if __name__ == '__main__':
    run_practical_examples()