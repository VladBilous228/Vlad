import numpy as np  # Імпортуємо бібліотеку numpy для роботи з масивами та математичними операціями

# Логічний вираз X1 AND (X2 AND X3) OR X4
def logical_expression(x1, x2, x3, x4):
    return x1 and (x2 and x3) or x4  # Визначаємо функцію, що реалізує логічний вираз

# Створення таблиці істинності
input_combinations = np.array([[x1, x2, x3, x4] for x1 in [0, 1] for x2 in [0, 1] for x3 in [0, 1] for x4 in [0, 1]])  # Створюємо всі можливі комбінації вхідних значень
output_values = np.array([logical_expression(*x) for x in input_combinations])  # Визначаємо вихідні значення за виразом

# Створення нейронної мережі
class NeuralNetwork:
    def __init__(self):
        self.input_size = 4  # Кількість вхідних нейронів
        self.hidden_size = 2  # Кількість нейронів прихованого шару
        self.output_size = 1  # Кількість вихідних нейронів

        # Ініціалізація ваг та зміщень з випадковими значеннями
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    # Сигмоїдна функція активації
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Похідна сигмоїдної функції
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # Пряме поширення
    def forward(self, X):
        self.hidden_sum = np.dot(X, self.weights_input_hidden) + self.bias_hidden  # Підсумкове значення прихованого шару
        self.activated_hidden = self.sigmoid(self.hidden_sum)  # Активація прихованого шару

        self.output_sum = np.dot(self.activated_hidden, self.weights_hidden_output) + self.bias_output  # Підсумкове значення вихідного шару
        self.activated_output = self.sigmoid(self.output_sum)  # Активація вихідного шару

        return self.activated_output  # Повертаємо вихідне значення

    # Зворотне поширення
    def backward(self, X, y, output):
        self.output_error = y - output  # Обчислюємо помилку на виході
        self.output_delta = self.output_error * self.sigmoid_derivative(output)  # Обчислюємо дельта для вихідного шару

        self.hidden_error = np.dot(self.output_delta, self.weights_hidden_output.T)  # Обчислюємо помилку прихованого шару
        self.hidden_delta = self.hidden_error * self.sigmoid_derivative(self.activated_hidden)  # Обчислюємо дельта для прихованого шару

        # Оновлюємо ваги шляхом використання градієнтного спуску
        self.weights_input_hidden += np.dot(X.T, self.hidden_delta)
        self.weights_hidden_output += np.dot(self.activated_hidden.T, self.output_delta)

    # Навчання
    def train(self, X, y, epochs):
        for epoch in range(epochs):  # Проводимо навчання протягом заданої кількості епох
            output = self.forward(X)  # Виконуємо пряме поширення
            self.backward(X, y, output)  # Виконуємо зворотнє поширення

# Навчання нейронної мережі
nn = NeuralNetwork()
nn.train(input_combinations, output_values.reshape(-1, 1), epochs=1000)  # Виконуємо навчання мережі

# Тестування нейронної мережі
test_input = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [1, 0, 1, 0]])  # Вхідні дані для тестування
predictions = nn.forward(test_input)  # Виконуємо передбачення за допомогою навченої мережі
print("\nTest Predictions:")
for i in range(len(test_input)):
    print(f"Input: {test_input[i]}, Output: {predictions[i]}")  # Виводимо передбачені значення

# Аналіз результатів
def accuracy(predictions, targets):
    predictions = np.round(predictions)  # Заокруглюємо передбачені значення до цілих чисел
    correct = np.sum(predictions == targets)  # Обчислюємо кількість правильних передбачень
    total = len(targets)  # Загальна кількість тестових прикладів
    return correct / total  # Повертаємо точність передбачень

print("\nModel Performance:")
print(f"Accuracy: {accuracy(predictions, output_values)}")  # Виводимо точність моделі
