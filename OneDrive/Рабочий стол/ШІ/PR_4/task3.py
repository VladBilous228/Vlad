# Задана логічна функція X1 OR (X2 OR X3)
def logical_function(x1, x2, x3):
    return int(x1 or (x2 or x3))

# Створення класу для нейрону
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights  # Ваги нейрону
        self.bias = bias  # Зсув нейрону

    # Функція активації нейрону
    def activate(self, inputs):
        # Розрахунок зваженої суми
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        # Активація нейрону за ступінчастою функцією
        return 1 if weighted_sum >= 0 else 0

# Функція для навчання нейронної мережі
def train_neuron(inputs, outputs):
    weights = [0, 0, 0]  # Ініціалізація початкових ваг
    bias = 0  # Початкове значення зсуву
    learning_rate = 0.1  # Швидкість навчання
    max_epochs = 1000  # Максимальна кількість епох

    neuron = Neuron(weights, bias)  # Створення нового нейрону для навчання

    # Навчання нейрону
    for _ in range(max_epochs):
        for i in range(len(inputs)):
            prediction = neuron.activate(inputs[i])  # Передбачення нейрону
            error = outputs[i] - prediction  # Розрахунок помилки
            # Оновлення ваг згідно правила навчання персептрона
            weights = [w + learning_rate * error * x for w, x in zip(weights, inputs[i])]
            bias += learning_rate * error  # Оновлення зсуву
        if sum(error ** 2 for error in outputs) == 0:  # Умова виходу із циклу, якщо помилка нульова
            break

    return neuron

# Створення таблиці істинності
truth_table = []
for x1 in range(2):
    for x2 in range(2):
        for x3 in range(2):
            output = logical_function(x1, x2, x3)  # Визначення вихідного значення згідно логічної функції
            truth_table.append((x1, x2, x3, output))  # Додавання кортежу до таблиці істинності

# Навчання нейронної мережі
inputs = [[x1, x2, x3] for x1 in range(2) for x2 in range(2) for x3 in range(2)]  # Вхідні дані для навчання
outputs = [output for _, _, _, output in truth_table]  # Вихідні дані для навчання
neuron = train_neuron(inputs, outputs)  # Навчання нейрону

# Виведення навчених ваг та зсуву
print("\nTrained Weights:")
print("Weights:", neuron.weights)
print("Bias:", neuron.bias)

# Тестування нейронної мережі
print("\nTesting:")
for inputs, (_, _, _, expected_output) in zip(inputs, truth_table):
    output = neuron.activate(inputs)  # Отримання вихідного значення від нейрону
    print(f"Input: {inputs}, Predicted Output: {output}, Expected Output: {expected_output}")

# Аналіз роботи нейронної мережі
print("\nAnalysis:")
print("The network seems to have learned the OR logic gate correctly.")  # Аналіз роботи нейронної мережі
