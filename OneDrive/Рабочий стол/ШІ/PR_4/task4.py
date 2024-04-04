import random
import matplotlib.pyplot as plt

# Функція активації (ступінчаста функція)
def activation_function(x):
    return 1 if x >= 0 else 0

# Клас для реалізації персептрону
class Perceptron:
    def __init__(self, num_inputs, activation_function):
        # Ініціалізація ваг та зсуву персептрону
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)
        self.activation_function = activation_function  # Функція активації

    # Функція для передбачення вихідного значення персептрону
    def forward(self, inputs):
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return self.activation_function(weighted_sum)

    # Функція для оновлення ваг персептрону за правилом навчання
    def update_weights(self, inputs, target, learning_rate):
        prediction = self.forward(inputs)
        error = target - prediction
        self.weights = [w + learning_rate * error * x for w, x in zip(self.weights, inputs)]
        self.bias += learning_rate * error

# Функції для визначення цільових лінійних функцій
def target_function1(x):
    return 2 * x - 2.5

def target_function2(x):
    return 2 * x - 2.5

num_points = 100
training_data = []
# Генерація випадкових точок та надання їм міток
for _ in range(num_points):
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    label1 = 1 if y > target_function1(x) else 0
    label2 = 1 if y > target_function2(x) else 0
    training_data.append(([x, y], label1, label2))

# Створення персептронів
perceptron1 = Perceptron(num_inputs=2, activation_function=activation_function)
perceptron2 = Perceptron(num_inputs=2, activation_function=activation_function)

learning_rate = 0.1  # Швидкість навчання
epochs = 100  # Кількість епох навчання
# Навчання персептронів
for epoch in range(epochs):
    for inputs, label1, label2 in training_data:
        perceptron1.update_weights(inputs, label1, learning_rate)
        perceptron2.update_weights(inputs, label2, learning_rate)

# Визначення значень X та Y для цільових функцій
x_values = [-1, 1]
y_values1 = [target_function1(x) for x in x_values]
y_values2 = [target_function2(x) for x in x_values]

# Побудова графіка цільових функцій
plt.plot(x_values, y_values1, label='Target Function 1')
plt.plot(x_values, y_values2, label='Target Function 2')

# Побудова точок на графіку для навчальних даних
for inputs, label1, label2 in training_data:
    color1 = 'red' if label1 == 1 else 'blue'
    color2 = 'green' if label2 == 1 else 'yellow'
    plt.scatter(inputs[0], inputs[1], color=color1, marker='o')
    plt.scatter(inputs[0], inputs[1], color=color2, marker='x')

plt.xlabel('X')  # Підпис осі X
plt.ylabel('Y')  # Підпис осі Y
plt.title('Linear Separation of Two Classes')  # Заголовок графіку
plt.legend()  # Додавання легенди
plt.show()  # Показ графіку
