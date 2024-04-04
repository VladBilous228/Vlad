from numpy import exp, array, random, dot  # Імпорт необхідних функцій та бібліотек з бібліотеки NumPy
from numpy import *  # Імпорт усіх функцій та об'єктів з бібліотеки NumPy під псевдонімом np
import numpy as np  # Імпорт бібліотеки NumPy під псевдонімом np

# Функція, яка рахує вихід персептрона
def calcOutput(inputs, weight):
    return 1 / (1 + exp(-(dot(inputs, weight))))

# Функція, яка рахує зміну ваги
def calcWeight(inputs, outputs, output):
    return dot(inputs.T, (outputs - output) * output * (1 - output))

# Функція, яка знаходить відповідь за заданою вхідною послідовністю
def find_answer(arr):
    # Вхідні та вихідні дані для навчання персептрона
    training_set_inputs = array([[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 1]])
    training_set_outputs = array([[1, 1, 1, 1, 1, 1]]).T

    random.seed(1)  # Фіксація випадкових значень для відтворюваності результатів

    weights = 2 * random.random((3, 1)) - 1  # Ініціалізація випадкових ваг

    # Навчання персептрона
    for iteration in range(100):
        output = 1 / (1 + exp(-(dot(training_set_inputs, weights))))
        weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))

    # Виведення відповіді за вхідною послідовністю arr
    print(1 / (1 + exp(-dot(arr, weights))))

# Функція, яка знаходить відповідь для логічної функції АБО
def find_OR_answer():
    # Вхідні та вихідні дані для навчання персептрона для логічної функції АБО
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    random.seed(1)  # Фіксація випадкових значень для відтворюваності результатів

    synaptic_weight = 2 * random.random((3, 1)) - 1  # Ініціалізація випадкових ваг

    # Навчання персептрона для логічної функції АБО
    for iteration in range(10000):
        output = calcOutput(training_set_inputs, synaptic_weight)
        synaptic_weight += calcWeight(training_set_inputs, training_set_outputs, output)

    # Виведення результату для певного вхідного значення
    print("Логічна функція АБО число 6 в двійковому форматі [1, 1, 0] ->  ?: ")
    print(1 / (1 + exp(-dot(array([0, 0, 1]), synaptic_weight))))

# Функція активації (ступінчаста функція)
def activation_function(x):
    return 1 if x >= 0 else 0

# Функція персептрона
def perceptron(X, w, b):
    return activation_function(np.dot(X, w) + b)

# Вхідні дані (X1, X2, X3) та вагові коефіцієнти для персептрона
X = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
               [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
w = np.array([1, 1, 1])  # Вагові коефіцієнти
b = -1  # Зсув

# Вивід результатів для кожної комбінації вхідних даних
for i in range(len(X)):
    result = perceptron(X[i], w, b)
    print(f"Вхід: {X[i]}, Вихід: {result}")

# Функція для тестування нейронної мережі
def test1():
    # Вхідні дані для тестування
    arr = array([1, 1, 0])
    arr1 = array([0, 0, 1])
    arr2 = array([0, 1, 1])
    arr3 = array([0, 0, 0])
    arr4 = array([0, 1, 0])

    # Тестування та виведення результатів
    find_answer(arr)
    find_answer(arr1)
    find_answer(arr2)
    find_answer(arr3)
    find_answer(arr4)

# Виклик функції для знаходження відповіді для логічної функції АБО
find_OR_answer()
