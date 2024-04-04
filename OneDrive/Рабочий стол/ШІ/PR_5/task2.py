import numpy as np  # Імпортуємо бібліотеку numpy для роботи з масивами та математичними операціями

# Генеруємо випадкові дані про мобільні телефони
np.random.seed(0)  # Встановлюємо початкове значення генератора випадкових чисел для відтворюваності результатів
num_samples = 1000  # Кількість зразків даних

# Генеруємо розмір екрану (від 4 до 9 дюймів)
screen_size = np.random.uniform(4, 9, size=num_samples)

# Генеруємо камеру (від 8 до 128 Мп)
camera = np.random.uniform(8, 128, size=num_samples)

# Генеруємо об'єм пам'яті (від 32 до 512 ГБ)
memory = np.random.choice([32, 64, 128, 256, 512], size=num_samples)

# Генеруємо ціну (від 500 до 1999)
price = np.random.randint(500, 1999, size=num_samples)

# Виводимо перші 7 записів для перевірки
print("First 7 records:")
print("Screen Size | Camera (MP) | Memory | Price")
for i in range(7):
    print(f"{screen_size[i]:.1f} inches  | {camera[i]:.1f} MP      | {memory[i]} GB  | ${price[i]}")

# Функція нормалізації ознак
def normalize_feature(feature):
    return (feature - np.min(feature)) / (np.max(feature) - np.min(feature))

# Нормалізуємо ознаки
screen_size = normalize_feature(screen_size)
camera = normalize_feature(camera)
memory = normalize_feature(memory)
price = normalize_feature(price)

# Виводимо перші 7 записів після нормалізації
print("\nFirst 7 records after normalization:")
print("Screen Size | Camera (MP) | Memory | Price")
for i in range(7):
    print(f"{screen_size[i]:.3f}       | {camera[i]:.3f}       | {memory[i]}       | ${price[i]}")

# Використовуємо попередній клас нейронної мережі для класифікації мобільних телефонів та їх цін

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Ініціалізуємо ваги
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    # Сигмоїдна функція активації
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Пряме поширення
    def forward(self, X):
        self.hidden_sum = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.activated_hidden = self.sigmoid(self.hidden_sum)

        self.output_sum = np.dot(self.activated_hidden, self.weights_hidden_output) + self.bias_output
        self.activated_output = self.sigmoid(self.output_sum)

        return self.activated_output

# Перевірка роботоспособності
input_size = 4  # Кількість функцій (розмір екрану, камера, пам'ять, ціна)
hidden_size = 5  # Кількість нейронів у прихованому шарі
output_size = 1  # Кількість вихідних класів (ціна)

# Навчаємо модель
epochs = 10  # Кількість епох навчання
model = NeuralNetwork(input_size, hidden_size, output_size)
output = model.forward(np.array([screen_size, camera, memory, price]).T)  # Початкове передбачення
for epoch in range(epochs):
    # Проходимо через кожен навчальний приклад
    for i in range(len(screen_size)):
        X = np.array([screen_size[i], camera[i], memory[i], price[i]]).reshape(1, -1)  # Вхідний приклад
        output = model.forward(np.array([screen_size, camera, memory, price]).T)  # Пряме поширення

    # Оцінюємо модель після кожної епохи
    if (epoch + 1) % 1 == 0:
        predictions = model.forward(np.array([screen_size, camera, memory, price]).T)
        loss = np.mean(np.square(price - predictions))
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

# Тестуємо модель
test_input = np.array([[0.6, 0.8, 0.3, 0.5],   # Розмір екрану: 6 дюймів, камера: 48 Мп, пам'ять: 128 ГБ, ціна: 800
                       [0.3, 0.5, 0.6, 0.8],   # Розмір екрану: 3 дюйми, камера: 40 Мп, пам'ять: 256 ГБ, ціна: 1800
                       [0.8, 0.6, 0.4, 0.7]])  # Розмір екрану: 8 дюймів, камера: 56 Мп, пам'ять: 160 ГБ, ціна: 1500
predictions = model.forward(test_input)
print("\nPredictions for test data:")
for i in range(len(test_input)):
    print(f"Screen size: {test_input[i, 0] * 3} inches, Camera: {test_input[i, 1] * 56 + 8} MP, Memory: {int(test_input[i, 2] * 480) + 32} GB, Price: ${int(test_input[i, 3] * 1800) + 200}")
