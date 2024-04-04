import keras.utils
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

mnist = tf.keras.datasets.mnist  # Завантаження набору даних MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # Розподіл набору даних на тренувальні та тестові дані

x_train = x_train / 255  # Нормалізація тренувальних даних
x_test = x_test / 255    # Нормалізація тестових даних

y_train_cat = keras.utils.to_categorical(y_train, 10)  # One-hot кодування міток класів для тренувальних даних
y_test_cat = keras.utils.to_categorical(y_test, 10)    # One-hot кодування міток класів для тестових даних

# Побудова моделі нейронної мережі
model = keras.Sequential([
    Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(28, 28, 1)),  # Згортковий шар з 32 фільтрами та активацією ReLU
    MaxPooling2D((2, 2), strides=2),  # Шар максимального пулінгу для зменшення розмірності зображення
    Conv2D(64, (3, 3), padding="same", activation="relu"),  # Згортковий шар з 64 фільтрами та активацією ReLU
    MaxPooling2D((2, 2), strides=2),  # Шар максимального пулінгу для зменшення розмірності зображення
    Flatten(),  # Перетворення зображення в одновимірний вектор перед подачею на повністю зв'язаний шар
    Dense(128, activation="relu"),  # Повністю зв'язаний шар з 128 нейронами та активацією ReLU
    Dense(10, activation="softmax")  # Вихідний шар з 10 нейронами та активацією softmax для класифікації
])

x_train = np.expand_dims(x_train, axis=3)  # Розширення розмірності тренувальних даних
x_test = np.expand_dims(x_test, axis=3)    # Розширення розмірності тестових даних
print(x_train.shape)

# Компіляція моделі
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Навчання моделі з 3 епохами
his = model.fit(x_train, y_train_cat, batch_size=32, epochs=3, validation_split=0.2)

# Оцінка точності моделі на тестових даних
model.evaluate(x_test, y_test_cat)

# Виведення структури моделі на консоль
print(model.summary())

# Функція для випадкового тестування зображення та візуалізації результатів
def testImage():
    index = np.random.randint(0, len(x_test))  # Випадковий вибір індексу зображення з тестових даних
    test_image = x_test[index]                 # Вибір тестового зображення
    test_label = y_test[index]                 # Вибір правильної мітки для тестового зображення

    prediction = model.predict(np.expand_dims(test_image, axis=0))  # Предикція класу за допомогою моделі

    # Візуалізація тестового зображення та його мітки
    plt.imshow(test_image.squeeze(), cmap="gray")
    plt.title(f"Actual: {test_label}, Predicted: {np.argmax(prediction)}, Number: {index}")
    plt.axis('off')
    plt.show()

# Цикл для випадкового тестування та візуалізації першого неправильно розпізнаного зображення
found_incorrect = False  # Змінна для відстеження того, чи було знайдено неправильно розпізнане зображення
while not found_incorrect:
    index = np.random.randint(0, len(x_test))  # Випадковий вибір індексу зображення з тестових даних
    test_image = x_test[index]                 # Вибір тестового зображення
    test_label = y_test[index]                 # Вибір правильної мітки для тестового зображення

    prediction = model.predict(np.expand_dims(test_image, axis=0))  # Предикція класу за допомогою моделі

    # Умова перевірки правильності передбачення
    if test_label != np.argmax(prediction):
        found_incorrect = True  # Позначення того, що було знайдено неправильно розпізнане зображення

        # Візуалізація неправильно розпізнаного зображення та його мітки
        plt.imshow(test_image.squeeze())
        plt.title(f"Actual: {test_label}, Predicted: {np.argmax(prediction)}, Number: {index}")
        plt.axis('off')
        plt.show()
