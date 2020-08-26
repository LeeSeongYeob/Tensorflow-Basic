import tensorflow as tf
import matplotlib.pyplot as plt

print(tf.__version__)

mnist = tf.keras.datasets.mnist
# 60,000 장의 data, size -> 28 * 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)

plt.figure(figsize=(7, 7))
plt.imshow(x_train[0])
plt.colorbar()
plt.grid(False)
plt.show()

# Data Nomalization
# x_train 한 원소의 픽셀당 0-255의 값을 가지므로 255로 나눔
x_train, x_test = x_train / 255, x_test / 255

# layer 만듬
# dropout 규제만듬. -> overfitting
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy", "binary_crossentropy"],
)

model.fit(x_train, y_train, batch_size=1024, epochs=5)

model.evaluate(x_test, y_test)
