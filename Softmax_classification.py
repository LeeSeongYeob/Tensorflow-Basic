import numpy as np
import tensorflow as tf

tf.random.set_seed(777)

x_data = [
    [1, 2, 1, 1],
    [2, 1, 3, 2],
    [3, 1, 3, 4],
    [4, 1, 5, 5],
    [1, 7, 5, 5],
    [1, 2, 5, 6],
    [1, 6, 6, 6],
    [1, 7, 7, 7],
]
y_data = [
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 0, 0],
]

x_data = np.asarray(x_data, dtype=np.float32)
y_data = np.asarray(y_data, dtype=np.float32)
x_data
y_data


nb_class = 3
# 최종 결과가 a b c 3개 이므로 3개의 클래스로 지정

W = tf.Variable(tf.random.normal([4, nb_class]), name="weight")
b = tf.Variable(tf.random.normal([nb_class]), name="bias")
Variable = [W, b]
print(W, b, sep="\n")


# softmax 함수를 이용한 가설 함수
def hypothesis(X):
    return tf.nn.softmax(tf.matmul(X, W) + b)


print(hypothesis(x_data))


# cost계산함수
def cost_func(hypothesis, y_data):
    cost = tf.reduce_mean(-tf.reduce_sum(y_data * tf.math.log(hypothesis), axis=1))
    return cost


print(cost_func(hypothesis(x_data), y_data))


# 기울기를 계산하는 함수
def grad_func(x_data, y_data):
    with tf.GradientTape() as tape:
        cost = cost_func(hypothesis(x_data), y_data)
        grad = tape.gradient(cost, Variable)
        return grad


print(grad_func(x_data, y_data))


# SGD를 이용하여 비용의 최소 값 찾음
def fit(X, Y, epochs=2000, verbose=100):
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

    for i in range(epochs):
        grads = grad_func(X, Y)
        optimizer.apply_gradients(zip(grads, Variable))
        if (i == 0) | ((i + 1) % verbose == 0):
            print("Loss at epoch %d: %f" % (i + 1, cost_func(hypothesis(X), Y).numpy()))


fit(x_data, y_data)

sample_data = np.array([[2, 1, 3, 2]], dtype=np.float32)
#  answer_label [[0,0,1]]

a = hypothesis(sample_data)

print(
    "{:<10}".format(("hypothesis")), a, "\n{:<10}".format(("result")), tf.argmax(a, 1)
)

b = hypothesis(x_data)
print(b)
print("{:<9}".format(("predict")), tf.argmax(b, 1))
print("{:<9}".format(("result")), tf.argmax(y_data, 1))

