#%%
import tensorflow as tf
import numpy as np

# H(X) = XW
data = np.array(
    [
        # X1,   X2,    X3,   y
        [73.0, 80.0, 75.0, 152.0],
        [93.0, 88.0, 93.0, 185.0],
        [89.0, 91.0, 90.0, 180.0],
        [96.0, 98.0, 100.0, 196.0],
        [73.0, 66.0, 70.0, 142.0],
    ],
    dtype=np.float32,
)

# 데이터 쪼개기
X = data[:, :-1]
Y = data[:, [-1]]

W = tf.Variable(tf.random.normal([3, 1]))
b = tf.Variable(tf.random.normal([1]))

learning_rate = 0.0000001


def predict(X):
    return tf.matmul(X, W) + b


for i in range(2000 + 1):
    with tf.GradientTape() as tape:
        cost = tf.reduce_mean(tf.square(predict(X) - Y))
    W_grad, b_grad = tape.gradient(cost, [W, b])
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)

    if i % 100 == 0:
        print("{:5} | {:10.4f}".format(i, cost.numpy()))

# %%
