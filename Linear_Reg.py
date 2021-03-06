#%%
import tensorflow as tf
import matplotlib.pyplot as plt


def linear(W, b, x_data):
    return W * x_data + b


x_data = [1, 2, 3, 4, 5]
y_data = [1, 2, 3, 4, 5]

W = tf.Variable(5.0)
b = tf.Variable(0.5)

learning_rate = 0.01

for i in range(100 + 1):
    # Gradient Descent
    with tf.GradientTape() as tape:
        hypothesis = W * x_data + b
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))
    W_grad, b_grad = tape.gradient(cost, [W, b])
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)
    if i % 10 == 0:
        print("{:5}|{:10.4f}|{:10.4}|{:10.6f}|".format(i, W.numpy(), b.numpy(), cost))
        plt.plot(x_data, linear(W, b, x_data))
        plt.show()

# %%
