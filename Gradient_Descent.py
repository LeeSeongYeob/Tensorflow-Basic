# Pure python
# import numpy as np

# X = np.array([1, 2, 3])
# Y = np.array([1, 2, 3])


# def cost_func(W, X, Y):
#     c = 0
#     for i in range(len(X)):
#         c += (W * X[i] - Y[i]) ** 2
#     return c / len(X)

# # W 의 값을 -3 부터 5까지 15구간으로 나누어 계산
# for feed_W in np.linspace(-3, 5, num=15):
#     curr_cost = cost_func(feed_W, X, Y)
#     print("{:6.3f} | {:10.5f}".format(feed_W, curr_cost))

# Cost func in tf
# import tensorflow as tf
# import numpy as np

# X = np.array([1, 2, 3])
# Y = np.array([1, 2, 3])


# def cost_fun(W, X, Y):
#     hypothesis = X * W
#     return tf.reduce_mean(tf.square(hypothesis - Y))


# W_valuse = np.linspace(-3, 5, num=15)
# cost_values = []

# for feed_W in W_valuse:
#     curr_cost = cost_fun(feed_W, X, Y)
#     cost_values.append(curr_cost)
#     print("{:6.3f} | {:10.5f}".format(feed_W, curr_cost))

#%%
import tensorflow as tf
import numpy as np

tf.random.set_seed(0)
X = np.array([1, 2, 3])
Y = np.array([1, 2, 3])

x_data = [1.0, 2.0, 3.0, 4.0]
y_data = [1.0, 3.0, 5.0, 7.0]

# W 에 임의 값 입력
W = tf.Variable(tf.random.normal((1,), -100.0, 100.0))
W = tf.Variable([6.0])


for step in range(300):
    hypothesis = X * W
    cost = tf.reduce_mean(tf.square(hypothesis - Y))

    # Gradient Descent
    alpha = 0.01
    gradient = tf.reduce_mean(tf.multiply(tf.multiply(W, X) - Y, X))
    descent = W - tf.multiply(gradient, alpha)
    W.assign(descent)

    if step % 10 == 0:
        print("{:5} | {:10.4f} | {:10.6f}".format(step, cost.numpy(), W.numpy()[0]))
