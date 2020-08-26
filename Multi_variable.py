#%%
import tensorflow as tf

x1 = [73.0, 93.0, 89.0, 96.0, 73.0]
x2 = [80.0, 88.0, 91.0, 98.0, 66.0]
x3 = [75.0, 93.0, 90.0, 100.0, 70.0]
Y = [152.0, 185.0, 180.0, 196.0, 142.0]

tf.random.set_seed(0)
W1 = tf.Variable(tf.random.normal((1,)))
W2 = tf.Variable(tf.random.normal((1,)))
W3 = tf.Variable(tf.random.normal((1,)))
b = tf.Variable(tf.random.normal((1,)))

learning_rate = 0.000001

for i in range(1000 + 1):
    with tf.GradientTape() as tape:
        hypothesis = W1 * x1 + W2 * x2 + W3 * x3 + b
        cost = tf.reduce_mean(tf.square(hypothesis - Y))

        # cost의 기울기 계산
    W1_grad, W2_grad, W3_grad, b_grad = tape.gradient(cost, [W1, W2, W3, b])
    # update 해줌
    W1.assign_sub(learning_rate * W1_grad)
    W2.assign_sub(learning_rate * W2_grad)
    W3.assign_sub(learning_rate * W3_grad)
    b.assign_sub(learning_rate * b_grad)
    if i % 50 == 0:
        print("{:5} | {:12.4f}".format(i, cost.numpy()))


# %%
