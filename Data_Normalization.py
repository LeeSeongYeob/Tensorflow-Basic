import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
tf.random.set_seed(333)


XY = np.array(
    [
        [828.659973, 833.450012, 908100, 828.349976, 831.659973],
        [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
        [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
        [816, 820.958984, 1008100, 815.48999, 819.23999],
        [819.359985, 823, 1188100, 818.469971, 818.97998],
        [819, 823, 1198100, 816, 820.450012],
        [811.700012, 815.25, 1098100, 809.780029, 813.669983],
        [809.51001, 816.659973, 1398100, 804.539978, 809.559998],
    ]
)

x_train = XY[:, 0:-1]
y_train = XY[:, [-1]]

plt.plot(x_train, "bo")
plt.plot(y_train)
plt.show()


# 최대 최솟값은 열 끼리 비교하므로 axis=0 값 주어줌
def normalization(XY):
    num = XY - np.min(XY, axis=0)
    return num / (np.max(XY, 0) - np.min(XY, 0))


XY = normalization(XY)
print(XY)
x_train = XY[:, 0:-1]
y_train = XY[:, [-1]]
plt.plot(x_train, "bo")
plt.plot(y_train)
plt.show()

# batch -> 8
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))
W = tf.Variable(tf.random.normal((4, 4)))
b = tf.Variable(tf.random.normal((4,)))


def linearReg_fn(X):
    return tf.matmul(X, W) + b


def l2_loss(cost, beta=0.01):
    W_reg = tf.nn.l2_loss(W)
    loss = tf.reduce_mean(cost + W_reg * beta)
    return loss


def loss_fn(X, label, flag):
    cost = tf.reduce_mean(tf.square(linearReg_fn(X) - label))
    if flag:
        cost = l2_loss(cost)
    return cost


is_decay = True
starter_learning_rate = 0.1

if is_decay:
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=starter_learning_rate,
        decay_steps=50,
        decay_rate=0.96,
        staircase=True,
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate)
else:
    optimizer = tf.keras.optimizers.SGD(starter_learning_rate)


def grad(features, labels, flag):
    with tf.GradientTape() as tape:
        loss_val = loss_fn(linearReg_fn(features), labels, flag)
    return tape.gradient(loss_val, [W, b]), loss_val


# dateset 의 피쳐와 라벨은 무엇인지?
# ->dataset 에 넣은 x_train 과 y_train
EPOCHS = 101
for step in range(EPOCHS):
    for features, labels in dataset:
        features = tf.cast(features, tf.float32)
        labels = tf.cast(labels, tf.float32)
        grads, loss_value = grad(features, labels, False)
        optimizer.apply_gradients(grads_and_vars=zip(grads, [W, b]))
    if step % 10 == 0:
        print("\nIter: {}, Loss: {:.4f}, ".format(step, loss_value))
        print("Leaning Rate: {:.8f}, ".format(optimizer._decayed_lr("float32")))
