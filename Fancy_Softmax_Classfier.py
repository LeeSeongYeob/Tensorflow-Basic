import tensorflow as tf
import numpy as np

print(tf.__version__)
tf.random.set_seed(333)

xy_data = np.loadtxt("data-04-zoo.csv", delimiter=",", dtype=np.float32)
x_data = xy_data[:, :-1]  # 마지막 열을 제외한 모든 열
y_data = xy_data[:, -1]  # 마지막 열

print(x_data.shape, y_data.shape)
nb_classes = 7  # 0~ 6 까지 분류
Y_one_hot = tf.one_hot(y_data.astype(np.int32), nb_classes)
W = tf.Variable(tf.random.normal((16, nb_classes)), name="Weight")
b = tf.Variable(tf.random.normal((nb_classes,)), name="Bias")
variables = [W, b]

print(Y_one_hot.shape)


def logit(X):
    return tf.matmul(X, W) + b


# hypothesis - softmax
def hypothesis(X):
    return tf.nn.softmax(logit(X))


# cost -> using logit
def cost_fn(X, Y):
    cost = tf.keras.losses.categorical_crossentropy(
        y_true=Y, y_pred=logit(X), from_logits=True
    )
    return tf.reduce_mean(cost)


# gradient
def grad_fn(X, Y):
    with tf.GradientTape() as tape:
        loss = cost_fn(X, Y)
        grad = tape.gradient(loss, variables)
        return grad


def predict(X, Y):
    # hypothesis(x).shape = (101, 7)
    pred = tf.argmax(hypothesis(X), 1)
    correct = tf.equal(pred, tf.argmax(Y, 1))
    # correct dtype = bool
    accurary = tf.reduce_mean(tf.cast(correct, np.float32))
    return accurary


def fit(X, Y, epochs=1000, verbose=100):
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

    for i in range(epochs):
        grads = grad_fn(X, Y)
        optimizer.apply_gradients(zip(grads, variables))
        if (i == 0) | ((i + 1) % verbose == 0):
            acc = predict(X, Y).numpy()
            loss = cost_fn(X, Y).numpy()
            print("Steps: {} Loss: {}, Acc: {}%".format(i + 1, loss, int(acc * 100)))


fit(x_data, Y_one_hot)
