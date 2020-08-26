import tensorflow as tf
import numpy as np
from tensorflow import keras

print(tf.__version__)

imdb = tf.keras.datasets.imdb

# train data = 25,000, test_data = 25,000
# num_words -> data size 유지. (상위 자주 등장하는 10,000개의 단어)
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print("train : ", train_data.shape, train_labels.shape)
print("test : ", test_data.shape, test_labels.shape)

# data 는 어휘사전의 특정 단어를 나타내는 정수
# labels은 0, 1로 구성. 0(부정) , 1(긍정)
print(train_data[0])
print(train_labels[0])
# 단어와 정수 index 매치
word_index = imdb.get_word_index()

# 3개의 index 추가
# 정수로 표현된 data_set 을 문자열로 변환.
# word_index -> "문자열" : index
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0  # 공백
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

# key 와 value 값 바꾸어 줌
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


# index 와 문자열 mapping
def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


print(decode_review(train_data[0]))
# data processing 과정. tensor 로 변환 해주어야함
# sol 1. one-hot encoding. data size -> num_word * train_data size
# sol 2. padding. data size -> max_length * train_data
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data, value=0, padding="post", maxlen=256
)

test_data = keras.preprocessing.sequence.pad_sequences(
    test_data, value=0, padding="post", maxlen=256
)

print(len(train_data[0]), len(test_data[0]))
print(train_data[0])
# layer 구성하기.
# 모델에서 얼마나 많은 층을 사용할 것인가?
# 각 층에서 얼마나 많은 은닉 유닛(hidden unit)을 사용할 것인가?
vocab_size = 10000
# model = tf.keras.Sequential(
#     [
#         tf.keras.layers.Embedding(vocab_size,16), # voca 사이즈 = word_num, 16차원
#         tf.keras.layers.GlobalAveragePooling1D(),
#         tf.keras.layers.Dense(16,activation="relu"),
#         tf.keras.layers.Dense(1,activation="sigmoid") # 0과 1
#     ]
# )
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16, input_shape=(None,)))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()

# 손실 함수, optimizer
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# validation set 만들기 -> train_set 에서 학습 및 검증
# test data는 마지막에 한번 만 검증 용으로 사용
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# 모델 훈련
# batch 사이즈는 메모리에 맞게 적절히 설정
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=40,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=1,
)
results = model.evaluate(test_data, test_labels, verbose=2)

print(results)
