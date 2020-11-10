# -*- coding: utf-8 -*-
# Project  : TensorFlowProjects
# File     : ep2.py
# Author   : guile
# Version  : v1.0
# Email    : lavender.lhy@gmail.com
# Date     : 2020-11-10 15:29
# Remarks  : 使用 tf.data 加载文本数据
# Tensorflow and keras
import tensorflow as tf
import tensorflow_datasets as tfds
import os

# 需要设置gpu的memory，，，否则报错
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
tf.config.experimental.set_memory_growth(gpus[0], True)


DIRECTORY_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
FILE_NAMES = ['cowper.txt', 'derby.txt', 'butler.txt']

for name in FILE_NAMES:
    text_dir = tf.keras.utils.get_file(name, origin=DIRECTORY_URL + name)

parent_dir = "/home/guile/.keras/datasets"


def labeler(example, index):
    return example, tf.cast(index, tf.int64)


labeled_data_sets = []

for i, file_name in enumerate(FILE_NAMES):
    line_dataset = tf.data.TextLineDataset(os.path.join(parent_dir, file_name))
    labeled_dataset = line_dataset.map(lambda ex: labeler(ex, i))
    labeled_data_sets.append(labeled_dataset)

BUFFER_SIZE = 50000
BATCH_SIZE = 65
TAKE_SIZE = 5000

all_labeled_data = labeled_data_sets[0]
for label_set in labeled_data_sets[1:]:
    all_labeled_data = all_labeled_data.concatenate(label_set)

all_labeled_data = all_labeled_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)

tokenizer = tfds.features.text.Tokenizer()

vocabulary_set = set()
for text_tensor, _ in all_labeled_data:
    some_tokens = tokenizer.tokenize(text_tensor.numpy())
    vocabulary_set.update(some_tokens)

vocab_size = len(vocabulary_set)

encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)
example_set = next(iter(all_labeled_data))[0].numpy()

encoded_example = encoder.encode(example_set)


def encode(tt, label):
    encoded_text = encoder.encode(tt.numpy())
    return encoded_text, label


def encode_map_fn(text, label):
    encoded_text, label = tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))
    encoded_text.set_shape([None])
    label.set_shape([])
    return encoded_text, label


all_encoded_data = all_labeled_data.map(encode_map_fn)

train_data = all_encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE)

test_data = all_encoded_data.take(TAKE_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE)

sample_text, sample_label = next(iter(test_data))
print(sample_text[0])
print()
print(sample_label[0])

vocab_size += 1

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 64))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))

for units in [64, 64]:
    model.add(tf.keras.layers.Dense(units, activation="relu"))

model.add(tf.keras.layers.Dense(3, activation="softmax"))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_data, epochs=3, validation_data=test_data)

eval_loss, eval_acc = model.evaluate(test_data)

print()
print("Eval Loss: {}, Eval Acc: {}".format(eval_loss, eval_acc))
