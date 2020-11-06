# -*- coding: utf-8 -*-
# Project  : TensorFlowProjects
# File     : ep3.py
# Author   : guile
# Version  : v1.0
# Email    : lavender.lhy@gmail.com
# Date     : 2020-11-06 15:50
# Remarks  : 使用 Keras 和 Tensorflow Hub 对电影评论进行文本分类
# Tensorflow and keras
import tensorflow as tf
from tensorflow import keras

import numpy as np
from matplotlib import pyplot as plt

import tensorflow_hub as hub
import tensorflow_datasets as tfds

# print(f"Version: {tf.__version__}")
# print(f"Eager mode: {tf.executing_eagerly()}")
# print(f"Hub version: {hub.__version__}")
# print(f"GPU is {'available' if tf.config.experimental.list_physical_devices('GPU') else 'NOT AVAILABLE'}")

train_data, validation_data, test_data, = tfds.load(name="imdb_reviews",
                                                    split=("train[:60%]", "train[60%:]", "test"),
                                                    as_supervised=True)

train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
# print(train_examples_batch)
# print(train_labels_batch)

embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)
print(hub_layer(train_examples_batch[:3]))

model = keras.Sequential()
model.add(hub_layer)
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1))
# model.summary()

model.compile(optimizer="adam", loss=keras.losses.BinaryCrossentropy(from_logits=True), metrics=["accuracy"])

history = model.fit(train_data.shuffle(10000).batch(512), epochs=20, validation_data=validation_data.batch(512), verbose=1)

results = model.evaluate(test_data.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.3f}")
