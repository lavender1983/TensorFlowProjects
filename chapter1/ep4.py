# -*- coding: utf-8 -*-
# Project  : TensorFlowProjects
# File     : ep4.py
# Author   : guile
# Version  : v1.0
# Email    : lavender.lhy@gmail.com
# Date     : 2020-11-06 16:37
# Remarks  : Basic regression: Predict fuel efficiency
# Tensorflow  keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# helpers
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

print(tf.__version__)

dataset_path = keras.utils.get_file("auto-mpg.data",
                                    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values="?", comment="\t", sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()

dataset = dataset.dropna()

origin = dataset.pop("Origin")

dataset["USA"] = (origin == 1) * 1.0
dataset["Europe"] = (origin == 2) * 1.0
dataset["Japan"] = (origin == 3) * 1.0

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")

# show
# plt.show()

train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_status = train_stats.transpose()
print(train_status.tail())

train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')


def norm(x, tt):
    return (x - tt['mean']) / tt['std']


normed_train_data = norm(train_dataset, train_status)
normed_test_data = norm(test_dataset, train_status)


def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation="relu", input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)
    ])
    optimizer = keras.optimizers.RMSprop(0.001)
    model.compile(loss="mse", optimizer=optimizer, metrics=["mae", "mse"])
    return model


md = build_model()
print(md.summary())

example_batch = normed_train_data[:10]
example_result = md.predict(example_batch)
print(example_result)
