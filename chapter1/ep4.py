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

# sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")

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
# print(md.summary())

example_batch = normed_train_data[:10]
example_result = md.predict(example_batch)
# print(example_result)


class PrintDot(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0:
            print(' ')
        print(".", end='')


EPOCHS = 1000

# history = md.fit(normed_train_data,
#                  train_labels,
#                  epochs=EPOCHS,
#                  validation_split=0.2,
#                  verbose=0,
#                  callbacks=[PrintDot()])

# hist = pd.DataFrame(history.history)
# hist["epoch"] = history.epoch
# print("\n")
# print(hist.tail())

# patience 值用来检查改进 epochs 的数量
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = md.fit(normed_train_data,
                 train_labels,
                 epochs=EPOCHS,
                 validation_split=0.2,
                 verbose=0,
                 callbacks=[early_stop, PrintDot()])

print("\n" * 2)


def plot_history(ht):
    hist = pd.DataFrame(ht.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'], label='Val Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'], label='Val Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()


# plot_history(history)

loss, mae, mse = md.evaluate(normed_test_data, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))


test_predictions = md.predict(normed_test_data).flatten()
# plt.scatter(test_labels, test_predictions)
# plt.xlabel("True Value [MPG]")
# plt.ylabel("Predictions [MPG]")
# plt.axis('equal')
# plt.axis('square')
# plt.ylim([0, plt.xlim()[1]])
# plt.ylim([0, plt.ylim()[1]])
# _ = plt.plot([-100, 100], [-100, 100])

error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel("Prediction Error [MPG]")
plt.ylabel("Count")

plt.show()