# -*- coding: utf-8 -*-
# Project  : TensorFlowProjects
# File     : ep5.py
# Author   : guile
# Version  : v1.0
# Email    : lavender.lhy@gmail.com
# Date     : 2020-11-09 10:29
# Remarks  : Overfit and underfit (过拟合，欠拟合)
# Tensorflow and keras
import tensorflow as tf

from tensorflow.keras import layers, regularizers

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

import numpy as np
import matplotlib.pyplot as plt
import pathlib
import shutil
import tempfile

log_dir = pathlib.Path(tempfile.mkdtemp()) / "tensorboard_logs"
print(log_dir)
shutil.rmtree(log_dir, ignore_errors=True)

gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')

FEATURES = 28

ds = tf.data.experimental.CsvDataset(gz, [
    float(),
] * (FEATURES + 1), compression_type="GZIP")


def pack_row(*row):
    label = row[0]
    features = tf.stack(row[1:], 1)
    return features, label


packed_ds = ds.batch(10000).map(pack_row).unbatch()

# for features, label in packed_ds.batch(1000).take(1):
#     print(features[0])
#     plt.hist(features.numpy().flatten(), bins=101)
#
# plt.show()

N_VALIDATION = int(1e3)
N_TRAIN = int(1e4)
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE

validate_ds = packed_ds.take(N_VALIDATION).cache()
train_ds = packed_ds.skip(N_VALIDATION).take(N_TRAIN).cache()

validate_ds = validate_ds.batch(BATCH_SIZE)
train_ds = train_ds.shuffle(BATCH_SIZE).repeat().batch(BATCH_SIZE)

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.001,
                                                             decay_steps=STEPS_PER_EPOCH * 1000,
                                                             decay_rate=1,
                                                             staircase=False)


def get_optimizer():
    return tf.keras.optimizers.Adam(lr_schedule)


# step = np.linspace(0, 100000)
# lr = lr_schedule(step)
# plt.figure(figsize=(8, 6))
# plt.plot(step / STEPS_PER_EPOCH, lr)
# plt.ylim([0, max(plt.ylim())])
# plt.xlabel('Epoch')
# plt.ylabel('Learning Rate')
# plt.show()


def get_callbacks(name):
    return [
        tfdocs.modeling.EpochDots(),
        tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
        tf.keras.callbacks.TensorBoard(log_dir / name)
    ]


def compile_and_fit(model, name, optimizer=None, max_epochs=10000):
    if not optimizer:
        optimizer = get_optimizer()
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.losses.BinaryCrossentropy(from_logits=True, name="binary_crossentropy"), "accuracy"])

    model.summary()

    history = model.fit(train_ds,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        epochs=max_epochs,
                        validation_data=validate_ds,
                        callbacks=get_callbacks(name),
                        verbose=0)
    return history


tiny_model = tf.keras.Sequential([layers.Dense(16, activation="elu", input_shape=(FEATURES,)), layers.Dense(1)])
size_histories = {}
size_histories['Tiny'] = compile_and_fit(tiny_model, 'sizes/Tiny')

# plotter = tfdocs.plots.HistoryPlotter(metric="binary_crossentropy", smoothing_std=10)
# plotter.plot(size_histories)
# plt.ylim([0.5, 0.7])
# plt.show()

small_model = tf.keras.Sequential(
    [layers.Dense(16, activation="elu", input_shape=(FEATURES,)),
     layers.Dense(16, activation="elu"),
     layers.Dense(1)])

size_histories["Small"] = compile_and_fit(small_model, "sizes/Small")

medium_model = tf.keras.Sequential([
    layers.Dense(64, activation="elu", input_shape=(FEATURES,)),
    layers.Dense(64, activation="elu"),
    layers.Dense(64, activation="elu"),
    layers.Dense(1)
])

size_histories['Medium'] = compile_and_fit(medium_model, "sizes/Medium")

large_model = tf.keras.Sequential([
    layers.Dense(512, activation="elu", input_shape=(FEATURES,)),
    layers.Dense(512, activation="elu"),
    layers.Dense(512, activation="elu"),
    layers.Dense(512, activation="elu"),
    layers.Dense(1)
])

size_histories["Large"] = compile_and_fit(large_model, "sizes/Large")

plotter = tfdocs.plots.HistoryPlotter(metric="binary_crossentropy", smoothing_std=10)
plotter.plot(size_histories)
a = plt.xscale('log')
plt.xlim([5, max(plt.xlim())])
plt.ylim([0.5, 0.7])
plt.xlabel("Epochs [Log Scale]")
plt.show()
