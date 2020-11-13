# -*- coding: utf-8 -*-
# Project  : TensorFlowProjects
# File     : ep5.py
# Author   : guile
# Version  : v1.0
# Email    : lavender.lhy@gmail.com
# Date     : 2020-11-11 10:42
# Remarks  : 使用 tf.data 加载 pandas dataframes
import pandas as pd
import tensorflow as tf

csv_file = tf.keras.utils.get_file("heart.csv", 'https://storage.googleapis.com/applied-dl/heart.csv')
print(csv_file)
df = pd.read_csv(csv_file)
df["thal"] = pd.Categorical(df["thal"])
df["thal"] = df.thal.cat.codes

target = df.pop("target")

dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))

# for feat, targ in dataset.take(5):
#     print("Features: {}, Target: {}".format(feat, targ))

train_dataset = dataset.shuffle(len(df)).batch(1)


def get_compiled_model():
    md = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    md.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return md


model = get_compiled_model()
model.fit(train_dataset, epochs=15)

inputs = {key: tf.keras.layers.Input(shape=(), name=key) for key in df.keys()}
x = tf.stack(list(inputs.values()), axis=1)

x = tf.keras.layers.Dense(10, activation="relu")(x)
output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model_func = tf.keras.Model(inputs=inputs, outputs=output)
model_func.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

dict_slices = tf.data.Dataset.from_tensor_slices((df.to_dict("list"), target.values)).batch(16)

for dict_slice in dict_slices.take(1):
    print(dict_slice)

model_func.fit(dict_slices, epochs=15)