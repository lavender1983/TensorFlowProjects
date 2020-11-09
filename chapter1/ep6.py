# -*- coding: utf-8 -*-
# Project  : TensorFlowProjects
# File     : ep6.py
# Author   : guile
# Version  : v1.0
# Email    : lavender.lhy@gmail.com
# Date     : 2020-11-09 14:56
# Remarks  : 保存和恢复模型
import os

import tensorflow as tf
from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


# define model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation="relu", input_shape=(784,)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(10),
    ])

    model.compile(optimizer="adam",
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    return model


md = create_model()
md.summary()

# checkpoint_path = "training_1/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
#
# # 创建一个保存模型权重的回调
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
#
# # 使用新的回调函数训练模型
# md.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels), callbacks=cp_callback)

# md.load_weights(checkpoint_path)
#
# loss, acc = md.evaluate(test_images, test_labels, verbose=2)
# print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

# checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# 创建一个回调，每5个epoch保存模型的权重
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True, period=5)

# save model weight and fit
# md.save_weights(checkpoint_path.format(epoch=0))
# md.fit(train_images,
#        train_labels,
#        epochs=50,
#        callbacks=[cp_callback],
#        validation_data=(test_images, test_labels),
#        verbose=0)

# latest = tf.train.latest_checkpoint(checkpoint_dir)
# md.load_weights(latest)
#
# loss, acc = md.evaluate(test_images, test_labels, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# md.fit(train_images, train_labels, epochs=5)
# md.save('save_model/my_model')

# new_model = tf.keras.models.load_model('save_model/my_model')
# new_model.summary()
# loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
# print("Restore model , accuracy: {:5.2f}%".format(acc * 100))
# print(new_model.predict(test_images).shape)

# md.fit(train_images, train_labels, epochs=5)
# md.save('h5/my_model.h5')

new_model = tf.keras.models.load_model('h5/my_model.h5')
new_model.summary()

loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print("Restore model, accuracy: {:5.2f}".format(acc*100))

