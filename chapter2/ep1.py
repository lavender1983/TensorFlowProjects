# -*- coding: utf-8 -*-
# Project  : TensorFlowProjects
# File     : ep1.py
# Author   : guile
# Version  : v1.0
# Email    : lavender.lhy@gmail.com
# Date     : 2020-11-09 16:56
# Remarks  : 用 tf.data 加载图片
# Tensorflow and keras
import pathlib
import random
import os
import time
import tensorflow as tf
from tensorflow import keras
from IPython.display import display, Image
import matplotlib.pyplot as plt

# 需要设置gpu的memory，，，否则报错
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
tf.config.experimental.set_memory_growth(gpus[0], True)

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

AUTOTUNE = tf.data.experimental.AUTOTUNE

data_root_orig = tf.keras.utils.get_file(
    origin="https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
    fname="flower_photos",
    untar=True)

data_root = pathlib.Path(data_root_orig)
print(data_root)

for item in data_root.iterdir():
    print(item)

all_image_path = list(data_root.glob("*/*"))
all_image_paths = [str(path) for path in all_image_path]

attributions = (data_root / "LICENSE.txt").open(encoding="utf-8").readlines()[4:]
attributions = [line.split(" CC-BY") for line in attributions]
attributions = dict(attributions)


def caption_image(image_path):
    image_rel = pathlib.Path(image_path).relative_to(data_root)
    return "Image (CC BY 2.0) " + ' - '.join(attributions[str(image_rel)].split(' - ')[:-1])


def change_range(image, label):
    return 2 * image - 1, label


def timeit(dataset, steps=2):
    steps = 2 * steps + 1
    overall_start = time.time()
    # 在开始计时之前
    # 取得单个batch来填充pipeline 填充随机缓冲区
    it = iter(dataset.take(steps + 1))
    next(it)

    start = time.time()
    for i, (images, labels) in enumerate(it):
        if i % 10 == 0:
            print(".", end="")
    print()
    end = time.time()

    duration = end - start
    print(f"{steps} batches: {duration} s")
    print("{:0.5f} Images/s".format(BATCH_SIZE * steps / duration))
    print("Total time: {}s".format(end - overall_start))


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0
    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


def load_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label


# 列出可用标签
label_names = sorted(item.name for item in data_root.glob("*/") if item.is_dir())
# 索引
label_to_index = dict((name, index) for index, name in enumerate(label_names))
all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]

# img_path = all_image_paths[0]
# img_raw = tf.io.read_file(img_path)
#
# img_tensor = tf.image.decode_image(img_raw)
# print(img_tensor.shape)
# print(img_tensor.dtype)
#
# # 调整大小
# img_final = tf.image.resize(img_tensor, [192, 192])
# img_final = img_final / 255.0

# path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
# image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

# plt.figure(figsize=(8, 8))
# for n, img in enumerate(image_ds.take(4)):
#     plt.subplot(2, 2, n + 1)
#     plt.imshow(img)
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#     plt.xlabel(caption_image(all_image_paths[n]))
# plt.show()


BATCH_SIZE = 32
image_count = len(all_image_paths)

label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
# image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
# print(image_label_ds)
#
# ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
#
# image_label_ds = ds.map(load_preprocess_from_path_label)
# print(image_label_ds)

# TFRecord

# 方法 1
# 231.0 batches: 11.517234086990356 s
# 641.82077 Images/s
# Total time: 16.730788230895996s

# image_ds = tf.data.Dataset.from_tensor_slices(all_image_paths).map(tf.io.read_file)
# tfrec = tf.data.experimental.TFRecordWriter('images.tfrec')
# tfrec.write(image_ds)
#
# image_ds = tf.data.TFRecordDataset('images.tfrec').map(preprocess_image)
#
# ds = tf.data.Dataset.zip((image_ds, label_ds))
# ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
# ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


# 方法2 添加预处理
path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds = path_ds.map(load_and_preprocess_image)
print(image_ds)

ds = image_ds.map(tf.io.serialize_tensor)
print(ds)

tfrec = tf.data.experimental.TFRecordWriter("images-2.tfrec")
tfrec.write(ds)

ds = tf.data.TFRecordDataset('images-2.tfrec')


def parse(x):
    result = tf.io.parse_tensor(x, out_type=tf.float32)
    result = tf.reshape(result, [192, 192, 3])
    return result


ds = ds.map(parse, num_parallel_calls=AUTOTUNE)

ds = tf.data.Dataset.zip((ds, label_ds))
ds = ds.apply(
    tf.data.experimental.shuffle_and_repeat(buffer_size=image_count)
)
ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
print(ds)
timeit(ds)
# end TFRecord

# 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据
# 被充分打乱。
# ds = image_label_ds.shuffle(buffer_size=image_count)
# ds = ds.repeat()
# ds = ds.batch(BATCH_SIZE)
# ds = ds.prefetch(buffer_size=AUTOTUNE)

# ds = image_label_ds.cache(filename="cache/cache.tf-data")
# ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
# ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
# print(ds)

mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
mobile_net.trainable = False
# help(keras.applications.mobilenet_v2.preprocess_input)

# def solve_cudnn_error():
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     if gpus:
#         try:
#             # Currently, memory growth needs to be the same across GPUs
#             for gpu in gpus:
#                 tf.config.experimental.set_memory_growth(gpu, True)
#             logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#             print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#         except RuntimeError as e:
#             # Memory growth must be set before GPUs have been initialized
#             print(e)
#
#
# solve_cudnn_error()

keras_ds = ds.map(change_range)

# 数据集可能需要几秒来启动，因为要填满其随机缓冲区。
image_batch, label_batch = next(iter(keras_ds))
feature_map_batch = mobile_net(image_batch)
print(feature_map_batch.shape)

model = tf.keras.Sequential([
    mobile_net,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(label_names), activation="softmax")
])

logit_batch = model(image_batch).numpy()
print(f"min logit: {logit_batch.min()}")
print(f"max logit: {logit_batch.max()}")
print()
print(f"Shape: {logit_batch.shape}")
print(len(model.trainable_variables))
print(model.summary())

model.compile(optimizer=tf.keras.optimizers.Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
steps_per_epoch = tf.math.ceil(len(all_image_paths) / BATCH_SIZE).numpy()
# 231.0 batches: 10.579161643981934 s
# 698.73212 Images/s
# Total time: 15.81494951248169s
timeit(ds, steps_per_epoch)

# model.fit(ds, epochs=1, steps_per_epoch=steps_per_epoch)
