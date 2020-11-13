# -*- coding: utf-8 -*-
# Project  : TensorFlowProjects
# File     : ep8.py
# Author   : guile
# Version  : v1.0
# Email    : lavender.lhy@gmail.com
# Date     : 2020-11-12 11:30
# Remarks  : TFRecord 和 tf.Example
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as display


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# print(_bytes_feature(b"test_string"))
# print(_bytes_feature("test_bytes".encode("utf-8")))
#
# feature = _float_feature(np.exp(1))
# b = feature.SerializeToString()
# print(b)
#
# print(_int64_feature(True))
# print(_int64_feature(1))
#
# # 创建 tf.Example 消息
#
# n_observations = int(1e4)
# print(n_observations)
#
# feature0 = np.random.choice([True, False], n_observations)
#
# feature1 = np.random.randint(0, 5, n_observations)
#
# strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
# feature2 = strings[feature1]
#
# feature3 = np.random.randn(n_observations)
#
# # print(feature0, feature1, feature2, feature3)
#
#
# def serialize_example(f0, f1, f2, f3):
#     f = {
#         "feature0": _int64_feature(f0),
#         "feature1": _int64_feature(f1),
#         "feature2": _bytes_feature(f2),
#         "feature3": _float_feature(f3),
#     }
#     example_proto = tf.train.Example(features=tf.train.Features(feature=f))
#     return example_proto.SerializeToString()
#
#
# example_observation = []
# serialized_example = serialize_example(False, 4, b'goat', 0.9876)
# print(serialized_example)
#
# example_proto = tf.train.Example.FromString(serialized_example)
# print(example_proto)
#
# # TFRecords 格式详细信息
#
# features_dataset = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2, feature3))
# print(features_dataset)
#
#
# def tf_serialize_example(f_0, f_1, f_2, f_3):
#     tf_string = tf.py_function(
#         serialize_example,
#         (f_0, f_1, f_2, f_3),    # pass these args to the above function.
#         tf.string)    # the return type is `tf.string`.
#     return tf.reshape(tf_string, ())    # The result is a scalar
#
#
# #
# # for a, b, c, d in features_dataset.take(1):
# #     t = tf_serialize_example(a, b, c, d)
# #     print(t)
#
#
# def generator():
#     for features in features_dataset:
#         yield serialize_example(*features)
#
#
# serialized_features_dataset = tf.data.Dataset.from_generator(generator, output_types=tf.string, output_shapes=())
#
# print(serialized_features_dataset)
#
# file_name = "test.tfrecord"
# writer = tf.data.experimental.TFRecordWriter(file_name)
# writer.write(serialized_features_dataset)
#
# filenames = [file_name]
# raw_dataset = tf.data.TFRecordDataset(filenames)
# print(raw_dataset)
#
# for raw_record in raw_dataset.take(10):
#     print(repr(raw_record))
#
# # Create a description of the features.
# feature_description = {
#     "feature0": tf.io.FixedLenFeature([], tf.int64, default_value=0),
#     "feature1": tf.io.FixedLenFeature([], tf.int64, default_value=0),
#     "feature2": tf.io.FixedLenFeature([], tf.string, default_value=''),
#     "feature3": tf.io.FixedLenFeature([], tf.float32, default_value=0.0)
# }
#
#
# def _parse_function(example_proto):
#     return tf.io.parse_sequence_example(example_proto, feature_description)
#
#
# parsed_dataset = raw_dataset.map(_parse_function)
# print(parsed_dataset)
#
# for parsed_record in parsed_dataset.take(10):
#     print(repr(parsed_record))

# 演练：读取和写入图像数据
cat_in_snow = tf.keras.utils.get_file(
    "320px-Felis_catus-cat_on_snow.jpg",
    'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg')
williamsburg_bridge = tf.keras.utils.get_file(
    '194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg'
)

print(cat_in_snow)
print(williamsburg_bridge)

images_labels = {cat_in_snow: 0, williamsburg_bridge: 1}
image_string = open(cat_in_snow, 'rb').read()

label = images_labels[cat_in_snow]


def image_example(img_string, img_label):
    image_shape = tf.image.decode_jpeg(img_string).shape

    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(img_label),
        'image_raw': _bytes_feature(img_string)
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


for line in str(image_example(image_string, label)).split("\n")[:15]:
    print(line)

print("...")

record_file = 'images.tfrecords'

with tf.io.TFRecordWriter(record_file) as writer:
    for filename, label in images_labels.items():
        image_string = open(filename, 'rb').read()
        tf_example = image_example(image_string, label)
        writer.write(tf_example.SerializeToString())

raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')
image_feature_description = {
    "height": tf.io.FixedLenFeature([], tf.int64),
    "width": tf.io.FixedLenFeature([], tf.int64),
    "depth": tf.io.FixedLenFeature([], tf.int64),
    "label": tf.io.FixedLenFeature([], tf.int64),
    "image_raw": tf.io.FixedLenFeature([], tf.string)
}


def _parse_image_function(example_proto):
    return tf.io.parse_single_example(example_proto, image_feature_description)


parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
print(parsed_image_dataset)

# for image_features in parsed_image_dataset:
#     image_raw = image_features['image_raw'].numpy()
#     display.display(display.Image(data=image_raw))
