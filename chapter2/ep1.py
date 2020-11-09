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
import tensorflow as tf
from tensorflow import keras
from IPython.display import display, Image
import matplotlib.pyplot as plt

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


for n in range(3):
    ipt = random.choice(all_image_paths)
    display(Image(ipt))
    print(caption_image(ipt))
    print()
