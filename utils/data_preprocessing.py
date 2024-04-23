from typing import Callable

import cv2
import tensorflow as tf


def load_data_cifake():
    train = tf.keras.preprocessing.image_dataset_from_directory(
        'datasets/CIFAKE/train',
        label_mode='binary',
        image_size=[32, 32]
    )

    valid = tf.keras.preprocessing.image_dataset_from_directory(
        'datasets/CIFAKE/test',
        label_mode='binary',
        image_size=[32, 32]
    )

    return train, valid


def load_data_artifact(dir_path):
    train, valid = tf.keras.preprocessing.image_dataset_from_directory(
        dir_path,
        label_mode='binary',
        image_size=[200, 200],
        validation_split=0.2,
        subset='both',
        shuffle=True,
        seed=42,
        interpolation='area',
        batch_size=256
    )

    return train, valid


def preprocess(
    image_size: tuple[int, int],
    preprocess_func: Callable[[cv2.Mat], cv2.Mat] | None = tf.keras.applications.imagenet_utils.preprocess_input,
    normalize: bool = False
) -> Callable[[cv2.Mat], cv2.Mat]:
    def func(image: cv2.Mat) -> cv2.Mat:
        resized = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
        preprocessed = preprocess_func(resized) if preprocess_func else resized
        return preprocessed if not normalize else preprocessed / 255

    return func
