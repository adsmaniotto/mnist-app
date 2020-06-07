"""
This script contains code for training of the MNIST model and it not used in the runtime application
"""

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from typing import NamedTuple

NUM_CLASSES = 10
IMG_ROWS = 28
IMG_COLS = 28


class MNISTData(NamedTuple):
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def build_model(input_shape) -> Sequential:
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(NUM_CLASSES, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer="rmsprop",
        metrics=["accuracy"]
    )
    return model


def reshape_image_data(train_data) -> np.ndarray:
    train_data = train_data.reshape(train_data.shape[0], IMG_ROWS, IMG_COLS, 1)
    train_data = train_data.astype("float32")
    train_data /= 255
    return train_data


def convert_labels_to_categorical(label_data) -> np.ndarray:
    return to_categorical(y=label_data, num_classes=NUM_CLASSES)


def import_data() -> MNISTData:
    """ Import MNIST data and transform it to a format fit for model training """

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = reshape_image_data(x_train)
    x_test = reshape_image_data(x_test)

    # convert class vectors to binary class matrices
    y_train = convert_labels_to_categorical(y_train)
    y_test = convert_labels_to_categorical(y_test)
    mnist_data = MNISTData(x_train, x_test, y_train, y_test)
    return mnist_data


def output_model_json(model) -> None:
    model_json = model.to_json()
    with open("model/model.json", "w") as json_file:
        json_file.write(model_json)


def train_model() -> Sequential:
    mnist_data = import_data()
    input_shape = (IMG_ROWS, IMG_COLS, 1)
    model = build_model(input_shape)

    model.fit(
        mnist_data.x_train,
        mnist_data.y_train,
        validation_data=(mnist_data.x_test, mnist_data.y_test),
        batch_size=128,
        epochs=10,
        verbose=1,
    )
    return model


if __name__ == "__main__":
    """ Train the MNIST model + save the model to JSON and model weights """
    trained_model = train_model()
    output_model_json(trained_model)
    trained_model.save_weights("model/weights.h5")
