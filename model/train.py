from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical

NUM_CLASSES = 10


def build_model(input_shape) -> Sequential:
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(
        loss="categorical_crossentropy",
        optimizer="rmsprop",
        metrics=['accuracy']
    )
    return model


def reshape_data(train_data, img_rows=28, img_cols=28):
    train_data = train_data.reshape(train_data.shape[0], img_rows, img_cols, 1)
    train_data = train_data.astype('float32')
    train_data /= 255

    return train_data


def import_data():
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    img_rows, img_cols = 28, 28
    x_train = reshape_data(x_train)
    x_test = reshape_data(x_test)
    input_shape = (img_rows, img_cols, 1)

    # convert class vectors to binary class matrices
    y_train = to_categorical(y=y_train, num_classes=NUM_CLASSES)
    y_test = to_categorical(y=y_test, num_classes=NUM_CLASSES)
    return x_train, x_test, y_train, y_test, input_shape


def train_model():
    x_train, x_test, y_train, y_test, input_shape = import_data()
    model = build_model(input_shape)

    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        batch_size=128,
        epochs=10,
        verbose=1,
    )
    return model


if __name__ == "__main__":
    trained_model = train_model()
    model_json = trained_model.to_json()
    with open("model/model.json", "w") as json_file:
        json_file.write(model_json)

    trained_model.save_weights("model/weights.h5")
