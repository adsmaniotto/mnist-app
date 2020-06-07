import numpy as np
from model.train import IMG_COLS, IMG_ROWS


def reshape_img_upload(img) -> np.ndarray:
    img = img.resize((IMG_ROWS, IMG_COLS))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1, IMG_ROWS, IMG_COLS, 1)
    return im2arr


def make_prediction(trained_model, img) -> str:
    pred = trained_model.predict_classes(img)
    return f"Predicted Digit: {pred[0]}"
