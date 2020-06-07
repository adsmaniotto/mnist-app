import numpy as np
import PIL

from model.train import IMG_ROWS, IMG_COLS


def upload_image(file):
    return PIL.Image.open(file).convert("L")


def reshape_img_upload(img) -> np.ndarray:
    img = img.resize((IMG_ROWS, IMG_COLS))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1, IMG_ROWS, IMG_COLS, 1)
    return im2arr