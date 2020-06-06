import numpy as np


def reshape_img_upload(img):
    img = img.resize((28, 28))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1, 28, 28, 1)
    return im2arr


def make_prediction(trained_model, img):
    pred = trained_model.predict_classes(img)
    return f"Predicted Number: {pred[0]}"
