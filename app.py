from flask import Flask, render_template, request
import logging
import PIL
from tensorflow.keras.models import model_from_json

from model.predict import make_prediction, reshape_img_upload

app = Flask(__name__, template_folder='templates')
_logger = logging.getLogger(__name__)


class InvalidFormatException(Exception):
    pass


def load_mnist_model():
    json_file = open('model/model.json', 'r')
    model_json = json_file.read()
    json_file.close()
    trained_model = model_from_json(model_json)
    trained_model.load_weights("model/weights.h5")
    return trained_model


def upload_image(file):
    return PIL.Image.open(file).convert("L")


@app.route('/')
def upload_file():
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    if request.method == 'GET':
        _logger.info('health status OK')
        return 'ok'


@app.route('/upload', methods=['POST'])
def run_prediction():
    try:
        img = upload_image(request.files['file'].stream)
    except PIL.UnidentifiedImageError:
        raise InvalidFormatException("Invalid image uploaded.")
    im2arr = reshape_img_upload(img)
    return make_prediction(model, im2arr)


if __name__ == '__main__':
    model = load_mnist_model()
    app.run(host='0.0.0.0', debug=True)
