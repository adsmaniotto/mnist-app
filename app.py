from flask import Flask, render_template, request
import logging
import PIL
from tensorflow.keras.models import model_from_json
from model.utils import reshape_img_upload, upload_image

app = Flask(__name__, template_folder='templates')
_logger = logging.getLogger(__name__)


class InvalidFormatException(Exception):
    pass


json_file = open('model/model.json', 'r')
model_json = json_file.read()
json_file.close()

MODEL = model_from_json(model_json)
MODEL.load_weights("model/weights.h5")


@app.route('/')
def render_html():
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    if request.method == 'GET':
        _logger.info('health status OK')
        return 'ok'


@app.route('/upload', methods=['POST'])
def run_prediction() -> str:
    try:
        img = upload_image(request.files['file'].stream)
    except PIL.UnidentifiedImageError:
        raise InvalidFormatException("Invalid image uploaded.")
    im2arr = reshape_img_upload(img)
    pred = MODEL.predict_classes(im2arr)
    return f"Predicted Digit: {pred[0]}"


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
