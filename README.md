# mnist-app
An MNIST prediction micro-service

This repo contains a trained MNIST classification model, hosted in a Flask app.

The user can upload a PNG file, and the app will return a predicted value of the handwritten
digit (0-9) in that image.

To run:
- In Python: `python3 app.py`
- In Docker: `docker build -t mnist . && docker run mnist:latest`