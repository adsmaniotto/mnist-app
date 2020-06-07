# mnist-app
An MNIST prediction micro-service

This repo contains a trained MNIST classification model, hosted in a Flask app.

The client requests an image upload from the user, and the app will return a predicted value of the handwritten
digit (0-9) contained in that image.

To run:
- [Heroku app](https://adsmaniotto-mnist.herokuapp.com)
- In Python: `python3 app.py`
- In Docker: `docker build -t mnist . && docker run mnist:latest`