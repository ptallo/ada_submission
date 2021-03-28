# app.py
import torch.nn
from flask import Flask, abort, flash, jsonify, request
from flask_cors import *
from skimage import io

from src import cleaning, dataset, model

app = Flask(__name__)
CORS(app, origins='*')

m: torch.nn = model.ImageCaptioningNet()

@app.route('/caption', methods=['POST'])
def caption():
    file = request.files['image']
    image = cleaning.clean_image(io.imread(file))
    return model(image), 200

if __name__ == '__main__':
    app.run(threaded=True, port=5000)
