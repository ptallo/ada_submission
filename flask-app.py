# app.py
from flask import Flask, request, jsonify, flash, abort
from flask_cors import *
from skimage import color
from skimage import io
from coco_model import clean_image, clean_text

app = Flask(__name__)
CORS(app, origins='*')

@app.route('/caption', methods=['POST'])
def caption():
    file = request.files['image']
    image = clean_image(io.imread(file))
    return "{}\n{}".format(image.shape, image[0][:10]), 200

if __name__ == '__main__':
    app.run(threaded=True, port=5000)
