# app.py
from flask import Flask, request, jsonify, flash, abort
from flask_cors import *
from skimage import color
from skimage import io

app = Flask(__name__)
app.secret_key = "USSR_SUPPER_SEKRET_KEZ"
CORS(app, origins='*')

@app.route('/caption', methods=['POST'])
def caption():
    file = request.files['image']
    image = io.imread(file)
    return "I captioned your image {}".format(image.shape), 200

if __name__ == '__main__':
    app.run(threaded=True, port=5000)
