# app.py
from flask import Flask, request, jsonify, flash, abort
from flask_cors import *
from skimage import io
from src import cleaning, dataset, model

app = Flask(__name__)
CORS(app, origins='*')

m = model.Model()

@app.route('/caption', methods=['POST'])
def caption():
    file = request.files['image']
    image = cleaning.clean_image(io.imread(file))
    return m.infer(image), 200

if __name__ == '__main__':
    app.run(threaded=True, port=5000)
