from typing import Tuple
import numpy as np
import pandas as pd
import json
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
from skimage import color
from skimage import io


def load_data(path: str) -> Tuple[np.ndarray, str]:
    with open(path, 'r+') as f:
        json_data = json.loads(f.read())
        for y in json_data['annotations']:
            image_json = [x for x in json_data['images']if y['image_id'] == x['id']][0]
            yield clean_image(load_image(image_json['file_name'])), cleanp.ndarray str_text(y)


def load_image(file_name: str, folder:str='coco/images/val2017') -> np.ndarray:
    return io.imread(os.path.join(folder, file_name))

def clean_image(img: np.ndarray) -> np.ndarray: 
    return color.rgb2gray(img)

def clean_text(txt: str) -> str:
    return txt

if __name__ == "__main__":
    for idx, (img, text) in enumerate(load_data('coco/annotations/captions_val2017.json')):
        if idx > 1:
            break

        plt.imshow(img, xcmap=plt.get_cmap('gray'), vmin=0, vmax=1)
        plt.show()


