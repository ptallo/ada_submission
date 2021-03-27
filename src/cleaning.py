from typing import Tuple
import numpy as np
from skimage import color
from skimage.transform import resize


def clean_image(img: np.ndarray) -> np.ndarray: 
    image = color.rgb2gray(img)
    image = [[x / 255 for x in y] for y in image]
    return np.array(image)

def clean_text(txt: str) -> str:
    return txt

