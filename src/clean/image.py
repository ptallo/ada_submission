import numpy as np
from torch import nn
from torchvision import transforms
from skimage import color

GAUSSIAN_KERNAL = 3
MEAN = 122
STD = 50
SIZE = 400

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(1),
    transforms.Resize((SIZE, SIZE)),
    transforms.GaussianBlur(GAUSSIAN_KERNAL),
    transforms.Normalize(MEAN, STD),
])


def clean(img: np.array) -> np.array:
    return transform(img)