from PIL import Image
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
    transforms.Resize((SIZE, SIZE)),
    transforms.GaussianBlur(GAUSSIAN_KERNAL),
    transforms.Normalize(MEAN, STD),
])


def clean(img: np.array) -> np.array:
    return transform(img)

if __name__ == "__main__":
    img = Image.open(r"C:\Users\tallo\OneDrive\Documents\projects\ada_submission\coco\images\val2017\000000061418.jpg")
    print(clean(img).shape)