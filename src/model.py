import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import io
from src import cleaning

class ImageCaptioningNet(nn.Module):
    def __init__(self, hidden_size=128):
      super(ImageCaptioningNet, self).__init__()
      self.hidden_size = hidden_size

    def forward(self, x):
      return x


if __name__ == "__main__":
    f = "coco\\images\\val2017\\000000000139.jpg"
    img = cleaning.clean_image(io.imread(f))
    my_nn = ImageCaptioningNet()
    print(img.shape)
    print(my_nn(img).shape)