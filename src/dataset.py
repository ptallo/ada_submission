import json
import os

import numpy as np
import torch
from PIL import Image

from clean.image import clean as clean_img
from clean.text import clean as clean_txt


class CocoAnnotationDataset(object):
    def __init__(self, path_to_images='coco/images/val2017', path_to_annotations='coco/annotations/captions_val2017.json'):
        json = self.get_annotations_json(path_to_annotations)

        self.annotations = json['annotations']

        self.images_json = json['images']
        self.images = [Image.open(os.path.join(path_to_images, x['file_name'])) for x in self.images_json]

    def get_annotations_json(self, path):
        f = open(path, 'r+')
        json_data = json.loads(f.read())
        f.close()
        return json_data

    def __getitem__(self, idx):
        image = clean_img(self.images[idx])
        annotations = [clean_txt(x['caption']) for x in self.annotations if x['image_id'] == self.images_json[idx]['id']]
        return image, annotations

    def __len__(self):
        return len(self.images_json)

if __name__ == "__main__":
    count = 0
    for (img, ann) in CocoAnnotationDataset():
        count += 1
        if count > 3:
            break
        print("Image Shape = {}, Number of Annotations = {}".format(img.shape, len(ann)))
