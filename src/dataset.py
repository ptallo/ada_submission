import json
import os

import numpy as np
import torch
from PIL import Image

from clean.image import clean as clean_img
from clean.text import clean as clean_txt, clean_batch
from clean.vocab import load_vocab


class CocoAnnotationDataset(object):
    def __init__(self, path_to_images='coco\\images\\val2017', path_to_annotations='coco\\annotations\\captions_val2017.json', vocab_path='vocab.txt'):
        json = self.get_annotations_json(path_to_annotations)
        self.vocab = load_vocab(vocab_path)
        self.annotations = json['annotations']

        self.images_json = json['images']
        self.path_to_images = path_to_images

    def get_annotations_json(self, path):
        f = open(path, 'r+')
        json_data = json.loads(f.read())
        f.close()
        return json_data

    def __getitem__(self, idx):
        img_path = self.images_json[idx]['file_name']
        annotations = [x['caption'] for x in self.annotations if x['image_id'] == self.images_json[idx]['id']]
        annotations = torch.transpose(clean_batch(annotations, self.vocab), 0, 1) # [seq_len, batch_size]
        
        image = clean_img(Image.open(os.path.join(self.path_to_images, img_path)))
        image = image.repeat(annotations.shape[1], 3, 1, 1) if image.shape[0] == 1 else image.repeat(annotations.shape[1], 1, 1, 1)

        return image, annotations

    def __len__(self):
        return len(self.annotations)

if __name__ == "__main__":
    count = 0
    for (img, ann) in CocoAnnotationDataset():
        count += 1
        if count > 3:
            break
        print("Image Shape = {}".format(img.shape))
        for annotation in ann:
            print("  {}".format(annotation))

        print("\n")
