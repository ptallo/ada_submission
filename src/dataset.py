import os
import json
import numpy as np
import torch
import cleaning
from PIL import Image


class CocoAnnotationDataset(object):
    def __init__(self, path_to_images='coco/images/val2017', path_to_annotations='coco/annotations/captions_val2017.json'):
        self.path_to_images = path_to_images
        json = self.get_annotations_json(path_to_annotations)
        self.annotations = json['annotations']
        self.images_json = json['images']

    def get_annotations_json(self, path):
        f = open(path, 'r+')
        json_data = json.loads(f.read())
        f.close()
        return json_data

    def __getitem__(self, idx):
        image_json = self.images_json[idx]

        image = Image.open(os.path.join(self.path_to_images, image_json['file_name']))
        image = cleaning.clean_image(np.array(image))
        annotations = [cleaning.clean_text(x['caption']) for x in self.annotations if x['image_id'] == image_json['id']]

        return image, annotations

    def __len__(self):
        return len(self.images_json)

if __name__ == "__main__":
    dataset = CocoAnnotationDataset()
    print(len(dataset))
    count = 0
    for (img, ann) in dataset:
        count += 1
        if count > 3:
            break

        print("Image Shape - {}".format(img.shape))
        print("Annotations...")
        for a in ann:
            print("  {}".format(a))
        print("\n")
