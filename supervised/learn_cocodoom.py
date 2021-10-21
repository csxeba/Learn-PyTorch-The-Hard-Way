import json
import os

import torch
import torchvision.models.vgg
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision.io import read_image

import utilities


class COCODoomDataset(data.Dataset):

    ENEMY_TYPE_IDS = [1, 2, 3, 5, 8, 10, 11, 12, 14, 15, 17, 18, 19, 20, 21, 22]
    ENEMY_TYPE_NAMES = ["POSSESSED", "SHOTGUY", "VILE", "UNDEAD", "FATSO", "CHAINGUY", "TROOP", "SERGEANT", "HEAD",
                        "BRUISER", "KNIGHT", "SKULL", "SPIDER", "BABY", "CYBORG", "PAIN"]
    IMAGE_WIDTH = 320
    IMAGE_HEIGHT = 200
    IMAGE_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, 3)

    def __init__(self, images_root: str, jsonfile_path: str):
        super().__init__()
        self.images_root = images_root
        self.json_data = json.load(open(jsonfile_path, "r"))

        self.image_meta_index = {}
        for meta in self.json_data["images"]:
            meta["annotations"] = []
            self.image_meta_index[meta["id"]] = meta

        for anno in (a for a in self.json_data["annotations"] if a["category_id"] in self.ENEMY_TYPE_IDS):
            self.image_meta_index[anno["image_id"]].append(anno)

    def _make_segmentation_mask(self, annotations):
        canvas = torch.zeros(self.IMAGE_WIDTH, self.IMAGE_HEIGHT, dtype=torch.int64)
        types = [self.ENEMY_TYPE_IDS.index(anno["category_id"]) for anno in annotations]
        segmentations = [anno["segmentation"] for anno in annotations]
        for class_idx, segmentation_repr in zip(types, segmentations):
            instance_mask = utilities.mask_from_representation(segmentation_repr, self.IMAGE_SHAPE)
            canvas[instance_mask] = class_idx+1
        return canvas

    def __getitem__(self, index):
        meta = self.image_meta_index[index]
        image = read_image(os.path.join(self.images_root, meta["file_name"]))
        segmentation = self._make_segmentation_mask(meta["annotations"])
        return image, segmentation

    def __len__(self):
        return len(self.image_meta_index)


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.vgg.vgg19(pretrained=True).features
        print("Anchor ME")


def main():
    net = Net()


if __name__ == '__main__':
    main()
