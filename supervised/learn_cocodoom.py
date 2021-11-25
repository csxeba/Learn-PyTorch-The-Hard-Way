import json
import os
from typing import List, Dict

from matplotlib import pyplot as plt
import torch
import torchvision.models.vgg
from torch import nn
from torch.nn import functional as F
from torch.utils import data

from utilities import DataDescriptor
from utilities import Masking
from utilities import Sample
from utilities import Metrics
from utilities import Result
from utilities import parse_args


class COCODoomDataset(data.Dataset):

    def __init__(self, images_root: str, jsonfile_path: str):
        super().__init__()
        self.images_root = images_root
        self.json_data = json.load(open(jsonfile_path, "r"))

        image_meta_index = {}
        for meta in self.json_data["images"]:
            meta["annotations"] = []
            image_meta_index[meta["id"]] = meta
            meta["image_path"] = os.path.join(self.images_root, meta["file_name"])

        valid_annos = filter(lambda annotation: annotation["category_id"] in DataDescriptor.ENEMY_TYPE_IDS,
                             self.json_data["annotations"])

        for anno in valid_annos:
            image_id = anno["image_id"]
            image_meta_index[image_id]["annotations"].append(anno)

        self.image_meta_index: List[dict] = list(image_meta_index.values())

    @staticmethod
    def _make_segmentation_mask(annotations: dict) -> torch.Tensor:
        """Generates a sparse mask"""
        canvas = torch.zeros(DataDescriptor.IMAGE_SHAPE_XY[:2], dtype=torch.int64)
        types = [DataDescriptor.ENEMY_TYPE_IDS.index(anno["category_id"]) for anno in annotations]
        segmentations = [anno["segmentation"] for anno in annotations]
        for class_idx, segmentation_repr in zip(types, segmentations):
            instance_mask = Masking.from_representation(segmentation_repr, DataDescriptor.IMAGE_SHAPE_XY[:2])
            canvas[instance_mask] = class_idx+1
        return canvas.long()

    @staticmethod
    def _make_input_image(image_meta: dict) -> torch.Tensor:
        image = plt.imread(image_meta["image_path"])
        image = torch.tensor(image)
        return torch.permute(image[..., :-1], (2, 0, 1))

    def __getitem__(self, index) -> Sample:
        meta = self.image_meta_index[index]
        image = self._make_input_image(meta)
        segmentation = self._make_segmentation_mask(meta["annotations"])
        return Sample(image=image, ground_truth=segmentation)

    def __len__(self) -> int:
        return len(self.image_meta_index)


class Backbone(nn.Module):

    def __init__(self, feature_layers: List[int]):
        super().__init__()
        self.wrapped_model = torchvision.models.vgg.vgg19(pretrained=True).features
        self.features = {}

        for layer_name in feature_layers:
            self._insert_forward_hooks(layer_name)

    def _insert_forward_hooks(self, layer_id):

        def hook(model, inputs, outputs):
            self.features[layer_id] = outputs

        layer = self.wrapped_model[layer_id]
        if layer is None:
            raise ValueError(f"Attempted to fetch non-existent layer from VGG19: {layer_id}")
        layer.register_forward_hook(hook)

    def _reset_forward_cache(self):
        self.features = {}

    @staticmethod
    def preprocess_input(x: torch.Tensor):
        x = x.float() / 255.
        x = F.pad(x, (0, 0, 0, 8))
        return x

    def forward(self, x):
        self._reset_forward_cache()
        x = self.preprocess_input(x)
        self.wrapped_model.forward(x)
        return self.features


class SemsegHead(nn.Module):

    def __init__(self, num_classes: int):
        super().__init__()
        # strides: 1, 2, 4, 8, 16
        # channels: 64, 128, 256, 512, 512

        # Stride 16
        self.conv_1 = nn.Conv2d(in_channels=512, out_channels=32, kernel_size=(3, 3), padding=1)
        self.relu_1 = nn.ReLU(inplace=True)
        self.upsc_1 = nn.UpsamplingBilinear2d(scale_factor=2)

        # Stride 8
        self.conv_2 = nn.Conv2d(in_channels=512 + 32, out_channels=32, kernel_size=(3, 3), padding=1)
        self.relu_2 = nn.ReLU(inplace=True)
        self.upsc_2 = nn.UpsamplingBilinear2d(scale_factor=2)

        # Stride 4
        self.conv_3 = nn.Conv2d(in_channels=256 + 32, out_channels=32, kernel_size=(3, 3), padding=1)
        self.relu_3 = nn.ReLU(inplace=True)
        self.upsc_3 = nn.UpsamplingBilinear2d(scale_factor=2)

        # Stride 2
        self.conv_4 = nn.Conv2d(in_channels=128 + 32, out_channels=32, kernel_size=(3, 3), padding=1)
        self.relu_4 = nn.ReLU(inplace=True)
        self.upsc_4 = nn.UpsamplingBilinear2d(scale_factor=2)

        # Stride 1
        self.conv_5 = nn.Conv2d(in_channels=64 + 32, out_channels=num_classes + 1, kernel_size=(3, 3), padding=1)

    def forward(self, inputs: Dict[int, torch.Tensor]):
        # 1, 6, 15, 24, 35

        feature_16 = inputs.pop(35)
        x = self.conv_1.forward(feature_16)
        x = self.relu_1.forward(x)
        x = self.upsc_1.forward(x)

        feature_8 = torch.cat([inputs.pop(24), x], dim=1)
        x = self.conv_2.forward(feature_8)
        x = self.relu_2.forward(x)
        x = self.upsc_2.forward(x)

        feature_4 = torch.cat([inputs.pop(15), x], dim=1)
        x = self.conv_3.forward(feature_4)
        x = self.relu_3.forward(x)
        x = self.upsc_3.forward(x)

        feature_2 = torch.cat([inputs.pop(6), x], dim=1)
        x = self.conv_4.forward(feature_2)
        x = self.relu_4.forward(x)
        x = self.upsc_4.forward(x)

        feature_1 = torch.cat([inputs.pop(1), x], dim=1)

        segmentation = self.conv_5.forward(feature_1)

        return segmentation


class Net(nn.Module):

    def __init__(self, adam_lr: float):
        super().__init__()
        # strides: 1, 2, 4, 8, 16
        # channels: 64, 128, 256, 512, 512
        self.backbone = Backbone(feature_layers=[1, 6, 15, 24, 35])
        self.head = SemsegHead(num_classes=DataDescriptor.NUM_CLASSES)
        optimizer = torch.optim.Adam(self.parameters(), lr=adam_lr)
        self.optimizer = optimizer

    def forward(self, x) -> torch.Tensor:
        features = self.backbone.forward(x)
        segmentation = self.head.forward(features)
        return segmentation

    def _inner_supervised_step(self, sample, optimize: bool):

        if optimize:
            self.train()
        else:
            self.eval()

        gt = F.pad(sample.ground_truth, (0, 0, 0, 8))

        segmentation_logits = self.forward(sample.image)
        classification_loss = F.cross_entropy(input=segmentation_logits, target=gt, reduction="mean")

        if optimize:
            self.optimizer.zero_grad()
            classification_loss.backward()
            self.optimizer.step()

        acc = Metrics.mean_accuracy(segmentation_logits, gt)
        iou = Metrics.mean_iou(segmentation_logits, gt, num_classes=DataDescriptor.NUM_CLASSES)

        result = Result.StepResult(classification_loss.item(), acc, iou)

        return result

    def train_step(self, sample: Sample) -> Result.StepResult:
        result = self._inner_supervised_step(sample, optimize=True)
        return result

    def eval_step(self, sample: Sample):
        result = self._inner_supervised_step(sample, optimize=False)
        return result


def train_loop(network: Net,
               train_loader: data.DataLoader,
               val_loader: data.DataLoader,
               epochs: int,
               metric_smoothing_window_size: int):

    for epoch in range(1, epochs+1):

        print(" [*] Training epoch", epoch)
        training_metric_aggregator = Result.Aggregator.make_empty()

        for i, sample in enumerate(train_loader, start=1):

            result = network.train_step(sample)
            training_metric_aggregator.push(result)
            report = training_metric_aggregator.report(smoothing_window_size=metric_smoothing_window_size)

            print(f"\r [*] Training step {i} / {len(train_loader)}"
                  f" - Loss: {report.loss:.4f}"
                  f" - Acc: {report.accuracy:>6.2%}"
                  f" - mIOU: {report.iou:.4f}", end="")

            if i % metric_smoothing_window_size == 0:
                print()

        print()

        eval_kpi_aggregator = Result.Aggregator.make_empty()
        for i, sample in enumerate(val_loader, start=1):
            print(f"\r [*] Eval step {i} / {len(val_loader)}", end="")
            eval_results = network.eval_step(sample)
            eval_kpi_aggregator.push(eval_results)
        print()

        report = eval_kpi_aggregator.report(smoothing_window_size=-1)
        print(f" [*] Eval step {len(train_loader)} / {len(train_loader)}"
              f" - Loss: {report.loss:.4f}"
              f" - Acc: {report.accuracy:>6.2%}"
              f" - mIOU: {report.iou:.4f}")

        print()


def main(images_root="/data/Datasets/cocodoom",
         train_json="/data/Datasets/cocodoom/run-train.json",
         val_json="/data/Datasets/cocodoom/run-val.json",
         epochs=30,
         batch_size=32,
         adam_lr=4e-3):

    if images_root is None:
        images_root, train_json, val_json, epochs, batch_size, adam_lr = parse_args()

    train_dataset = COCODoomDataset(images_root=images_root, jsonfile_path=train_json)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataset = COCODoomDataset(images_root=images_root, jsonfile_path=val_json)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    net = Net(adam_lr)

    train_loop(net, train_loader, val_loader, epochs=epochs, metric_smoothing_window_size=100)


if __name__ == '__main__':
    main()
