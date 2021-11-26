"""
Several helpers are included here for the learn_cocodoom script.

Normally the helpers would be factored into different submodules,
but for simplicity's sake they are just be wrapped into class definitions.
Outer classes are just namespaces without much OOP functionality.
"""
import argparse
import dataclasses
from typing import NamedTuple

from matplotlib import pyplot as plt
import numpy as np
import cv2
import torch


class Masking:

    """
    MSCOCO/COCODoom mask generating utils taken from
    https://github.com/csxeba/Verres.git
    """

    @staticmethod
    def _decode_poly(poly, shape):
        full_mask = np.zeros(shape, dtype="uint8")
        pts = [np.round(np.array(p).reshape(-1, 2)).astype(int) for p in poly]
        return cv2.fillPoly(full_mask, pts, color=1).astype(bool)

    @staticmethod
    def _decode_rle(rle, shape):
        full_mask = np.zeros(np.prod(shape), dtype=bool)
        fill = False
        start = 0
        for num in rle["counts"]:
            end = start + num
            full_mask[start:end] = fill
            fill = not fill
            start = end
        return full_mask.reshape(shape[::-1]).T

    @classmethod
    def from_representation(cls, segmentation_repr, image_shape):
        if isinstance(segmentation_repr, list):
            return cls._decode_poly(segmentation_repr, image_shape).astype(bool)
        elif "counts" in segmentation_repr:
            return cls._decode_rle(segmentation_repr, image_shape)


class Metrics:

    @staticmethod
    def mean_accuracy(prediction: torch.Tensor, ground_truth: torch.Tensor) -> float:
        predicted_classes = prediction.argmax(dim=1)
        equalities = torch.eq(predicted_classes, ground_truth)
        accuracy = torch.mean(equalities.float())
        return float(accuracy.item())

    @staticmethod
    def mean_iou(predicition: torch.Tensor, ground_truth: torch.Tensor, num_classes: int) -> float:
        predicted_classes = predicition.argmax(dim=1)
        ious = []
        for class_idx in range(num_classes):
            pred_class = torch.eq(predicted_classes, class_idx)
            gt_class = torch.eq(ground_truth, class_idx)
            intersection = torch.sum(torch.logical_and(pred_class, gt_class))
            union = torch.sum(torch.logical_or(pred_class, gt_class))
            iou = float(intersection.item()) / max(float(union.item()), 1.)
            ious.append(iou)
        return float(np.mean(ious))


class Visualizer:

    @staticmethod
    def visualize_segmentation_mask(mask: torch.Tensor,
                                    image: torch.Tensor = None,
                                    alpha: float = 0.5):

        mask = mask.detach().numpy()
        if image is None:
            image = np.zeros(mask.shape + (3,), dtype="uint8")
        else:
            image = image.detach().numpy()

        assert mask.ndim == 2
        assert image.ndim == 3
        assert mask.shape[:2] == image.shape[1:]

        image = image.transpose((1, 2, 0))

        for i in range(1, DataDescriptor.NUM_CLASSES):
            x, y = np.where(mask == i)
            image[x, y] = DataDescriptor.COLORS[i] * alpha + image[x, y] * (1 - alpha)

        plt.imshow(image)
        plt.show()


class Result:

    class StepResult(NamedTuple):

        """
        Class for holding the KPIs for a single (training or validation) step.
        """

        loss: float
        accuracy: float
        iou: float

    @dataclasses.dataclass
    class Aggregator:

        """
        Class for logging KPIs.
        """

        losses: np.ndarray
        accs: np.ndarray
        ious: np.ndarray
        maxlen: int

        @classmethod
        def make_empty(cls, maxlen: int = -1):
            return cls(np.array([], dtype="float32"), np.array([], dtype="float32"), np.array([], dtype="float32"),
                       maxlen=maxlen)

        def push(self, step_result: "Result.StepResult"):
            self.losses = np.concatenate([self.losses, [step_result.loss]])
            self.accs = np.concatenate([self.accs, [step_result.accuracy]])
            self.ious = np.concatenate([self.ious, [step_result.iou]])
            if self.maxlen > 0:
                self.losses = self.losses[-self.maxlen:]
                self.accs = self.accs[-self.maxlen:]
                self.ious = self.ious[-self.maxlen:]

        def report(self, smoothing_window_size: int = 0) -> "Result.StepResult":
            if smoothing_window_size == 0:
                return Result.StepResult(self.losses[-1], self.accs[-1], self.ious[-1])
            else:
                if smoothing_window_size < 0:
                    smoothing_window_size = len(self.losses)
                return Result.StepResult(
                    float(np.mean(self.losses[-smoothing_window_size:])),
                    float(np.mean(self.accs[-smoothing_window_size:])),
                    float(np.mean(self.ious[-smoothing_window_size:])))


class Sample(NamedTuple):

    """
    Class for holding a single or a batch of samples.
    """

    image: torch.Tensor
    ground_truth: torch.Tensor


class DataDescriptor:

    """
    Constants Describing the COCODoom dataset.
    """

    RED = 255, 0, 0
    GREEN = 0, 255, 0
    BLUE = 0, 0, 255
    WHITE = 255, 255, 255
    YELLOW = 255, 255, 0
    PINK = 100, 255, 100
    GREY = 128, 128, 128

    ENEMY_TYPE_IDS = [1, 2, 3, 5, 8, 10, 11, 12, 14, 15, 17, 18, 19, 20, 21, 22]  # N = 16 + 1
    ENEMY_TYPE_NAMES = ["POSSESSED", "SHOTGUY", "VILE",  "UNDEAD", "FATSO", "CHAINGUY",  "TROOP", "SERGEANT", "HEAD",
                        "BRUISER", "KNIGHT", "SKULL",  "SPIDER", "BABY", "CYBORG",  "PAIN"]
    COLORS = [RED, BLUE, RED,  BLUE, WHITE, GREEN,  YELLOW, PINK, RED,
              WHITE, GREY, RED,  PINK, WHITE,  RED,  WHITE]
    COLORS = np.array(COLORS) / 255.
    NUM_CLASSES = len(ENEMY_TYPE_IDS)

    IMAGE_WIDTH = 320
    IMAGE_HEIGHT = 200
    IMAGE_SHAPE_YX = IMAGE_WIDTH, IMAGE_HEIGHT, 3
    IMAGE_SHAPE_XY = IMAGE_HEIGHT, IMAGE_WIDTH, 3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_root", help="Path to the COCODoom images root")
    parser.add_argument("--train_json", help="Path to the COCODoom train json file")
    parser.add_argument("--val_json", help="Path to the COCODoom val json file")
    parser.add_argument("--epochs", type=int, help="Number of training epochs", default=30)
    parser.add_argument("--batch_size", type=int, help="Training minibatch size", default=32)
    parser.add_argument("--adam_lr", type=float, help="Learning rate for the Adam optimizer", default=3e-4)

    args = parser.parse_args()

    return (
        args.images_root,
        args.train_json,
        args.val_json,
        args.epochs,
        args.batch_size,
        args.adam_lr)
