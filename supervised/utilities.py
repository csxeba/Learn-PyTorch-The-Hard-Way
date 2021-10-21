"""
MSCOCO/COCODoom masking utils taken from
https://github.com/csxeba/Verres.git
"""

import numpy as np
import cv2


def decode_poly(poly, shape):
    full_mask = np.zeros(shape, dtype="uint8")
    pts = [np.round(np.array(p).reshape(-1, 2)).astype(int) for p in poly]
    return cv2.fillPoly(full_mask, pts, color=1).astype(bool)


def decode_rle(rle, shape):
    full_mask = np.zeros(np.prod(shape), dtype=bool)
    fill = False
    start = 0
    for num in rle["counts"]:
        end = start + num
        full_mask[start:end] = fill
        fill = not fill
        start = end
    return full_mask.reshape(shape[::-1]).T


def mask_from_annotation(annotation, image_shape):
    return mask_from_representation(annotation["segmentation"], image_shape)


def mask_from_representation(segmentation_repr, image_shape):
    if isinstance(segmentation_repr, list):
        return decode_poly(segmentation_repr, image_shape).astype(bool)
    elif "counts" in segmentation_repr:
        return decode_rle(segmentation_repr, image_shape)
