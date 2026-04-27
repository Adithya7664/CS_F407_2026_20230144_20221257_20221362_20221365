import math
import cv2
import numpy as np
import torch
 
import config

#1. Binary safe zone mask
def build_safe_mask(seg_map: np.ndarray) -> np.ndarray:
    """
    Convert a (H, W) integer class-ID map (at 800×600) into a binary
    uint8 mask:  255 = safe pixel,  0 = unsafe pixel.
 
    Safe classes (from config.SAFE_CLASSES): 1, 3, 4
    (paved-area, low-vegetation/grass, dirt/gravel)
 
    Parameters
    ----------
    seg_map : np.ndarray  shape (H, W),  dtype int  — class IDs at 800×600
 
    Returns
    -------
    mask : np.ndarray  shape (H, W),  dtype uint8
    """
    mask = np.zeros(seg_map.shape, dtype=np.uint8)
    for cls_id in config.SAFE_CLASSES:
        mask[seg_map == cls_id] = 255
    return mask

