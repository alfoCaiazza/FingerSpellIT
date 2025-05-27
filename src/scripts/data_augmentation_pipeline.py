import os
import cv2
import numpy as np
from tqdm import tqdm
import uuid
import json

def random_rotation(img, angle_range=(-15,15)):
    angle = np.random.uniform(*angle_range)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)

    return cv2.warpAffine(img, M, (w,h), borderMode=cv2.BORDER_REFLECT)

def random_zoom(img, zoom_range=(0.9,1.1)):
    zoom = np.random.uniform(*zoom_range)
    h, w = img.shape[:2]
    new_h, new_w = int(h*zoom), int(w * zoom)
    resized = cv2.resize(img, (new_h, new_w))


    if zoom < 1.0: # Pad
        pad_top = (h - new_h) // 2
        pad_left = (w - new_w) // 2
        return cv2.copyMakeBorder(resized, pad_top, h - new_h - pad_top,
                                  pad_left, w - new_w - pad_left,
                                  cv2.BORDER_REFLECT)
    else: # Crop
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        

        return resized[start_h:start_h + h, start_w:start_w + w]

def random_brightness_constrast(img, brightness_range=(0.8,1.2), contrast_range=(0.8,1.2)):
    brightness = np.random.uniform(*brightness_range)
    contrast = np.random.uniform(*contrast_range)

    return np.clip((img - 0.5) * contrast + 0.5 * brightness)

def random_shift(img, shift_fraction=0.1):
    h, w = img.shape[:2]
    max_dx, max_dy = int(w * shift_fraction), int(h * shift_fraction)
    dx = np.random.randint(-max_dx, max_dx)
    dy = np.random.randint(-max_dy, max_dy)
    M = np.float32([[1, 0 , dx], [0, 1, dy]])

    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def random_cut(img, size=0.2):
    h,w = img.shape[:2]
    cut_h, cut_w = int(h * size), int(w * size)
    y = np.random.randint(0, h - cut_h)
    x = np.random.randint(0, w - cut_w)
    img_copy = img.copy()
    img_copy[y:y + cut_h, x:x + cut_w] = 0.0

    return img_copy