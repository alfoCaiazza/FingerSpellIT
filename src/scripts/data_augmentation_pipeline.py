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
    