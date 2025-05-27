import os
import cv2
import numpy as np
from tqdm import tqdm
import uuid

def random_rotation(img, angle_range=(-15, 15)):
    angle = np.random.uniform(*angle_range)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def random_zoom(img, zoom_range=(0.9, 1.1)):
    zoom = np.random.uniform(*zoom_range)
    h, w = img.shape[:2]
    new_h, new_w = int(h * zoom), int(w * zoom)
    resized = cv2.resize(img, (new_w, new_h))

    if zoom < 1.0:  # Pad
        pad_top = (h - new_h) // 2
        pad_left = (w - new_w) // 2
        return cv2.copyMakeBorder(resized, pad_top, h - new_h - pad_top,
                                  pad_left, w - new_w - pad_left,
                                  cv2.BORDER_REFLECT)
    else:  # Crop
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        return resized[start_h:start_h + h, start_w:start_w + w]

def random_brightness_contrast(img, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2)):
    brightness = np.random.uniform(*brightness_range)
    contrast = np.random.uniform(*contrast_range)
    return np.clip((img - 0.5) * contrast + 0.5 * brightness, 0, 1)

def random_shift(img, shift_fraction=0.1):
    h, w = img.shape[:2]
    max_dx, max_dy = int(w * shift_fraction), int(h * shift_fraction)
    dx = np.random.randint(-max_dx, max_dx)
    dy = np.random.randint(-max_dy, max_dy)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def random_cutout(img, size=0.2):
    h, w = img.shape[:2]
    cut_h, cut_w = int(h * size), int(w * size)
    y = np.random.randint(0, h - cut_h)
    x = np.random.randint(0, w - cut_w)
    img_copy = img.copy()
    img_copy[y:y+cut_h, x:x+cut_w] = 0.0
    return img_copy

def augment_and_save(data_dir, output_dir, img_size=(224, 224), noise_std=0.03):
    os.makedirs(output_dir, exist_ok=True)
    class_names = sorted(os.listdir(data_dir))

    for class_name in tqdm(class_names):
        print(f"Processing letter: {class_name}")
        class_dir = os.path.join(data_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        for filename in os.listdir(class_dir):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            path = os.path.join(class_dir, filename)
            img = cv2.imread(path)
            if img is None:
                continue

            img = cv2.resize(img, img_size)
            img = img[..., ::-1]
            img = img.astype(np.float32) / 255.0

            source_id = str(uuid.uuid4())

            original_name = f"original_{filename}"
            cv2.imwrite(os.path.join(output_class_dir, original_name),(img * 255).astype(np.uint8)[..., ::-1])

            flipped = cv2.flip(img, 1)
            noise = np.random.normal(0, noise_std, flipped.shape).astype(np.float32)
            noisy_flipped = np.clip(flipped + noise, 0, 1)
            flipped_name = f"flipped_noisy_{filename}"
            cv2.imwrite(os.path.join(output_class_dir, flipped_name),(noisy_flipped * 255).astype(np.uint8)[..., ::-1])

            for i in range(3):
                aug = img.copy()
                if np.random.rand() < 0.5:
                    aug = random_rotation(aug)
                if np.random.rand() < 0.5:
                    aug = random_zoom(aug)
                if np.random.rand() < 0.5:
                    aug = random_brightness_contrast(aug)
                if np.random.rand() < 0.5:
                    aug = random_shift(aug)
                if np.random.rand() < 0.3:
                    aug = random_cutout(aug)

                aug_filename = f"aug_{i}_{filename}"
                cv2.imwrite(os.path.join(output_class_dir, aug_filename),(aug * 255).astype(np.uint8)[..., ::-1])

data_dir = "src/data/raw_imgs/LIS-fingerspelling-dataset"
output_dir = "src/data/processed_imgs/"
augment_and_save(data_dir, output_dir)
