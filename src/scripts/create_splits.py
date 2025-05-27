import os
import random
import shutil
from sklearn.model_selection import train_test_split

dataset_dir = 'src/data/raw_imgs/LIS-fingerspelling-dataset'
output_dir = 'src/data/raw_imgs/splits'

classes = os.listdir(dataset_dir)
os.makedirs(output_dir, exist_ok=True)

def copy_files(file_list, split, output_dir, cls):
    split_dir = os.path.join(output_dir, split, cls)
    os.makedirs(split_dir, exist_ok=True)
    for file in file_list:
        shutil.copy(file, split_dir)

for cls in classes:
    cls_dir = os.path.join(dataset_dir, cls)
    images = [os.path.join(cls_dir, img) for img in os.listdir(cls_dir)]

    # Images shuffling to prevent data unbalancing
    random.shuffle(images)

    # Train/Test split
    train_imgs, temp_imgs = train_test_split(images, test_size=0.2, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

    copy_files(train_imgs, "train", output_dir, cls)
    copy_files(val_imgs, "val", output_dir, cls)
    copy_files(test_imgs, "test", output_dir, cls)


