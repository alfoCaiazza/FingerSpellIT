import os
import csv
import cv2
import mediapipe as mp
import random
from tqdm import tqdm
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

dataset_dir = 'src/data/processed_imgs/LIS-augmented'
output_dir = 'src/data/csv'
LANDMARK_SAMPLES = 'src/artifacts'
DATASET_NAME = 'aug_landmark_dataset.csv'
HAND_LANDMARK_TASK_PATH = 'src/artifacts/hand_landmarker.task'

os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, DATASET_NAME)

# Mediapipe Initialization
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

base_options = python.BaseOptions(model_asset_path=HAND_LANDMARK_TASK_PATH)
options = vision.HandLandmarkerOptions(
    base_options = base_options,
    num_hands = 1,
    running_mode=VisionRunningMode.IMAGE
)
detector = vision.HandLandmarker.create_from_options(options)

# Creating CSV file with detected hand landmark
classes = [cls for cls in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, cls))]

sample_counter = 0

with open(csv_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    header = ['label']
    for i in range(21):
        header.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z']) # 3 spatial coordinates for each hand landmark
    csv_writer.writerow(header)

    for cls in tqdm(classes, desc="Class Processing"):
        cls_dir = os.path.join(dataset_dir, cls)
        imgs = [os.path.join(cls_dir, img) for img in os.listdir(cls_dir) if img.lower().endswith(('.jpg'))]

        for img_path in imgs:
            try:
                image = mp.Image.create_from_file(img_path)
                detection_result = detector.detect(image)
            except:
                continue

            if not detection_result.hand_landmarks:
                continue

            for hand_landmarks in detection_result.hand_landmarks:
                row = [cls]
                for landmark in hand_landmarks:
                    row.extend([landmark.x, landmark.y, landmark.z])
                csv_writer.writerow(row)
                break

            if sample_counter < 4:
                    if random.random() > 0.85:
                        img_bgr = cv2.imread(img_path)
                        h, w, _ = img_bgr.shape
                        for landmark in hand_landmarks:
                            cx, cy = int(landmark.x * w), int(landmark.y * h)
                            cv2.circle(img_bgr, (cx, cy), 4, (0, 255, 0), -1)

                        sample_name = f'aug_sample_{sample_counter + 1}.jpg'
                        sample_path = os.path.join(LANDMARK_SAMPLES, sample_name)
                        cv2.imwrite(sample_path, img_bgr)   
                        sample_counter = sample_counter + 1

print(f"CSV file successfully saved to: {csv_path}")