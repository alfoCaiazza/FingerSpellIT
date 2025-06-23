import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

INPUT_DIR = 'tests/in_the_wild/create_dataset' 
OUTPUT_DIR = 'hands_frame'
os.makedirs(OUTPUT_DIR, exist_ok=True)
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']

# MediaPipe Initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.8
)

# Extract function for the bounding box
def process_image(image_path, output_path):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[ERRORE] Unable to read image: {image_path}")
        return

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        h, w, _ = image.shape
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]

        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))

        margin = 20
        x_min = max(x_min - margin, 0)
        y_min = max(y_min - margin, 0)
        x_max = min(x_max + margin, w)
        y_max = min(y_max + margin, h)

        cropped = image[y_min:y_max, x_min:x_max]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, cropped)
        print(f"[OK] Bounding Box saved in : {output_path}")
    else:
        print(f"[INFO] No hand detected: {image_path}")

# Sequential folder images extraction
for root, _, files in os.walk(INPUT_DIR):
    for file in files:
        if Path(file).suffix.lower() in IMAGE_EXTENSIONS:
            img_path = Path(root) / file

            relative_path = img_path.relative_to(INPUT_DIR)
            output_img_path = Path(OUTPUT_DIR) / relative_path

            process_image(img_path, output_img_path)

hands.close()
print("[FINE] Fully completed extraction.")
