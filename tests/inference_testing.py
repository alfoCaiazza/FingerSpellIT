import os
import time
import psutil
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import pandas as pd
import joblib
import mediapipe as mp
from tqdm import tqdm
from tensorflow import keras
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.metrics import accuracy_score

MODEL_DIR = os.path.join( 'src', 'models')
RESNET_PATH = os.path.join(MODEL_DIR, 'augmented_image_model_v1.keras')
MLP_PATH = os.path.join(MODEL_DIR, 'augmented_andmark_model_v1.keras')
TEST_IMAGE_PATH = os.path.join('src', 'data', 'raw_imgs', 'splits', 'test')
IN_THE_WILD_IMAGE_PATH = os.path.join('tests', 'in_the_wild', 'processed_imgs') # CAMBIA PATH PER TEST
HAND_LANDMARK_TASK_PATH = os.path.join('src','artifacts','hand_landmarker.task')

# Initializing scaler and label econder for mlp model
try:
    resnet_model = load_model(RESNET_PATH)
    mlp_model = load_model(MLP_PATH)

    scaler = joblib.load('src/artifacts/csv_model/augmented/scaler.pkl')
    feature_names = scaler.feature_names_in_

    label_encoder = joblib.load('src/artifacts/csv_model/augmented/label_encoder.pkl')
    labels = label_encoder.classes_.tolist()

    print(f"Successfully loaded models from: {MODEL_DIR}")
except Exception as e:
    raise RuntimeError(f"Error in loading model: {e}")

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

# Loading in-the-wild images
def load_images_from_folder(root_folder, target_size=(224,224)):
    images, filenames, labels = [], [], []

    for class_name in sorted(os.listdir(root_folder)):
        class_path = os.path.join(root_folder, class_name)
        if not os.path.isdir(class_path):
            continue

        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, target_size)
                    images.append(img)
                    labels.append(class_name)
                    filenames.append(filename)

    return images, labels, filenames

def normalize_img(img):
    img = img.astype(np.float32) / 255.0
    input_data = np.expand_dims(img, axis=0)
    return input_data

# Benchmark function
def benchmark_model(model, images, model_type, image_paths=None):
    latencies, cpu_percents = [], []
    successful_runs = 0

    for i, img in enumerate(tqdm(images, desc=f"Benchmark {model_type}")):
        try:
            start = time.time()

            if model_type == 'resnet':
                # Considering image normalization process for real-world latency
                img_norm = normalize_img(img)
                _ = model.predict(img_norm, verbose=0)

            elif model_type == 'landmark':
                # Considering landmarks detection for real-world latency
                image = mp.Image.create_from_file(image_paths[i])
                results = detector.detect(image)

                if not results.hand_landmarks:
                    continue

                hand_landmarks = results.hand_landmarks[0]
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks]
                input_array = np.array(landmarks).flatten().reshape(1, -1)
                
                input_df = pd.DataFrame(input_array, columns=feature_names)
                input_scaled = scaler.transform(input_df)   

                _ = model.predict(input_scaled, verbose=0)


            dt = (time.time() - start) * 1000
            latencies.append(dt)
            cpu_percents.append(psutil.cpu_percent(interval=None))
            successful_runs += 1

        except Exception as e:
            print(f"Inference Error ({model_type}): {e}")
            continue

    return (
        np.mean(latencies) if latencies else float('nan'),
        np.mean(cpu_percents) if cpu_percents else float('nan'),
        successful_runs
    )

# Comparing accuracy prediction based on in-the-wild images
def test_failure_scenarios(model, model_type, test_pairs, image_paths=None):
    y_true, y_pred = [], []

    for i, (img, label) in enumerate(tqdm(test_pairs, desc=f"Testing {model_type}")):
        try:
            if model_type == 'resnet':
                img_norm = normalize_img(img)
                pred = model.predict(img_norm, verbose=0)
                pred_class = np.argmax(pred)
            
            elif model_type == 'landmark':
                if image_paths is None:
                    raise ValueError("image_paths must be provided for landmark model.")

                mp_image = mp.Image.create_from_file(image_paths[i])
                results = detector.detect(mp_image)

                if not results.hand_landmarks:
                    continue

                hand_landmarks = results.hand_landmarks[0]
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks]
                input_array = np.array(landmarks).flatten().reshape(1, -1)
                input_df = pd.DataFrame(input_array, columns=feature_names)
                input_scaled = scaler.transform(input_df)
                pred = model.predict(input_scaled, verbose=0)
                pred_class = np.argmax(pred)

            y_true.append(label)
            y_pred.append(label_encoder.inverse_transform([pred_class])[0])

        except Exception as e:
            print(f"Prediction error: {e}")
            continue

    return accuracy_score(y_true, y_pred)

# Loading Images
test_images, test_labels, filenames = load_images_from_folder(TEST_IMAGE_PATH)
test_image_paths = [os.path.join(TEST_IMAGE_PATH, label, fname) for label, fname in zip(test_labels, filenames)]

if not test_images:
    raise RuntimeError("No image found in TEST_IMAGE_PATH")

in_the_wild_images, in_the_wild_labels, in_the_wild_filenames = load_images_from_folder(IN_THE_WILD_IMAGE_PATH)
in_the_wild_data = list(zip(in_the_wild_images, in_the_wild_labels, in_the_wild_filenames))
test_pairs = [(img, label) for img, label, _ in in_the_wild_data]

in_the_wild_image_paths = [
    os.path.join(IN_THE_WILD_IMAGE_PATH, label, fname)
    for _, label, fname in in_the_wild_data
]

# Benchmark Execution
print("\n=== Benchmark ResNet ===")
resnet_latency, resnet_cpu, resnet_success = benchmark_model(resnet_model, test_images, model_type='resnet')
resnet_acc = test_failure_scenarios(resnet_model, 'resnet', test_pairs)

print("\n=== Benchmark MediaPipe+MLP ===")
mp_latency, mp_cpu, mp_success = benchmark_model(mlp_model, test_images, model_type='landmark', image_paths=test_image_paths)
mlp_acc = test_failure_scenarios(mlp_model, 'landmark', test_pairs, image_paths=in_the_wild_image_paths)


# Results
print("\n=== Results ===")
print(f"ResNet - Completed Runs: {resnet_success}/{len(test_images)}")
print(f"Avg Latency: {resnet_latency:.2f}ms")
print(f"Avg CPU usage: {resnet_cpu:.2f}%")

print(f"\nMediaPipe+MLP - Completed Runs: {mp_success}/{len(test_images)}")
print(f"Avg Latency: {mp_latency:.2f}ms")
print(f"Avg CPU usage: {mp_cpu:.2f}%")

# Speedup
if not np.isnan(mp_latency) and not np.isnan(resnet_latency):
    speedup = resnet_latency / mp_latency
    print(f"\nSpeedup MediaPipe vs ResNet: {speedup:.2f}x")

print("\nAccuracy")
print(f"ResNet - Critic Scenarios Accuracy: {resnet_acc:.2%}")
print(f"MediaPipe - Critic Scenarios Accuracy: {mlp_acc:.2%}")