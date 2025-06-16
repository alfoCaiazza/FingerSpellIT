import cv2
import numpy as np
import mediapipe as mp
import time
import joblib
import requests
from tensorflow.keras.models import load_model

# Loading scaler and label encoder
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')
gesture_labels = label_encoder.classes_.tolist()

# Parameters
k = 2  # Predicitons step in seconds
last_prediction = ""
last_sample_time = time.time()
build_string = [] 

# MediaPipe init
mp_hands = mp.solutions.hands

# Webcam init
print("Webcam Initialization")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERRORE: Webcam non avaiable.")
    exit()
print("Webcam found. Press 'q' to exit.")

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.9,
    min_tracking_confidence=0.9
) as hands:

    while cap.isOpened():
        current_time = time.time()
        success, frame = cap.read()
        if not success:
            print("Frame not read.")
            continue

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            if current_time - last_sample_time >= k:
                last_sample_time = current_time

                for hand_landmarks in results.multi_hand_landmarks:
                    landmark_array = np.array([
                        [lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark
                    ]).flatten().reshape(1, -1)

                    # landmark_array_scaled = scaler.transform(landmark_array)

                    try:
                        payload = {
                            "landmarks": landmark_array.flatten().tolist()
                        }

                        response = requests.post("http://localhost:8000/predict_landmarks", json=payload)
                        result = response.json()

                        if response.status_code == 200:
                            class_id = result["class"]
                            confidence = result["confidence"]
                            predicted_label = gesture_labels[class_id]

                            if confidence > 0.8:
                                predicted_label = gesture_labels[class_id]
                                last_prediction = f"Letter: {predicted_label} ({confidence:.2f})"
                                build_string.append(predicted_label)
                                print(f"Gesture: {last_prediction}")
                        
                        else:
                            print(f"API ERROR: {result.get('detail', 'Unknown Error')}")
                    except Exception as e:
                        print(f"Error occurred during API request: {e}")

        # === Visualizza risultato sull'immagine ===
        if cv2.waitKey(5) & 0xFF == ord(' '):
            build_string.append(' ')

        if build_string:
            display_text = ''.join(build_string)
            
            cv2.putText(
                frame,
                display_text,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),  # Colore Rosso
                3,
                cv2.LINE_AA
            )

            # Istruzioni (in basso a destra)
            instructions = "SPACE: spazio | Q: esci"
            (text_width, text_height), _ = cv2.getTextSize(instructions, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(
                frame,
                instructions,
                (frame.shape[1] - text_width - 10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 209, 102),  # Colore Giallo
                2,
                cv2.LINE_AA
            )

        # Visualizza
        cv2.imshow('Hand Landmark Recognition', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            print("[INFO] Exit requested.")
            break


cap.release()
cv2.destroyAllWindows()