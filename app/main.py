from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from tensorflow import keras
import joblib

app = FastAPI()

# Load model and the scaler when starting app
try:
    model = keras.models.load_model('src/models/augmented_andmark_model_v1.keras')
    scaler = joblib.load('src/artifacts/csv_model/augmented/scaler.pkl')

    label_encoder = joblib.load('src/artifacts/csv_model/augmented/label_encoder.pkl')
    labels = label_encoder.classes_.tolist()
except Exception as e:
    raise RuntimeError(f"Error in loading model: {e}")

# Input structure
class LandmarkInput(BaseModel):
    landmarks: list[float]

@app.post("/predict_landmarks")
def predict_landmarks(data: LandmarkInput):
    try:
        if len(data.landmarks) != 63:
            raise HTTPException(status_code=400, detail="Expected 63 values for landmarks.")
        
        input_array = np.array(data.landmarks).reshape(1,-1)
        input_scaled = scaler.transform(input_array)

        prediction = model.predict(input_scaled)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return {
            "class": predicted_class,
            "letter": labels[predicted_class],
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/health")
def healthcheck():
    return {"status": "ok"}
