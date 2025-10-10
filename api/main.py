import os
import io
import json  
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from PIL import Image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

app = FastAPI(title="Crop Disease Detection API")

CONFIG_PATH = "api\config.json"
MODELS_CONFIG = {}
try:
    with open(CONFIG_PATH, 'r') as f:
        MODELS_CONFIG = json.load(f)
    print("--- Model configuration loaded successfully from config.json ---")
except FileNotFoundError:
    print(f"ERROR: Configuration file not found at {CONFIG_PATH}")
except json.JSONDecodeError:
    print(f"ERROR: Could not decode JSON from {CONFIG_PATH}")

MODELS = {}

@app.on_event("startup")
def load_models():
    """Load all TFLite models into memory at startup for fast access."""
    print("--- Loading all models into memory ---")
    for crop_name, config in MODELS_CONFIG.items():
        model_path = config["path"]
        if os.path.exists(model_path):
            print(f"Loading model for: {crop_name} from {model_path}")
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            MODELS[crop_name] = interpreter
        else:
            print(f"WARNING: Model file not found for {crop_name} at {model_path}")
    print("--- All available models loaded ---")


@app.post("/predict")
async def predict(
    crop_name: str = Form(...),
    image: UploadFile = File(...)
):
    if crop_name not in MODELS:
        raise HTTPException(status_code=404, detail=f"Model for crop '{crop_name}' not found.")

    print(f"Received prediction request for crop: {crop_name}")

    interpreter = MODELS[crop_name]
    config = MODELS_CONFIG[crop_name]
    input_size = tuple(config["input_size"])
    class_names = config["class_names"]
    model_type = config.get("model_type", "from_scratch") 

    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    img = img.resize(input_size)
    img_array = np.array(img)
    img_batch = np.expand_dims(img_array, axis=0).astype(np.float32)
    
    processed_image = None
    if model_type == "transfer_learning":
        print("Applying transfer learning preprocessing...")
        processed_image = preprocess_input(img_batch)
    else: 
        print("Applying standard preprocessing (scaling to [0, 1])...")
        processed_image = img_batch / 255.0

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    if processed_image.dtype != input_details['dtype']:
        processed_image = processed_image.astype(input_details['dtype'])

    interpreter.set_tensor(input_details['index'], processed_image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details['index'])

    predicted_class_index = np.argmax(output_data)
    prediction = class_names[predicted_class_index]
    confidence = float(np.max(output_data))

    print(f"Prediction: {prediction}, Confidence: {confidence:.2%}")

    return {
        "crop": crop_name,
        "prediction": prediction,
        "confidence": f"{confidence:.2%}"
    }

@app.get("/")
def read_root():
    return {"message": "Welcome to the Crop Disease Detection API!"}