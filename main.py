from fastapi import FastAPI, UploadFile, File
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

app = FastAPI()
session = ort.InferenceSession("cabbage_model.onnx")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Decode and resize the image
    image = Image.open(io.BytesIO(await file.read())).convert('RGB').resize((224, 224))

    # 2. Convert to float32 (matches what we send from Flutter)
    input_data = np.array(image, dtype=np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0) # Add batch dimension [1, 224, 224, 3]

    # 3. Run inference
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: input_data})

    # 4. Process the results (Map to your labels)
    labels = ['Alternaria Leaf Spot', 'Black Rot', 'Downy Mildew', 'Healthy']
    max_idx = np.argmax(result[0])

    return {"disease": labels[max_idx], "confidence": float(np.max(result[0]))}
