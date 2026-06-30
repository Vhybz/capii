from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import traceback

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
try:
    session = ort.InferenceSession("cabbage_model.onnx")
except Exception as e:
    session = None

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if session is None:
        return {"error": "Model failed to load"}

    try:
        # 1. Decode and resize the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB').resize((224, 224))

        # 2. Convert to float32 and normalize
        input_data = np.array(image, dtype=np.float32) / 255.0

        # Apply ImageNet normalization (Standard for PyTorch models)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        input_data = (input_data - mean) / std

        # 3. TRANSPOSE: Move channels to front for PyTorch [224, 224, 3] -> [3, 224, 224]
        input_data = np.transpose(input_data, (2, 0, 1))

        # 4. Add batch dimension -> [1, 3, 224, 224]
        input_data = np.expand_dims(input_data, axis=0)

        # 5. Run inference
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: input_data})

        # 6. Process results
        labels = ['Alternaria Leaf Spot', 'Black Rot', 'Downy Mildew', 'Healthy']
        output = result[0][0]
        max_idx = np.argmax(output)

        return {"disease": labels[max_idx], "confidence": float(np.max(output))}

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}
