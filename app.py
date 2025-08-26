import torch
import torch.nn as nn
from torchvision import transforms
from model import ModifiedMobileNetV2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from io import BytesIO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Class names provided by user
class_names = ['Gallstones', 'Cholecystitis', 'Gangrenous_Cholecystitis', 'Perforation', 'Polyps&Cholesterol_Crystal', 'WallThickening', 'Adenomyomatosis', 'Carcinoma', 'Intra-abdominal&Retroperitoneum', 'Normal']

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Load model
try:
    model = ModifiedMobileNetV2(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load('GB_stu_mob.pth', map_location=device))
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Inference function
def predict(image):
    with torch.no_grad():
        if not torch.is_tensor(image):
            image = preprocess(image).unsqueeze(0)
        image = image.to(device)
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        confidence_score = probabilities[0, predicted_class.item()].item()
    return class_names[predicted_class.item()], confidence_score

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('RGB')
        # Run prediction
        class_name, confidence_score = predict(image)
        return {
            "filename": file.filename,
            "predicted_class": class_name,
            "confidence_score": confidence_score
        }
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to the ModifiedMobileNetV2 API. Use POST /predict to upload an image."}