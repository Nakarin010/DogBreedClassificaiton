from pathlib import Path
from io import BytesIO

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "dog_breed_resnet50.pth"

# Same normalization as training
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    class_names = checkpoint["class_names"]

    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, len(class_names))
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, class_names

MODEL, CLASS_NAMES = load_model()

def predict_image_bytes(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = MODEL(tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    # list of (class_name, prob)
    return list(zip(CLASS_NAMES, probs))