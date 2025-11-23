from pathlib import Path
from io import BytesIO

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH_5 = "dog_5breed_resnet50.pth"
MODEL_PATH_120 = "dog_120breed_resnet50.pth"

# Same normalization as training
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Cache for loaded models
_model_cache = {}

def load_model(model_type="5breed"):
    """
    Load a model based on type.
    Args:
        model_type: "5breed" or "120breed"
    Returns:
        tuple of (model, class_names)
    """
    # Return cached model if already loaded
    if model_type in _model_cache:
        return _model_cache[model_type]

    # Select model path
    model_path = MODEL_PATH_5 if model_type == "5breed" else MODEL_PATH_120

    checkpoint = torch.load(model_path, map_location=DEVICE)
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

    # Cache the model
    _model_cache[model_type] = (model, class_names)

    return model, class_names

def predict_image_bytes(image_bytes, model_type="5breed"):
    """
    Predict dog breed from image bytes.
    Args:
        image_bytes: Raw image data
        model_type: "5breed" or "120breed"
    Returns:
        list of (class_name, probability) tuples
    """
    model, class_names = load_model(model_type)

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    # list of (class_name, prob)
    return list(zip(class_names, probs))
