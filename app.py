from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from model_inference import predict_image_bytes

app = FastAPI()

# Serve static frontend from ./static
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = Path("static/index.html")
    return index_path.read_text(encoding="utf-8")

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_type: str = Form(default="5breed")
):
    if not file.content_type.startswith("image/"):
        return JSONResponse({"error": "File must be an image"}, status_code=400)

    # Validate model_type
    if model_type not in ["5breed", "120breed"]:
        return JSONResponse({"error": "Invalid model_type. Must be '5breed' or '120breed'"}, status_code=400)

    image_bytes = await file.read()
    preds = predict_image_bytes(image_bytes, model_type=model_type)

    # sort by prob descending and convert to percentages
    preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)
    result = [
        {"breed": name, "probability": float(prob) * 100.0}
        for name, prob in preds_sorted
    ]
    return {"predictions": result}

# run: uvicorn app:app --reload
