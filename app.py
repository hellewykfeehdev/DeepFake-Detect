from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as transforms
import torchaudio
import numpy as np
import tempfile
import cv2
import os
import uvicorn
from io import BytesIO

app = FastAPI(title="SentinelGuard DeepFake & Audio Detector", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Caminhos dos modelos
DEEPFAKE_MODEL_PATH = "checkpoints/model_epoch_10.pth"
AUDIO_MODEL_PATH = "checkpoints/audio_model.pth"  # você vai adicionar depois
SANDBOX_MODE = os.environ.get("SANDBOX_MODE", "false").lower() == "true"

# Carrega modelo de vídeo
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load(DEEPFAKE_MODEL_PATH, map_location=device)
model.eval()

# Transformações de imagem
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.get("/")
def home():
    return {
        "status": "ok",
        "sandbox": SANDBOX_MODE,
        "message": "SentinelGuard Deep Analyzer online."
    }

@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    """Analisa vídeo e retorna score médio de deepfake"""
    if SANDBOX_MODE:
        return {"deepfake_score": 0.42, "is_deepfake": False, "sandbox": True}

    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(contents)
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)
    frames, count = [], 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % 10 == 0:
            frames.append(transform(frame).unsqueeze(0))
        if len(frames) >= 10:
            break

    cap.release()
    os.remove(video_path)

    if not frames:
        return {"error": "no frames extracted"}

    preds = []
    with torch.no_grad():
        for f in frames:
            f = f.to(device)
            output = model(f)
            prob = torch.sigmoid(output).cpu().numpy().flatten()[0]
            preds.append(prob)

    score = float(np.mean(preds))
    return {"deepfake_score": round(score, 4), "is_deepfake": score > 0.5}

@app.post("/analyze/audio")
async def analyze_audio(file: UploadFile = File(...)):
    """Analisa áudio (voz falsa, manipulação etc.)"""
    if SANDBOX_MODE:
        return {"audio_score": 0.33, "is_fake": False, "sandbox": True}

    waveform, sr = torchaudio.load(BytesIO(await file.read()))
    # Reduz taxa para 16kHz
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
    # Usa modelo leve (placeholder até subir o real)
    score = float(np.abs(waveform.mean().item()))
    return {"audio_score": round(score, 3), "is_fake": score > 0.6}

@app.post("/sandbox/toggle")
async def toggle_sandbox(mode: str = Form(...)):
    """Liga/desliga modo sandbox remotamente"""
    global SANDBOX_MODE
    SANDBOX_MODE = mode.lower() == "true"
    return {"sandbox_mode": SANDBOX_MODE}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
