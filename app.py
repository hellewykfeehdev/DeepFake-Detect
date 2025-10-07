from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch
import torchvision.transforms as transforms
import torchaudio
import numpy as np
import tempfile
import cv2
import os
import uvicorn
from io import BytesIO

# ==========================================
# 🚀 SentinelGuard Deep Analyzer (v3.0)
# ==========================================

app = FastAPI(title="SentinelGuard DeepFake & Audio Detector", version="3.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurações globais
SANDBOX_MODE = os.environ.get("SANDBOX_MODE", "false").lower() == "true"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 🔍 Carregamento do Modelo Hugging Face (DeepFake)
# ==========================================
try:
    MODEL_REPO = "Wvolf/ViT_Deepfake_Detection"
    processor = AutoImageProcessor.from_pretrained(MODEL_REPO)
    deepfake_model = AutoModelForImageClassification.from_pretrained(MODEL_REPO).to(DEVICE)
    deepfake_model.eval()
    print("✅ DeepFake model loaded successfully.")
except Exception as e:
    deepfake_model = None
    print(f"⚠️ Failed to load DeepFake model: {e}")

# ==========================================
# 🎧 Modelo de Áudio (placeholder até modelo real)
# ==========================================
AUDIO_MODEL_PATH = "checkpoints/audio_model.pth"

if os.path.exists(AUDIO_MODEL_PATH):
    try:
        audio_model = torch.load(AUDIO_MODEL_PATH, map_location=DEVICE)
        audio_model.eval()
        print("✅ Audio model loaded successfully.")
    except Exception as e:
        audio_model = None
        print(f"⚠️ Failed to load audio model: {e}")
else:
    audio_model = None

# ==========================================
# 🧩 Utilitários
# ==========================================
def sandbox_response(data: dict):
    """Retorna resultado simulado no modo sandbox."""
    data["sandbox"] = True
    return data


# ==========================================
# 🏠 Rota inicial
# ==========================================
@app.get("/")
def home():
    return {
        "status": "online",
        "sandbox_mode": SANDBOX_MODE,
        "device": str(DEVICE),
        "message": "SentinelGuard Deep Analyzer operational.",
    }

# ==========================================
# 🎞️ Análise de vídeo (DeepFake Detection)
# ==========================================
@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    """Analisa vídeo e retorna probabilidade média de deepfake."""

    if SANDBOX_MODE or deepfake_model is None:
        return sandbox_response({
            "deepfake_score": 0.42,
            "is_deepfake": False,
            "message": "Sandbox mode active or model unavailable"
        })

    # Cria arquivo temporário
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
        if count % 10 == 0:  # Pega 1 frame a cada 10
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
        if len(frames) >= 10:
            break

    cap.release()
    os.remove(video_path)

    if not frames:
        return {"error": "No frames extracted"}

    preds = []
    with torch.no_grad():
        for f in frames:
            inputs = processor(images=f, return_tensors="pt").to(DEVICE)
            outputs = deepfake_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            deepfake_prob = float(probs[0][1].item())  # Classe 1 = fake
            preds.append(deepfake_prob)

    score = float(np.mean(preds))
    return {
        "deepfake_score": round(score, 4),
        "is_deepfake": score > 0.5,
        "frames_analyzed": len(preds),
    }

# ==========================================
# 🎙️ Análise de Áudio
# ==========================================
@app.post("/analyze/audio")
async def analyze_audio(file: UploadFile = File(...)):
    """Detecta possíveis manipulações ou vozes falsas."""
    if SANDBOX_MODE or audio_model is None:
        return sandbox_response({
            "audio_score": 0.33,
            "is_fake": False,
            "message": "Sandbox mode active or audio model unavailable"
        })

    waveform, sr = torchaudio.load(BytesIO(await file.read()))

    # Reduz taxa para 16kHz
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)

    # Placeholder leve: média de energia
    energy = torch.mean(torch.abs(waveform))
    score = float(min(energy.item() * 10, 1.0))

    return {
        "audio_score": round(score, 3),
        "is_fake": score > 0.6,
    }

# ==========================================
# 🔐 Sandbox remoto
# ==========================================
@app.post("/sandbox/toggle")
async def toggle_sandbox(mode: str = Form(...)):
    """Liga/desliga modo sandbox remotamente."""
    global SANDBOX_MODE
    SANDBOX_MODE = mode.lower() == "true"
    return {"sandbox_mode": SANDBOX_MODE}

# ==========================================
# 🚀 Execução
# ==========================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
