import os
import json
import sys
import torch
import logging
from typing import Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("inference")

app = FastAPI(
    title="GitHub Issues Classifier",
    description="Classifies GitHub issues into Bug / Feature / Support / Spam using fine-tuned DistilBERT",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "my-space/models")

tokenizer = None
model     = None
id2label  = None

@app.on_event("startup")
def load_model():
    global tokenizer, model, id2label
    try:
        logger.info(f"Loading model from: {MODEL_DIR}")
        artifacts_path = os.path.join(MODEL_DIR, "artifacts.json")
        with open(artifacts_path) as f:
            artifacts = json.load(f)
        id2label  = {int(k): v for k, v in artifacts["id2label"].items()}
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
        model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        model.eval()
        logger.info("Model and tokenizer loaded successfully ✓")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label:         str
    label_id:      int
    confidence:    float
    probabilities: Dict[str, float]


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "GitHub Issues Classifier is running 🚀"}

@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(request: PredictRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Text cannot be empty.")
    try:
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs    = model(**inputs)
            probs      = torch.softmax(outputs.logits, dim=-1).squeeze()
            pred_id    = probs.argmax().item()
            confidence = probs[pred_id].item()
        probabilities = {id2label[i]: round(probs[i].item(), 4) for i in range(len(id2label))}
        return PredictResponse(
            label=id2label[pred_id],
            label_id=pred_id,
            confidence=round(confidence, 4),
            probabilities=probabilities,
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))