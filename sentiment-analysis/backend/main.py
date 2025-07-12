# main.py
# Point d'entrée FastAPI
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.models.stt import SpeechToText
from backend.models.nlp import SentimentAnalyzer
import tempfile
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

stt = SpeechToText()
sentiment = SentimentAnalyzer()

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            text = stt.transcribe(tmp.name)
            os.unlink(tmp.name)
        
        sentiment_result = sentiment.analyze(text)
        return {
            "text": text,
            "sentiment": sentiment_result[0],
            "confidence": sentiment_result[1]
        }
    except Exception as e:
        raise HTTPException(500, str(e))