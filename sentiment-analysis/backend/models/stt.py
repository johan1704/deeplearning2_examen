# stt.py
# Reconnaissance vocale avec Wav2Vec 2.0
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import librosa

class SpeechToText:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        #self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        #self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)

        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-french").to(self.device)
    
    def transcribe(self, audio_path):
        speech, _ = librosa.load(audio_path, sr=16000)
        inputs = self.processor(speech, return_tensors="pt", sampling_rate=16000).input_values.to(self.device)
        with torch.no_grad():
            logits = self.model(inputs).logits
        return self.processor.batch_decode(torch.argmax(logits, dim=-1))[0]
