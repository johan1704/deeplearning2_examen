# nlp.py
# Analyse de sentiment avec BERT
from transformers import BertTokenizer, BertForSequenceClassification
import torch

class SentimentAnalyzer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
        self.model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment').to(self.device)
    
    def analyze(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs).item()
        return ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"][pred], float(probs[0][pred])