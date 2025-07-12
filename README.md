# deeplearning2_examen
# 🎧 Détection Automatique de Sentiment dans des Appels Vocaux

Ce projet propose un pipeline intelligent permettant de **transcrire des appels vocaux** et **détecter automatiquement le sentiment** (positif, négatif ou neutre) des clients, en combinant **Wav2Vec 2.0** pour la reconnaissance vocale et **BERT** pour l'analyse de sentiment.

---

## 🧠 Contexte

Les entreprises reçoivent chaque jour des centaines d'appels contenant des informations précieuses : satisfaction, frustration, attentes des clients, etc.

Analyser ces appels **manuellement** est **long, coûteux** et inefficace.

🔍 **Objectif** : Automatiser ce processus à l’aide d’un pipeline de traitement audio-texte-sentiment.

---

## 🏗️ Architecture du Pipeline

Le traitement suit les étapes suivantes :

1. **Chargement de l'audio**  
   Le fichier audio (appel vocal) est chargé et prétraité.

2. **Transcription vocale avec Wav2Vec 2.0**  
   Le modèle convertit la voix en texte.

3. **Analyse de sentiment avec BERT**  
   Le texte obtenu est analysé pour détecter l’émotion exprimée :  
   ✅ **Positif**, ❌ **Négatif**, 😐 **Neutre**.

4. **Résultat final**  
   Le système retourne la **transcription** + le **sentiment détecté**.

---

## ⚙️ Technologies utilisées

- **Python**
- **FastAPI** : backend API pour servir les prédictions
- **Gradio** : interface web interactive
- **Wav2Vec 2.0** : modèle de transcription vocale
- **BERT** : modèle de NLP pour l’analyse de sentiment

---

## 🚀 Lancement rapide

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer le backend FastAPI
uvicorn app.backend.main:app --reload

# Lancer l'interface Gradio (frontend)
python app/frontend/app.py
