# deeplearning2_examen
# 🎧 Détection Automatique de Sentiment dans des Appels Vocaux

Ce projet propose un pipeline intelligent permettant de **transcrire des appels vocaux** et **détecter automatiquement le sentiment** (positif, négatif ou neutre) des clients, en combinant **Wav2Vec 2.0** pour la reconnaissance vocale et **BERT** pour l'analyse de sentiment.

---

## 🧠 Contexte

Les entreprises reçoivent chaque jour des centaines d'appels contenant des informations précieuses : satisfaction, frustration, attentes des clients, etc.

Analyser ces appels **manuellement** est **long, coûteux** et inefficace.

🔍 **Objectif** : Automatiser ce processus à l’aide d’un pipeline de traitement audio-texte-sentiment.



## 🌐 Architecture du Système

### Composants Principaux
1. **Module de Transcription** :
   - Basé sur Wav2Vec 2.0 (lien : https://huggingface.co/facebook/wav2vec2-large-xlsr-53-french ) ----sur huggingface
   - le modèle huggingface.co/facebook/wav2vec2-large-xlsr-53-french est optimisé pour la langue française seulement
   - Conversion parole-texte en temps
   - Supporte les formats WAV/MP3/ogg/m4a ..

2. **Moteur d'Analyse** :
   - Modèle BERT fine-tuné (lien : https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment )
   - Détection de 5 niveaux de sentiment ( très mecontentt", "mecontent","Neutre", "satisfait", "très satisfait" )
   - Analyse multilingue

3. **Interface Utilisateur** :
   - Dashboard interactif
   - Visualisation des résultats
   - Gestion des historiques

### Flux de Données
1. Soumission audio → 2. Transcription → 3. Analyse NLP → 4. Génération rapport → 5. Stockage résultats

## 🖥 Prérequis Techniques

### Configuration Minimum
- Docker 20.10+
- 4 CPU cores
- 4GB de RAM
- 5GB d'espace disque

### Dépendances
- Bibliothèques Python (gérées automatiquement)
- Modèles Hugging Face (téléchargés automatiquement)

## 🚀 Installation et Lancement

### Construction des Conteneurs
bash

docker-compose build

### Lancement des Services
bash

docker-compose up 

Accès aux Interfaces
Backend (API)

    URL : http://localhost:8000

    Endpoints :

        POST /analyze : Soumission des fichiers audio


Frontend (Web)

    URL : http://localhost:7860


