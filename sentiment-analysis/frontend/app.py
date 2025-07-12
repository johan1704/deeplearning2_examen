# app.py
# Interface utilisateur Gradio
import gradio as gr
import requests
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

def analyze_audio(audio):
    if audio is None:
        return "Please upload audio first"
    
    try:
        with open(audio, "rb") as f:
            response = requests.post(
                f"{BACKEND_URL}/analyze",
                files={"file": f}
            )
        if response.status_code == 200:
            data = response.json()
            return f"""
            **Transcription:** {data['text']}
            **Sentiment:** {data['sentiment']}
            **Confidence:** {data['confidence']:.2%}
            """
        return f"Error: {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

demo = gr.Interface(
    fn=analyze_audio,
    inputs=gr.Audio(source="upload", type="filepath"),
    outputs=gr.Markdown(),
    title="Customer Call Sentiment Analysis"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)