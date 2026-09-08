import os
import time
import datetime
import threading
import requests
import wave
import pyaudio
import webview
from flask import Flask, render_template, jsonify
import mlx_whisper

app = Flask(__name__)

# --- State Management ---
app_state = {
    "status": "Idle",
    "class_name": "Detecting...",
    "notes": [],
    "is_recording": False,
    "lecture_active": True
}

BASE_DIR = os.path.dirname(os.path.abspath(__name__))
NOTES_DIR = os.path.join(BASE_DIR, "notes")
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "DWS:Aurora"

def ask_ollama(prompt):
    """Zero-API key local call to DWS:Aurora."""
    try:
        res = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        })
        return res.json().get("response", "").strip()
    except Exception as e:
        return f"[AI Error: {e}]"

def get_formatted_date():
    """Returns format like: September 3rd - Tuesday"""
    now = datetime.datetime.now()
    day = now.day
    suffix = 'th' if 11 <= day <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
    return now.strftime(f"%B {day}{suffix} - %A")

def save_note_to_disk():
    """Saves the current session to the dynamically named folder."""
    if app_state["class_name"] == "Detecting..." or not app_state["notes"]:
        return
    
    class_dir = os.path.join(NOTES_DIR, app_state["class_name"])
    os.makedirs(class_dir, exist_ok=True)
    
    file_path = os.path.join(class_dir, f"{get_formatted_date()}.md")
    with open(file_path, "w") as f:
        f.write("\n\n".join(app_state["notes"]))

def audio_processing_loop():
    """Background thread that captures audio and runs the AI pipeline."""
    # Note: For production, implement WebRTCVAD here to chunk audio on silence.
    # This is a simulated chunking loop for the architecture blueprint.
    
    while app_state["lecture_active"]:
        if not app_state["is_recording"]:
            time.sleep(1)
            continue
            
        app_state["status"] = "Listening..."
        # 1. Capture 15-30 seconds of audio using PyAudio (simulated here)
        time.sleep(5) 
        temp_audio = "temp_chunk.wav" # Assume PyAudio saved the chunk here
        
        if not os.path.exists(temp_audio):
            continue

        # 2. M4 Transcription
        app_state["status"] = "Transcribing (M4)..."
        raw_text = mlx_whisper.transcribe(temp_audio)["text"]
        
        if not raw_text.strip():
            continue

        # 3. Polish formatting via DWS:Aurora
        app_state["status"] = "Polishing (DWS:Aurora)..."
        polish_prompt = f"Format this lecture excerpt into a clean, cohesive paragraph. Fix grammar, remove filler words. Excerpt: {raw_text}"
        clean_text = ask_ollama(polish_prompt)
        app_state["notes"].append(clean_text)

        # 4. Zero-Input Classification (Run only on the first chunk)
        if app_state["class_name"] == "Detecting...":
            app_state["status"] = "Classifying Subject..."
            class_prompt = f"Based on this lecture excerpt, what is the academic class name? Respond ONLY with the class name (e.g. 'Calculus 1', 'Psychology'). Excerpt: {clean_text}"
            app_state["class_name"] = ask_ollama(class_prompt).replace('"', '')
        
        save_note_to_disk()
        app_state["status"] = "Idle"

# --- API Routes for Web UI ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/state")
def get_state():
    return jsonify(app_state)

@app.route("/api/toggle_pause", methods=["POST"])
def toggle_pause():
    app_state["is_recording"] = not app_state["is_recording"]
    app_state["status"] = "Listening..." if app_state["is_recording"] else "Paused"
    return jsonify({"success": True})

@app.route("/api/end_lecture", methods=["POST"])
def end_lecture():
    app_state["is_recording"] = False
    app_state["lecture_active"] = False
    app_state["status"] = "Summarizing (DWS:Aurora)..."
    
    # Generate full summary
    full_text = "\n".join(app_state["notes"])
    summary_prompt = f"Summarize this entire lecture. Include: 1. Core topics learned. 2. Key points to remember. 3. Important notes/action items. Lecture: {full_text}"
    summary = ask_ollama(summary_prompt)
    
    app_state["notes"].append("\n### Lecture Summary\n" + summary)
    save_note_to_disk()
    app_state["status"] = "Lecture Saved & Completed."
    return jsonify({"success": True})

def start_server():
    app.run(host="127.0.0.1", port=5000, debug=False)

if __name__ == "__main__":
    # Ensure notes directory exists
    os.makedirs(NOTES_DIR, exist_ok=True)
    
    # Start Flask API in background
    t = threading.Thread(target=start_server)
    t.daemon = True
    t.start()
    
    # Start Audio/AI processing pipeline in background
    app_state["is_recording"] = True
    audio_thread = threading.Thread(target=audio_processing_loop)
    audio_thread.daemon = True
    audio_thread.start()
    
    # Launch Native Glassmorphism Window
    webview.create_window("DWS-SwiftNote", "http://127.0.0.1:5000/", width=1100, height=750, transparent=True)
    webview.start()
