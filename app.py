import os
import time
import datetime
import threading
import multiprocessing
import requests
import wave
import pyaudio
import uuid
import queue
import webview
from flask import Flask, render_template, jsonify, request

# --- Initialization ---
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__name__))
NOTES_DIR = os.path.join(BASE_DIR, "notes")
OLLAMA_URL = "http://localhost:11434/api/generate" # Default local Ollama port
MODEL_NAME = "DWS:Aurora"

audio_queue = queue.Queue()

# --- State Management ---
app_state = {
    "status": "Idle",
    "class_name": "Detecting...",
    "notes": [],
    "is_recording": False,
    "lecture_active": True,
    "activity": "idle", # listening, polishing, summarizing
    "search_query": "",
    "existing_classes": []
}

# --- Helper Functions ---
def ask_ollama(prompt):
    """Zero-API key local call to DWS:Aurora via Ollama."""
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

def update_existing_classes():
    if os.path.exists(NOTES_DIR):
        app_state["existing_classes"] = [d for d in os.listdir(NOTES_DIR) if os.path.isdir(os.path.join(NOTES_DIR, d))]

def save_note_to_disk():
    if app_state["class_name"] == "Detecting..." or not app_state["notes"]:
        return
    
    class_dir = os.path.join(NOTES_DIR, app_state["class_name"])
    os.makedirs(class_dir, exist_ok=True)
    
    file_path = os.path.join(class_dir, f"{get_formatted_date()}.md")
    
    # Tag it as unprocessed if the lecture isn't finished yet
    status_tag = "" if not app_state["lecture_active"] else "\n\n> **[UNPROCESSED - Lecture Ongoing]**\n\n"
    
    with open(file_path, "w") as f:
        f.write(status_tag + "\n\n".join(app_state["notes"]))

# --- Background Processing Threads ---
def audio_capture_thread():
    """Producer: Strictly handles recording and queuing audio."""
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 15
    
    p = pyaudio.PyAudio()
    
    while app_state["lecture_active"]:
        if not app_state["is_recording"]:
            time.sleep(1)
            continue
            
        try:
            stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, 
                            input=True, frames_per_buffer=CHUNK)
            
            frames = []
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                
            stream.stop_stream()
            stream.close()
            
            chunk_id = uuid.uuid4().hex
            temp_audio = f"temp_chunk_{chunk_id}.wav"
            
            wf = wave.open(temp_audio, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            audio_queue.put(temp_audio)
            
        except Exception as e:
            app_state["status"] = "Mic Error"
            time.sleep(2)
            
    p.terminate()

def ai_processing_thread():
    """Consumer: Pulls audio, transcribes on M4, polishes with Aurora."""
    import mlx_whisper # Imported here so it doesn't block startup
    
    while app_state["lecture_active"] or not audio_queue.empty():
        try:
            current_audio_file = audio_queue.get(timeout=1)
        except queue.Empty:
            continue
            
        app_state["activity"] = "polishing"
        
        # 1. Local M4 Transcription
        raw_text = mlx_whisper.transcribe(current_audio_file)["text"]
        
        if os.path.exists(current_audio_file):
            os.remove(current_audio_file)
            
        if not raw_text.strip():
            audio_queue.task_done()
            app_state["activity"] = "listening" if app_state["is_recording"] else "idle"
            continue

        # 2. DWS:Aurora Polish
        polish_prompt = f"Format this lecture excerpt into a clean paragraph. Fix grammar. Excerpt: {raw_text}"
        clean_text = ask_ollama(polish_prompt)
        app_state["notes"].append(clean_text)

        # 3. Intelligent Google Search Trigger
        search_prompt = f"Does this excerpt mention a highly specific term, historical event, or complex equation that requires visual reference? If yes, respond with ONLY the search query. If no, respond with 'NO'. Excerpt: {clean_text}"
        search_decision = ask_ollama(search_prompt)
        if search_decision != "NO" and len(search_decision) < 40:
             app_state["search_query"] = search_decision

        # 4. Zero-Input Classification (Run only on first chunk)
        if app_state["class_name"] == "Detecting...":
            class_prompt = f"Based on this excerpt, what is the class name? Respond ONLY with the name (e.g. 'Calculus 1'). Excerpt: {clean_text}"
            app_state["class_name"] = ask_ollama(class_prompt).replace('"', '')
        
        save_note_to_disk()
        app_state["activity"] = "listening" if app_state["is_recording"] else "idle"
        audio_queue.task_done()

# --- Flask API Routes ---
@app.route("/")
def index():
    update_existing_classes()
    return render_template("index.html")

@app.route("/api/state")
def get_state():
    return jsonify(app_state)

@app.route("/api/toggle_pause", methods=["POST"])
def toggle_pause():
    app_state["is_recording"] = not app_state["is_recording"]
    app_state["activity"] = "listening" if app_state["is_recording"] else "idle"
    return jsonify({"success": True})

@app.route("/api/set_class", methods=["POST"])
def set_class():
    app_state["class_name"] = request.json.get("class_name")
    save_note_to_disk()
    return jsonify({"success": True})

@app.route("/api/end_lecture", methods=["POST"])
def end_lecture():
    app_state["is_recording"] = False
    app_state["lecture_active"] = False
    app_state["activity"] = "summarizing"
    
    def run_summary():
        full_text = "\n".join(app_state["notes"])
        summary_prompt = f"Summarize this entire lecture. Include: 1. Core topics. 2. Key points. 3. Action items. Lecture: {full_text}"
        summary = ask_ollama(summary_prompt)
        
        app_state["notes"].append("\n### Lecture Summary\n" + summary)
        save_note_to_disk()
        app_state["activity"] = "idle"
        
    threading.Thread(target=run_summary).start()
    return jsonify({"success": True})

@app.route("/api/clear_search", methods=["POST"])
def clear_search():
    app_state["search_query"] = ""
    return jsonify({"success": True})

# -- Viewer Routes --
@app.route("/api/files")
def list_files():
    files = []
    if os.path.exists(NOTES_DIR):
        for class_name in os.listdir(NOTES_DIR):
            class_path = os.path.join(NOTES_DIR, class_name)
            if os.path.isdir(class_path):
                for file in os.listdir(class_path):
                    if file.endswith(".md"):
                        files.append({
                            "class": class_name, 
                            "date": file.replace(".md", ""), 
                            "path": os.path.join(class_path, file)
                        })
    return jsonify(files)

@app.route("/api/file_content", methods=["POST"])
def file_content():
    path = request.json.get("path")
    with open(path, "r") as f:
        content = f.read()
    return jsonify({
        "content": content, 
        "unprocessed": "[UNPROCESSED]" in content
    })

@app.route("/api/process_backlog", methods=["POST"])
def process_backlog():
    path = request.json.get("path")
    with open(path, "r") as f:
        raw_content = f.read().replace("> **[UNPROCESSED - Lecture Ongoing]**", "")
        
    summary_prompt = f"Summarize this transcript. Include core topics, key points, and action items. Transcript: {raw_content}"
    summary = ask_ollama(summary_prompt)
    
    clean_content = f"{raw_content}\n\n### Backlog Summary (DWS:Aurora)\n{summary}"
    with open(path, "w") as f:
        f.write(clean_content)
        
    return jsonify({"success": True})

def start_server():
    app.run(host="127.0.0.1", port=5000, debug=False)

# --- macOS Menu Bar (Isolated Process) ---
def run_menu_bar():
    import rumps
    class SwiftNoteMenuBar(rumps.App):
        def __init__(self):
            super(SwiftNoteMenuBar, self).__init__("🎙️ DWS", icon=None)
            self.menu = ["Toggle Pause", "Quit App"]
            
        @rumps.clicked("Toggle Pause")
        def on_pause(self, _):
            try:
                requests.post("http://127.0.0.1:5000/api/toggle_pause")
            except:
                pass
            
        @rumps.clicked("Quit App")
        def on_quit(self, _):
            rumps.quit_application()
            
    SwiftNoteMenuBar().run()

# --- Main Execution ---
if __name__ == "__main__":
    multiprocessing.freeze_support()
    os.makedirs(NOTES_DIR, exist_ok=True)
    
    # 1. Start Flask API
    t = threading.Thread(target=start_server)
    t.daemon = True
    t.start()
    
    # 2. Start AI/Audio Threads
    app_state["is_recording"] = True
    app_state["activity"] = "listening"
    
    threading.Thread(target=audio_capture_thread, daemon=True).start()
    threading.Thread(target=ai_processing_thread, daemon=True).start()
    
    # 3. Start macOS Menu Bar (Separate process to prevent thread crashing)
    menu_process = multiprocessing.Process(target=run_menu_bar)
    menu_process.start()
    
    # 4. Launch Frameless Webview
    webview.create_window("DWS:SwiftNote", "http://127.0.0.1:5000/", width=700, height=650, frameless=True, transparent=True)
    webview.start()
    
    # Cleanup when window closes
    menu_process.terminate()
