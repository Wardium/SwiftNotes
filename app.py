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
import json
import re
from flask import Flask, render_template, jsonify, request
import tkinter as tk
from tkinter import filedialog

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__name__))
NOTES_DIR = os.path.join(BASE_DIR, "notes")
OLLAMA_URL = "https://ai-super.teamexist.com/api/generate"
MODEL_NAME = "DWS:Aurora"

audio_queue = queue.Queue()

DEFAULT_SAVE_DIR = os.path.join(os.path.expanduser("~"), "Documents", "SwiftNotes")
os.makedirs(DEFAULT_SAVE_DIR, exist_ok=True) # Ensure it exists immediately

app_state = {
    "current_save_dir": DEFAULT_SAVE_DIR,
    "status": "Idle",
    "class_name": "Detecting...",
    "notes": [], 
    "lecture_summary": "", 
    "is_recording": False,
    "lecture_active": True,
    "activity": "idle",
    "search_query": "",
    "existing_classes": []
}

def ask_ollama(prompt, require_json=False):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    if require_json:
        payload["format"] = "json"
        
    try:
        res = requests.post(OLLAMA_URL, json=payload)
        return res.json().get("response", "").strip()
    except Exception as e:
        return f"[AI Error: {e}]"

def get_formatted_date():
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
    
    # CHANGE IS HERE: Use the dynamic directory from app_state
    class_dir = os.path.join(app_state["current_save_dir"], app_state["class_name"])
    os.makedirs(class_dir, exist_ok=True)
    file_path = os.path.join(class_dir, f"{get_formatted_date()}.md")
    
    status_tag = "" if not app_state["lecture_active"] else "> **[UNPROCESSED - Lecture Ongoing]**\n\n"
    
    formatted_notes = [n["paragraph"] for n in app_state["notes"]]
    
    with open(file_path, "w") as f:
        f.write(status_tag + "\n\n---\n\n".join(formatted_notes))
        
        # Append the master summary to the file if it exists
        if app_state["lecture_summary"]:
            f.write(f"\n\n### Lecture Summary\n{app_state['lecture_summary']}")
            
def audio_capture_thread():
    CHUNK, FORMAT, CHANNELS, RATE, RECORD_SECONDS = 1024, pyaudio.paInt16, 1, 16000, 15
    p = pyaudio.PyAudio()
    
    while app_state["lecture_active"]:
        if not app_state["is_recording"]:
            time.sleep(1)
            continue
            
        try:
            stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
            frames = [stream.read(CHUNK, exception_on_overflow=False) for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS))]
            stream.stop_stream()
            stream.close()
            
            temp_audio = f"temp_chunk_{uuid.uuid4().hex}.wav"
            with wave.open(temp_audio, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
            
            audio_queue.put(temp_audio)
        except Exception as e:
            app_state["status"] = "Mic Error"
            time.sleep(2)
    p.terminate()

def ai_processing_thread():
    import mlx_whisper
    
    while app_state["lecture_active"] or not audio_queue.empty():
        try:
            current_audio_file = audio_queue.get(timeout=1)
        except queue.Empty:
            continue
            
        app_state["activity"] = "polishing"
        raw_text = mlx_whisper.transcribe(current_audio_file)["text"]
        
        if os.path.exists(current_audio_file):
            os.remove(current_audio_file)
            
        if not raw_text.strip():
            audio_queue.task_done()
            app_state["activity"] = "listening" if app_state["is_recording"] else "idle"
            continue

        prompt = f"""
        Analyze this raw lecture transcript: "{raw_text}"
        Available previous classes: {app_state['existing_classes']}
        
        Return ONLY a valid JSON object with the following keys:
        - "class_name": Determine the academic class. Use an existing one if it matches, otherwise create a new short name.
        - "paragraph": The raw text polished into a clean, well-formatted paragraph.
        - "search_term": A single specific concept, term, or entity mentioned that is worth looking up. If none, return null.
        """
        
        raw_response = ask_ollama(prompt, require_json=True)
        
        if raw_response.startswith("[AI Error:"):
            audio_queue.task_done()
            continue

        try:
            clean_json_str = re.sub(r'```json|```', '', raw_response).strip()
            data = json.loads(clean_json_str)
            
            if app_state["class_name"] == "Detecting...":
                app_state["class_name"] = data.get("class_name", "Unknown Class")
                
            note_entry = {
                "paragraph": data.get("paragraph", raw_text),
                "search_term": data.get("search_term")
            }
            app_state["notes"].append(note_entry)
            
            # Trigger the search query if one is provided
            if note_entry.get("search_term"):
                app_state["search_query"] = note_entry["search_term"]
                
        except Exception as e:
            print(f"JSON Parse Error: {e}")

        save_note_to_disk()
        app_state["activity"] = "listening" if app_state["is_recording"] else "idle"
        audio_queue.task_done()

# Flask Routes
@app.route("/")
def index():
    update_existing_classes()
    return render_template("index.html")

@app.route("/api/state")
def get_state(): return jsonify(app_state)

@app.route("/api/toggle_pause", methods=["POST"])
def toggle_pause():
    app_state["is_recording"] = not app_state["is_recording"]
    app_state["activity"] = "listening" if app_state["is_recording"] else "idle"
    return jsonify({"success": True})

@app.route("/api/end_lecture", methods=["POST"])
def end_lecture():
    app_state["is_recording"] = False
    app_state["lecture_active"] = False
    app_state["activity"] = "summarizing"
    
    def run_summary():
        full_text = "\n".join([n["paragraph"] for n in app_state["notes"]])
        summary_prompt = f"Summarize this entire lecture. Include: 1. Core topics. 2. Key points. 3. Action items. Lecture: {full_text}"
        summary = ask_ollama(summary_prompt, require_json=False)
        
        app_state["lecture_summary"] = summary
        app_state["activity"] = "idle"
        save_note_to_disk()
        
    threading.Thread(target=run_summary).start()
    return jsonify({"success": True})

@app.route("/api/clear_search", methods=["POST"])
def clear_search():
    app_state["search_query"] = ""
    return jsonify({"success": True})

@app.route("/api/files")
def list_files():
    files = []
    if os.path.exists(NOTES_DIR):
        for class_name in os.listdir(NOTES_DIR):
            class_path = os.path.join(NOTES_DIR, class_name)
            if os.path.isdir(class_path):
                for file in os.listdir(class_path):
                    if file.endswith(".md"):
                        files.append({"class": class_name, "date": file.replace(".md", ""), "path": os.path.join(class_path, file)})
    return jsonify(files)

@app.route("/api/file_content", methods=["POST"])
def file_content():
    with open(request.json.get("path"), "r") as f:
        return jsonify({"content": f.read()})

@app.route('/api/change_dir', methods=['POST'])
def change_dir():
    # Initialize tkinter and hide the main window
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True) # Force the window to open on top of the app
    
    # Open the folder picker
    selected_dir = filedialog.askdirectory(
        title="Select SwiftNotes Save Directory",
        initialdir=app_state["current_save_dir"]
    )
    
    root.destroy()
    
    # If the user selected a folder (didn't click cancel), update the state
    if selected_dir:
        app_state["current_save_dir"] = selected_dir
        
    return jsonify({"status": "success", "new_dir": app_state["current_save_dir"]})

@app.route("/api/rename_class", methods=["POST"])
def rename_class():
    new_name = request.json.get("class_name", "").strip()
    if not new_name:
        return jsonify({"success": False})

    old_name = app_state["class_name"]
    
    # If notes already exist and we aren't just stuck on 'Detecting...', move the file to the new folder
    if app_state["notes"] and old_name != "Detecting..." and old_name != new_name:
        old_dir = os.path.join(NOTES_DIR, old_name)
        old_file = os.path.join(old_dir, f"{get_formatted_date()}.md")
        
        new_dir = os.path.join(NOTES_DIR, new_name)
        new_file = os.path.join(new_dir, f"{get_formatted_date()}.md")
        
        os.makedirs(new_dir, exist_ok=True)
        
        if os.path.exists(old_file):
            os.rename(old_file, new_file)
            # Clean up the old directory if it's now empty
            if not os.listdir(old_dir):
                os.rmdir(old_dir)
    
    app_state["class_name"] = new_name
    update_existing_classes() # Refresh the global classes list
    save_note_to_disk()
    return jsonify({"success": True})

def start_server(): app.run(host="127.0.0.1", port=5000, debug=False)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    os.makedirs(NOTES_DIR, exist_ok=True)
    threading.Thread(target=start_server, daemon=True).start()
    app_state["is_recording"] = True
    app_state["activity"] = "listening"
    threading.Thread(target=audio_capture_thread, daemon=True).start()
    threading.Thread(target=ai_processing_thread, daemon=True).start()
    webview.create_window("Swift Note", "http://127.0.0.1:5000/", width=1100, height=800)
    webview.start()
