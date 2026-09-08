import os
import time
import datetime
import threading
import requests
import wave
import pyaudio
import uuid
import queue
import webbrowser
import customtkinter as ctk
import rumps

# --- State & Config ---
app_state = {
    "status": "Idle",
    "class_name": "Detecting...",
    "notes": [],
    "is_recording": False,
    "lecture_active": True,
    "current_activity": "idle" # Options: idle, listening, polishing, summarizing
}

BASE_DIR = os.path.dirname(os.path.abspath(__name__))
NOTES_DIR = os.path.join(BASE_DIR, "notes")
OLLAMA_URL = "https://ai-super.teamexist.com/api/generate"
MODEL_NAME = "DWS:Aurora"
audio_queue = queue.Queue()

# --- AI & File Helpers ---
def ask_ollama(prompt):
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
    now = datetime.datetime.now()
    day = now.day
    suffix = 'th' if 11 <= day <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
    return now.strftime(f"%B {day}{suffix} - %A")

def get_existing_classes():
    if not os.path.exists(NOTES_DIR):
        return []
    return [d for d in os.listdir(NOTES_DIR) if os.path.isdir(os.path.join(NOTES_DIR, d))]

def save_note_to_disk():
    if app_state["class_name"] == "Detecting..." or not app_state["notes"]:
        return
    
    class_dir = os.path.join(NOTES_DIR, app_state["class_name"])
    os.makedirs(class_dir, exist_ok=True)
    
    file_path = os.path.join(class_dir, f"{get_formatted_date()}.md")
    # If the lecture isn't active, it's finished processing
    status_tag = "" if not app_state["lecture_active"] else "\n\n> **[UNPROCESSED - Lecture Ongoing]**\n\n"
    
    with open(file_path, "w") as f:
        f.write(status_tag + "\n\n".join(app_state["notes"]))

def trigger_ai_search(query_topic):
    """Background task to search Google and open the top result if DWS:Aurora thinks it's important."""
    search_url = f"https://www.google.com/search?q={query_topic.replace(' ', '+')}"
    webbrowser.open(search_url)

# --- Background Processing Threads ---
def audio_capture_thread():
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
    import mlx_whisper
    while app_state["lecture_active"] or not audio_queue.empty():
        try:
            current_audio_file = audio_queue.get(timeout=1)
        except queue.Empty:
            continue
            
        app_state["current_activity"] = "polishing"
        app_state["status"] = f"Processing Queue: {audio_queue.qsize()}..."
        
        raw_text = mlx_whisper.transcribe(current_audio_file)["text"]
        if os.path.exists(current_audio_file):
            os.remove(current_audio_file)
            
        if not raw_text.strip():
            audio_queue.task_done()
            app_state["current_activity"] = "listening" if app_state["is_recording"] else "idle"
            continue

        polish_prompt = f"Format this lecture excerpt into a clean paragraph. Fix grammar. Excerpt: {raw_text}"
        clean_text = ask_ollama(polish_prompt)
        app_state["notes"].append(clean_text)

        # Intelligent Search Trigger
        search_prompt = f"Does this excerpt mention a highly specific term, historical event, or complex equation that requires visual reference? If yes, respond with ONLY the search query. If no, respond with 'NO'. Excerpt: {clean_text}"
        search_decision = ask_ollama(search_prompt)
        if search_decision != "NO" and len(search_decision) < 40:
             threading.Thread(target=trigger_ai_search, args=(search_decision,)).start()

        if app_state["class_name"] == "Detecting...":
            class_prompt = f"Based on this excerpt, what is the class name? Respond ONLY with the name. Excerpt: {clean_text}"
            app_state["class_name"] = ask_ollama(class_prompt).replace('"', '')
        
        save_note_to_disk()
        app_state["current_activity"] = "listening" if app_state["is_recording"] else "idle"
        app_state["status"] = "Listening..." if app_state["is_recording"] else "Paused"
        audio_queue.task_done()

# --- Square Frameless UI (CustomTkinter) ---
class SwiftNoteUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("DWS-SwiftNote")
        self.geometry("400x400") # Perfectly Square
        self.overrideredirect(True) # Removes macOS Window Frame (frameless)
        self.configure(fg_color="#1e1b4b") # Deep glassmorphism blue
        self.attributes('-alpha', 0.95) # Slight transparency
        
        # Make the frameless window draggable
        self.bind("<ButtonPress-1>", self.start_move)
        self.bind("<ButtonRelease-1>", self.stop_move)
        self.bind("<B1-Motion>", self.do_move)
        self.bind("<space>", self.toggle_pause_key)
        
        # --- UI Elements ---
        # Top Icons
        self.icon_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.icon_frame.pack(pady=(20, 0))
        
        self.icon_listen = ctk.CTkLabel(self.icon_frame, text="🎙️", font=("Arial", 24), text_color="gray")
        self.icon_listen.pack(side="left", padx=10)
        self.icon_polish = ctk.CTkLabel(self.icon_frame, text="✨", font=("Arial", 24), text_color="gray")
        self.icon_polish.pack(side="left", padx=10)
        self.icon_summary = ctk.CTkLabel(self.icon_frame, text="📝", font=("Arial", 24), text_color="gray")
        self.icon_summary.pack(side="left", padx=10)
        
        # Status
        self.lbl_class = ctk.CTkLabel(self, text="Detecting Class...", font=("Arial", 20, "bold"), text_color="white")
        self.lbl_class.pack(pady=(30, 5))
        
        self.lbl_status = ctk.CTkLabel(self, text="Initializing...", font=("Arial", 14), text_color="#cbd5e1")
        self.lbl_status.pack(pady=5)
        
        # Class Selection Dropdown (Populated dynamically)
        self.class_var = ctk.StringVar(value="Select Existing Class")
        existing_classes = get_existing_classes()
        if existing_classes:
            self.class_dropdown = ctk.CTkOptionMenu(self, values=existing_classes, variable=self.class_var, command=self.set_class_manual)
            self.class_dropdown.pack(pady=10)
            
        # Viewer Button
        self.btn_viewer = ctk.CTkButton(self, text="Open Notes Viewer", command=self.open_viewer, fg_color="#475569")
        self.btn_viewer.pack(pady=(20, 10))
        
        # Controls
        self.btn_end = ctk.CTkButton(self, text="End Lecture", command=self.end_lecture, fg_color="#ef4444", hover_color="#dc2626")
        self.btn_end.pack(side="bottom", pady=30)
        
        self.update_loop()

    # Window Drag Logic
    def start_move(self, event):
        self.x = event.x
        self.y = event.y

    def stop_move(self, event):
        self.x = None
        self.y = None

    def do_move(self, event):
        deltax = event.x - self.x
        deltay = event.y - self.y
        x = self.winfo_x() + deltax
        y = self.winfo_y() + deltay
        self.geometry(f"+{x}+{y}")

    # App Logic
    def toggle_pause_key(self, event):
        app_state["is_recording"] = not app_state["is_recording"]
        app_state["status"] = "Listening..." if app_state["is_recording"] else "Paused"
        
    def set_class_manual(self, choice):
        app_state["class_name"] = choice
        
    def open_viewer(self):
        # Simply opens the macOS Finder to the notes directory for now
        os.system(f"open '{NOTES_DIR}'")

    def end_lecture(self):
        app_state["is_recording"] = False
        app_state["lecture_active"] = False
        app_state["current_activity"] = "summarizing"
        app_state["status"] = "Summarizing (DWS:Aurora)..."
        
        def run_summary():
            full_text = "\n".join(app_state["notes"])
            summary_prompt = f"Summarize this entire lecture. Include: 1. Core topics. 2. Key points. 3. Action items. Lecture: {full_text}"
            summary = ask_ollama(summary_prompt)
            app_state["notes"].append("\n### Lecture Summary\n" + summary)
            save_note_to_disk()
            app_state["status"] = "Lecture Saved & Completed."
            app_state["current_activity"] = "idle"
            time.sleep(2)
            self.quit() # Closes UI
            
        threading.Thread(target=run_summary).start()

    def update_loop(self):
        """Polls app_state and updates the UI icons every 500ms."""
        self.lbl_class.configure(text=app_state["class_name"])
        self.lbl_status.configure(text=app_state["status"])
        
        # Reset Icon Colors
        self.icon_listen.configure(text_color="gray")
        self.icon_polish.configure(text_color="gray")
        self.icon_summary.configure(text_color="gray")
        
        # Highlight Active Icon
        if app_state["current_activity"] == "listening" or (app_state["is_recording"] and app_state["current_activity"] == "idle"):
            self.icon_listen.configure(text_color="#38bdf8") # Blue
        elif app_state["current_activity"] == "polishing":
            self.icon_polish.configure(text_color="#f59e0b") # Yellow/Gold
        elif app_state["current_activity"] == "summarizing":
            self.icon_summary.configure(text_color="#10b981") # Green
            
        self.after(500, self.update_loop)


# --- macOS Menu Bar App (Rumps) ---
class SwiftNoteMenuBar(rumps.App):
    def __init__(self):
        super(SwiftNoteMenuBar, self).__init__("DWS", icon=None)
        self.menu = ["Status: Idle", "Toggle Pause", "End Lecture"]
        
    @rumps.timer(1)
    def update_status(self, _):
        # Update the text in the drop-down menu
        self.menu["Status: Idle"].title = f"Status: {app_state['status']}"
        # Update the actual menu bar icon text at the top of the screen
        if app_state["is_recording"]:
            self.title = "🎙️ DWS"
        else:
            self.title = "⏸️ DWS"
            
    @rumps.clicked("Toggle Pause")
    def on_pause(self, _):
        app_state["is_recording"] = not app_state["is_recording"]
        app_state["status"] = "Listening..." if app_state["is_recording"] else "Paused"
        
    @rumps.clicked("End Lecture")
    def on_quit(self, _):
        rumps.quit_application()


if __name__ == "__main__":
    os.makedirs(NOTES_DIR, exist_ok=True)
    
    # 1. Start the Background AI Threads
    app_state["is_recording"] = True
    app_state["current_activity"] = "listening"
    
    capture_thread = threading.Thread(target=audio_capture_thread)
    capture_thread.daemon = True
    capture_thread.start()
    
    ai_thread = threading.Thread(target=ai_processing_thread)
    ai_thread.daemon = True
    ai_thread.start()
    
    # 2. Start the Menu Bar App in a Background Thread
    # Rumps uses PyObjC which *prefers* main thread, but can run in background
    # if we don't block the main thread.
    def run_menu_bar():
        SwiftNoteMenuBar().run()
        
    menu_thread = threading.Thread(target=run_menu_bar)
    menu_thread.daemon = True
    menu_thread.start()
    
    # 3. Start the Square Frameless UI on the MAIN Thread (macOS requirement)
    ui = SwiftNoteUI()
    ui.mainloop()
