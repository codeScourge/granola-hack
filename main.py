import torch
import pickle
import numpy as np
from pyannote.audio import Pipeline, Inference
from pyannote.audio import Audio
from scipy.spatial.distance import cosine
import pyaudio
import wave
import threading
import queue
import time
from datetime import datetime
import re
from collections import deque
import google.generativeai as genai
from PIL import ImageGrab
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
CONFIG = {
    'SPEAKER_THRESHOLD': 0.3,
    'CHUNK_DURATION': 2,  # seconds per audio chunk
    'SILENCE_TIMEOUT': 30,  # X2 seconds of silence to end conversation
    'TODO_CHECK_INTERVAL': 10,  # X seconds - check for todos
    'TODO_CHECK_WINDOW': 30,  # Y seconds - window to analyze
    'SCREENSHOT_CHECK_INTERVAL': 5,
    'SCREENSHOT_CHECK_WINDOW': 15,
    'VAD_THRESHOLD': 0.02,  # voice activity detection threshold
    'HF_TOKEN': os.environ.get('HF_KEY', ''),
    'GEMINI_API_KEY': os.environ.get('GEMINI_KEY', ''),
    'SAMPLE_RATE': 16000
}

# Initialize Gemini
genai.configure(api_key=CONFIG['GEMINI_API_KEY'])
model = genai.GenerativeModel('gemini-1.5-flash')

# Load speaker database
with open("speaker_db.pkl", "rb") as f:
    speaker_db = pickle.load(f)

# Load models
embedding_model = Inference("pyannote/embedding", window="whole")
audio_processor = Audio(sample_rate=CONFIG['SAMPLE_RATE'], mono="downmix")

# Global state
class ConversationState:
    def __init__(self):
        self.active = False
        self.speakers = set()
        self.transcript = []  # [(timestamp, speaker, text)]
        self.screenshots = []  # [(timestamp, filepath)]
        self.last_activity = time.time()
        self.audio_buffer = deque(maxlen=100)  # rolling buffer
        self.lock = threading.Lock()
        
state = ConversationState()

# Queues for inter-thread communication
audio_queue = queue.Queue()
ui_notification_queue = queue.Queue()

def identify_speaker(embedding):
    """Compare embedding against database"""
    best_match = None
    min_distance = float('inf')
    
    for name, ref_embedding in speaker_db.items():
        dist = cosine(embedding.flatten(), ref_embedding.flatten())
        if dist < min_distance:
            min_distance = dist
            best_match = name
    
    if min_distance < CONFIG['SPEAKER_THRESHOLD']:
        return best_match
    return "UNKNOWN"

def has_voice_activity(waveform):
    """Simple VAD based on energy"""
    energy = np.abs(waveform).mean()
    return energy > CONFIG['VAD_THRESHOLD']

def take_screenshot(label=""):
    """Capture and save screenshot"""
    timestamp = datetime.now()
    filename = f"screenshots/screenshot_{timestamp.strftime('%Y%m%d_%H%M%S')}_{label}.png"
    os.makedirs("screenshots", exist_ok=True)
    
    screenshot = ImageGrab.grab()
    screenshot.save(filename)
    return timestamp, filename

def transcribe_audio(waveform_data):
    """Placeholder - integrate Whisper or similar STT"""
    # You'll need to implement actual STT here
    # Example with whisper:
    # import whisper
    # result = whisper_model.transcribe(waveform_data)
    # return result['text']
    return "[TRANSCRIPTION PLACEHOLDER]"

def check_for_todos(text_window):
    """Check text for TODO indicators"""
    prompt = f"""Analyze this conversation fragment for action items or todos.
Only return the actual todo items, one per line. If none, return empty.

Conversation:
{text_window}

Todo items:"""
    
    try:
        response = model.generate_content(prompt)
        todos = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
        return todos
    except Exception as e:
        print(f"Todo check error: {e}")
        return []

def check_for_visual_cues(text_window):
    """Check for visual reference cues"""
    prompt = f"""Does this conversation fragment contain references to visual content?
Look for phrases like "look at this", "see that", "on the screen", etc.
Do NOT trigger on "gemini, screenshot this".
Answer only YES or NO.

Conversation:
{text_window}"""
    
    try:
        response = model.generate_content(prompt)
        return "YES" in response.text.upper()
    except Exception as e:
        print(f"Visual cue check error: {e}")
        return False

def summarize_conversation(transcript, screenshots):
    """Generate final summary"""
    # Build context with positioned screenshots
    context = "# Conversation Transcript\n\n"
    
    screenshot_dict = {ts: path for ts, path in screenshots}
    
    for ts, speaker, text in transcript:
        context += f"[{ts.strftime('%H:%M:%S')}] {speaker}: {text}\n"
        
        # Check if screenshot happened around this time
        for ss_ts, ss_path in screenshots:
            if abs((ss_ts - ts).total_seconds()) < 5:
                context += f"  📸 Screenshot: {ss_path}\n"
    
    prompt = f"""Summarize this conversation clearly and concisely. 
Do NOT include todos in the summary - they will be listed separately.

{context}

Summary:"""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Summarization error: {e}")
        return "Summary generation failed"

def audio_capture_thread():
    """Continuously capture audio"""
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=CONFIG['SAMPLE_RATE'],
        input=True,
        frames_per_buffer=int(CONFIG['SAMPLE_RATE'] * CONFIG['CHUNK_DURATION'])
    )
    
    print("🎤 Audio capture started")
    
    try:
        while True:
            audio_chunk = stream.read(int(CONFIG['SAMPLE_RATE'] * CONFIG['CHUNK_DURATION']))
            waveform = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            audio_queue.put((time.time(), waveform))
    except KeyboardInterrupt:
        stream.stop_stream()
        stream.close()
        p.terminate()

def speaker_identification_thread():
    """Process audio chunks for speaker ID and transcription"""
    global state
    
    while True:
        timestamp, waveform = audio_queue.get()
        
        # Check for voice activity
        if not has_voice_activity(waveform):
            with state.lock:
                if state.active and (time.time() - state.last_activity) > CONFIG['SILENCE_TIMEOUT']:
                    # End conversation
                    print("\n🔴 Conversation ended - processing...")
                    end_conversation()
                    state.active = False
                    state.speakers.clear()
            continue
        
        # Voice detected
        waveform_torch = torch.tensor(waveform).unsqueeze(0)
        embedding = embedding_model({"waveform": waveform_torch, "sample_rate": CONFIG['SAMPLE_RATE']})
        speaker = identify_speaker(embedding)
        
        # Transcribe
        text = transcribe_audio(waveform)
        
        # Check for manual screenshot command
        if "gemini" in text.lower() and "screenshot" in text.lower():
            ts, path = take_screenshot("manual")
            with state.lock:
                state.screenshots.append((datetime.fromtimestamp(timestamp), path))
            ui_notification_queue.put(("screenshot", f"Screenshot saved: {path}"))
        
        with state.lock:
            state.last_activity = time.time()
            state.audio_buffer.append((timestamp, speaker, text))
            
            # Start conversation if not active
            if not state.active:
                state.active = True
                state.transcript = []
                state.screenshots = []
                print("\n🟢 Conversation started")
            
            # Add/update speaker
            if speaker not in state.speakers:
                state.speakers.add(speaker)
                ui_notification_queue.put(("speaker_join", speaker))
            
            # Add to transcript
            state.transcript.append((datetime.fromtimestamp(timestamp), speaker, text))

def todo_monitor_thread():
    """Periodically check for todos"""
    global state
    last_check_idx = 0
    
    while True:
        time.sleep(CONFIG['TODO_CHECK_INTERVAL'])
        
        with state.lock:
            if not state.active:
                continue
            
            # Get recent window
            recent = state.transcript[last_check_idx:]
            if not recent:
                continue
            
            text_window = "\n".join([f"{s}: {t}" for _, s, t in recent[-10:]])
            last_check_idx = len(state.transcript)
        
        todos = check_for_todos(text_window)
        for todo in todos:
            ui_notification_queue.put(("todo", todo))
            print(f"✓ TODO detected: {todo}")

def visual_cue_monitor_thread():
    """Periodically check for visual references"""
    global state
    last_check_idx = 0
    
    while True:
        time.sleep(CONFIG['SCREENSHOT_CHECK_INTERVAL'])
        
        with state.lock:
            if not state.active:
                continue
            
            recent = state.transcript[last_check_idx:]
            if not recent:
                continue
            
            text_window = "\n".join([f"{s}: {t}" for _, s, t in recent[-5:]])
            last_check_idx = len(state.transcript)
        
        if check_for_visual_cues(text_window):
            ts, path = take_screenshot("auto")
            with state.lock:
                state.screenshots.append((ts, path))
            ui_notification_queue.put(("screenshot", "Auto screenshot taken"))
            print(f"📸 Auto screenshot: {path}")

def ui_notification_thread():
    """Display UI notifications (placeholder - implement actual UI)"""
    while True:
        notification_type, data = ui_notification_queue.get()
        
        # This is where you'd integrate actual UI framework
        # For now, just print
        if notification_type == "speaker_join":
            print(f"👤 {data} joined")
        elif notification_type == "todo":
            print(f"✓ TODO: {data}")
        elif notification_type == "screenshot":
            print(f"📸 {data}")
        
        # Auto-dismiss after 1 second would happen in real UI
        time.sleep(1)

def end_conversation():
    """Process and summarize ended conversation"""
    global state
    
    with state.lock:
        transcript = state.transcript.copy()
        screenshots = state.screenshots.copy()
    
    # Extract all todos from full transcript
    full_text = "\n".join([f"{s}: {t}" for _, s, t in transcript])
    all_todos = check_for_todos(full_text)
    
    # Generate summary
    summary = summarize_conversation(transcript, screenshots)
    
    # Save to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"summaries/conversation_{timestamp}.txt"
    os.makedirs("summaries", exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write(f"Conversation Summary - {timestamp}\n")
        f.write(f"Participants: {', '.join(state.speakers)}\n\n")
        f.write(summary)
        f.write("\n\n--- ACTION ITEMS ---\n")
        for todo in all_todos:
            f.write(f"• {todo}\n")
    
    print(f"\n📝 Summary saved: {output_file}")
    print(f"📋 {len(all_todos)} todos identified")

def main():
    """Start all threads"""
    threads = [
        threading.Thread(target=audio_capture_thread, daemon=True),
        threading.Thread(target=speaker_identification_thread, daemon=True),
        threading.Thread(target=todo_monitor_thread, daemon=True),
        threading.Thread(target=visual_cue_monitor_thread, daemon=True),
        threading.Thread(target=ui_notification_thread, daemon=True),
    ]
    
    for t in threads:
        t.start()
    
    print("🚀 AI Notetaker started. Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")

if __name__ == "__main__":
    main()
