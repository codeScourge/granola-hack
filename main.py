import pyannote_load_patch  # noqa: F401 — before any pyannote model load

import torch
import pickle
import numpy as np
from pyannote.audio import Pipeline, Inference
from pyannote.audio import Audio
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
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
import cv2
import os
from dotenv import load_dotenv
from huggingface_hub import login
import whisper


# --- auth
load_dotenv()

hf_token = os.environ.get("HF_KEY")
if not hf_token:
    raise SystemExit(
        "HF_KEY must be set. "
        "Accept conditions at https://hf.co/pyannote/embedding and "
        "https://hf.co/pyannote/speaker-diarization-3.1"
    )
login(token=hf_token)

# --- things
CONFIG = {
    'SPEAKER_THRESHOLD': 0.3,
    'CHECK_INTERVAL': 3,       # run VAD/speaker/transcribe every 1s
    'CONTEXT_SECONDS': 3,      # use last 5s of audio when processing
    'CAPTURE_CHUNK_DURATION': 1,  # capture in 1s chunks for the rolling buffer
    'SILENCE_TIMEOUT': 15,
    'TODO_CHECK_INTERVAL': 5, 
    'TODO_CHECK_WINDOW': 5,
    'PHOTO_CHECK_INTERVAL': 5,
    'PHOTO_CHECK_WINDOW': 5,
    'CAMERA_INDEX': 0,
    'HF_TOKEN': os.environ.get('HF_KEY'),
    'GEMINI_API_KEY': os.environ.get('GEMINI_KEY'),
    'SAMPLE_RATE': 16000
}


whisper_model = whisper.load_model("base")  # or "small", "medium", "large"
genai.configure(api_key=CONFIG['GEMINI_API_KEY'])
model = genai.GenerativeModel('gemini-2.0-flash')


def _gemini_text(response):
    """Safely get text from Gemini response; empty candidates (e.g. safety block) return ''."""
    if not response.candidates:
        return ""
    return (response.text or "")


# --- getting annote 
with open("speaker_db.pkl", "rb") as f:
    speaker_db = pickle.load(f)

embedding_model = Inference("pyannote/embedding", window="whole")
audio_processor = Audio(sample_rate=CONFIG['SAMPLE_RATE'], mono="downmix")

vad_model = Model.from_pretrained("pyannote/segmentation", token=hf_token)
vad_pipeline = VoiceActivityDetection(segmentation=vad_model)
vad_pipeline.instantiate({'onset': 0.5, 'offset': 0.5, 'min_duration_on': 0.1, 'min_duration_off': 0.1})

# --- app stuff
class ConversationState:
    def __init__(self):
        self.active = False
        self.speakers = set()
        self.transcript = []  # [(timestamp, speaker, text)]
        self.photos = []  # [(timestamp, filepath)]
        self.last_activity = time.time()
        self.last_manual_photo_time = 0.0  # cooldown to avoid repeated triggers
        self.last_auto_photo_time = 0.0
        self.emitted_todos = set()  # dedupe TODOs within conversation
        self.audio_buffer = deque(maxlen=100)  # rolling buffer (transcript)
        # Raw audio: last CONTEXT_SECONDS (e.g. 5) chunks of CAPTURE_CHUNK_DURATION each
        self.raw_audio_buffer = deque(maxlen=CONFIG['CONTEXT_SECONDS'])
        self.lock = threading.Lock()
        
state = ConversationState()

# Queues for inter-thread communication
ui_notification_queue = queue.Queue()

# Latency metrics (append from threads, read after shutdown)
_metrics_lock = threading.Lock()
_latency_metrics = {
    "vad": [],
    "speaker_diarization": [],
    "todo_check": [],
    "visual_cue_check": [],
    "photo_keyword_check": [],
}

def _embedding_vector(arr):
    """Normalize to (512,) for cosine: handle both whole-window (512,) and sliding (T, 512)."""
    x = np.asarray(arr).flatten()
    if x.size == 512:
        return x
    x = x.reshape(-1, 512)
    return x.mean(axis=0)


def identify_speaker(embedding):
    """Compare embedding against database"""
    best_match = None
    min_distance = float('inf')
    emb = _embedding_vector(embedding)
    for name, ref_embedding in speaker_db.items():
        ref = _embedding_vector(ref_embedding)
        dist = cosine(emb, ref)
        if dist < min_distance:
            min_distance = dist
            best_match = name
    
    if min_distance < CONFIG['SPEAKER_THRESHOLD']:
        return best_match
    return "UNKNOWN"

def has_voice_activity(waveform):
    file_like = {"waveform": torch.tensor(waveform).unsqueeze(0), "sample_rate": CONFIG['SAMPLE_RATE']}
    vad_result = vad_pipeline(file_like)
    return len(list(vad_result.itertracks())) > 0

def take_camera_photo(label=""):
    """Capture and save a photo from the camera."""
    timestamp = datetime.now()
    os.makedirs("photos", exist_ok=True)
    filename = f"photos/photo_{timestamp.strftime('%Y%m%d_%H%M%S')}_{label}.png"
    cap = cv2.VideoCapture(CONFIG.get("CAMERA_INDEX", 0))
    try:
        ret, frame = cap.read()
        if not ret or frame is None:
            return timestamp, None
        cv2.imwrite(filename, frame)
        return timestamp, filename
    finally:
        cap.release()

def transcribe_audio(waveform):
    """Transcribe audio using Whisper"""
    try:
        # Whisper expects float32 numpy array
        audio_np = np.array(waveform, dtype=np.float32)
        
        # Whisper prefers longer segments, but will work with 2s chunks
        result = whisper_model.transcribe(
            audio_np,
            language='en',  # or None for auto-detect
            fp16=False  # set True if using GPU
        )
        return result['text'].strip()
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""

def check_for_todos(text_window):
    """Check text for TODO indicators"""
    prompt = f"""Analyze this conversation fragment for action items or todos.
Only return the actual todo items, one per line. If none, return empty.

Conversation:
{text_window}

Todo items:"""
    
    try:
        response = model.generate_content(prompt)
        text = _gemini_text(response)
        todos = [line.strip() for line in text.strip().split('\n') if line.strip()]
        return todos
    except Exception as e:
        print(f"Todo check error: {e}")
        return []

def check_for_visual_cues(text_window):
    """Check for visual reference cues (e.g. "look at this", "take a photo")"""
    prompt = f"""Does this conversation fragment contain references to showing or capturing something visually?
Look for phrases like "look at this", "see that", "take a photo", "snap a picture", etc.
Do NOT trigger on "gemini, take a photo" (that is an explicit command).
Answer only YES or NO.

Conversation:
{text_window}"""
    
    try:
        response = model.generate_content(prompt)
        return "YES" in _gemini_text(response).upper()
    except Exception as e:
        print(f"Visual cue check error: {e}")
        return False

def summarize_conversation(transcript, photos):
    """Generate final summary"""
    # Build context with positioned camera photos
    context = "# Conversation Transcript\n\n"
    
    for ts, speaker, text in transcript:
        context += f"[{ts.strftime('%H:%M:%S')}] {speaker}: {text}\n"
        # Check if a photo was taken around this time
        for photo_ts, photo_path in photos:
            if abs((photo_ts - ts).total_seconds()) < 5:
                context += f"  📷 Photo: {photo_path}\n"
    
    prompt = f"""Summarize this conversation clearly and concisely. 
Do NOT include todos in the summary - they will be listed separately.

{context}

Summary:"""
    
    try:
        response = model.generate_content(prompt)
        return _gemini_text(response) or "Summary generation failed"
    except Exception as e:
        print(f"Summarization error: {e}")
        return "Summary generation failed"

def audio_capture_thread():
    """Continuously capture audio in 1s chunks into rolling buffer (last CONTEXT_SECONDS)."""
    p = pyaudio.PyAudio()
    frames_per_chunk = int(CONFIG['SAMPLE_RATE'] * CONFIG['CAPTURE_CHUNK_DURATION'])
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=CONFIG['SAMPLE_RATE'],
        input=True,
        frames_per_buffer=frames_per_chunk,
    )
    print("🎤 Audio capture started (1s chunks → last %ds buffer)" % CONFIG['CONTEXT_SECONDS'])
    try:
        while True:
            try:
                audio_chunk = stream.read(frames_per_chunk, exception_on_overflow=False)
            except OSError as e:
                if e.errno == -9981:
                    continue
                raise
            if len(audio_chunk) < frames_per_chunk * 2:
                continue
            waveform = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            with state.lock:
                state.raw_audio_buffer.append((time.time(), waveform))
    except KeyboardInterrupt:
        stream.stop_stream()
        stream.close()
        p.terminate()

def speaker_identification_thread():
    """Every CHECK_INTERVAL s, process last CONTEXT_SECONDS of audio for speaker ID and transcription."""
    global state
    current_speaker = None

    while True:
        time.sleep(CONFIG['CHECK_INTERVAL'])

        with state.lock:
            if len(state.raw_audio_buffer) == 0:
                continue
            chunks = list(state.raw_audio_buffer)
        # Concatenate last CONTEXT_SECONDS (e.g. 5s) into one waveform
        timestamps = [c[0] for c in chunks]
        waveform = np.concatenate([c[1] for c in chunks])
        timestamp = timestamps[-1]  # use end of window

        if len(waveform) < CONFIG['SAMPLE_RATE']:  # need at least 1s
            continue

        t0 = time.perf_counter()
        vad_result = has_voice_activity(waveform)
        with _metrics_lock:
            _latency_metrics["vad"].append(time.perf_counter() - t0)
        if not vad_result:
            if current_speaker is not None:
                print(f"\n🔇 Silence...")
                current_speaker = None
            with state.lock:
                if state.active and (time.time() - state.last_activity) > CONFIG['SILENCE_TIMEOUT']:
                    print("\n🔴 Conversation ended - processing...")
                    end_conversation()
                    state.active = False
                    state.speakers.clear()
                    current_speaker = None
            continue

        # Voice detected - identify speaker (on full context window)
        waveform_torch = torch.tensor(waveform).unsqueeze(0)
        t0 = time.perf_counter()
        embedding = embedding_model({"waveform": waveform_torch, "sample_rate": CONFIG['SAMPLE_RATE']})
        speaker = identify_speaker(embedding)
        with _metrics_lock:
            _latency_metrics["speaker_diarization"].append(time.perf_counter() - t0)

        text = transcribe_audio(waveform)
        if not text:
            continue

        # Dedupe: skip if same as last transcript entry (overlapping windows)
        with state.lock:
            if state.transcript and state.transcript[-1][1] == speaker and state.transcript[-1][2] == text:
                continue

        if speaker != current_speaker:
            print(f"\n💬 {speaker}:")
            current_speaker = speaker
        print(f"   {text}")

        t0 = time.perf_counter()
        photo_triggered = ("gemini" in text.lower() and
                          ("photo" in text.lower() or "camera" in text.lower()))
        with _metrics_lock:
            _latency_metrics["photo_keyword_check"].append(time.perf_counter() - t0)
        if photo_triggered:
            with state.lock:
                if time.time() - state.last_manual_photo_time < 15:
                    photo_triggered = False  # cooldown: same utterance in overlapping windows
                else:
                    state.last_manual_photo_time = time.time()
            if photo_triggered:
                ts, path = take_camera_photo("manual")
                if path:
                    with state.lock:
                        state.photos.append((datetime.fromtimestamp(timestamp), path))
                    ui_notification_queue.put(("photo", f"Camera photo saved: {path}"))

        with state.lock:
            state.last_activity = time.time()
            state.audio_buffer.append((timestamp, speaker, text))
            if not state.active:
                state.active = True
                state.transcript = []
                state.photos = []
                state.emitted_todos = set()
                current_speaker = speaker
                print(f"\n🟢 Conversation started")
                print(f"💬 {speaker}:")
                print(f"   {text}")
            if speaker not in state.speakers:
                state.speakers.add(speaker)
                print(f"👤 {speaker} joined the conversation")
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
        
        t0 = time.perf_counter()
        todos = "" # TODO: check_for_todos(text_window)
        with _metrics_lock:
            _latency_metrics["todo_check"].append(time.perf_counter() - t0)
        with state.lock:
            seen = state.emitted_todos
            new_todos = [t for t in todos if t and t not in seen]
            for t in new_todos:
                seen.add(t)
        for todo in new_todos:
            ui_notification_queue.put(("todo", todo))

def visual_cue_monitor_thread():
    """Periodically check for visual references"""
    global state
    last_check_idx = 0
    
    while True:
        time.sleep(CONFIG['PHOTO_CHECK_INTERVAL'])
        
        with state.lock:
            if not state.active:
                continue
            
            recent = state.transcript[last_check_idx:]
            if not recent:
                continue
            
            text_window = "\n".join([f"{s}: {t}" for _, s, t in recent[-5:]])
            last_check_idx = len(state.transcript)
        
        t0 = time.perf_counter()
        visual_cue = check_for_visual_cues(text_window)
        with _metrics_lock:
            _latency_metrics["visual_cue_check"].append(time.perf_counter() - t0)
        if visual_cue:
            with state.lock:
                if time.time() - state.last_auto_photo_time < 15:
                    visual_cue = False  # cooldown
                else:
                    state.last_auto_photo_time = time.time()
            if visual_cue:
                ts, path = take_camera_photo("auto")
                if path:
                    with state.lock:
                        state.photos.append((ts, path))
                    ui_notification_queue.put(("photo", "Auto camera photo taken"))

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
        elif notification_type == "photo":
            print(f"📷 {data}")
        
        # Auto-dismiss after 1 second would happen in real UI
        time.sleep(1)

def end_conversation():
    """Process and summarize ended conversation"""
    global state
    
    with state.lock:
        transcript = state.transcript.copy()
        photos = state.photos.copy()
    

    full_text = "\n".join([f"{s}: {t}" for _, s, t in transcript])
    all_todos = "" # TODO: check_for_todos(full_text)
    
    # Generate summary
    summary = summarize_conversation(transcript, photos)
    
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

def _print_latency_metrics():
    """Print latency stats for VAD, speaker diarization, and the 3 keyword checks."""
    with _metrics_lock:
        data = {k: list(v) for k, v in _latency_metrics.items()}
    if not any(data.values()):
        print("No latency metrics collected.")
        return
    print("\n" + "=" * 60)
    print("LATENCY METRICS")
    print("=" * 60)
    for name, times in data.items():
        if not times:
            print(f"  {name}: (no samples)")
            continue
        arr = np.array(times) * 1000  # ms
        n = len(arr)
        print(f"  {name}:")
        print(f"    count={n}, mean={arr.mean():.2f} ms, min={arr.min():.2f} ms, max={arr.max():.2f} ms")
        if n >= 2:
            p50 = np.percentile(arr, 50)
            p95 = np.percentile(arr, 95)
            print(f"    p50={p50:.2f} ms, p95={p95:.2f} ms")
    print("=" * 60)


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
        _print_latency_metrics()

if __name__ == "__main__":
    main()
