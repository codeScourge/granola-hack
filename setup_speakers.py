import pickle
import numpy as np
from pathlib import Path
from pyannote.audio import Inference
from pyannote.audio import Audio

SPEAKERS_SAMPLES_DIR = Path("speakers_samples")

# Load embedding model
model = Inference("pyannote/embedding", window="whole")
audio = Audio(sample_rate=16000, mono="downmix")

# Store speaker embeddings
speaker_db = {}

def enroll_speaker(name, audio_file):
    """Extract and store speaker embedding from whole file"""
    waveform, sr = audio(audio_file)
    embedding = model({"waveform": waveform, "sample_rate": sr})
    if name not in speaker_db:
        speaker_db[name] = []
    speaker_db[name].append(embedding)

# Enroll all .wav files from speakers_samples; label each as its filename (stem)
wav_files = list(SPEAKERS_SAMPLES_DIR.glob("*.wav"))
if not wav_files:
    raise SystemExit(f"No .wav files found in {SPEAKERS_SAMPLES_DIR}")

for wav_path in wav_files:
    name = wav_path.stem  # filename without .wav
    enroll_speaker(name, str(wav_path))

# Average embeddings per speaker
for name in speaker_db:
    speaker_db[name] = np.mean(speaker_db[name], axis=0)

# Save
with open("speaker_db.pkl", "wb") as f:
    pickle.dump(speaker_db, f)

print(f"Enrolled {len(speaker_db)} speakers")
