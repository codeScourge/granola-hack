import pyannote_load_patch  # noqa: F401 — before any pyannote model load

import os
from pathlib import Path
from dotenv import load_dotenv
import pickle
import numpy as np
from pyannote.audio import Model, Inference, Audio
from pyannote.core import Segment


# --- auth
load_dotenv()

hf_token = os.environ.get("HF_KEY")
if not hf_token:
    raise SystemExit(
        "HF_KEY must be set. "
        "Accept conditions at https://hf.co/pyannote/embedding"
    )

from huggingface_hub import login
login(token=hf_token)

# --- things
SPEAKER_SAMPLES_DIR = Path("speakers_samples")
ACCEPTABLE_EXTENSIONS = {".mp3"}

model = Model.from_pretrained("pyannote/embedding")
inference = Inference(model)
audio = Audio(sample_rate=16000, mono="downmix")

speaker_db = {}


# --- running shit
def enroll_speaker_multi_condition(name, samples):
    embeddings = []
    for audio_file, start, end, condition in samples:
        if start is not None and end is not None:
            segment = Segment(start, end)
            emb = inference.crop(audio_file, segment)
        else:
            emb = inference(audio_file)
        embeddings.append(emb)
    
    if not embeddings:
        return
    speaker_db[name] = np.mean(embeddings, axis=0)

# --- main
if __name__ == "__main__":
    if not SPEAKER_SAMPLES_DIR.exists():
        raise SystemExit(f"Directory {SPEAKER_SAMPLES_DIR} not found")

    subdirs = [d for d in SPEAKER_SAMPLES_DIR.iterdir() if d.is_dir()]
    if not subdirs:
        raise SystemExit(f"No speaker folders found in {SPEAKER_SAMPLES_DIR}")

    for speaker_dir in sorted(subdirs):
        name = speaker_dir.name
        files = [p for p in speaker_dir.iterdir() 
                if p.is_file() and p.suffix.lower() in ACCEPTABLE_EXTENSIONS]
        if not files:
            print(f"Warning: no audio files in {speaker_dir}")
            continue
        
        samples = [(str(p), None, None, p.stem) for p in sorted(files)]
        enroll_speaker_multi_condition(name, samples)
        print(f"Enrolled '{name}' with {len(samples)} sample(s)")

    with open("speaker_db.pkl", "wb") as f:
        pickle.dump(speaker_db, f)

    print(f"Enrolled {len(speaker_db)} speakers → speaker_db.pkl")
