import os
from pathlib import Path
from dotenv import load_dotenv
import pickle
import numpy as np

# PyTorch 2.6+ defaults to weights_only=True; pyannote checkpoints contain
# PyTorch Lightning classes (e.g. EarlyStopping). Allow loading (trusted HF source).
import torch
_torch_load = torch.load
def _load_trusted(*args, **kwargs):
    kwargs["weights_only"] = False  # override Lightning's explicit weights_only=True
    return _torch_load(*args, **kwargs)
torch.load = _load_trusted

from pyannote.audio import Model, Inference, Audio
from pyannote.core import Segment

load_dotenv()

hf_token = os.environ.get("HF_KEY")
if not hf_token:
    raise SystemExit(
        "HF_KEY must be set. "
        "Accept conditions at https://hf.co/pyannote/embedding"
    )

# Login BEFORE importing pyannote
from huggingface_hub import login
login(token=hf_token)

SPEAKER_SAMPLES_DIR = Path("speakers_samples")
ACCEPTABLE_EXTENSIONS = {".mp3"}

# Model should load without needing token parameter after login
model = Model.from_pretrained("pyannote/embedding")
inference = Inference(model)
audio = Audio(sample_rate=16000, mono="downmix")

speaker_db = {}

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
