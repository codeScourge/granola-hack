import pyannote_load_patch  # noqa: F401 — before any pyannote model load

import os
from pathlib import Path

import torchaudio
from dotenv import load_dotenv
from huggingface_hub import login
from pyannote.audio import Audio, Inference, Pipeline
from pyannote.core import Segment

from conversation_speakers import (
    ConversationSpeakers,
    clamp_segment,
    load_speaker_db,
)

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

# --- build helper (chunk in → who is talking; or run file for diarize + identify)
speaker_db = load_speaker_db()
audio = Audio(sample_rate=16000, mono="downmix")
helper = ConversationSpeakers(
    embedding_model=Inference("pyannote/embedding", window="whole"),
    diarization_pipeline=Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    ),
    audio=audio,
    speaker_db=speaker_db,
)


def test_conversation_visual(audio_file):
    """
    Auto-segment and identify, then export for manual review.
    Uses conversation_speakers helper; segments clamped to file bounds.
    SHORT = segment too short for embedding model; distance inf = not computed.
    """
    results = helper.run(audio_file, clamp_to_bounds=True)

    print(f"{'Time':<15} {'Diar':<12} {'Identified':<12} {'Distance':<10}")
    print("=" * 50)
    for r in results:
        dist_str = f"{r['distance']:.3f}" if r["distance"] != float("inf") else "inf"
        print(
            f"{r['start']:5.1f}-{r['end']:5.1f}s  {r['diar_label']:<12} "
            f"{r['identified']:<12} {dist_str}"
        )

    # Save clips for manual review (clamp segment to file duration to avoid crop errors)
    output_dir = Path("review_segments")
    output_dir.mkdir(exist_ok=True)
    for old in output_dir.iterdir():
        if old.is_file():
            old.unlink()
    file_duration = audio.get_duration(audio_file)

    for i, r in enumerate(results):
        seg = clamp_segment(Segment(r["start"], r["end"]), file_duration)
        try:
            waveform, sr = audio.crop(audio_file, seg)
        except ValueError:
            continue
        dist_str = f"{r['distance']:.2f}" if r["distance"] != float("inf") else "inf"
        filename = f"{i:03d}_{r['start']:.1f}s_{r['identified']}_dist{dist_str}.wav"
        torchaudio.save(output_dir / filename, waveform, sr)

    print(f"\n{len(results)} segments saved to {output_dir}/")
    return results


if __name__ == "__main__":
    results = test_conversation_visual("tests/00.mp3")
