import pickle
import numpy as np
from pathlib import Path
from pyannote.audio import Inference, Audio
from scipy.spatial.distance import cosine

# Load enrolled speakers
with open("speaker_db.pkl", "rb") as f:
    speaker_db = pickle.load(f)

model = Inference("pyannote/embedding", window="whole")
audio = Audio(sample_rate=16000, mono="downmix")

THRESHOLD = 0.3  # Adjust based on testing

def identify_speaker(embedding, threshold=THRESHOLD):
    """Compare embedding against database"""
    best_match = None
    min_distance = float('inf')
    
    for name, ref_embedding in speaker_db.items():
        dist = cosine(embedding.flatten(), ref_embedding.flatten())
        if dist < min_distance:
            min_distance = dist
            best_match = name
    
    if min_distance < threshold:
        return best_match, min_distance
    return "UNKNOWN", min_distance


def test_conversation_visual(audio_file):
    """
    Auto-segment and identify, then export for manual review.
    No pre-labeling needed - you verify after seeing results.
    """
    from pyannote.audio import Pipeline
    from pyannote.core import Segment
    
    # Diarization auto-finds speaker segments
    diarization = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token="YOUR_TOKEN"
    )
    
    diar_result = diarization(audio_file)
    
    results = []
    print(f"{'Time':<15} {'Diar':<12} {'Identified':<12} {'Distance':<10}")
    print("="*50)
    
    for turn, _, diar_label in diar_result.itertracks(yield_label=True):
        waveform, sr = audio.crop(audio_file, turn)
        embedding = model({"waveform": waveform, "sample_rate": sr})
        
        identified, distance = identify_speaker(embedding)
        
        results.append({
            "start": turn.start,
            "end": turn.end,
            "duration": turn.duration,
            "diar_label": diar_label,
            "identified": identified,
            "distance": distance,
        })
        
        print(f"{turn.start:5.1f}-{turn.end:5.1f}s  {diar_label:<12} "
              f"{identified:<12} {distance:.3f}")
    
    # Save with audio clips for listening
    output_dir = Path("review_segments")
    output_dir.mkdir(exist_ok=True)
    
    for i, r in enumerate(results):
        segment = Segment(r["start"], r["end"])
        waveform, sr = audio.crop(audio_file, segment)
        
        # Save clip with descriptive name
        filename = f"{i:03d}_{r['start']:.1f}s_{r['identified']}_dist{r['distance']:.2f}.wav"
        import torchaudio
        torchaudio.save(
            output_dir / filename,
            waveform,
            sr
        )
    
    print(f"\n{len(results)} segments saved to {output_dir}/")
    print("Listen to clips and check if identified names are correct")
    
    return results

results = test_conversation_visual("4person_chat.mp3")
