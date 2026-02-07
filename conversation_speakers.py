"""
External helper: diarize an audio file and identify who is talking per segment.
Also supports single-chunk identification for live use (chunk in → who is talking).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pyannote.audio import Audio, Inference, Pipeline
from pyannote.core import Segment
from scipy.spatial.distance import cosine


# Defaults; can be overridden when building the helper
DEFAULT_THRESHOLD = 0.3
# Smallest duration the pyannote/embedding (xvector) model accepts: receptive field
# for 1 output frame ≈ 4771 samples at 16kHz ≈ 0.298s. Below this, conv1d fails.
# Shorter segments are returned as identified="SHORT", distance=inf.
DEFAULT_MIN_SEGMENT_DURATION = 0.30  # seconds (~model minimum)


def embedding_vector(arr: np.ndarray) -> np.ndarray:
    """Normalize to (512,) for cosine: handle both whole-window (512,) and sliding (T, 512)."""
    x = np.asarray(arr).flatten()
    if x.size == 512:
        return x
    x = x.reshape(-1, 512)
    return x.mean(axis=0)


def identify_speaker(
    embedding: np.ndarray,
    speaker_db: Dict[str, np.ndarray],
    threshold: float = DEFAULT_THRESHOLD,
) -> Tuple[str, float]:
    """Compare one embedding to speaker DB. Returns (name, distance)."""
    best_match = None
    min_distance = float("inf")
    emb = embedding_vector(embedding)
    for name, ref_embedding in speaker_db.items():
        ref = embedding_vector(ref_embedding)
        dist = cosine(emb, ref)
        if dist < min_distance:
            min_distance = dist
            best_match = name
    if min_distance < threshold:
        return best_match, min_distance
    return "UNKNOWN", min_distance


def clamp_segment(segment: Segment, file_duration: float) -> Segment:
    """Clamp segment to [0, file_duration] to avoid crop out-of-bounds."""
    start = max(0.0, min(segment.start, file_duration))
    end = max(0.0, min(segment.end, file_duration))
    if end <= start:
        end = start + 1e-6  # avoid zero-length
    return Segment(start, end)


class ConversationSpeakers:
    """
    Diarize + speaker identification. Use for file-based "who spoke when" or
    single-chunk "who is talking" (e.g. live).
    """

    def __init__(
        self,
        *,
        embedding_model: Inference,
        diarization_pipeline: Pipeline,
        audio: Audio,
        speaker_db: Dict[str, np.ndarray],
        threshold: float = DEFAULT_THRESHOLD,
        min_segment_duration: float = DEFAULT_MIN_SEGMENT_DURATION,
    ):
        self.embedding_model = embedding_model
        self.diarization_pipeline = diarization_pipeline
        self.audio = audio
        self.speaker_db = speaker_db
        self.threshold = threshold
        self.min_segment_duration = min_segment_duration

    def identify_chunk(
        self,
        waveform: Any,
        sample_rate: int,
    ) -> Tuple[str, float]:
        """
        Identify speaker for a single chunk (e.g. live buffer).
        Returns (name, distance). Use for "chunk in → who is talking".
        """
        if getattr(waveform, "shape", None) is not None:
            duration = waveform.shape[-1] / sample_rate
        else:
            duration = len(waveform) / sample_rate
        if duration < self.min_segment_duration:
            return "SHORT", float("inf")
        file_like = {"waveform": waveform, "sample_rate": sample_rate}
        embedding = self.embedding_model(file_like)
        return identify_speaker(embedding, self.speaker_db, self.threshold)

    def run(
        self,
        audio_file: Union[str, Path],
        clamp_to_bounds: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Diarize the file and identify speaker for each segment.
        Returns list of dicts: start, end, duration, diar_label, identified, distance.
        If clamp_to_bounds is True (default), segments that extend past file duration
        are clamped so crop does not raise.
        """
        diar_result = self.diarization_pipeline(audio_file)
        file_duration = self.audio.get_duration(audio_file)
        results = []

        for turn, _, diar_label in diar_result.itertracks(yield_label=True):
            seg = clamp_segment(turn, file_duration) if clamp_to_bounds else turn
            try:
                waveform, sr = self.audio.crop(audio_file, seg)
            except ValueError:
                # still out of bounds or empty; skip embedding
                results.append({
                    "start": turn.start,
                    "end": turn.end,
                    "duration": turn.duration,
                    "diar_label": diar_label,
                    "identified": "SKIP",
                    "distance": float("inf"),
                })
                continue

            if seg.duration < self.min_segment_duration:
                identified, distance = "SHORT", float("inf")
            else:
                embedding = self.embedding_model({"waveform": waveform, "sample_rate": sr})
                identified, distance = identify_speaker(
                    embedding, self.speaker_db, self.threshold
                )

            results.append({
                "start": turn.start,
                "end": turn.end,
                "duration": turn.duration,
                "diar_label": diar_label,
                "identified": identified,
                "distance": distance,
            })

        return results


def load_speaker_db(path: Union[str, Path] = "speaker_db.pkl") -> Dict[str, np.ndarray]:
    """Load speaker DB from pickle path."""
    with open(path, "rb") as f:
        return pickle.load(f)
