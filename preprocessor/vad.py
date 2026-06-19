import numpy as np
import torch
from pydub import AudioSegment


class VoiceActivityDetector:
    """Trims a segment to its speech regions via Silero VAD, dropping sighs/breaths/pauses."""

    def __init__(self, threshold=0.5, speech_pad_ms=100):
        self.threshold = threshold
        self.speech_pad_ms = speech_pad_ms
        self._model = None
        self._get_speech_timestamps = None

    def _load(self):
        if self._model is None:
            self._model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
            self._get_speech_timestamps = utils[0]

    def trim(self, segment: AudioSegment) -> AudioSegment:
        self._load()
        seg16 = segment.set_channels(1).set_frame_rate(16000)
        samples = np.array(seg16.get_array_of_samples(), dtype=np.float32) / 32768.0
        spans = self._get_speech_timestamps(
            torch.from_numpy(samples), self._model,
            sampling_rate=16000, threshold=self.threshold, speech_pad_ms=self.speech_pad_ms,
        )
        out = AudioSegment.empty()
        for s in spans:
            out += segment[s["start"] * 1000 // 16000:s["end"] * 1000 // 16000]
        return out
