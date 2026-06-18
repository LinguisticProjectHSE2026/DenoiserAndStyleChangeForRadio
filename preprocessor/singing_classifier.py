import numpy as np
from pydub import AudioSegment


class SingingClassifier:
    """Classifies an AudioSegment as 'singing' or 'speech' using pitch CV."""

    def __init__(
        self,
        singing_cv_threshold: float = 0.6,
        window_duration: float = 0.05,   # seconds
        max_pitch_hz: int = 1000,
        min_pitch_hz: int = 70,
    ):
        self.singing_cv_threshold = singing_cv_threshold
        self.window_duration = window_duration
        self.max_pitch_hz = max_pitch_hz
        self.min_pitch_hz = min_pitch_hz

    def process(self, segment: AudioSegment) -> str:
        cv = self._pitch_cv(segment)
        return "singing" if cv < self.singing_cv_threshold else "speech"

    def _pitch_cv(self, segment: AudioSegment) -> float:
        """Coefficient of variation of pitch over voiced frames.

        Low CV = stable pitch = singing. High CV = variable pitch = speech.
        """
        samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
        if segment.channels == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)

        sr = segment.frame_rate
        win = int(sr * self.window_duration)
        hop = win // 2
        min_period = int(sr / self.max_pitch_hz)
        max_period = int(sr / self.min_pitch_hz)

        pitches = []
        for i in range(0, len(samples) - win, hop):
            w = samples[i : i + win]
            corr = np.correlate(w, w, mode="full")[len(w):]
            if max_period >= len(corr):
                continue
            peak = int(np.argmax(corr[min_period:max_period])) + min_period
            if corr[peak] > 0.15 * corr[0]:  # voiced frame check
                pitches.append(sr / peak)

        if len(pitches) < 5:
            return 1.0  # too few voiced frames → treat as speech

        arr = np.array(pitches)
        return float(np.std(arr) / (np.mean(arr) + 1e-8))
