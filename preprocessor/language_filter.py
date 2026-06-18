import numpy as np
import whisper
from pydub import AudioSegment
from pydub.silence import detect_nonsilent


class LanguageFilter:
    """Detects whether an AudioSegment belongs to any of the given language codes."""

    def __init__(
        self,
        languages: set[str],
        prob_threshold: float = 0.5,
        use_median: bool = True,
        median_silence_thresh_dbfs: int = -30,
        median_min_silence_ms: int = 200,
        debug: bool = False,
    ):
        self.languages = languages
        self.prob_threshold = prob_threshold
        self.use_median = use_median
        self.median_silence_thresh_dbfs = median_silence_thresh_dbfs
        self.median_min_silence_ms = median_min_silence_ms
        self.debug = debug
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = whisper.load_model("base")
        return self._model

    def _lang_prob(self, segment: AudioSegment) -> float:
        """Returns the max probability across target languages for a segment."""
        seg16 = segment.set_frame_rate(16000).set_channels(1)
        samples = np.array(seg16.get_array_of_samples(), dtype=np.float32) / 32768.0
        audio = whisper.pad_or_trim(samples)
        mel = whisper.log_mel_spectrogram(audio).to(self._get_model().device)
        _, probs = self._get_model().detect_language(mel)
        return max(probs.get(lang, 0.0) for lang in self.languages)

    def _process_single(self, segment: AudioSegment) -> bool:
        prob = self._lang_prob(segment)
        if self.debug:
            print(f"    lang_filter: prob={prob:.3f} threshold={self.prob_threshold}")
        return prob > self.prob_threshold

    def _process_median(self, segment: AudioSegment) -> bool:
        ranges = detect_nonsilent(
            segment,
            min_silence_len=self.median_min_silence_ms,
            silence_thresh=self.median_silence_thresh_dbfs,
        )
        if not ranges:
            return self._process_single(segment)
        probs = [self._lang_prob(segment[start:end]) for start, end in ranges]
        median = float(np.median(probs))
        if self.debug:
            print(f"    lang_filter: median={median:.3f} ({len(probs)} sub-chunks) threshold={self.prob_threshold}")
        return median > self.prob_threshold

    def process(self, segment: AudioSegment) -> bool:
        if self.use_median:
            return self._process_median(segment)
        return self._process_single(segment)
