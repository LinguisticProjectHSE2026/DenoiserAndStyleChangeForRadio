import numpy as np
import torch
from pydub import AudioSegment

MODEL_ID = "facebook/mms-lid-4017"
ISO1_TO_3 = {"ru": "rus", "en": "eng", "be": "bel", "uk": "ukr"}


class MMSLanguageFilter:
    """Discards a segment when Meta MMS-LID puts it in `languages` above prob_threshold.

    Unlike Whisper, MMS-LID has a Chukchi class, so genuine Chukchi rarely scores high on
    Russian. Same interface as LanguageFilter: process(segment) -> True means discard.
    """

    def __init__(self, languages, prob_threshold=0.5, debug=False):
        self.codes = {ISO1_TO_3.get(c, c) for c in languages}
        self.prob_threshold = prob_threshold
        self.debug = debug
        self._model = None

    def _load(self):
        if self._model is None:
            from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

            self._fe = AutoFeatureExtractor.from_pretrained(MODEL_ID)
            self._model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID).eval()
            self._ids = [i for i, c in self._model.config.id2label.items() if c in self.codes]

    @torch.no_grad()
    def process(self, segment: AudioSegment) -> bool:
        self._load()
        seg = segment.set_frame_rate(16000).set_channels(1)
        x = np.array(seg.get_array_of_samples(), dtype=np.float32)[:16000 * 30] / 32768.0
        logits = self._model(**self._fe(x, sampling_rate=16000, return_tensors="pt")).logits[0]
        probs = torch.softmax(logits, -1)
        p = max((probs[i].item() for i in self._ids), default=0.0)
        if self.debug:
            print(f"    mms_lang_filter: p={p:.3f} threshold={self.prob_threshold}")
        return p > self.prob_threshold
