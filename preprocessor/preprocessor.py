from pathlib import Path

from pydub.silence import detect_nonsilent

from .voice_denoiser import VoiceDenoiser
from .singing_classifier import SingingClassifier
from .language_filter import LanguageFilter


class Preprocessor:
    """Pipeline: background removal -> silence splitting -> classification."""

    def __init__(
        self,
        silence_thresh_dbfs: int = -40,
        min_silence_ms: int = 650,
        min_chunk_ms: int = 600,
        singing_cv_threshold: float = 0.6,
        strip_languages_speech: set[str] = {"ru"},
        strip_languages_prob_threshold_speech: float = 0.4,
        strip_languages_use_median_speech: bool = True,
        strip_languages_singing: set[str] = None,
        strip_languages_prob_threshold_singing: float = 0.7,
        strip_languages_use_median_singing: bool = False,
        debug: bool = False,
    ):
        self.voice_denoiser = VoiceDenoiser()
        self.singing_classifier = SingingClassifier(singing_cv_threshold)
        self.speech_lang_filter = LanguageFilter(strip_languages_speech, strip_languages_prob_threshold_speech, strip_languages_use_median_speech, debug=debug) if strip_languages_speech else None
        self.singing_lang_filter = LanguageFilter(strip_languages_singing, strip_languages_prob_threshold_singing, use_median=strip_languages_use_median_singing, debug=debug) if strip_languages_singing else None
        self.silence_thresh_dbfs = silence_thresh_dbfs
        self.min_silence_ms = min_silence_ms
        self.min_chunk_ms = min_chunk_ms
        self.debug = debug

    def process(self, input_path: Path, output_base: Path) -> dict[str, list[Path]]:
        """Remove background music, split by silence, classify each chunk.

        Outputs to output_base/{speech,singing}/. Segments of chosen languages are discarded.
        """
        audio = self.voice_denoiser.process(input_path)
        ranges = detect_nonsilent(
            audio,
            min_silence_len=self.min_silence_ms,
            silence_thresh=self.silence_thresh_dbfs,
        )

        stem = input_path.stem
        for old_file in output_base.rglob(f"{stem}_*.wav"):
            old_file.unlink()

        result: dict[str, list[Path]] = {}
        for i, (start_ms, end_ms) in enumerate(ranges):
            duration_ms = end_ms - start_ms
            if duration_ms < self.min_chunk_ms:
                if self.debug:
                    print(f"  chunk {i:04d} {start_ms/1000:.2f}s-{end_ms/1000:.2f}s ({duration_ms}ms) -> skipped (too short)")
                continue
            if self.debug:
                print(f"  chunk {i:04d} {start_ms/1000:.2f}s-{end_ms/1000:.2f}s ({duration_ms}ms)")

            chunk = audio[start_ms:end_ms]
            label = self.singing_classifier.process(chunk)

            lang_filter = self.speech_lang_filter if label == "speech" else self.singing_lang_filter
            if lang_filter is not None and lang_filter.process(chunk):
                if self.debug:
                    print(f"  -> {label} (discarded: filtered language)")
                continue

            if self.debug:
                print(f"  -> {label}")

            out_dir = output_base / label
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{stem}_{i:04d}.wav"
            chunk.export(out_path, format="wav")
            result.setdefault(label, []).append(out_path)

        return result
