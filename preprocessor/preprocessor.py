import json
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
        lang_filter_backend: str = "whisper",
        remove_nonspeech: bool = False,
        vad_threshold: float = 0.5,
        merge_short_ms: int = 0,
        debug: bool = False,
    ):
        self.merge_short_ms = merge_short_ms
        self.voice_denoiser = VoiceDenoiser()
        self.singing_classifier = SingingClassifier(singing_cv_threshold)
        if remove_nonspeech:
            from .vad import VoiceActivityDetector

            self.vad = VoiceActivityDetector(threshold=vad_threshold)
        else:
            self.vad = None
        if not strip_languages_speech:
            self.speech_lang_filter = None
        elif lang_filter_backend == "mms":
            from .mms_language_filter import MMSLanguageFilter

            self.speech_lang_filter = MMSLanguageFilter(strip_languages_speech, strip_languages_prob_threshold_speech, debug=debug)
        else:
            self.speech_lang_filter = LanguageFilter(strip_languages_speech, strip_languages_prob_threshold_speech, strip_languages_use_median_speech, debug=debug)
        self.singing_lang_filter = LanguageFilter(strip_languages_singing, strip_languages_prob_threshold_singing, use_median=strip_languages_use_median_singing, debug=debug) if strip_languages_singing else None
        self.silence_thresh_dbfs = silence_thresh_dbfs
        self.min_silence_ms = min_silence_ms
        self.min_chunk_ms = min_chunk_ms
        self.debug = debug

    def _merge_short_ranges(self, ranges):
        """Merge short ranges into a neighbour across a small gap, for stabler classification/LID."""
        if self.merge_short_ms <= 0:
            return ranges
        merged = []
        for s, e in ranges:
            short = e - s < self.merge_short_ms
            if merged and s - merged[-1][1] <= self.merge_short_ms and (short or merged[-1][1] - merged[-1][0] < self.merge_short_ms):
                merged[-1][1] = e
            else:
                merged.append([s, e])
        return merged

    def process(self, input_path: Path, output_base: Path) -> dict[str, list[Path]]:
        """Remove background music, split by silence, classify each chunk.

        Outputs to output_base/{speech,singing}/. Segments of chosen languages are discarded.
        """
        audio = self.voice_denoiser.process(input_path)
        ranges = self._merge_short_ranges(detect_nonsilent(
            audio,
            min_silence_len=self.min_silence_ms,
            silence_thresh=self.silence_thresh_dbfs,
        ))

        stem = input_path.stem
        for old_file in output_base.rglob(f"{stem}_*.wav"):
            old_file.unlink()

        manifest_path = output_base / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        manifest = {k: v for k, v in manifest.items() if not k.startswith(f"{stem}_")}

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

            if label == "speech" and self.vad is not None:
                chunk = self.vad.trim(chunk)
                if len(chunk) < self.min_chunk_ms:
                    if self.debug:
                        print(f"  -> speech (discarded: no speech after VAD)")
                    continue

            if self.debug:
                print(f"  -> {label} ({len(chunk)}ms)")

            out_dir = output_base / label
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{stem}_{i:04d}.wav"
            chunk.export(out_path, format="wav")
            result.setdefault(label, []).append(out_path)
            manifest[f"{stem}_{i:04d}"] = [start_ms, end_ms]

        output_base.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest))
        return result
