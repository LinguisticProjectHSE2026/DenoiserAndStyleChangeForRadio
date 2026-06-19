import os
from pathlib import Path

import numpy as np
from pydub import AudioSegment

from style_changer.device import select_device

DEFAULT_TARGET_DIR = "target"
REFERENCE_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
MODEL_SAMPLE_RATE = 16000
TOPK = 4


def resolve_target_dir() -> Path:
    """Target speaker reference dir, read at call time (STYLE_TARGET_DIR override)."""
    return Path(os.environ.get("STYLE_TARGET_DIR", DEFAULT_TARGET_DIR))


class VoiceConverter:
    """Zero-shot voice conversion via kNN-VC. Loads model + target matching set once."""

    def __init__(self) -> None:
        import torch

        self.device = select_device()
        target_dir = resolve_target_dir()

        refs = sorted(
            p for p in target_dir.glob("*") if p.suffix.lower() in REFERENCE_EXTENSIONS
        )
        if not refs:
            raise FileNotFoundError(
                f"No target reference audio in '{target_dir}'. Add target-speaker wavs "
                f"(≈5-10 min) or set STYLE_TARGET_DIR. Accepted: {sorted(REFERENCE_EXTENSIONS)}"
            )

        self.model = torch.hub.load(
            "bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, device=self.device,
        )
        # feed tensors (not paths) so kNN-VC avoids torchaudio.load -> torchcodec
        self.matching_set = self.model.get_matching_set([self._load_16k(p) for p in refs])

    @staticmethod
    def _load_16k(path: Path):
        """Load audio as a (1, T) float32 tensor at 16 kHz mono."""
        import torch

        seg = AudioSegment.from_file(str(path)).set_channels(1).set_frame_rate(MODEL_SAMPLE_RATE)
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0
        return torch.from_numpy(samples)[None, :]

    def convert(self, chunk_path: Path) -> np.ndarray:
        """Convert one chunk to the target voice. Returns int16 mono @ 16 kHz."""
        query_seq = self.model.get_features(self._load_16k(chunk_path))
        wav = self.model.match(query_seq, self.matching_set, topk=TOPK)
        wav = wav.clamp(-1.0, 1.0).cpu().numpy()
        return (wav * 32767.0).astype(np.int16)


_CONVERTER: VoiceConverter | None = None


def get_converter() -> VoiceConverter:
    """Lazy singleton so the model loads once per run."""
    global _CONVERTER
    if _CONVERTER is None:
        _CONVERTER = VoiceConverter()
    return _CONVERTER
