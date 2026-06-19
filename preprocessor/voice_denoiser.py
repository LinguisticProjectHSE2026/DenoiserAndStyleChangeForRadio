from pathlib import Path

import numpy as np
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model
from pydub import AudioSegment


class VoiceDenoiser:
    """Strips background music via demucs (in-process), returns vocals as AudioSegment."""

    def __init__(self) -> None:
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = get_model("htdemucs").eval()
        return self._model

    def process(self, input_path: Path) -> AudioSegment:
        model = self._get_model()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        seg = AudioSegment.from_file(str(input_path)).set_frame_rate(model.samplerate)
        x = np.array(seg.get_array_of_samples(), dtype=np.float32).reshape(-1, seg.channels).T / 32768.0
        wav = torch.from_numpy(x)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)

        ref = wav.mean(0)
        normed = (wav - ref.mean()) / (ref.std() + 1e-9)
        with torch.no_grad():
            sources = apply_model(model, normed[None], shifts=0, device=device, progress=False)[0]
        sources = sources * ref.std() + ref.mean()

        vocals = sources[model.sources.index("vocals")].clamp(-1.0, 1.0)
        samples = (vocals.T.cpu().numpy() * 32767.0).astype(np.int16)
        return AudioSegment(
            data=samples.tobytes(), sample_width=2,
            frame_rate=model.samplerate, channels=samples.shape[1],
        )
