import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from pydub import AudioSegment

from style_changer.converter import MODEL_SAMPLE_RATE, REFERENCE_EXTENSIONS, resolve_target_dir

KNN_SVC_DIR = Path(os.environ.get("KNN_SVC_DIR", Path(__file__).resolve().parents[2] / "knn-svc"))
KNN_SVC_CKPT_DIR = Path(os.environ.get("KNN_SVC_CKPT_DIR", "/tmp/knnsvc_ckpt"))
CKPT_TYPE = "mix_harm_no_amp"
POST_OPT = "post_opt_0.2"

_WRAPPER = Path(__file__).resolve().parent / "_knnsvc_infer.py"


def resolve_singing_target_dir() -> Path:
    """Sung-reference dir for singing (STYLE_SINGING_TARGET_DIR), else the speech target."""
    override = os.environ.get("STYLE_SINGING_TARGET_DIR")
    return Path(override) if override else resolve_target_dir()


def _unavailable_reason() -> str | None:
    if not (KNN_SVC_DIR / "ddsp_hubconf.py").is_file():
        return f"kNN-SVC repo not found at {KNN_SVC_DIR} (clone bshall/knn-svc, set KNN_SVC_DIR)"
    if not list(KNN_SVC_CKPT_DIR.glob(f"*{CKPT_TYPE}*.pt")):
        return f"no '{CKPT_TYPE}' checkpoint in {KNN_SVC_CKPT_DIR} (set KNN_SVC_CKPT_DIR)"
    return None


class SingingConverter:
    """kNN-SVC singing voice conversion (batch).

    Converts all singing chunks in one subprocess via kNN-SVC's folder mode, so the
    model and target pool load once. Returns int16 mono @ 16 kHz per chunk.
    """

    def __init__(self) -> None:
        self._target = self._build_target()

    def _build_target(self) -> Path:
        target_dir = resolve_singing_target_dir()
        refs = sorted(p for p in target_dir.glob("*") if p.suffix.lower() in REFERENCE_EXTENSIONS)
        if not refs:
            raise FileNotFoundError(f"No singing target reference audio in '{target_dir}'.")
        combined = AudioSegment.empty()
        for ref in refs:
            combined += AudioSegment.from_file(str(ref))
        target = Path(tempfile.gettempdir()) / "style_singing_target_16k.wav"
        combined.set_channels(1).set_frame_rate(MODEL_SAMPLE_RATE).export(str(target), format="wav")
        return target

    def convert_batch(self, chunk_paths: list[Path]) -> dict[Path, np.ndarray]:
        if not chunk_paths:
            return {}
        work = Path(tempfile.mkdtemp())
        src_dir = work / "src" / "even"
        tgt_dir = work / "tgt" / "target"
        src_dir.mkdir(parents=True)
        tgt_dir.mkdir(parents=True)
        shutil.copy(self._target, tgt_dir / "target.wav")

        for chunk in chunk_paths:
            AudioSegment.from_file(str(chunk)).set_channels(1).set_frame_rate(
                MODEL_SAMPLE_RATE
            ).export(str(src_dir / f"{chunk.stem}.wav"), format="wav")

        converted_dir = work / "out"
        subprocess.run(
            [sys.executable, str(_WRAPPER), str(work / "src"), str(work / "tgt"),
             str(converted_dir), str(KNN_SVC_CKPT_DIR), CKPT_TYPE, POST_OPT],
            check=True,
            cwd=str(KNN_SVC_DIR),
        )

        results: dict[Path, np.ndarray] = {}
        for chunk in chunk_paths:
            out = converted_dir / "even" / chunk.stem / "target.wav"
            if out.is_file():
                seg = AudioSegment.from_file(str(out)).set_channels(1).set_frame_rate(MODEL_SAMPLE_RATE)
                results[chunk] = np.array(seg.get_array_of_samples(), dtype=np.int16)
        return results


_SINGING: SingingConverter | None = None


def get_singing_converter() -> SingingConverter | None:
    global _SINGING
    if _SINGING is None:
        reason = _unavailable_reason()
        if reason:
            print(f"[style_changer] kNN-SVC unavailable: {reason}. Using kNN-VC for singing.")
            return None
        _SINGING = SingingConverter()
    return _SINGING
