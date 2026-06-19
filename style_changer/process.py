import json
import os
import shutil
from pathlib import Path

import numpy as np
from pydub import AudioSegment

from style_changer.converter import MODEL_SAMPLE_RATE, get_converter

OUTPUT_SAMPLE_RATE = 44100
LABELS = ("speech", "singing")
DEFAULT_SPEED = 1.0


def _labels() -> tuple[str, ...]:
    """Chunk labels to include in the output (STYLE_LABELS); defaults to speech only."""
    return tuple(s.strip() for s in os.environ.get("STYLE_LABELS", "speech").split(",") if s.strip())


def _speed() -> float:
    return float(os.environ.get("STYLE_SPEED", DEFAULT_SPEED))


def _vc_mode(label: str) -> str:
    """Voice-conversion mode per branch: 'off' (keep speaker), 'knn-vc', or 'knn-svc'."""
    env = "STYLE_SPEECH_VC" if label == "speech" else "STYLE_SINGING_VC"
    return os.environ.get(env, "off")


def _chunk_index(path: Path) -> int:
    """Global timeline index from a `{stem}_{NNNN}.wav` filename."""
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def _discover_chunks(input_path: Path) -> list[Path]:
    chunks: list[Path] = []
    for label in _labels():
        chunks.extend((input_path / label).glob("*.wav"))
    return sorted(chunks, key=_chunk_index)


def _singing_svc_results(chunks: list[Path]) -> dict[Path, np.ndarray]:
    """Batch-convert singing chunks via kNN-SVC when STYLE_SINGING_VC=knn-svc."""
    if os.environ.get("STYLE_SINGING_VC", "off") != "knn-svc":
        return {}
    singing = [c for c in chunks if c.parent.name == "singing"]
    if not singing:
        return {}
    from style_changer.singing_converter import get_singing_converter

    converter = get_singing_converter()
    return converter.convert_batch(singing) if converter is not None else {}


def _segment_for(chunk: Path, svc_results: dict[Path, np.ndarray]) -> AudioSegment | None:
    """One chunk as a 44.1 kHz mono segment: voice-converted or passed through."""
    mode = _vc_mode(chunk.parent.name)
    if mode == "off":
        seg = AudioSegment.from_file(str(chunk))
    else:
        samples = svc_results[chunk] if chunk in svc_results else get_converter().convert(chunk)
        if samples.size == 0:
            return None
        seg = AudioSegment(
            data=samples.tobytes(), sample_width=2, frame_rate=MODEL_SAMPLE_RATE, channels=1
        )
    return seg.set_channels(1).set_frame_rate(OUTPUT_SAMPLE_RATE)


def _load_manifest(input_path: Path) -> dict | None:
    if os.environ.get("STYLE_RESTORE_GAPS", "off").lower() not in ("1", "true", "on", "yes"):
        return None
    path = input_path / "manifest.json"
    return json.loads(path.read_text()) if path.is_file() else None


def _slow_down(segment: AudioSegment, speed: float) -> AudioSegment:
    """Pitch-preserving time stretch (speed < 1 = slower): Rubber Band if installed, else WSOLA."""
    if speed == 1.0:
        return segment
    sr = segment.frame_rate
    y = np.array(segment.get_array_of_samples(), dtype=np.float32) / 32768.0
    if shutil.which("rubberband"):
        import pyrubberband as pyrb

        y = pyrb.time_stretch(y, sr, speed)
    else:
        from audiotsm import wsola
        from audiotsm.io.array import ArrayReader, ArrayWriter

        writer = ArrayWriter(1)
        wsola(1, speed=speed).run(ArrayReader(y.reshape(1, -1)), writer)
        y = writer.data[0]
    out = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16)
    return AudioSegment(data=out.tobytes(), sample_width=2, frame_rate=sr, channels=1)


def process(input_path: Path, output_path: Path) -> None:
    """Render each Chukchi chunk to a neutral, slowed-down clip and reassemble into output_path.

    Only the labels in STYLE_LABELS are included (default 'speech'; set 'speech,singing' for both).
    Per branch, voice conversion is optional (STYLE_SPEECH_VC / STYLE_SINGING_VC; default 'off'
    keeps the original speaker). STYLE_SPEED slows playback for easier word recognition (default
    1.0 = no change). Chunks are concatenated in timeline order; silence gaps between them are
    dropped unless STYLE_RESTORE_GAPS is set (needs the preprocessor manifest).
    """
    chunks = _discover_chunks(input_path)
    if not chunks:
        raise FileNotFoundError(f"No chunks found under {input_path}/{{{','.join(_labels())}}}/.")

    svc_results = _singing_svc_results(chunks)
    manifest = _load_manifest(input_path)

    segments: list[AudioSegment] = []
    prev_end = None
    for chunk in chunks:
        if manifest is not None and chunk.stem in manifest:
            start_ms, end_ms = manifest[chunk.stem]
            if prev_end is not None and start_ms > prev_end:
                segments.append(
                    AudioSegment.silent(duration=start_ms - prev_end, frame_rate=OUTPUT_SAMPLE_RATE)
                )
            prev_end = end_ms
        seg = _segment_for(chunk, svc_results)
        if seg is not None and len(seg) > 0:
            segments.append(seg)

    if not segments:
        raise RuntimeError("Style change produced no audio for any chunk.")

    combined = segments[0]
    for segment in segments[1:]:
        combined += segment
    combined = _slow_down(combined, _speed())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(str(output_path), format=output_path.suffix.lstrip(".") or "wav")
