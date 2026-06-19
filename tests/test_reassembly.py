import json
from pathlib import Path

import numpy as np
import pytest
from pydub import AudioSegment

from style_changer import process as sc_process
from style_changer.converter import MODEL_SAMPLE_RATE


def _make_chunk(path: Path, ms: int = 100) -> None:
    """Write a small 44.1 kHz stereo wav, mirroring real denoiser chunks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    AudioSegment.silent(duration=ms, frame_rate=44100).set_channels(2).export(
        str(path), format="wav"
    )


class _FakeConverter:
    """Identity-ish stand-in: records call order, returns 1 s mono @ model rate."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def convert(self, chunk_path: Path) -> np.ndarray:
        self.calls.append(chunk_path.stem)
        return np.full(MODEL_SAMPLE_RATE, 1000, dtype=np.int16)


def test_discover_orders_by_global_index(tmp_path, monkeypatch):
    monkeypatch.setenv("STYLE_LABELS", "speech,singing")
    # speech and singing share one global index; 10 must sort after 2 (not lexically)
    _make_chunk(tmp_path / "speech" / "rec_0000.wav")
    _make_chunk(tmp_path / "singing" / "rec_0001.wav")
    _make_chunk(tmp_path / "speech" / "rec_0002.wav")
    _make_chunk(tmp_path / "singing" / "rec_0010.wav")

    ordered = sc_process._discover_chunks(tmp_path)
    assert [sc_process._chunk_index(p) for p in ordered] == [0, 1, 2, 10]


def test_process_reassembles_in_timeline_order(tmp_path, monkeypatch):
    # force kNN-VC on both branches so chunks route through the mock; no slow-down
    monkeypatch.setenv("STYLE_LABELS", "speech,singing")
    monkeypatch.setenv("STYLE_SPEECH_VC", "knn-vc")
    monkeypatch.setenv("STYLE_SINGING_VC", "knn-vc")
    monkeypatch.setenv("STYLE_SPEED", "1.0")
    _make_chunk(tmp_path / "speech" / "rec_0000.wav")
    _make_chunk(tmp_path / "singing" / "rec_0001.wav")
    _make_chunk(tmp_path / "speech" / "rec_0002.wav")

    fake = _FakeConverter()
    monkeypatch.setattr(sc_process, "get_converter", lambda: fake)

    out = tmp_path / "out.wav"
    sc_process.process(tmp_path, out)

    # converted across both folders in global-index order
    assert fake.calls == ["rec_0000", "rec_0001", "rec_0002"]

    result = AudioSegment.from_file(str(out))
    assert result.frame_rate == sc_process.OUTPUT_SAMPLE_RATE
    assert result.channels == 1
    assert abs(result.duration_seconds - 3.0) < 0.05  # 3 chunks * 1 s


def test_passthrough_slows_without_a_model(tmp_path, monkeypatch):
    # default VC 'off' keeps the speaker (no model); STYLE_SPEED stretches the timeline
    monkeypatch.setenv("STYLE_LABELS", "speech,singing")
    monkeypatch.setenv("STYLE_SPEED", "0.5")
    _make_chunk(tmp_path / "speech" / "rec_0000.wav", ms=200)
    _make_chunk(tmp_path / "singing" / "rec_0001.wav", ms=200)

    out = tmp_path / "out.wav"
    sc_process.process(tmp_path, out)

    result = AudioSegment.from_file(str(out))
    assert result.frame_rate == sc_process.OUTPUT_SAMPLE_RATE
    assert result.channels == 1
    # 0.4 s of audio at 0.5x speed -> ~0.8 s
    assert abs(result.duration_seconds - 0.8) < 0.1


def test_restore_gaps_inserts_silence(tmp_path, monkeypatch):
    monkeypatch.setenv("STYLE_SPEED", "1.0")
    monkeypatch.setenv("STYLE_RESTORE_GAPS", "on")
    _make_chunk(tmp_path / "speech" / "rec_0000.wav", ms=200)
    _make_chunk(tmp_path / "speech" / "rec_0001.wav", ms=200)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"rec_0000": [0, 200], "rec_0001": [700, 900]})
    )

    out = tmp_path / "out.wav"
    sc_process.process(tmp_path, out)

    result = AudioSegment.from_file(str(out))
    assert abs(result.duration_seconds - 0.9) < 0.1  # 0.2 + 0.5 gap + 0.2


def test_default_excludes_singing(tmp_path):
    # default STYLE_LABELS = speech only; singing chunks must not reach the output
    _make_chunk(tmp_path / "speech" / "rec_0000.wav", ms=200)
    _make_chunk(tmp_path / "singing" / "rec_0001.wav", ms=200)

    out = tmp_path / "out.wav"
    sc_process.process(tmp_path, out)

    result = AudioSegment.from_file(str(out))
    assert abs(result.duration_seconds - 0.2) < 0.05  # speech chunk only


def test_empty_tmp_raises(tmp_path):
    (tmp_path / "speech").mkdir()
    (tmp_path / "singing").mkdir()
    with pytest.raises(FileNotFoundError):
        sc_process.process(tmp_path, tmp_path / "out.wav")
