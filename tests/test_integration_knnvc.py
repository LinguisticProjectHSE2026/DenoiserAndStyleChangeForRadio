import shutil
from pathlib import Path

import pytest

pytest.importorskip("torch")  # skip entirely if torch isn't installed
from pydub import AudioSegment  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEECH_DIR = REPO_ROOT / "tmp" / "speech"
SINGING_DIR = REPO_ROOT / "tmp" / "singing"


@pytest.mark.integration
def test_knnvc_converts_one_chunk(tmp_path, monkeypatch):
    """Run the real kNN-VC model end-to-end on real denoised chunks (CPU).

    Slow: downloads WavLM + HiFi-GAN weights on first run. Uses real audio
    (pure synthetic tones get fully VAD-trimmed by kNN-VC's matching set).
    Skipped if torch/network/model or the sample chunks are unavailable.
    """
    speech = sorted(SPEECH_DIR.glob("*.wav"))
    singing = sorted(SINGING_DIR.glob("*.wav"))
    if not speech or not singing:
        pytest.skip("No real denoised chunks under tmp/ to use as source/target.")

    monkeypatch.setenv("STYLE_DEVICE", "cpu")
    monkeypatch.setenv("STYLE_LABELS", "speech,singing")  # this test exercises both branches
    monkeypatch.setenv("STYLE_SPEECH_VC", "knn-vc")
    monkeypatch.setenv("STYLE_SINGING_VC", "knn-vc")
    monkeypatch.setenv("STYLE_SPEED", "1.0")

    # Target = one real singing chunk (real speech, so VAD trimming behaves).
    target = tmp_path / "target"
    target.mkdir()
    shutil.copy(singing[0], target / "ref.wav")
    monkeypatch.setenv("STYLE_TARGET_DIR", str(target))

    # reset the cached singleton so it re-reads the env above
    from style_changer import converter as conv_mod

    conv_mod._CONVERTER = None

    # Source = one speech + one singing chunk, with a global index ordering.
    src = tmp_path / "tmp"
    (src / "speech").mkdir(parents=True)
    (src / "singing").mkdir(parents=True)
    shutil.copy(speech[0], src / "speech" / "rec_0000.wav")
    shutil.copy(singing[0], src / "singing" / "rec_0001.wav")

    from style_changer import process as sc_process

    out = tmp_path / "out.wav"
    try:
        sc_process.process(src, out)
    except Exception as exc:  # no network / model unavailable in CI
        pytest.skip(f"kNN-VC model unavailable: {exc}")

    result = AudioSegment.from_file(str(out))
    assert result.frame_rate == sc_process.OUTPUT_SAMPLE_RATE
    assert result.channels == 1
    assert result.duration_seconds > 0
