import subprocess
import shutil
import tempfile
from pathlib import Path

from pydub import AudioSegment


class VoiceDenoiser:
    """Strips background music via demucs, returns vocals as AudioSegment."""

    def process(self, input_path: Path) -> AudioSegment:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            vocals_path = Path(f.name)
        try:
            demucs_out = vocals_path.parent / "_demucs_tmp"
            subprocess.run(
                ["python", "-m", "demucs", "--two-stems=vocals", "-o", str(demucs_out), str(input_path)],
                check=True,
            )
            src = demucs_out / "htdemucs" / input_path.stem / "vocals.wav"
            shutil.move(str(src), str(vocals_path))
            shutil.rmtree(demucs_out)
            return AudioSegment.from_file(vocals_path)
        finally:
            vocals_path.unlink(missing_ok=True)
