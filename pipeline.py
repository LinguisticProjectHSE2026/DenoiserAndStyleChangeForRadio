from pathlib import Path

from denoiser.process import process as denoise
from style_changer.process import process as change_style

INPUT_DIR = Path("input")
TMP_DIR = Path("tmp")
OUTPUT_DIR = Path("output")

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def run() -> None:
    for input_file in INPUT_DIR.iterdir():
        if input_file.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        tmp_file = TMP_DIR / input_file.name
        output_file = OUTPUT_DIR / input_file.name

        print(f"Denoising:      {input_file} -> {tmp_file}")
        denoise(input_file, tmp_file)

        print(f"Style changing: {tmp_file} -> {output_file}")
        change_style(tmp_file, output_file)

        print(f"Done:           {output_file}\n")


if __name__ == "__main__":
    run()
