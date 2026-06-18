from pathlib import Path

from preprocessor import Preprocessor
from style_changer.process import process as change_style

INPUT_DIR = Path("input")
TMP_DIR = Path("tmp")
OUTPUT_DIR = Path("output")

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def run() -> None:
    preprocessor = Preprocessor(debug=True)
    for input_file in INPUT_DIR.iterdir():
        if input_file.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        output_file = OUTPUT_DIR / input_file.name

        print(f"Denoising:      {input_file} -> {TMP_DIR}/{{speech,singing}}/")
        preprocessor.process(input_file, TMP_DIR)

        print(f"Style changing: {TMP_DIR}/ -> {output_file}")
        change_style(TMP_DIR, output_file)

        print(f"Done:           {output_file}\n")


if __name__ == "__main__":
    run()
