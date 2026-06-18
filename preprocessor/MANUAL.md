# preprocessor

Audio preprocessing pipeline for splitting recordings into labeled speech and singing chunks, with optional language filtering.

---

## preprocessor.Preprocessor

```
preprocessor.Preprocessor(silence_thresh_dbfs=-40, min_silence_ms=650,
    min_chunk_ms=600, singing_cv_threshold=0.6,
    strip_languages_speech={"ru"}, strip_languages_prob_threshold_speech=0.4,
    strip_languages_use_median_speech=True,
    strip_languages_singing=None, strip_languages_prob_threshold_singing=0.7,
    strip_languages_use_median_singing=False, debug=False)
```

Full pipeline. Strips background music, splits by silence, classifies each chunk as `"speech"` or `"singing"`, and optionally discards chunks in filtered languages. Previous output files with the same stem are deleted before writing new ones.

**Parameters:**

`silence_thresh_dbfs` *int, optional*

Loudness threshold in dBFS below which audio is considered silence. Default is `-40`.

`min_silence_ms` *int, optional*

Minimum duration of silence in milliseconds required to split a chunk. Default is `650`.

`min_chunk_ms` *int, optional*

Minimum duration of a chunk in milliseconds to be processed. Shorter chunks are discarded. Default is `600`.

`singing_cv_threshold` *float, optional*

Pitch coefficient of variation threshold for singing/speech classification. Values below this are classified as `"singing"`, above as `"speech"`. Default is `0.6`.

`strip_languages_speech` *set[str] or None, optional*

ISO 639-1 language codes to filter from speech chunks. Pass `None` or an empty set to disable. Default is `{"ru"}`.

`strip_languages_prob_threshold_speech` *float, optional*

Minimum language probability to trigger filtering for speech chunks. Default is `0.4`.

`strip_languages_use_median_speech` *bool, optional*

If `True`, splits the speech chunk into sub-chunks and uses the median language probability across them. More robust for mixed-language chunks. Default is `True`.

`strip_languages_singing` *set[str] or None, optional*

ISO 639-1 language codes to filter from singing chunks. Pass `None` or an empty set to disable. Default is `None`.

`strip_languages_prob_threshold_singing` *float, optional*

Minimum language probability to trigger filtering for singing chunks. Default is `0.7`.

`strip_languages_use_median_singing` *bool, optional*

If `True`, uses median sub-chunk probability for singing. Default is `False`.

`debug` *bool, optional*

If `True`, prints per-chunk classification and language filter scores to stdout. Default is `False`.

**Methods:**

`process(input_path, output_base)` *dict[str, list[Path]]*

Run the pipeline on a single audio file. Returns a dict mapping labels (`"speech"`, `"singing"`) to lists of output file paths. Output files are written to `output_base/speech/` and `output_base/singing/`.

**Examples:**

```python
from pathlib import Path
from preprocessor import Preprocessor

p = Preprocessor(debug=True)
result = p.process(Path("input/track.wav"), Path("tmp"))
# {"speech": [Path("tmp/speech/track_0001.wav"), ...], "singing": [...]}

# Filter Russian speech only, keep Russian singing
p = Preprocessor(strip_languages_speech={"ru"}, strip_languages_singing=None)

# Filter multiple languages
p = Preprocessor(strip_languages_speech={"ru", "uk"})
```

---

## preprocessor.VoiceDenoiser

```
preprocessor.VoiceDenoiser()
```

Strips background music from an audio file, returning the vocals-only track as an `AudioSegment`.

**Methods:**

`process(input_path)` *AudioSegment*

Isolate vocals from `input_path` and return them.

---

## preprocessor.SingingClassifier

```
preprocessor.SingingClassifier(singing_cv_threshold=0.6, window_duration=0.05,
    max_pitch_hz=1000, min_pitch_hz=70)
```

Classifies an `AudioSegment` as `"singing"` or `"speech"` based on pitch stability. Stable pitch classifies as `"singing"`, variable pitch as `"speech"`.

**Parameters:**

`singing_cv_threshold` *float, optional*

Pitch variation threshold below which a segment is classified as `"singing"`. Default is `0.6`.

`window_duration` *float, optional*

Analysis window size in seconds. Default is `0.05` (50 ms).

`max_pitch_hz` *int, optional*

Upper bound for pitch detection in Hz. Default is `1000`.

`min_pitch_hz` *int, optional*

Lower bound for pitch detection in Hz. Default is `70`.

**Methods:**

`process(segment)` *str*

Returns `"singing"` or `"speech"`.

---

## preprocessor.LanguageFilter

```
preprocessor.LanguageFilter(languages, prob_threshold=0.5, use_median=True,
    median_silence_thresh_dbfs=-30, median_min_silence_ms=200, debug=False)
```

Detects whether an `AudioSegment` belongs to any of the specified languages. Returns `True` if the segment is of selected language.

**Parameters:**

`languages` *set[str]*

Set of ISO 639-1 language codes to detect, e.g. `{"ru"}` or `{"ru", "uk"}`.

`prob_threshold` *float, optional*

Detection threshold. A segment is flagged if the max probability across target languages exceeds this value. Default is `0.5`.

`use_median` *bool, optional*

If `True`, splits the segment into sub-chunks and takes the median language probability, making detection more robust for mixed-language audio. Falls back to single-shot if no sub-chunks are found. Default is `True`.

`median_silence_thresh_dbfs` *int, optional*

Silence threshold in dBFS used for sub-chunk splitting in median mode. Default is `-30`.

`median_min_silence_ms` *int, optional*

Minimum silence duration in milliseconds for sub-chunk splitting. Default is `200`.

`debug` *bool, optional*

If `True`, prints the detected probability (or median) and threshold to stdout. Default is `False`.

**Methods:**

`process(segment)` *bool*

Returns `True` if the segment matches a filtered language.
