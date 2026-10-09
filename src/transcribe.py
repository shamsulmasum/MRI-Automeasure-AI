"""Transcribe an audio file with faster-whisper (CPU).

Usage: python transcribe.py <wav_path> <out_txt>
"""
import sys

from faster_whisper import WhisperModel

wav, out = sys.argv[1], sys.argv[2]
model = WhisperModel("small", device="cpu", compute_type="int8")
segments, info = model.transcribe(wav, vad_filter=True)
lines = [f"detected language: {info.language} (p={info.language_probability:.2f})"]
for s in segments:
    lines.append(f"[{int(s.start)//60:02d}:{int(s.start)%60:02d} -> "
                 f"{int(s.end)//60:02d}:{int(s.end)%60:02d}] {s.text.strip()}")
text = "\n".join(lines)
with open(out, "w", encoding="utf-8") as fh:
    fh.write(text)
print(text)
