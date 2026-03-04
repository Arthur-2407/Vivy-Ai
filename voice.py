import os
import re
import sys
import numpy as np
import sounddevice as sd
import soundfile as sf
from TTS.api import TTS
from contextlib import contextmanager

# ===============================
# Create recordings folder
# ===============================
RECORDINGS_DIR = "vivy_recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)

recording_count = 0

# ===============================
# Silent stdout/stderr context
# ===============================
@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# ===============================
# Load CPU-friendly Coqui TTS
# (Compatible with your installed version)
# ===============================
with suppress_output():
    tts = TTS(
        model_name="tts_models/en/ljspeech/tacotron2-DDC",
        progress_bar=False,
        gpu=False
    )

# ===============================
# Available voices
# ===============================
print("Available voices:")
print("0: ljspeech_female (default)")

voices = ["ljspeech_female"]
selected_voice = voices[0]
print(f"Selected voice: {selected_voice}")

# ===============================
# Voice properties
# ===============================
speech_rate = 1.0   # Recommended: 0.9 – 1.1
volume = 1.0        # 0.0 – 1.0

# ===============================
# Text sanitization & pacing
# ===============================
def clean_text(text: str) -> str:
    text = text.strip()

    text = re.sub(r"\.{2,}", "...", text)
    text = re.sub(r"\s+", " ", text)

    # Natural pacing
    text = text.replace(",", ", ")
    text = text.replace(";", "; ")
    text = text.replace(":", ": ")

    if not text.endswith((".", "!", "?")):
        text += "."

    return text

# ===============================
# Audio Processing Helpers
# ===============================
def normalize_audio(audio):
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95
    return audio

def soft_compress(audio, threshold=0.6, ratio=4.0):
    compressed = np.copy(audio)
    mask = np.abs(audio) > threshold
    compressed[mask] = np.sign(audio[mask]) * (
        threshold + (np.abs(audio[mask]) - threshold) / ratio
    )
    return compressed

def time_stretch(audio, rate):
    if rate == 1.0:
        return audio

    # High-quality resampling (no robotic artifacts)
    new_length = int(len(audio) / rate)
    x_old = np.linspace(0, 1, len(audio))
    x_new = np.linspace(0, 1, new_length)
    return np.interp(x_new, x_old, audio).astype(np.float32)

def trim_silence(audio, threshold=0.0005):
    non_silent = np.where(np.abs(audio) > threshold)[0]
    if len(non_silent) == 0:
        return audio
    return audio[non_silent[0]:non_silent[-1]]

# ===============================
# Core Speak Function
# ===============================
def speak(text):
    global recording_count
    recording_count += 1

    text = clean_text(text)

    output_file = os.path.join(
        RECORDINGS_DIR, f"vivy_{recording_count}.wav"
    )

    # Generate speech
    with suppress_output():
        tts.tts_to_file(
            text=text,
            file_path=output_file,
            speaker=None
        )

    # Load audio
    data, samplerate = sf.read(output_file, dtype="float32")

    # Post-processing for natural voice
    data = trim_silence(data)
    data = time_stretch(data, speech_rate)
    data = soft_compress(data)
    data = normalize_audio(data)
    data *= volume

    # Play audio
    sd.play(data, samplerate)
    sd.wait()

# ===============================
# Standalone test loop
# ===============================
if __name__ == "__main__":
    while True:
        text_to_speak = input(
            "\nEnter text to speak (or type 'exit'): "
        )
        if text_to_speak.lower() == "exit":
            break

        print("Speaking...")
        speak(text_to_speak)
