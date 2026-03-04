import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import scipy.interpolate
import webrtcvad
import subprocess
import queue
import sys
import time
import os
from colorama import Fore, Style

# ================= CONFIG =================
SAMPLE_RATE = 48000  # Enhanced: Increased from 16000 to 48000 for better quality
CHANNELS = 1
FRAME_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)

# Audio Quality Settings
AUDIO_GAIN = 1.2  # Slight gain boost for better clarity
HIGH_PASS_FILTER_HZ = 80  # Remove very low frequency rumble
LOW_PASS_FILTER_HZ = 12000  # Optimize for speech clarity

# ENC (Echo Noise Cancellation) Parameters
ENC_ENABLED = True
ENC_TAIL_LENGTH = 0.5  # Echo tail length in seconds

# ANC (Active Noise Cancellation) Parameters
ANC_ENABLED = True
ANC_UPDATE_RATE = 0.1  # Update rate for noise estimation

BASE_SILENCE_TIMEOUT = 2.5  # 2.5s delay before stopping to ensure completion
MAX_SILENCE_LIMIT = 10.0  # Allow longer recordings
MIN_RECORDING_DURATION = 0.5  # Minimum 0.5s to avoid false triggers

MAX_SILENCE_FRAMES = int(BASE_SILENCE_TIMEOUT / (FRAME_MS / 1000))

# VAD Configuration for better speech detection
VAD_AGGRESSIVENESS = 3  # Level 3 for better discrimination of speech vs noise
MIN_SPEECH_FRAMES = 10   # Require 10 consecutive frames (300ms) to trigger recording - strong protection

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WHISPER_PATH = os.path.join(BASE_DIR, "whisper.cpp", "whisper-cli.exe")  # Use whisper-cli.exe which is not deprecated
MODEL_PATH = os.path.join(BASE_DIR, "models", "ggml-medium.bin")
RECORD_DIR = os.path.join(BASE_DIR, "recordings")
OUT_PREFIX = os.path.join(RECORD_DIR, "last")

# Verify paths exist
if not os.path.exists(WHISPER_PATH):
    print(Fore.RED + f"Whisper executable not found at: {WHISPER_PATH}" + Style.RESET_ALL)
    # Try alternative paths
    whisper_alt_paths = [
        os.path.join(BASE_DIR, "whisper.cpp", "main.exe"),  # Fallback to main.exe if whisper-cli.exe not found
        os.path.join(BASE_DIR, "whisper.cpp", "command.exe")
    ]
    for alt_path in whisper_alt_paths:
        if os.path.exists(alt_path):
            WHISPER_PATH = alt_path
            print(Fore.YELLOW + f"Using alternative Whisper executable: {WHISPER_PATH}" + Style.RESET_ALL)
            break
    else:
        print(Fore.RED + "No suitable Whisper executable found!" + Style.RESET_ALL)

os.makedirs(RECORD_DIR, exist_ok=True)
# =========================================
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)  # Configurable VAD aggressiveness level
audio_queue = queue.Queue()

recording = False
audio_buffer = []
start_time = 0
silence_frames = 0
speech_frames = 0
recording_cooldown = 0  # Cooldown frames after processing before allowing new recording
COOLDOWN_FRAMES = 60  # 1800ms cooldown (60 * 30ms frames) - maximum protection against false triggers

# New variable to track if we're in a cooldown period where speech should still trigger recording
cooldown_allows_new_recording = False

noise_profile = np.zeros(FRAME_SIZE // 2 + 1, dtype=np.float32)
ANC_ALPHA = 0.02  # Slower update for better noise estimation

# ENC Echo Cancellation Parameters
enc_buffer = np.zeros(int(SAMPLE_RATE * ENC_TAIL_LENGTH), dtype=np.float32)
enc_coefficients = np.zeros(int(SAMPLE_RATE * ENC_TAIL_LENGTH / 10), dtype=np.float32)
enc_index = 0

# DC Offset and Pop Removal
dc_filter_state = 0.0
dc_filter_alpha = 0.95  # High-pass filter for DC removal

# ============= MIC SELECTION =============
def list_mics():
    print(Fore.YELLOW + "\nAvailable Microphones:")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"[{i}] {d['name']}")
    print(Style.RESET_ALL)

def select_mic():
    list_mics()
    attempt_count = 0
    max_attempts = 3
    
    while attempt_count < max_attempts:
        try:
            user_input = input("Select mic index: ").strip()
            # Handle case where terminal command gets passed as input
            if user_input.startswith('&') or 'Scripts\\Activate.ps1' in user_input:
                print(Fore.RED + "Invalid input detected. Please enter a valid microphone index number." + Style.RESET_ALL)
                attempt_count += 1
                if attempt_count < max_attempts:
                    list_mics()
                continue
            mic_index = int(user_input)
            # Validate that the mic index exists
            devices = sd.query_devices()
            if 0 <= mic_index < len(devices) and devices[mic_index]["max_input_channels"] > 0:
                return mic_index
            else:
                print(Fore.RED + f"Microphone index {mic_index} not found or not available. Please try again." + Style.RESET_ALL)
                attempt_count += 1
                if attempt_count < max_attempts:
                    list_mics()
        except ValueError:
            print(Fore.RED + "Please enter a valid number for the microphone index." + Style.RESET_ALL)
            attempt_count += 1
            if attempt_count < max_attempts:
                list_mics()
        except KeyboardInterrupt:
            print(Fore.YELLOW + "\nMicrophone selection cancelled." + Style.RESET_ALL)
            sys.exit(0)
    
    # If we've exhausted attempts, try to use the default system microphone
    print(Fore.YELLOW + "Too many invalid attempts. Trying to use default system microphone..." + Style.RESET_ALL)
    try:
        default_device = sd.default.device[0]  # Input device
        devices = sd.query_devices()
        if 0 <= default_device < len(devices) and devices[default_device]["max_input_channels"] > 0:
            print(Fore.GREEN + f"Using default microphone: {devices[default_device]['name']}" + Style.RESET_ALL)
            return default_device
        else:
            # Fallback to first available microphone
            for i, device in enumerate(devices):
                if device["max_input_channels"] > 0:
                    print(Fore.GREEN + f"Using first available microphone: {device['name']}" + Style.RESET_ALL)
                    return i
            # No microphones found
            raise sd.PortAudioError("No input devices found")
    except Exception as e:
        print(Fore.RED + f"Could not select a microphone automatically: {e}" + Style.RESET_ALL)
        print(Fore.RED + "Please check your microphone connections and permissions." + Style.RESET_ALL)
        sys.exit(1)

# MIC_INDEX will be set in the main execution block
MIC_INDEX = None

# =============== AUDIO IO =================
def callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())

def show_timer():
    elapsed = time.time() - start_time
    sys.stdout.write(
        Fore.CYAN + f"\rRecording: {elapsed:05.1f}s " + Style.RESET_ALL
    )
    sys.stdout.flush()

# ========== AUDIO ENHANCEMENT SUITE ==========
def apply_high_pass_filter(audio):
    """Remove low frequency rumble and DC offset for cleaner audio"""
    # Design high-pass filter
    nyquist = SAMPLE_RATE / 2
    normalized_freq = HIGH_PASS_FILTER_HZ / nyquist
    if normalized_freq >= 1.0:
        return audio
    b, a = signal.butter(4, normalized_freq, btype='high')
    return signal.filtfilt(b, a, audio)

def apply_low_pass_filter(audio):
    """Optimize audio for speech clarity"""
    # Design low-pass filter
    nyquist = SAMPLE_RATE / 2
    normalized_freq = LOW_PASS_FILTER_HZ / nyquist
    if normalized_freq >= 1.0:
        return audio
    b, a = signal.butter(4, normalized_freq, btype='low')
    return signal.filtfilt(b, a, audio)

def apply_anc(frame):
    """Active Noise Cancellation using spectral subtraction with better parameters"""
    global noise_profile
    spectrum = np.fft.rfft(frame)
    mag = np.abs(spectrum)
    
    # Update noise profile slowly for better estimation
    noise_profile[:len(mag)] = ((1 - ANC_ALPHA) * noise_profile[:len(mag)] + 
                                  ANC_ALPHA * mag)
    
    # Aggressive spectral subtraction for cleaner output
    clean_mag = np.maximum(mag - noise_profile[:len(mag)] * 1.5, 0.1 * mag)
    clean_spec = clean_mag * np.exp(1j * np.angle(spectrum))
    
    return np.fft.irfft(clean_spec, n=len(frame)).astype(np.float32)

def apply_enc(frame):
    """Echo Noise Cancellation to reduce echo and reverberation"""
    global enc_buffer, enc_index
    
    # Ensure buffer has enough space
    if len(enc_buffer) < len(frame):
        enc_buffer = np.zeros(int(SAMPLE_RATE * ENC_TAIL_LENGTH), dtype=np.float32)
    
    # Detect potential echo by comparing with delayed signal
    enc_delay = int(SAMPLE_RATE * 0.05)  # 50ms echo delay
    frame_energy = np.sum(frame ** 2)
    
    if frame_energy > 1e-6 and len(enc_buffer) >= enc_delay:
        delayed_signal = enc_buffer[-enc_delay:]
        if len(delayed_signal) == enc_delay:
            # Cross-correlation to detect echo
            correlation = np.correlate(frame[:min(len(frame), len(delayed_signal))], delayed_signal[:min(len(frame), len(delayed_signal))], mode='valid')
            if len(correlation) > 0 and np.max(correlation) > frame_energy * 0.3:
                # Echo detected, reduce it gently
                frame = frame - 0.2 * delayed_signal[:len(frame)]
    
    # Update buffer safely
    enc_buffer = np.concatenate([enc_buffer[len(frame):], frame])
    
    return frame.astype(np.float32)

def apply_dc_removal(frame):
    """Remove DC offset to prevent pops and clicks with improved IIR filter"""
    global dc_filter_state
    output = np.zeros_like(frame)
    # Use a faster alpha for better DC removal at frame boundaries
    fast_alpha = 0.98
    for i in range(len(frame)):
        dc_filter_state = fast_alpha * dc_filter_state + (1 - fast_alpha) * frame[i]
        output[i] = frame[i] - dc_filter_state
    return output.astype(np.float32)

def apply_de_clicker(frame):
    """Remove clicks and pops caused by transients and clipping with improved algorithm"""
    if len(frame) < 5:
        return frame
    
    output = frame.copy()
    
    # First pass: detect and smooth extreme transients
    window_size = 5
    for i in range(window_size, len(frame) - window_size):
        window = frame[i-window_size:i+window_size+1]
        median_val = np.median(window)
        
        # If sample deviates significantly from median, smooth it
        deviation = np.abs(frame[i] - median_val)
        window_std = np.std(window)
        
        # More aggressive click detection
        if window_std > 0.01 and deviation > 3 * window_std:
            output[i] = 0.7 * frame[i] + 0.3 * median_val
    
    # Second pass: reduce clipping artifacts
    for i in range(1, len(output) - 1):
        # Detect sudden sign changes (common in clipped audio)
        if i > 0 and i < len(output) - 1:
            prev_sign = np.sign(output[i-1])
            curr_sign = np.sign(output[i])
            next_sign = np.sign(output[i+1])
            
            # If sign changes abruptly, interpolate
            if prev_sign != curr_sign and curr_sign != next_sign:
                output[i] = (output[i-1] + output[i+1]) / 2.0
    
    return output.astype(np.float32)

def enhance_audio(frame):
    """Apply comprehensive audio enhancement pipeline optimized for pop reduction"""
    # Remove DC offset first to prevent pops at frame boundaries
    frame = apply_dc_removal(frame)
    
    # Apply de-clicking early to catch input artifacts
    frame = apply_de_clicker(frame)
    
    # Apply high-pass filter to remove rumble and DC
    frame = apply_high_pass_filter(frame)
    
    # Apply ANC for noise reduction
    if ANC_ENABLED:
        frame = apply_anc(frame)
    
    # Apply ENC for echo cancellation (with improved buffering)
    if ENC_ENABLED:
        frame = apply_enc(frame)
    
    # Apply low-pass filter for speech clarity
    frame = apply_low_pass_filter(frame)
    
    # Apply gain boost for better volume (reduced to prevent clipping)
    frame = frame * (AUDIO_GAIN * 0.9)  # Slightly reduce gain to avoid clipping pops
    
    # Prevent clipping with soft limiting instead of hard clipping
    frame = np.tanh(frame)  # Soft clipping using tanh
    
    return frame.astype(np.float32)

# ============= WHISPER RUN ================
def resample_audio(audio_data, original_sr, target_sr):
    """Resample audio from original sample rate to target sample rate"""
    if original_sr == target_sr:
        return audio_data
    
    # Calculate resampling ratio
    ratio = target_sr / original_sr
    num_samples = int(len(audio_data) * ratio)
    
    # Create interpolation function
    x_old = np.linspace(0, 1, len(audio_data))
    x_new = np.linspace(0, 1, num_samples)
    f = scipy.interpolate.interp1d(x_old, audio_data, kind='linear')
    
    return f(x_new).astype(np.float32)

def run_whisper(wav_file):
    print(Fore.GREEN + "\nTranscribing..." + Style.RESET_ALL)
    
    # Verify model file exists
    if not os.path.exists(MODEL_PATH):
        print(Fore.RED + f"Model file not found: {MODEL_PATH}" + Style.RESET_ALL)
        return "[Transcription failed: Model file not found]"
    
    # Read the 48kHz audio file
    try:
        sr, audio_int16 = wav.read(wav_file)
        audio = audio_int16.astype(np.float32) / 32768.0
    except Exception as e:
        print(Fore.RED + f"Error reading WAV file: {e}" + Style.RESET_ALL)
        return "[Transcription failed: Error reading WAV file]"
    
    # Resample from 48kHz to 16kHz for Whisper compatibility
    if sr != 16000:
        audio = resample_audio(audio, sr, 16000)
        sr = 16000
    
    # Save temporary 16kHz WAV file for Whisper
    temp_wav_path = wav_file.replace(".wav", "_16k.wav")
    audio_int16 = np.clip(audio * 32768, -32768, 32767).astype(np.int16)
    
    try:
        wav.write(temp_wav_path, sr, audio_int16)
        print(Fore.YELLOW + f"Temporary file created: {temp_wav_path}" + Style.RESET_ALL)
        print(Fore.YELLOW + f"Temporary file size: {os.path.getsize(temp_wav_path)} bytes" + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"Error saving temporary WAV: {e}" + Style.RESET_ALL)
        return "[Transcription failed: Error saving temporary file]"
    
    # Run Whisper on the 16kHz file
    try:
        print(Fore.CYAN + "Processing audio with Whisper..." + Style.RESET_ALL)
        # Use stable Whisper parameters with minimal flags to avoid crashes
        whisper_cmd = [
            WHISPER_PATH,
            "-m", MODEL_PATH,
            "-f", temp_wav_path,
            "-t", "1",  # Single thread
            "-l", "en",  # English
            "-pp",  # Print progress
            "-otxt",  # Output as text file
            "-of", os.path.join(RECORD_DIR, os.path.splitext(os.path.basename(temp_wav_path))[0].replace("_16k", "")),  # Output file
            "-ng",  # No GPU
            "-nfa"  # No flash attention
        ]
        
        print(Fore.YELLOW + f"Running command: {' '.join(whisper_cmd)}" + Style.RESET_ALL)
        
        result = subprocess.run(
            whisper_cmd,
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,  # Increased timeout to 60 seconds
        )
        
        # Check if temp file still exists after command
        if os.path.exists(temp_wav_path):
            print(Fore.YELLOW + f"Temporary file still exists after command: {temp_wav_path}" + Style.RESET_ALL)
        else:
            print(Fore.YELLOW + f"Temporary file was deleted during command execution" + Style.RESET_ALL)
        
        # Silently handle Whisper - no return code or stdout printing for clean output
        text = result.stdout.strip()
        
        # Filter out model loading messages and only show if there are real errors
        if result.returncode != 0 and not text:
            # Only show errors if there's no transcription at all
            stderr_lines = result.stderr.split('\n') if result.stderr else []
            error_lines = [l for l in stderr_lines if not any(skip in l.lower() for skip in 
                ['loading model', 'use gpu', 'flash attn', 'whisper_init', 'log_mel', 'system_info', 'processing', 'compute buffer', 'kv self', 'kv cross', 'kv pad', 'main:', 'n_vocab', 'n_audio'])]
            if error_lines and any(line.strip() for line in error_lines):
                pass  # Silently skip error output
        
        # Try to read output from generated text file
        output_txt_path = os.path.join(RECORD_DIR, os.path.basename(temp_wav_path).replace("_16k.wav", ".txt"))
        if os.path.exists(output_txt_path):
            try:
                with open(output_txt_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                os.remove(output_txt_path)  # Clean up after reading
            except:
                pass
        
        # Print debug information if transcription failed
        if not text or "[Transcription failed]" in text:
            print(Fore.RED + f"Whisper command failed with return code: {result.returncode}" + Style.RESET_ALL)
            if result.stderr:
                print(Fore.RED + f"Whisper stderr: {result.stderr[:500]}..." + Style.RESET_ALL)
                
        # Only clean up temp file if successful (for debugging)
        # if result.returncode == 0:
        #     try:
        #         os.remove(temp_wav_path)
        #     except:
        #         pass
        
        # Handle deprecation warning but still extract transcription
        if "WARNING: The binary" in text or "deprecated" in text.lower():
            # Extract actual transcription from stdout, ignoring warning
            lines = text.split('\n')
            text_lines = [line for line in lines if "WARNING" not in line.upper() and "deprecated" not in line.lower() and line.strip()]
            if text_lines:
                text = '\n'.join(text_lines).strip()
            else:
                text = ""
        
        # If no output from stdout, try to extract from stderr (sometimes Whisper outputs here)
        if not text and result.stderr:
            # Look for transcription patterns in stderr
            stderr_lines = result.stderr.split('\n')
            for line in stderr_lines:
                # Skip model loading info and other debug lines
                if any(skip in line.lower() for skip in ['loading model', 'use gpu', 'flash attn', 'whisper_init', 'log_mel_spectrogram', 'system_info', 'processing', 'deprecated', 'warning: the binary', 'unknown argument']):
                    continue
                # Look for actual transcription content (typically indented or starts with specific patterns)
                if line.strip() and not line.startswith('[') and not line.startswith('whisper'):
                    if len(line.strip()) > 2:  # Reasonable text length
                        text = line.strip()
                        break
        
        # If still no text, check if there's actual error
        if not text:
            if result.returncode != 0:
                # Get first actual error line
                if result.stderr:
                    error_lines = [l for l in result.stderr.split('\n') if l.strip() and not any(skip in l.lower() for skip in ['loading model', 'use gpu', 'flash attn', 'whisper_init', 'log_mel_spectrogram', 'system_info', 'processing'])]
                    if error_lines:
                        print(Fore.RED + f"Whisper error: {error_lines[0][:150]}" + Style.RESET_ALL)
                    else:
                        print(Fore.RED + "Whisper processing failed with no transcription." + Style.RESET_ALL)
                else:
                    print(Fore.RED + "No transcription produced." + Style.RESET_ALL)
            else:
                print(Fore.YELLOW + "No speech detected in audio." + Style.RESET_ALL)
            # Clean up temp file
            try:
                os.remove(temp_wav_path)
            except:
                pass
            return ""
        
        # Clean up temp file
        try:
            os.remove(temp_wav_path)
        except:
            pass
    
    except subprocess.TimeoutExpired:
        print(Fore.RED + "Whisper transcription timeout (60s exceeded)." + Style.RESET_ALL)
        try:
            os.remove(temp_wav_path)
        except:
            pass
        return ""
    except FileNotFoundError:
        print(Fore.RED + f"Whisper executable not found: {WHISPER_PATH}" + Style.RESET_ALL)
        try:
            os.remove(temp_wav_path)
        except:
            pass
        return ""
    except Exception as e:
        print(Fore.RED + f"Whisper error: {e}" + Style.RESET_ALL)
        try:
            os.remove(temp_wav_path)
        except:
            pass
        return ""

    # FORCE creation of transcript file
    transcript_dir = os.path.join(BASE_DIR, "transcripts")
    os.makedirs(transcript_dir, exist_ok=True)

    txt_path = os.path.join(
        transcript_dir,
        os.path.basename(wav_file).replace(".wav", ".txt")
    )

    # Write transcription result or error message
    with open(txt_path, "w", encoding="utf-8") as f:
        if text:
            f.write(text)
        else:
            f.write("[Transcription failed or no speech detected]")

    if text and not text.startswith("["):
        print(Fore.WHITE + text + Style.RESET_ALL)
    elif text:
        print(Fore.YELLOW + text + Style.RESET_ALL)
    else:
        print(Fore.YELLOW + "No transcription produced." + Style.RESET_ALL)
        
    print(Fore.CYAN + f"Transcript saved: {txt_path}" + Style.RESET_ALL)

    return text if text else ""

# ================= MAIN ==================
print(Fore.GREEN + "\nAuto-listening started. Speak anytime.\n" + Style.RESET_ALL)

# Select microphone when starting
MIC_INDEX = select_mic()

print(Fore.CYAN + f"Using microphone index: {MIC_INDEX}" + Style.RESET_ALL)

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    blocksize=FRAME_SIZE,
    dtype="int16",
    callback=callback,
    device=MIC_INDEX,
) as stream:
    print(Fore.CYAN + "Audio stream started. Waiting for speech..." + Style.RESET_ALL)
    while True:
        data = audio_queue.get()
        pcm_bytes = data.reshape(-1).tobytes()
        speech = vad.is_speech(pcm_bytes, SAMPLE_RATE)

        # Handle recording cooldown - prevent false triggers completely
        if recording_cooldown > 0:
            recording_cooldown -= 1
            # During cooldown, ignore all speech detection
            speech = False
            speech_frames = 0
            # Don't process anything during cooldown
            continue
        
        # Speech detected and we're not recording
        if speech and not recording:
            # Speech detected and we're not in cooldown
            speech_frames += 1
            # Require minimum consecutive speech frames to trigger recording
            if speech_frames >= MIN_SPEECH_FRAMES:
                silence_frames = 0
                recording = True
                audio_buffer = []
                start_time = time.time()
                print(Fore.GREEN + "\nVoice detected. Recording..." + Style.RESET_ALL)
        else:
            # No speech detected (or already recording), reset speech counter
            speech_frames = 0
            # Continue counting silence only when recording
            if recording:
                silence_frames += 1

        # Continue recording and capture audio
        if recording:
            audio_buffer.append(data.copy())
            show_timer()

        # Stop recording when silence threshold is reached
        if recording and silence_frames > MAX_SILENCE_FRAMES:
            # Check if recording is long enough
            recording_duration = (len(audio_buffer) * FRAME_SIZE) / SAMPLE_RATE
            if recording_duration >= MIN_RECORDING_DURATION:
                print(Fore.YELLOW + "\nSilence detected. Processing..." + Style.RESET_ALL)
                recording = False
                silence_frames = 0
                recording_cooldown = COOLDOWN_FRAMES  # Start cooldown

                raw = (
                    np.concatenate(audio_buffer, axis=0)
                    .reshape(-1)
                    .astype(np.float32)
                    / 32768.0
                )

                cleaned = np.zeros_like(raw)
                total_frames = len(raw) // FRAME_SIZE

                # Apply comprehensive audio enhancement
                for i in range(total_frames):
                    s = i * FRAME_SIZE
                    e = s + FRAME_SIZE
                    cleaned[s:e] = enhance_audio(raw[s:e])

                if len(raw) % FRAME_SIZE:
                    cleaned[total_frames * FRAME_SIZE:] = enhance_audio(raw[total_frames * FRAME_SIZE:])

                pcm16 = np.clip(cleaned * 32768, -32768, 32767).astype(np.int16)

                filename = os.path.join(RECORD_DIR, f"rec_{int(time.time())}.wav")
                wav.write(filename, SAMPLE_RATE, pcm16)

                print(Fore.CYAN + f"Saved: {filename}" + Style.RESET_ALL)

                if os.path.getsize(filename) > 1000:
                    transcription = run_whisper(filename)
                    if not transcription or "Transcription failed" in transcription:
                        print(Fore.YELLOW + "Continuing despite transcription failure..." + Style.RESET_ALL)
                else:
                    print(Fore.RED + "Invalid WAV, skipped." + Style.RESET_ALL)
            
            # Reset counters after processing to prevent false triggers
            recording = False
            silence_frames = 0
            speech_frames = 0  # Reset speech frames to prevent immediate re-triggering
            recording_cooldown = COOLDOWN_FRAMES  # Start cooldown
            
            # Clear audio buffer to prevent residual data
            audio_buffer = []
