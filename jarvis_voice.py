# jarvis_voice.py — Jarvis Voice Interface v4.0
# Paul speaks. Jarvis listens, thinks, speaks back.
# STT: OpenAI Whisper (local, offline) | TTS: pyttsx3 (local, no API cost)
# Wake word: "Hey Jarvis" | Continuous listening mode
# Personality: calm, direct, British-adjacent. Like the real Jarvis.

import asyncio
import io
import json
import os
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# Voice config
WAKE_WORDS = ["hey jarvis", "jarvis", "yo jarvis", "ok jarvis"]
SILENCE_THRESHOLD = 500      # audio energy threshold
SILENCE_DURATION = 1.8       # seconds of silence = end of speech
MAX_RECORDING_SECS = 30      # max single utterance
VOICE_RATE = 175              # TTS words per minute (175 = calm, natural)
VOICE_VOLUME = 0.95
AUDIO_SAMPLE_RATE = 16000
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")  # tiny/base/small/medium

# Import guard with helpful errors
def check_deps() -> dict:
    """Check which voice deps are available."""
    status = {}
    try:
        import pyttsx3
        status["tts"] = True
    except ImportError:
        status["tts"] = False

    try:
        import whisper
        status["stt"] = True
    except ImportError:
        status["stt"] = False

    try:
        import pyaudio
        status["audio"] = True
    except ImportError:
        status["audio"] = False

    try:
        import numpy as np
        status["numpy"] = True
    except ImportError:
        status["numpy"] = False

    return status


class JarvisVoice:
    """Text-to-Speech engine. Paul hears Jarvis."""

    def __init__(self):
        self.engine = None
        self.available = False
        self._init_engine()

    def _init_engine(self):
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", VOICE_RATE)
            self.engine.setProperty("volume", VOICE_VOLUME)

            # Pick the best available voice — prefer deep male voice
            voices = self.engine.getProperty("voices")
            chosen = None
            # Priority: David (Windows) > any male > first available
            for v in voices:
                name = v.name.lower()
                if "david" in name or "mark" in name or "george" in name:
                    chosen = v
                    break
            if not chosen:
                for v in voices:
                    if "male" in v.name.lower() or "zira" not in v.name.lower():
                        chosen = v
                        break
            if chosen:
                self.engine.setProperty("voice", chosen.id)

            self.available = True
        except Exception as e:
            print(f"  TTS unavailable: {e}")
            self.available = False

    def speak(self, text: str, blocking: bool = True):
        """Speak text aloud."""
        if not self.available:
            print(f"[Jarvis would say]: {text}")
            return

        # Clean text for speech — remove markdown, symbols
        clean = text
        for symbol in ["🜏", "**", "__", "##", "```", "*", "_", "~"]:
            clean = clean.replace(symbol, "")
        # Truncate very long responses for speech
        if len(clean) > 800:
            clean = clean[:800] + "... I'll show you the rest on screen."

        try:
            self.engine.say(clean)
            if blocking:
                self.engine.runAndWait()
            else:
                threading.Thread(target=self.engine.runAndWait, daemon=True).start()
        except Exception as e:
            print(f"  TTS error: {e}")

    def speak_async(self, text: str):
        """Speak without blocking the main thread."""
        self.speak(text, blocking=False)

    def set_rate(self, wpm: int):
        if self.engine:
            self.engine.setProperty("rate", wpm)

    def stop(self):
        if self.engine:
            try:
                self.engine.stop()
            except Exception:
                pass


class JarvisEar:
    """Speech-to-Text engine. Jarvis hears Paul."""

    def __init__(self, model_name: str = WHISPER_MODEL):
        self.model = None
        self.model_name = model_name
        self.available = False
        self._load_model()

    def _load_model(self):
        try:
            import whisper
            print(f"  Loading Whisper '{self.model_name}'...", end="", flush=True)
            self.model = whisper.load_model(self.model_name)
            self.available = True
            print(" ✅")
        except ImportError:
            print("  Whisper not installed. Run: pip install openai-whisper")
        except Exception as e:
            print(f"  Whisper load failed: {e}")

    def transcribe_file(self, audio_path: str) -> str:
        """Transcribe an audio file to text."""
        if not self.available:
            return ""
        try:
            result = self.model.transcribe(audio_path, language="en", fp16=False)
            return result["text"].strip()
        except Exception as e:
            return f"[transcription error: {e}]"

    def transcribe_bytes(self, audio_bytes: bytes, sample_rate: int = AUDIO_SAMPLE_RATE) -> str:
        """Transcribe raw audio bytes."""
        if not self.available:
            return ""
        try:
            import numpy as np
            import whisper
            # Convert bytes to numpy float32
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            result = self.model.transcribe(audio_np, language="en", fp16=False)
            return result["text"].strip()
        except Exception as e:
            return f"[transcription error: {e}]"


class AudioListener:
    """Microphone listener with VAD (Voice Activity Detection).
    Records until silence. Returns audio bytes.
    """

    def __init__(self):
        self.available = False
        self.pyaudio = None
        self._init_audio()

    def _init_audio(self):
        try:
            import pyaudio
            self.pyaudio = pyaudio.PyAudio()
            self.available = True
        except ImportError:
            print("  pyaudio not installed. Run: pip install pyaudio")
        except Exception as e:
            print(f"  Audio init failed: {e}")

    def record_until_silence(self, timeout: float = MAX_RECORDING_SECS) -> bytes | None:
        """Record from mic until silence detected. Returns raw PCM bytes."""
        if not self.available:
            return None

        import pyaudio
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1

        stream = self.pyaudio.open(
            format=FORMAT, channels=CHANNELS,
            rate=AUDIO_SAMPLE_RATE, input=True,
            frames_per_buffer=CHUNK
        )

        frames = []
        silent_chunks = 0
        silent_limit = int(SILENCE_DURATION * AUDIO_SAMPLE_RATE / CHUNK)
        max_chunks = int(timeout * AUDIO_SAMPLE_RATE / CHUNK)
        recording_started = False

        for _ in range(max_chunks):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

            # Simple energy-based VAD
            energy = sum(abs(b - 128) for b in data) / len(data)

            if energy > SILENCE_THRESHOLD:
                recording_started = True
                silent_chunks = 0
            elif recording_started:
                silent_chunks += 1
                if silent_chunks >= silent_limit:
                    break

        stream.stop_stream()
        stream.close()

        if not recording_started:
            return None

        return b"".join(frames)

    def listen_for_wake_word(self, ear: JarvisEar, timeout: float = 60.0) -> bool:
        """Listen continuously until wake word detected. Returns True if heard."""
        if not self.available or not ear.available:
            return False

        import pyaudio
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        WAKE_WINDOW_SECS = 3.0  # Check every 3 seconds of audio

        stream = self.pyaudio.open(
            format=FORMAT, channels=CHANNELS,
            rate=AUDIO_SAMPLE_RATE, input=True,
            frames_per_buffer=CHUNK
        )

        chunks_per_window = int(WAKE_WINDOW_SECS * AUDIO_SAMPLE_RATE / CHUNK)
        start = time.time()

        while time.time() - start < timeout:
            window = []
            for _ in range(chunks_per_window):
                data = stream.read(CHUNK, exception_on_overflow=False)
                window.append(data)

            audio_bytes = b"".join(window)
            text = ear.transcribe_bytes(audio_bytes).lower()

            if any(wake in text for wake in WAKE_WORDS):
                stream.stop_stream()
                stream.close()
                return True

        stream.stop_stream()
        stream.close()
        return False

    def cleanup(self):
        if self.pyaudio:
            try:
                self.pyaudio.terminate()
            except Exception:
                pass


class VoicePersonalityEngine:
    """Maps response type to voice delivery style.
    Jarvis doesn't just speak — he speaks appropriately.
    """

    # Prefixes Jarvis uses based on context
    ACKNOWLEDGMENTS = [
        "Got it.", "On it.", "Sure thing.", "Right.",
        "Understood.", "Of course.", "Absolutely."
    ]

    THINKING_PHRASES = [
        "Let me check that.", "One moment.",
        "Looking into it.", "Give me a second."
    ]

    COMPLETION_PHRASES = [
        "Done.", "There you go.", "All set.",
        "Complete.", "Finished."
    ]

    ERROR_PHRASES = [
        "I hit a snag.", "Something went wrong.",
        "I ran into an issue."
    ]

    def __init__(self, voice: JarvisVoice):
        self.voice = voice
        self._idx = {"ack": 0, "think": 0, "done": 0, "err": 0}

    def _next(self, key: str, phrases: list) -> str:
        phrase = phrases[self._idx[key] % len(phrases)]
        self._idx[key] += 1
        return phrase

    def acknowledge(self):
        """Speak an acknowledgment before processing."""
        self.voice.speak_async(self._next("ack", self.ACKNOWLEDGMENTS))

    def thinking(self):
        """Let Paul know Jarvis is working."""
        self.voice.speak(self._next("think", self.THINKING_PHRASES))

    def done(self):
        self.voice.speak_async(self._next("done", self.COMPLETION_PHRASES))

    def error(self, detail: str = ""):
        phrase = self._next("err", self.ERROR_PHRASES)
        self.voice.speak(f"{phrase} {detail}" if detail else phrase)

    def respond(self, text: str, response_type: str = "normal"):
        """Full contextual response."""
        if response_type == "tool_result":
            self.voice.speak(text)
        elif response_type == "error":
            self.error(text[:100])
        elif response_type == "completion":
            self.done()
            if text:
                self.voice.speak(text)
        else:
            self.voice.speak(text)


class JarvisVoiceInterface:
    """The full voice loop. Paul speaks → Jarvis hears → Jarvis thinks → Jarvis speaks.
    Two modes:
      - Wake word mode: always listening, activates on 'Hey Jarvis'
      - Push-to-talk mode: Press Enter to speak, Enter again to stop
    """

    def __init__(self):
        self.voice = JarvisVoice()
        self.ear = JarvisEar()
        self.listener = AudioListener()
        self.personality = VoicePersonalityEngine(self.voice)
        self.session_count = 0

    def _fallback_input(self) -> str:
        """Text fallback when audio hardware unavailable."""
        return input("Paul (text): ").strip()

    async def voice_turn(self, get_response_fn) -> bool:
        """One full voice turn: listen → transcribe → respond. Returns False to quit."""
        # Listen
        if self.listener.available and self.ear.available:
            print("  🎙️  Listening...", end="", flush=True)
            audio = self.listener.record_until_silence()
            if audio is None:
                print(" (silence)")
                return True
            print(" transcribing...")
            text = self.ear.transcribe_bytes(audio)
        else:
            text = self._fallback_input()

        if not text:
            return True

        text_clean = text.strip()
        # Remove wake word from command if present
        for wake in WAKE_WORDS:
            if text_clean.lower().startswith(wake):
                text_clean = text_clean[len(wake):].strip()
                break

        if not text_clean:
            return True

        if text_clean.lower() in ["exit", "quit", "goodbye", "bye jarvis", "shut down"]:
            self.voice.speak("Goodbye Paul. Jarvis offline.")
            return False

        print(f"Paul: {text_clean}")
        self.personality.acknowledge()

        # Get AI response
        try:
            response = await get_response_fn(text_clean)
            print(f"Jarvis: {response}")
            self.personality.respond(response)
            self.session_count += 1
        except Exception as e:
            self.personality.error(str(e)[:80])

        return True

    async def run_wake_word_mode(self, get_response_fn):
        """Always-on mode: listens for wake word, then activates."""
        if not self.listener.available or not self.ear.available:
            print("  Audio not available. Falling back to text mode.")
            await self.run_ptt_mode(get_response_fn)
            return

        self.voice.speak("Jarvis online. Say 'Hey Jarvis' to activate.")
        print("\n  🜏 Wake word mode active. Say 'Hey Jarvis'")
        print("  Ctrl+C to exit\n")

        while True:
            try:
                heard = self.listener.listen_for_wake_word(self.ear, timeout=300)
                if heard:
                    self.voice.speak("Yes?")
                    continue_loop = await self.voice_turn(get_response_fn)
                    if not continue_loop:
                        break
            except KeyboardInterrupt:
                self.voice.speak("Goodbye Paul.")
                break

        self.listener.cleanup()

    async def run_ptt_mode(self, get_response_fn):
        """Push-to-talk: press Enter to speak."""
        self.voice.speak("Jarvis online. Press Enter to speak.")
        print("\n  🜏 Push-to-talk mode. Press Enter to speak, Ctrl+C to exit\n")

        while True:
            try:
                input("  [Press Enter to speak]")

                if self.listener.available and self.ear.available:
                    print("  🎙️  Recording... (speak now)")
                    audio = self.listener.record_until_silence()
                    if audio:
                        text = self.ear.transcribe_bytes(audio)
                    else:
                        text = self._fallback_input()
                else:
                    text = self._fallback_input()

                if not text.strip():
                    continue

                if text.lower().strip() in ["exit", "quit", "bye"]:
                    self.voice.speak("Goodbye Paul.")
                    break

                print(f"Paul: {text}")
                self.personality.acknowledge()
                response = await get_response_fn(text)
                print(f"Jarvis: {response}")
                self.personality.respond(response)

            except KeyboardInterrupt:
                self.voice.speak("Goodbye Paul.")
                break

        self.listener.cleanup()


async def run_voice_mode(mode: str = "ptt"):
    """Entry point: launch Jarvis in full voice mode."""
    # Check deps
    deps = check_deps()
    print("\n🜏 JARVIS VOICE v4.0")
    print(f"  TTS (pyttsx3):  {'✅' if deps['tts'] else '❌ pip install pyttsx3'}")
    print(f"  STT (Whisper):  {'✅' if deps['stt'] else '❌ pip install openai-whisper'}")
    print(f"  Audio (PyAudio):{'✅' if deps['audio'] else '❌ pip install pyaudio'}")
    print(f"  NumPy:          {'✅' if deps['numpy'] else '❌ pip install numpy'}")
    print()

    if not deps["tts"] and not deps["stt"]:
        print("  Installing voice dependencies...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install",
                       "pyttsx3", "openai-whisper", "pyaudio", "numpy",
                       "--quiet"], check=False)
        print("  Restart to activate voice mode.")
        return

    # Load Jarvis brain
    from paul_core import load_memory, jarvis_respond
    mem = load_memory()

    async def get_response(text: str) -> str:
        return await jarvis_respond(text, mem)

    interface = JarvisVoiceInterface()

    if mode == "wake":
        await interface.run_wake_word_mode(get_response)
    else:
        await interface.run_ptt_mode(get_response)


if __name__ == "__main__":
    mode = "wake" if "--wake" in sys.argv else "ptt"
    asyncio.run(run_voice_mode(mode))
