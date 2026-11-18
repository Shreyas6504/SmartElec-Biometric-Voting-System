"""
SmartElec Audio Feedback Module
Provides auditory prompts (Text-to-Speech & Acoustic chimes) for voter interaction.
Supports pyttsx3, Windows System.Speech synthesis, and winsound fallbacks.
"""

import sys
import threading
import subprocess

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False


class AudioFeedback:
    """
    Manages non-blocking audio cues and speech announcements for voting events.
    """

    def __init__(self, enable_speech: bool = True, enable_beeps: bool = True):
        self.enable_speech = enable_speech
        self.enable_beeps = enable_beeps
        self._engine = None

        if self.enable_speech and PYTTSX3_AVAILABLE:
            try:
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", 160)
            except Exception:
                self._engine = None

    def _speak_worker(self, text: str) -> None:
        """Background worker to speak without freezing GUI / video feed."""
        if not self.enable_speech:
            return

        # 1. Try pyttsx3
        if self._engine is not None:
            try:
                self._engine.say(text)
                self._engine.runAndWait()
                return
            except Exception:
                pass

        # 2. Windows PowerShell SpeechSynthesizer Fallback (Zero external dependency)
        if sys.platform == "win32":
            try:
                clean_text = text.replace("'", "").replace('"', "")
                ps_command = f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{clean_text}')"
                subprocess.run(
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_command],
                    capture_output=True,
                    timeout=4
                )
                return
            except Exception:
                pass

        # 3. Simple console echo fallback
        print(f"[AUDIO PROMPT]: {text}")

    def speak(self, text: str, async_mode: bool = True) -> None:
        """Announces text using text-to-speech."""
        if async_mode:
            thread = threading.Thread(target=self._speak_worker, args=(text,), daemon=True)
            thread.start()
        else:
            self._speak_worker(text)

    def play_success_beep(self) -> None:
        """Acoustic tone for successful vote / registration."""
        if not self.enable_beeps:
            return
        if WINSOUND_AVAILABLE and sys.platform == "win32":
            def beep():
                try:
                    winsound.Beep(1200, 150)
                    winsound.Beep(1800, 200)
                except Exception:
                    pass
            threading.Thread(target=beep, daemon=True).start()

    def play_warning_beep(self) -> None:
        """Warning buzz for double voting or unregistered attempt."""
        if not self.enable_beeps:
            return
        if WINSOUND_AVAILABLE and sys.platform == "win32":
            def beep():
                try:
                    winsound.Beep(400, 300)
                except Exception:
                    pass
            threading.Thread(target=beep, daemon=True).start()
