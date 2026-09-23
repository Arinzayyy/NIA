"""tts.py -- NIA's voice. Swappable TTS behind one interface.

One class shape, speak(text) / stop(), so the orchestrator never cares which
engine is running. Engine is config-selected so the expressive upgrade drops in
without touching the loop (PRD 5 / Phase 1 handoff):

  - sapi    : Windows built-in voice via pyttsx3. ZERO setup, talks immediately.
              Use this for the first voice test so you are not blocked on a model.
  - piper   : local Piper neural voice. Better quality. Needs a voice model file.
  - styletts2 / xtts : reserved for the expressive build (Phase 5 / PRD 9).

Playback runs on a background thread and is interruptible via stop(), which is
what makes barge-in possible: the orchestrator calls stop() the instant Lazy
starts talking and her current line dies mid-word.
"""
import threading
import queue
import subprocess
import wave
import io


class BaseTTS:
    def __init__(self, config):
        self.config = config.get("tts", {})
        self._stop = threading.Event()
        self._thread = None

    @property
    def is_speaking(self):
        return self._thread is not None and self._thread.is_alive()

    def speak(self, text, blocking=False):
        """Speak text. Non-blocking by default so the loop can keep listening
        for barge-in. Cancels any line already in progress."""
        if not text:
            return
        self.stop()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(text,), daemon=True)
        self._thread.start()
        if blocking:
            self._thread.join()

    def stop(self):
        """Interrupt the current line immediately (barge-in)."""
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def _run(self, text):
        raise NotImplementedError


class SapiTTS(BaseTTS):
    """Windows speech via the native SAPI voice (comtypes -> SAPI.SpVoice).

    We drive SAPI directly instead of through pyttsx3, whose runAndWait() is
    flaky on repeated calls (audio dies after the first line). Native SAPI does
    back-to-back speech with no loop state to corrupt.

    COM rules: the SpVoice object is created AND used on one dedicated worker
    thread (COM is apartment-bound, so cross-thread use kills the audio). Lines
    arrive on a queue and are spoken synchronously, one at a time. stop() just
    drops queued lines; the short current line finishes on its own. (Piper, via
    sounddevice, is the engine with true mid-line barge-in.)"""
    def __init__(self, config):
        super().__init__(config)
        self._q = queue.Queue()
        self._speaking = False
        self._ready = threading.Event()
        self._init_error = None
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()
        self._ready.wait(timeout=10)
        if self._init_error:
            raise self._init_error

    def _make_voice(self):
        import comtypes.client
        voice = comtypes.client.CreateObject("SAPI.SpVoice")
        # SAPI Rate is an integer -10..10 (NOT words-per-minute). If the config
        # holds a pyttsx3-style WPM number, ignore it and use a mild speed-up.
        rate = self.config.get("sapi_rate")
        if isinstance(rate, (int, float)):
            r = int(rate)
            voice.Rate = r if -10 <= r <= 10 else 1
        name = self.config.get("sapi_voice")
        if name:
            tokens = voice.GetVoices()
            for i in range(tokens.Count):
                tok = tokens.Item(i)
                if name.lower() in tok.GetDescription().lower():
                    voice.Voice = tok
                    break
        return voice

    def _loop(self):
        # create AND use the SAPI object on this one thread (COM apartment rule)
        try:
            import comtypes
            comtypes.CoInitialize()
        except Exception:
            pass
        try:
            voice = self._make_voice()
        except Exception as e:
            self._init_error = e
            self._ready.set()
            return
        self._ready.set()
        while True:
            text = self._q.get()
            if text is None:        # shutdown sentinel
                break
            self._speaking = True
            try:
                voice.Speak(text)   # synchronous; SAPI handles repeats natively
            except Exception:
                pass
            self._speaking = False

    @property
    def is_speaking(self):
        return self._speaking

    def speak(self, text, blocking=False):
        if not text:
            return
        self.stop()                 # drop any stale queued lines first
        self._q.put(text)
        if blocking:
            while self._speaking or not self._q.empty():
                threading.Event().wait(0.02)

    def stop(self):
        # drain anything queued so stale lines never play. We do NOT interrupt
        # the line currently playing -- cross-thread COM would break SAPI audio,
        # and her lines are short. (Piper supports true mid-line barge-in.)
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass


class PiperTTS(BaseTTS):
    """Local Piper neural voice via the PERSISTENT Python API. The voice model
    is loaded ONCE at startup (PiperVoice.load) and reused for every line, so we
    never pay the multi-second model reload that a fresh subprocess costs on each
    line. Audio plays in interruptible chunks via sounddevice for clean barge-in."""
    def __init__(self, config):
        super().__init__(config)
        self.model = self.config.get("piper_model")  # path to a .onnx voice
        self.output_device = self.config.get("output_device")  # None = default
        if not self.model:
            raise ValueError(
                "tts.piper_model not set. Download a Piper voice (.onnx) and "
                "point config.tts.piper_model at it, or use engine: sapi for now."
            )
        from piper import PiperVoice
        # loads the .onnx and its .onnx.json sidecar once; reused every call
        self._voice = PiperVoice.load(self.model)
        # one persistent playback worker, fed by a queue. Reusing a single global
        # sounddevice stream (sd.play/sd.wait/sd.stop) avoids the "plays once then
        # silent" failure of opening a fresh stream per line.
        self._q = queue.Queue()
        self._speaking = False
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def _synth(self, text):
        """Synthesize via the loaded voice. Returns (samples float32, sample_rate).
        Handles the AudioChunk shapes the piper API may return, defensively."""
        import numpy as np
        chunks = []
        sr = 22050
        for chunk in self._voice.synthesize(text):
            sr = getattr(chunk, "sample_rate", sr)
            b = getattr(chunk, "audio_int16_bytes", None)
            if b is not None:
                chunks.append(np.frombuffer(b, dtype=np.int16).astype("float32") / 32768.0)
                continue
            arr = getattr(chunk, "audio_float_array", None)
            if arr is not None:
                chunks.append(np.asarray(arr, dtype="float32"))
                continue
            if isinstance(chunk, (bytes, bytearray)):
                chunks.append(np.frombuffer(bytes(chunk), dtype=np.int16).astype("float32") / 32768.0)
        if not chunks:
            return np.zeros(0, dtype="float32"), sr
        return np.concatenate(chunks), sr

    def _loop(self):
        import sounddevice as sd
        while True:
            text = self._q.get()
            if text is None:        # shutdown sentinel
                break
            self._speaking = True
            try:
                data, sr = self._synth(text)
                if len(data) > 0:
                    sd.play(data, sr, device=self.output_device)
                    sd.wait()       # returns when done OR when stop() calls sd.stop()
            except Exception:
                pass
            self._speaking = False

    @property
    def is_speaking(self):
        return self._speaking

    def speak(self, text, blocking=False):
        if not text:
            return
        # just enqueue; the worker plays lines in order. We do NOT drain here, so
        # back-to-back lines are never dropped. Barge-in goes through stop().
        self._q.put(text)
        if blocking:
            while self._speaking or not self._q.empty():
                threading.Event().wait(0.02)

    def stop(self):
        import sounddevice as sd
        # barge-in: drop queued lines AND cut the one currently playing (sd.stop
        # makes the worker's sd.wait() return). Called by the orchestrator the
        # instant Lazy starts talking.
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        try:
            sd.stop()
        except Exception:
            pass


def get_tts(config):
    """Factory. Reads tts.engine from config and returns the right backend."""
    engine = config.get("tts", {}).get("engine", "sapi").lower()
    if engine == "sapi":
        return SapiTTS(config)
    if engine == "piper":
        return PiperTTS(config)
    raise ValueError(
        f"tts.engine '{engine}' not built yet. Use 'sapi' (no setup) or 'piper'. "
        "styletts2 / xtts are reserved for the expressive build."
    )
