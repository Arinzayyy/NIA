"""stt.py -- NIA's ears.

Captures Lazy's mic, segments speech on his pauses with Voice Activity Detection
(VAD), and transcribes each utterance with faster-whisper. Two things come out:

  - on_speech_start callback: fires the instant Lazy starts talking, BEFORE
    transcription. The orchestrator uses this for barge-in (kill NIA's TTS now).
  - listen() generator: yields finalized transcript strings once he pauses.

faster-whisper runs on CPU by default here (PRD 9: 32 GB RAM lets Whisper sit on
CPU and keep VRAM free for the brain). Model size and device are config-driven.
"""
import collections
import threading
import queue


class STT:
    def __init__(self, config, on_speech_start=None):
        s = config.get("stt", {})
        self.model_size = s.get("model", "base")          # tiny|base|small|...
        self.device = s.get("device", "cpu")              # cpu|cuda
        self.compute_type = s.get("compute_type", "int8")  # int8 is fast on CPU
        self.language = s.get("language", "en")
        self.beam_size = s.get("beam_size", 5)
        self.initial_prompt = s.get("initial_prompt") or None
        self.input_device = s.get("input_device")          # None = default mic
        self.sample_rate = 16000                            # whisper + webrtcvad want 16k
        self.frame_ms = s.get("frame_ms", 30)              # webrtcvad: 10/20/30
        self.vad_aggressiveness = s.get("vad_aggressiveness", 2)  # 0..3
        # how much trailing silence (ms) ends an utterance
        self.silence_tail_ms = s.get("silence_tail_ms", 600)
        # speech must persist this long before we believe it (debounces noise)
        self.speech_onset_ms = s.get("speech_onset_ms", 180)
        # utterances with less real speech than this are ignored as blips
        self.min_speech_ms = s.get("min_speech_ms", 350)
        # known Whisper hallucinations on silence, dropped outright
        self.drop_phrases = {p.strip().lower() for p in s.get("drop_phrases", [])}
        self.on_speech_start = on_speech_start

        self._audio_q = queue.Queue()
        self._running = threading.Event()
        self._model = None
        self._vad = None

    def _lazy_init(self):
        """Import heavy deps only when we actually start listening."""
        from faster_whisper import WhisperModel
        import webrtcvad
        self._model = WhisperModel(
            self.model_size, device=self.device, compute_type=self.compute_type
        )
        self._vad = webrtcvad.Vad(self.vad_aggressiveness)

    def _mic_callback(self, indata, frames, time_info, status):
        # sounddevice pushes raw int16 bytes; queue them for the VAD loop
        self._audio_q.put(bytes(indata))

    def listen(self):
        """Generator yielding finalized transcripts. Blocks between utterances.

        Run this on its own thread in the orchestrator. It opens the mic, runs
        VAD to find speech segments, and transcribes each completed segment.
        """
        import sounddevice as sd

        self._lazy_init()
        self._running.set()
        frame_bytes = int(self.sample_rate * (self.frame_ms / 1000.0)) * 2  # int16
        ring = collections.deque()
        triggered = False
        silence_frames = 0
        speech_frames = 0          # real speech frames in the current utterance
        onset_speech = 0           # consecutive speech frames while not yet triggered
        max_silence = self.silence_tail_ms // self.frame_ms
        onset_needed = max(1, self.speech_onset_ms // self.frame_ms)
        min_speech = max(1, self.min_speech_ms // self.frame_ms)
        # keep a short pre-roll so the onset frames are not lost once we trigger
        pre = collections.deque(maxlen=onset_needed)

        with sd.RawInputStream(samplerate=self.sample_rate, blocksize=frame_bytes // 2,
                               dtype="int16", channels=1, device=self.input_device,
                               callback=self._mic_callback):
            buf = b""
            while self._running.is_set():
                buf += self._audio_q.get()
                while len(buf) >= frame_bytes:
                    frame, buf = buf[:frame_bytes], buf[frame_bytes:]
                    is_speech = self._vad.is_speech(frame, self.sample_rate)
                    if not triggered:
                        pre.append(frame)
                        if is_speech:
                            onset_speech += 1
                            # only believe it is real speech once it persists,
                            # so a single noisy blip never triggers her
                            if onset_speech >= onset_needed:
                                triggered = True
                                ring.clear()
                                ring.extend(pre)         # include the onset frames
                                speech_frames = onset_speech
                                silence_frames = 0
                                if self.on_speech_start:
                                    self.on_speech_start()
                        else:
                            onset_speech = 0
                    else:
                        ring.append(frame)
                        if is_speech:
                            silence_frames = 0
                            speech_frames += 1
                        else:
                            silence_frames += 1
                            if silence_frames >= max_silence:
                                # utterance complete
                                audio = b"".join(ring)
                                triggered = False
                                onset_speech = 0
                                ring.clear()
                                # ignore blips with too little actual speech
                                if speech_frames >= min_speech:
                                    text = self._transcribe(audio)
                                    if text and not self._is_hallucination(text):
                                        yield text
                                speech_frames = 0

    def _is_hallucination(self, text):
        """True if the transcript is a known Whisper-on-silence artifact."""
        t = text.strip().lower().strip(".!?, ")
        return (not t) or (t in self.drop_phrases)

    def _transcribe(self, audio_bytes):
        import numpy as np
        data = np.frombuffer(audio_bytes, dtype=np.int16).astype("float32") / 32768.0
        segments, _ = self._model.transcribe(
            data, language=self.language, beam_size=self.beam_size,
            vad_filter=False, no_speech_threshold=0.6,
            initial_prompt=self.initial_prompt,
        )
        out = []
        for seg in segments:
            # drop low-confidence / likely-silence segments (hallucination guard)
            if getattr(seg, "no_speech_prob", 0.0) > 0.6:
                continue
            if getattr(seg, "avg_logprob", 0.0) < -1.0:
                continue
            out.append(seg.text.strip())
        return " ".join(out).strip()

    def stop(self):
        self._running.clear()
