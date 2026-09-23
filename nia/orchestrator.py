"""orchestrator.py -- the real-time loop.

Wires the pieces from Phase 0-1 into one turn-taking loop:

    mic -> STT -> gate -> (assemble context) -> brain -> TTS

with two hard rules layered on top (PRD 4.1 / 4.2):

  - Barge-in: the instant Lazy starts talking, NIA's current TTS line is killed.
    The STT speech-start callback calls tts.stop() directly, so it fires before
    a word of his is even transcribed. The streamer always wins the floor.
  - Cooldown: the gate vetoes any line within cooldown_seconds of her last one.

The loop body is split into handle_transcript() so it can be unit-tested with
fakes (no mic, no model) -- that is what we verify in the sandbox. run() is the
live entry that needs real hardware.
"""
import threading


class Orchestrator:
    def __init__(self, brain, gate, state, tts, stt=None, debug=True):
        self.brain = brain
        self.gate = gate
        self.state = state
        self.tts = tts
        self.stt = stt
        self.vision = None          # set by voice.py if Phase 2 vision is enabled
        self.debug = debug          # print what she hears + why she stays quiet
        self._last_reason = ""
        self.latest_vision = ""     # most recent on-screen description (Phase 2)
        # one turn at a time: STT (mic) and vision run on separate threads, so a
        # lock keeps them from both making her speak at the same instant.
        self._turn_lock = threading.Lock()
        self._running = threading.Event()

    # ---- barge-in -------------------------------------------------------
    def on_speech_start(self):
        """Called by STT the moment Lazy starts talking. Kill her line."""
        if self.tts.is_speaking:
            self.tts.stop()

    # ---- one turn from the mic (pure-ish, unit-testable) ----------------
    def handle_transcript(self, transcript, event=""):
        """Process one finalized utterance (+ optional game event). Returns the
        line NIA spoke this turn, or None if the gate kept her quiet."""
        with self._turn_lock:
            # any speech from Lazy resets the silence-fill timer
            if transcript:
                self.gate.note_lazy_voice()

            if event == "death":
                self.state.record_death()

            decision = self.gate.evaluate(transcript=transcript, event=event)
            self._last_reason = decision.reason
            if not decision.speak:
                return None

            note = self.state.lock_in_note()
            # hand her what's currently on screen too, so a spoken reply is aware
            # of the game even when she is answering Lazy.
            try:
                line = self.brain.respond(
                    transcript=decision.transcript or None,
                    vision=self.latest_vision or None,
                    session_note=note,
                )
            except Exception as e:
                # a brain blip (e.g. Ollama hiccup) must not crash the stream
                if self.debug:
                    print(f"  [brain error -- skipping] {e}")
                return None
            if line:
                self.tts.speak(line)        # non-blocking, interruptible
                self.gate.note_spoke()      # arm the cooldown
            return line

    # ---- one turn from the game (Phase 2 vision) ------------------------
    def handle_vision(self, event, description):
        """Called by the VisionWatcher each time it describes the screen. A
        notable event (kill/death/win/loss/boss) can make her react; a death
        always increments the counter regardless of whether she speaks."""
        with self._turn_lock:
            if description:
                self.latest_vision = description

            # show EVERY glance so you can confirm her eyes are actually working
            if self.debug:
                print(f"  [vision] {event}: {description}")

            # count deaths the moment they happen, even if she stays quiet
            if event == "death":
                self.state.record_death()

            if not event or event == "none":
                return None

            decision = self.gate.evaluate(event=event)
            self._last_reason = decision.reason
            if not decision.speak:
                if self.debug:
                    print(f"  [saw {event}: {description}] (quiet -- {decision.reason})")
                return None

            note = self.state.lock_in_note()
            try:
                line = self.brain.respond(vision=self.latest_vision or None,
                                          session_note=note)
            except Exception as e:
                if self.debug:
                    print(f"  [brain error -- skipping] {e}")
                return None
            if line:
                self.tts.speak(line)
                self.gate.note_spoke()
            if self.debug:
                print(f"  [saw {event}: {description}]")
                if line:
                    print(f"  NIA> {line}")
            return line

    # ---- live loop (needs hardware) -------------------------------------
    def run(self):
        if self.stt is None:
            raise RuntimeError("No STT attached. Orchestrator.run() needs a mic.")
        self._running.set()
        print("NIA is listening. Talk to her. Ctrl+C to stop.")
        try:
            for transcript in self.stt.listen():
                if not self._running.is_set():
                    break
                if self.debug:
                    print(f"  [heard] {transcript!r}")
                line = self.handle_transcript(transcript)
                if line:
                    print(f"  NIA> {line}")
                elif self.debug:
                    print(f"  [stayed quiet -- gate reason: {self._last_reason}]")
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        self._running.clear()
        if self.stt:
            self.stt.stop()
        if self.vision:
            self.vision.stop()
        self.tts.stop()
