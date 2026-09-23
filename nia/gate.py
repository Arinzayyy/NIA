"""gate.py -- the rule-based speak-gate.

Decides whether NIA should even speak this tick, WITHOUT calling the LLM. This
is what protects the latency budget and keeps her from chattering (PRD 4.1).
She speaks only when one of these is true:

  - Direct address: Lazy says her name or asks a question aimed at her.
  - Game event: the vision step flags a notable change (Phase 2; stubbed here).
  - Silence fill: Lazy has been quiet for N seconds and the air needs filling.

On top of that:
  - Hard cooldown: she physically cannot speak again within X seconds of her
    last line.
  - Barge-in is handled in the orchestrator (mic interrupts TTS), not here.

The gate returns a Decision telling the orchestrator whether to speak and why,
so silence-fill vs direct-address vs event can be assembled into context.
"""
import re
import time
import random
from dataclasses import dataclass

# game events that count as a trigger once vision is live (Phase 2)
NOTABLE_EVENTS = {"kill", "death", "win", "loss", "boss"}


@dataclass
class Decision:
    speak: bool
    reason: str          # "address" | "event" | "silence" | "cooldown" | "idle"
    transcript: str = ""  # what Lazy said, if anything
    event: str = ""       # the game event that fired, if any


class SpeakGate:
    def __init__(self, config, name_aliases=None):
        g = config.get("gate", {})
        self.cooldown = g.get("cooldown_seconds", 6.0)
        self.silence_fill_after = g.get("silence_fill_seconds", 25.0)
        self.silence_fill_enabled = g.get("silence_fill_enabled", True)
        # occasional unprompted chime-in on something Lazy said (not addressed)
        self.chime_chance = g.get("chime_in_chance", 0.25)
        self.chime_min_gap = g.get("chime_in_min_gap_seconds", 20.0)
        self._last_chime = 0.0
        self._rand = random.random  # swappable for tests
        # names she answers to. lowercased for matching.
        aliases = name_aliases or g.get("name_aliases", ["nia", "neela", "nya"])
        self.aliases = [a.lower() for a in aliases]
        self._last_spoke = 0.0          # monotonic time she last spoke
        self._last_lazy_voice = self._now()  # last time Lazy said anything

    def _now(self):
        return time.monotonic()

    def _is_direct_address(self, transcript):
        """True if Lazy said her name or asked a question aimed at her."""
        if not transcript:
            return False
        low = transcript.lower()
        # her name appears as a whole word
        for alias in self.aliases:
            if re.search(rf"\b{re.escape(alias)}\b", low):
                return True
        # a question (ends with ?, or starts with a question word) is treated as
        # aimed at her when she is the only other party in the room
        if transcript.strip().endswith("?"):
            return True
        if re.match(r"^\s*(you|are you|did you|do you|can you|what|why|how|huh)\b",
                    low):
            return True
        return False

    def on_cooldown(self, now=None):
        now = now if now is not None else self._now()
        return (now - self._last_spoke) < self.cooldown

    def note_spoke(self, now=None):
        """Orchestrator calls this right after NIA speaks, to arm the cooldown."""
        self._last_spoke = now if now is not None else self._now()

    def note_lazy_voice(self, now=None):
        """Orchestrator calls this whenever Lazy speaks, to reset the silence timer."""
        self._last_lazy_voice = now if now is not None else self._now()

    def evaluate(self, transcript="", event="", now=None):
        """Decide whether NIA speaks this tick.

        transcript: finalized speech from Lazy this tick (may be empty)
        event:      game event flag from vision this tick (Phase 2; "" for now)
        """
        now = now if now is not None else self._now()

        # cooldown is an absolute veto, EXCEPT we still report why
        if self.on_cooldown(now):
            return Decision(False, "cooldown", transcript=transcript, event=event)

        # 1. direct address always wins
        if self._is_direct_address(transcript):
            return Decision(True, "address", transcript=transcript, event=event)

        # 2. a notable game event (Phase 2)
        if event and event.lower() in NOTABLE_EVENTS:
            return Decision(True, "event", transcript=transcript, event=event)

        # 2.5 occasional unprompted chime-in: react to what he said even though
        #     he did not address her. Rate-limited by chime_in_min_gap so she is
        #     present, not annoying. The hard cooldown still applies on top.
        if (transcript and self.chime_chance > 0
                and (now - self._last_chime) >= self.chime_min_gap
                and self._rand() < self.chime_chance):
            self._last_chime = now
            return Decision(True, "chime", transcript=transcript, event=event)

        # 3. silence fill: Lazy quiet for too long
        if (self.silence_fill_enabled
                and (now - self._last_lazy_voice) >= self.silence_fill_after):
            return Decision(True, "silence", transcript=transcript, event=event)

        # otherwise stay quiet
        return Decision(False, "idle", transcript=transcript, event=event)
