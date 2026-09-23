"""state.py -- session state.

The orchestrator owns session state, NOT the LLM (PRD Section 4.4). NIA just
reacts to a flag we inject. Phase 0 implements the death-to-lock-in loop: a
rolling counter tracks deaths inside a time window, and when the count crosses a
threshold we hand the brain a lock-in note so NIA flips to tough-love mode.

In Phase 0 deaths are fed by hand (the REPL :death command). In Phase 2 a real
`vision event == death` calls record_death() instead. Same logic either way.
"""
import time


class SessionState:
    def __init__(self, config):
        s = config["state"]
        self.window = s.get("death_window_seconds", 180)
        self.threshold = s.get("lock_in_threshold", 4)
        self.lock_in_cooldown = s.get("lock_in_cooldown_seconds", 120)
        self._deaths = []            # list of monotonic timestamps
        self._last_lock_in = 0.0     # when we last emitted the note (0 = never)

    def _now(self):
        return time.monotonic()

    def _prune(self, now=None):
        """Drop deaths that have aged out of the rolling window."""
        now = now if now is not None else self._now()
        cutoff = now - self.window
        self._deaths = [t for t in self._deaths if t >= cutoff]

    def record_death(self, now=None):
        """Register one death. Returns the current in-window death count."""
        now = now if now is not None else self._now()
        self._deaths.append(now)
        self._prune(now)
        return len(self._deaths)

    def deaths_in_window(self, now=None):
        self._prune(now)
        return len(self._deaths)

    def lock_in_note(self, now=None):
        """Return the session note to inject into NIA's context, or None.

        Fires only when in-window deaths >= threshold AND we have not already
        scolded within lock_in_cooldown_seconds (so she does not nag every line).
        """
        now = now if now is not None else self._now()
        count = self.deaths_in_window(now)
        if count < self.threshold:
            return None
        if now - self._last_lock_in < self.lock_in_cooldown:
            return None
        self._last_lock_in = now
        return (
            f"{count} deaths in {self.window // 60} min -- escalate to lock-in "
            "mode. Tough love. Get on him."
        )

    def reset(self):
        self._deaths.clear()
        self._last_lock_in = 0.0
