"""vision.py -- NIA's eyes. Two-stage so we only pay for the API when something
actually changed (PRD 4.3 / Phase 2 handoff).

  Stage A (free, fast, no API): every sample_interval, grab a downscaled frame
    and diff it against the last. If little changed, do nothing. A cooldown caps
    how often we ever call the API, protecting latency and the Gemini rate limit.

  Stage B (Gemini Flash, only when A fires): send the frame with a tight prompt
    asking for a one-line description plus an event flag. Parse it into
    {event, description}. On a 429 / quota error, back off and skip rather than
    crash -- NIA going briefly quiet is fine, NIA crashing the stream is not.
"""
import os
import time
import threading

VISION_PROMPT = (
    "You are watching a live video game stream. Call out the single most notable "
    "thing in this frame. Reply EXACTLY as: "
    "EVENT: <kill|death|win|loss|boss|none> | <one short vivid clause>.\n"
    "Cues for each event:\n"
    "- kill: a kill-feed entry, 'eliminated', a knockdown, an enemy dropping, +score popup.\n"
    "- death: a death or respawn screen, 'you died', greyed-out / spectator view, downed.\n"
    "- win: a victory / 'win' / 'victory royale' / results-win screen.\n"
    "- loss: a defeat / 'you lost' / game-over screen.\n"
    "- boss: a boss on screen or a large boss health bar.\n"
    "Be willing to flag an event when you see its cue. Use none only when nothing "
    "notable is happening. Under 20 words total."
)

NOTABLE = {"kill", "death", "win", "loss", "boss"}


def parse_response(text):
    """Parse 'EVENT: kill | triple kill' -> ('kill', 'triple kill').

    Defensive: handles missing 'EVENT:', missing pipe, stray casing/whitespace.
    Returns (event, description); event is always one of NOTABLE or 'none'.
    """
    t = (text or "").strip()
    low = t.lower()
    if "event:" in low:
        t = t[low.index("event:") + len("event:"):].strip()
    event = "none"
    desc = ""
    if "|" in t:
        ev, desc = t.split("|", 1)
        event = ev.strip().lower()
        desc = desc.strip()
    else:
        words = t.split()
        first = words[0].lower().strip(".,!|") if words else "none"
        if first in NOTABLE:
            event = first
            desc = t
        else:
            desc = t
    if event not in NOTABLE:
        event = "none"
    return event, desc


def frame_diff(a, b):
    """Normalized mean absolute difference between two frames (0..1).
    1.0 (treat as fully changed) if either is missing or shapes differ."""
    import numpy as np
    if a is None or b is None or a.shape != b.shape:
        return 1.0
    d = np.abs(a.astype("int16") - b.astype("int16")).mean()
    return float(d) / 255.0


class VisionWatcher:
    def __init__(self, config, source, on_event):
        v = config.get("vision", {})
        self.sample_interval = v.get("sample_interval", 2.0)
        self.scene_threshold = v.get("scene_change_threshold", 0.15)
        self.cooldown = v.get("vision_cooldown", 6.0)
        self.model = v.get("model", "gemini-2.5-flash")
        self.max_tokens = v.get("max_output_tokens", 50)
        self.api_key_env = v.get("api_key_env", "GEMINI_API_KEY")
        self.source = source         # a FrameSource
        self.on_event = on_event     # callback(event, description)
        self._running = threading.Event()
        self._client = None
        self._last_frame = None
        self._last_call = 0.0

    def _ensure_client(self):
        if self._client is None:
            from google import genai
            key = os.environ.get(self.api_key_env)
            if not key:
                raise RuntimeError(
                    f"{self.api_key_env} not set. Put your Gemini key in the "
                    "environment (or .env) before enabling vision.")
            self._client = genai.Client(api_key=key)

    def _describe(self, frame):
        from google.genai import types
        from PIL import Image
        import cv2
        self._ensure_client()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        # gemini-2.5-flash is a THINKING model; with a tiny token budget its
        # reasoning eats the whole output and the visible text comes back empty.
        # Disable thinking for this fast one-line reflex (Flash supports budget=0).
        kwargs = {"max_output_tokens": self.max_tokens}
        try:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        except Exception:
            pass
        resp = self._client.models.generate_content(
            model=self.model,
            contents=[img, VISION_PROMPT],
            config=types.GenerateContentConfig(**kwargs),
        )
        return parse_response(getattr(resp, "text", ""))

    def run(self):
        self._running.set()
        backoff = 1.0
        while self._running.is_set():
            time.sleep(self.sample_interval)
            frame = self.source.downscale(self.source.get_frame())
            if frame is None:
                continue
            # Stage A: cheap local scene-change check
            diff = frame_diff(self._last_frame, frame)
            self._last_frame = frame
            if diff < self.scene_threshold:
                continue
            if time.monotonic() - self._last_call < self.cooldown:
                continue
            # Stage B: pay for Gemini only now
            try:
                event, desc = self._describe(frame)
                self._last_call = time.monotonic()
                backoff = 1.0
                if self.on_event:
                    self.on_event(event, desc)
            except Exception as e:
                msg = str(e).lower()
                if "429" in msg or "quota" in msg or "resource" in msg or "rate" in msg:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)
                # any error: skip this cycle, never crash the stream

    def stop(self):
        self._running.clear()
        try:
            self.source.release()
        except Exception:
            pass
