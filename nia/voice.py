"""voice.py -- live entry point (Phases 1 + 2).

Run from the repo root:   python -m nia.voice

Brings up the full loop: she hears you through the mic, the speak-gate decides
whether to answer, and she talks back out loud, yielding the floor the instant
you start talking. If vision is enabled in config (Phase 2), she also watches
the game on a second thread and reacts to kills, deaths, wins, and bosses.
"""
import os
import threading

from .brain import Brain
from .config import load_config
from .gate import SpeakGate
from .state import SessionState
from .tts import get_tts
from .stt import STT
from .orchestrator import Orchestrator


def _load_dotenv():
    """Tiny .env reader so GEMINI_API_KEY can live in .env, not the shell."""
    path = os.path.join(os.getcwd(), ".env")
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    except FileNotFoundError:
        pass


def main():
    _load_dotenv()
    config = load_config()

    brain = Brain(config)
    gate = SpeakGate(config)
    state = SessionState(config)

    try:
        tts = get_tts(config)
    except Exception as e:
        print(f"[TTS init failed] {e}")
        return

    # orchestrator first (it owns the barge-in callback), then wire STT to it
    orch = Orchestrator(brain, gate, state, tts, stt=None)
    stt = STT(config, on_speech_start=orch.on_speech_start)
    orch.stt = stt

    # Phase 2: vision. Optional -- only starts if enabled AND deps/key are present,
    # so the voice loop still runs fine without it.
    vision_on = False
    if config.get("vision", {}).get("enabled"):
        try:
            from .capture import FrameSource
            from .vision import VisionWatcher
            source = FrameSource(config)
            watcher = VisionWatcher(config, source, on_event=orch.handle_vision)
            orch.vision = watcher
            threading.Thread(target=watcher.run, daemon=True).start()
            vision_on = True
        except Exception as e:
            print(f"[vision failed to start, continuing without it] {e}")

    print("NIA voice loop. Engine:", config.get("tts", {}).get("engine", "sapi"),
          "| STT:", config.get("stt", {}).get("model", "base"),
          "| vision:", "ON" if vision_on else "off")
    try:
        orch.run()
    except Exception as e:
        print(f"[voice loop error] {e}")
        orch.stop()


if __name__ == "__main__":
    main()
