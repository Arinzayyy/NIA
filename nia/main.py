"""main.py -- Phase 0 terminal REPL.

Prove the personality before anything real-time. You type what Lazy says; NIA
answers in text, in character. Debug commands let you exercise the parts that
later phases drive automatically (deaths, vision, chat) without any hardware.

Run from the repo root:   python -m nia.main

Commands:
  <anything>        speak to NIA as Lazy
  :death            register one death (drives the lock-in escalation)
  :deaths N         register N deaths at once
  :screen <text>    inject a vision description as if she saw it (Phase 2 stub)
  :chat <text>      inject chat as untrusted viewer data (Phase 4 stub)
  :state            print current death count / window
  :reset            clear memory + death state (new session)
  :help             show this list
  :quit / :exit     leave
"""
import sys

from .brain import Brain
from .config import load_config
from .state import SessionState

BANNER = r"""
  NIA -- Neural Interactive Avatar  (Phase 0: brain on text)
  Type what Lazy says. ':help' for commands, ':quit' to exit.
"""

HELP = """
  <anything>      speak to NIA as Lazy
  :death          register one death (drives lock-in escalation)
  :deaths N       register N deaths at once
  :screen <text>  inject a vision description (Phase 2 stub)
  :chat <text>    inject chat as untrusted viewer data (Phase 4 stub)
  :state          print death count / window
  :reset          clear memory + death state
  :help           this list
  :quit / :exit   leave
"""


def main():
    config = load_config()
    state = SessionState(config)
    try:
        brain = Brain(config)
    except Exception as e:
        print(f"\n[could not init brain] {e}")
        print("Check config.yaml brain.base_url and that your model server is up.")
        sys.exit(1)

    print(BANNER)

    # one-shot stubs consumed on the next spoken turn, then cleared
    pending_screen = None
    pending_chat = None

    while True:
        try:
            line = input("Lazy> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nlater.")
            break

        if not line:
            continue

        if line in (":quit", ":exit"):
            print("later.")
            break
        if line == ":help":
            print(HELP)
            continue
        if line == ":state":
            print(f"  deaths in window: {state.deaths_in_window()} "
                  f"(threshold {state.threshold}, window {state.window}s)")
            continue
        if line == ":reset":
            state.reset()
            brain.reset()
            print("  [session reset]")
            continue
        if line == ":death" or line.startswith(":deaths"):
            n = 1
            if line.startswith(":deaths"):
                parts = line.split()
                if len(parts) > 1 and parts[1].isdigit():
                    n = int(parts[1])
            count = 0
            for _ in range(n):
                count = state.record_death()
            print(f"  [death recorded -- {count} in window]")
            continue
        if line.startswith(":screen"):
            pending_screen = line[len(":screen"):].strip() or None
            print(f"  [vision queued: {pending_screen}]")
            continue
        if line.startswith(":chat"):
            pending_chat = line[len(":chat"):].strip() or None
            print(f"  [chat queued: {pending_chat}]")
            continue

        # a normal spoken turn
        note = state.lock_in_note()
        try:
            reply = brain.respond(
                transcript=line,
                vision=pending_screen,
                chat=pending_chat,
                session_note=note,
            )
        except Exception as e:
            print(f"  [brain error] {e}")
            continue
        finally:
            pending_screen = None
            pending_chat = None

        print(f"NIA> {reply}\n")


if __name__ == "__main__":
    main()
