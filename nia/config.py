"""Config loader. Single source of truth so nothing is hardcoded in modules."""
from pathlib import Path
import os
import yaml


def load_config(path=None):
    """Load config.yaml from the repo root (one level up from this package)."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_system_prompt(path=None):
    """Load the exact NIA system prompt (PRD Section 8) from system_prompt.txt."""
    if path is None:
        path = Path(__file__).resolve().parent / "system_prompt.txt"
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def resolve_api_key(env_name, default="local-no-key-needed"):
    """Read the brain API key from an env var. Local servers ignore it, but the
    OpenAI client still needs a non-empty value, so fall back to a placeholder."""
    return os.environ.get(env_name) or default


def _pkg_path(filename):
    from pathlib import Path
    return Path(__file__).resolve().parent / filename


def load_vibe(path=None):
    """Load the editable slang/vibe palette (vibe.txt). Returns '' if missing."""
    p = path or _pkg_path("vibe.txt")
    try:
        with open(p, "r", encoding="utf-8") as f:
            # strip leading comment lines so only the palette reaches the model
            lines = [ln for ln in f.read().splitlines() if not ln.strip().startswith("#")]
        return "\n".join(lines).strip()
    except FileNotFoundError:
        return ""


def load_few_shot(path=None):
    """Parse few_shot.txt into a list of {role, content} chat messages.

    Lines starting with 'USER:' become user turns, 'NIA:' become assistant turns.
    Blank lines and '#' comments are ignored. Returns [] if the file is missing.
    """
    p = path or _pkg_path("few_shot.txt")
    messages = []
    try:
        with open(p, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.rstrip("\n")
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.startswith("USER:"):
                    messages.append({"role": "user", "content": stripped[5:].strip()})
                elif stripped.startswith("NIA:"):
                    messages.append({"role": "assistant", "content": stripped[4:].strip()})
        return messages
    except FileNotFoundError:
        return []
