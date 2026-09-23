# Project NIA -- Phase 0 (Brain on Text)

NIA's brain, on text, in a terminal. You type what Lazy says; she answers in
character. This proves the personality before any audio, vision, avatar, or
chat exists. It is config-driven and modular so Phases 1+ bolt on without
refactoring.

This is the scaffold from the Phase 0-1 Claude Code handoff. It deliberately
bakes in NONE of the two open hardware decisions (model size, vision local/API):
the brain talks to any OpenAI-compatible endpoint via a configurable base URL
and model name, so swapping the served model is a one-line config change.

## What's here

```
NIA/
  config.yaml            all tunables (endpoint, model, temperature, death thresholds)
  requirements.txt       Phase 0 deps (openai, PyYAML)
  .env.example           copy to .env; holds the brain API key (local servers ignore it)
  .gitignore
  nia/
    system_prompt.txt    the EXACT NIA persona (PRD Section 8). Do not edit casually.
    config.py            loads config.yaml + system_prompt.txt
    brain.py             OpenAI-compatible client + context assembler + rolling memory
    state.py             rolling death counter -> lock-in escalation note
    main.py              the terminal REPL
```

## Prereqs: stand up a model (one-time, on the main PC)

Nothing serves a model yet, so do this first. The PRD's recommended Windows path
is Ollama installed natively (good GPU support on the 3060).

1. Install Ollama from https://ollama.com (native Windows installer).
2. Pull a 7-8B instruct model (PRD Section 9 resolved the brain at 7-8B):
   ```
   ollama pull qwen2.5:7b-instruct
   ```
   (Llama 3.1 8B is the alternate: `ollama pull llama3.1:8b-instruct`.)
3. Ollama serves an OpenAI-compatible endpoint at `http://localhost:11434/v1`
   automatically. That is what `config.yaml` already points at.

When Odysseus is up later, you can repoint `brain.base_url` at its endpoint
instead. The code does not change. For lowest latency the PRD suggests pointing
`brain.py` straight at the fastest server (Ollama/vLLM) and using Odysseus for
model management, memory, and MCP. Decide that by measuring in Phase 0.

## Run it

```
cd NIA
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
copy .env.example .env                               # Windows (cp on mac/linux)
python -m nia.main
```

## Talk to her

```
Lazy> yo NIA you seeing this?
NIA> W yeah Lazy I'm locked on. let's go.

Lazy> :deaths 4        # simulate feeding -> flips her to lock-in mode
Lazy> man this boss
NIA> Lazy. Lock in. We are NOT doing this today.
```

REPL commands: `:death`, `:deaths N`, `:screen <text>` (Phase 2 vision stub),
`:chat <text>` (Phase 4 chat stub, injected as untrusted viewer data), `:state`,
`:reset`, `:help`, `:quit`.

## Phase 0 acceptance (from the handoff)

- [x] Chat with NIA in the terminal; she stays in character and stays SHORT.
- [x] She calls him Lazy.
- [x] She deflates on losses.
- [x] Feeding a fake death count flips her to lock-in mode.

The death-to-lock-in logic, context assembly, output cleaning, and end-to-end
endpoint wiring are unit-verified against a mock server. The only thing the
sandbox cannot test is her actual in-character voice -- that needs your live
Ollama model, which is why you run `python -m nia.main` on the PC.

## Tuning

Everything lives in `config.yaml`:
- `brain.temperature` -- her spark. Higher = wilder.
- `brain.max_tokens` -- hard length cap (keeps her from monologuing).
- `state.lock_in_threshold` / `death_window_seconds` -- how fast feeding triggers tough love.
- `state.lock_in_cooldown_seconds` -- how long before she can scold again.

## Stubs left for later phases (do not refactor)

- `vision` input into the gate + context assembly (Phase 2 fills via Gemini Flash).
- `chat` input into context assembly, already labeled untrusted (Phase 4).

---

# Phase 1 -- Voice Loop

She hears you through the mic and talks back out loud. Same brain, gate, and
state as Phase 0, now wired into a real-time turn-taking loop.

New modules: `gate.py` (when she speaks), `stt.py` (ears, faster-whisper),
`tts.py` (voice, swappable), `orchestrator.py` (the loop + barge-in), and
`voice.py` (the live entry point).

## Setup (one-time, on top of Phase 0)

```
.venv\Scripts\activate
pip install -r requirements-voice.txt
```

Then find your device indices and put them in `config.yaml` (`stt.input_device`,
`tts.output_device`). Leave them blank to use the system defaults:

```
python -m sounddevice          # lists input/output devices with indices
```

## Run her live

```
python -m nia.voice
```

She listens. Talk to her. Say her name or ask a question and she answers out
loud. Go quiet for ~25 s and she fills the air. Start talking while she is
speaking and she shuts up immediately (barge-in). Ctrl+C to stop.

She starts on the **sapi** voice (Windows built-in, no download) so she can
talk on the first run. It is robotic on purpose -- it is the dev voice, not the
final one.

## Phase 1 acceptance (from the handoff)

- [x] Mic -> STT -> gate -> brain -> TTS loop runs.
- [x] Gate decides whether she responds (direct address / silence-fill / cooldown).
- [x] Barge-in: she stops the instant you start talking.
- [x] She respects the hard cooldown.

The gate logic and the full orchestrator turn (address, cooldown, barge-in,
death-to-lock-in, silence-fill) are unit-verified with fakes in the sandbox.
STT and TTS audio can only be validated on your PC with a real mic and speakers,
which is what `python -m nia.voice` is for.

## Tuning the voice loop (`config.yaml`)

- `gate.cooldown_seconds` -- minimum gap between her lines. Lower = chattier.
- `gate.silence_fill_seconds` -- how long you can be quiet before she jumps in.
- `gate.name_aliases` -- what she answers to (whisper sometimes mishears "NIA").
- `stt.model` -- `tiny`/`base`/`small`. Bigger is more accurate but slower.
- `stt.vad_aggressiveness` -- raise if it misses your speech, lower if it triggers on noise.
- `tts.engine` -- `sapi` (now) or `piper` (the upgrade).
- `tts.sapi_rate` / `sapi_voice` -- speed and which Windows voice.

## Upgrading her voice (Piper)

The sapi voice is a placeholder. For a real voice, switch to Piper:

1. Download the piper binary and a voice `.onnx` from the Piper releases.
2. In `config.yaml`: set `tts.engine: piper`, `tts.piper_binary` to the exe
   path, and `tts.piper_model` to the voice file.
3. Re-run `python -m nia.voice`. No code change -- the engine swaps under the
   same interface. (StyleTTS2 / XTTS, the expressive build, slot in here later.)

---

# Phase 2 -- Eyes (game vision)

She watches the game and reacts to kills, deaths, wins, and bosses on her own.
This also makes the death-to-lock-in loop fire for real (no more `:deaths`).

New modules: `capture.py` (the frame source) and `vision.py` (a cheap local
scene-change check, then Gemini Flash to describe only when something changed).
Both feed the same speak-gate via the orchestrator's `handle_vision`.

## Setup

```
.venv\Scripts\activate
pip install -r requirements-vision.txt
```

1. Get a free Gemini key from https://aistudio.google.com (free tier is fine for
   development). Put it in `.env`:  `GEMINI_API_KEY=your_key_here`
2. Pick a feed in `config.yaml` under `capture.source`:
   - `screen` (dev): she watches your monitor. Easiest -- just have the game
     visible. Set `capture.monitor` or a `capture.region` to focus on the game.
   - `capture_card` (live): the PS5 through the card. Set `capture.device_index`
     to the card's video device.
3. Turn vision on: set `vision.enabled: true` in `config.yaml`.
4. Run `python -m nia.voice`. The header should say `vision: ON`.

## How it behaves

- A real game moment (kill/death/win/loss/boss) becomes a gate trigger, so she
  pops off on a kill, deflates on a death, etc.
- Several deaths inside the window flip her to lock-in mode unprompted.
- A calm screen does not spam the API (the local scene-change check suppresses
  it) and does not make her chatter (the cooldown holds).
- A Gemini rate-limit (429) does not crash her -- she backs off and skips.

## Dev vs live (Gemini tier)

- Dev: free key, no card. The cooldown keeps call volume low.
- Live: enable billing on the Google project and the same code uses paid Flash
  (a few cents per streaming hour). No code change -- just the billing state.

## Tuning (`config.yaml`)

- `vision.scene_change_threshold` -- lower = more sensitive to small changes.
- `vision.vision_cooldown` -- minimum seconds between Gemini calls.
- `vision.sample_interval` -- how often the cheap local check runs.
- `capture.max_width` -- smaller = faster + cheaper, less detail.
- `state.lock_in_threshold` -- deaths in the window before tough-love mode.
