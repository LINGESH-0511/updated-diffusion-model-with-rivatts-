# A2F + Avatar Rendering — Status & Continuation Notes

**Date:** 2026-07-08
**Machine:** Windows, conda env `a2f`, GPU: NVIDIA GeForce RTX 4050 Laptop GPU
**Project root:** `D:\A2F`
**Supersedes:** `A2F_PROJECT_STATUS (1).md` (2026-07-08 morning) — that doc's "IMMEDIATE NEXT STEPS" are now DONE. This doc picks up from there.

---

## 1. What's DONE and VERIFIED working (do not re-debug these)

### 1.1 Backend Python pipeline — fully working, end-to-end tested twice
- `backend/__init__.py` — created (was missing, was the real cause of `ModuleNotFoundError: No module named 'backend'`)
- `backend/audio2face/{client,stream,receiver,wav_sender,test_audio,test_tts,__init__}.py` — all correct, no bugs remaining
- `backend/utils/{config,logger,exceptions,__init__}.py` — all correct, no bugs remaining
- `backend/tts/piper_streamer.py` — **fixed**: `piper-tts` 1.4.2's `PiperVoice` has no `synthesize_stream_raw()` method (that's from an older API). Fixed to use `voice.synthesize(text)` which yields `AudioChunk` objects; use `.audio_int16_bytes` property to get raw PCM bytes. Confirmed via extracting the actual installed wheel source.

**Verified test #1 — WAV path:**
```powershell
cd D:\A2F
python -m backend.audio2face.test_audio --wav "Audio2Face-3D-Samples/example_audio/Mark_neutral.wav" --config "Audio2Face-3D-Samples/scripts/audio2face_3d_microservices_interaction_app/config/config_james.yml"
```
Result: `Status code: 0`, `55 blendshapes`, `148 animation frames`, SUCCESS.

**Verified test #2 — Piper TTS path:**
```powershell
python -m backend.audio2face.test_tts --text "Hello, this is a test of the avatar pipeline." --config "Audio2Face-3D-Samples/scripts/audio2face_3d_microservices_interaction_app/config/config_james.yml"
```
Result: `Status code: 0`, `55 blendshapes`, `88 animation frames`, `128008 audio bytes`, SUCCESS.

### 1.2 Docker / NIM — fully working
- **Model in use: `james_v2.3`** (NVIDIA's default, TRT engine already cached at `Audio2Face-3D-Samples\quick-start\a2f-3d-init-data\james_v2.3.trt`) — decision made and final, do not revisit Mark.
- Fixed: `quick-start\configs\` was missing `james_stylization_config.yaml` — copied from `Audio2Face-3D-Samples\configs\james_stylization_config.yaml`.
- **Fixed (the real root cause of days of `Connection refused` errors): `network_mode: "host"` in `docker-compose.yml` does not work on Docker Desktop for Windows** (host networking binds to the WSL2 VM, not to Windows' `127.0.0.1`). Replaced with explicit port mapping:
  ```yaml
  ports:
    - "52000:52000"
    - "50000:50000"
    - "51000:51000"
  ```
  This was the single biggest time-sink of the whole project. **If A2F connection issues ever recur, check this first before anything else.**

**Known-good startup sequence (2 terminals):**
```powershell
# Terminal A — Docker (from quick-start dir specifically, not D:\A2F)
cd D:\A2F\Audio2Face-3D-Samples\quick-start
docker compose down
$env:A2F_3D_MODEL_NAME = "james"
$env:NGC_API_KEY = "<your rotated key>"
docker compose up --force-recreate
# wait for: [global] [info] Running...

# Terminal B — Python (from D:\A2F specifically)
cd D:\A2F
python -m backend.audio2face.test_audio --wav "..." --config "..."
```
**Common mistake made repeatedly this session: running Docker commands or Python commands from the wrong directory.** `docker compose` commands only work from `quick-start\`. `python -m backend...` commands only work from `D:\A2F`. Always double check `pwd`/prompt before running either.

### 1.3 Avatar rendering — working, visually confirmed
- Avatar source: ReadyPlayerMe `.glb` (mesh names `Wolf3D_Head`, `Wolf3D_Teeth`, `EyeLeft`, `EyeRight`, all with **72 morph targets** including full ARKit set + visemes). File: `D:\A2F\frontend\assets\avatar.glb`.
- **Earlier failed attempt (documented so it's not repeated):** an Avaturn `.glb` was tried first and had **0 morph targets** — this was because it was exported as Avaturn body-type **T1** (static face, cannot be animated). If Avaturn is ever used again, must select body-type **T2** (animatable face) at avatar-creation time, not at export.
- Built `frontend/avatar_viewer.html` — Three.js (via CDN import map) + GLTFLoader + OrbitControls. Loads `avatar.glb`, auto-matches A2F blendshape names (PascalCase, e.g. `JawOpen`) against the mesh's morph target names (camelCase, e.g. `jawOpen`) via case-folding fallback in `resolveMorphName()`.
- Built `backend/audio2face/save_animation.py` — runs the existing A2F pipeline once and saves `frontend/assets/test_animation.json` (blendshape frames) + `frontend/assets/test_animation_audio.wav` (matching audio), so the renderer can be tested/iterated on without needing Docker running every time.
- **Verified result: 52/55 blendshapes matched.** The 3 "unmatched" (`HeadRoll`, `HeadPitch`, `HeadYaw`) are NOT a bug — these are head rotation angles A2F sends alongside blendshapes, meant to drive a bone/joint rotation, not a morph target. No avatar anywhere has these as blendshapes. Not currently wired up (see open items).
- Screenshot confirms: avatar loads correctly, textures/hair/glasses/proportions all correct, sitting in what appears to be a resting/neutral pose in the still captured.

**How to view it right now:**
```powershell
cd D:\A2F\frontend
python -m http.server 8080
# open http://localhost:8080/avatar_viewer.html in browser
# click once on the page to unblock audio autoplay
```

---

## 2. OPEN ITEMS — pick up here next session

### 2.1 Not yet visually confirmed: is the mouth actually animating?
Last screenshot was a single still frame — could not confirm from it alone whether blendshape weights are changing frame-to-frame in sync with audio, or whether the avatar is just sitting in neutral pose the whole time. **This is the very next thing to check.** If it's not animating:
- Check browser console for JS errors
- Check that `frame.blendShapes` values in `test_animation.json` actually have non-zero values for `JawOpen`/`MouthSmileLeft`/etc mid-clip (they should, since A2F reported real animation data — 148 frames — in the backend test)
- Check `applyBlendshapes()` is actually being called every `requestAnimationFrame` tick (add a `console.log` temporarily if needed)

### 2.2 Head rotation (HeadRoll/HeadPitch/HeadYaw) not wired up
Low priority — cosmetic. To add: find the head/neck bone in the avatar's skeleton (likely `Head` or similar in the `.glb`'s node hierarchy), and each frame set its `.rotation` from these three A2F values (need to confirm units — likely radians or degrees, check A2F docs/proto).

### 2.3 Rotate API keys — still not done
`GROQ_API_KEY`, `NVIDIA_API_KEY`, `NGC_API_KEY`, `CARTESIA_API_KEY` were pasted in plaintext in an earlier chat session and were flagged as compromised at the start of this whole debugging saga. **Still not rotated as of this doc.** Do this whenever convenient — not blocking technical work, but shouldn't be forgotten indefinitely.

### 2.4 Live mode not built yet — biggest remaining piece of work
Everything above is either (a) a one-shot batch test (`test_audio.py`, `test_tts.py`) or (b) an offline replay of saved data (`avatar_viewer.html`'s current `OFFLINE replay` mode). **Nothing currently streams live blendshapes from a live TTS/LLM response to the browser in real time.** To get there:

1. **Rewrite `receiver.py` to stream, not batch.** Currently `read_animation_stream()` collects ALL frames into one `AnimationResult` and only returns after the stream fully ends. Needs an async-generator variant that yields each frame the instant it arrives, so it can be pushed to a WebSocket immediately instead of waiting for the whole utterance to finish.
2. **Build a WebSocket bridge** — likely a small FastAPI/websockets server that: receives text → runs Piper TTS → streams to A2F via `send_audio_stream()` → receives blendshape frames via the new streaming `receiver.py` → forwards each frame as JSON to the connected browser client, roughly in real time.
3. **Switch `avatar_viewer.html` to live mode** — the code already has this half-built: set `const WS_URL = "ws://localhost:8765";` (currently `null`) and `startLiveMode()` will take over from `startOfflineReplay()`. The WebSocket message format expected is `{"blendShapes": {...}}` per frame — needs to match whatever the bridge server actually sends.
4. **Sync audio playback with the blendshape stream** — both need to start at the same moment. Web Audio API on the frontend, timed against incoming frame timecodes (or simplest first version: just play the full audio via `<audio>` tag same as the offline replay does, in parallel with the live blendshape WebSocket — may be "good enough" for a first working version even if not perfectly frame-accurate).

### 2.5 Not yet integrated into main VRA/ARIA app
This whole session's work has been a **standalone** A2F leg (`D:\A2F`), separate from the main VRA/ARIA FastAPI + WebSocket app (`D:\vra` per earlier project history) that has Groq Whisper STT, BGE-M3 embeddings, Qdrant, Groq LLM already working. Once 2.4 above is solid, the next big step is wiring this A2F backend into that existing app to fully replace the old D-ID avatar layer. Not started.

### 2.6 Still empty / untouched
- `backend/rag/`, `backend/api/` — empty, not part of this session's work
- `backend/audio2face/test_health.py` import bug mentioned in the original status doc — never actually confirmed broken this session (all files were checked individually and found correct); low priority, check only if `python -m backend.test_health` is actually run and fails

---

## 3. Known environment gotchas (avoid repeating)

- **Two separate Python environments exist in this project**: conda env `a2f` (what's actually been used all session) and a `.venv` folder with some packages installed only there (e.g. `piper`, `opencv`). Always confirm `(a2f)` shows in the PowerShell prompt before running anything. If a package is missing despite appearing to be "installed" somewhere, it's likely in the wrong environment — `pip install <package>` again inside the active `(a2f)` env.
- **Directory matters for every command.** `docker compose *` → must run from `Audio2Face-3D-Samples\quick-start\`. `python -m backend.*` → must run from `D:\A2F`. Both errors ("no configuration file provided" and "No module named 'backend'") were hit multiple times this session purely from being in the wrong folder.
- **Docker Desktop must be manually running** before any `docker compose up` — it does not auto-start. If you see `failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine`, Docker Desktop itself isn't open; launch it and wait ~30-90s before retrying.
- **`.env` files for Docker vs Python are separate.** `D:\A2F\.env` (Python `dotenv`, has GROQ/NVIDIA/CARTESIA keys etc.) is unrelated to `A2F_3D_MODEL_NAME`/`NGC_API_KEY`, which must be set as **PowerShell session environment variables** (`$env:...`) before `docker compose up` each time Docker Desktop restarts (these are lost on Docker/PowerShell restart, no persistent `.env` currently used for this).

---

## 4. Quick reference — all working commands in one place

```powershell
# Start Docker (Terminal A, from quick-start)
cd D:\A2F\Audio2Face-3D-Samples\quick-start
$env:A2F_3D_MODEL_NAME = "james"
$env:NGC_API_KEY = "<key>"
docker compose up --force-recreate
# wait for: [global] [info] Running...

# Test WAV path (Terminal B, from D:\A2F)
cd D:\A2F
python -m backend.audio2face.test_audio --wav "Audio2Face-3D-Samples/example_audio/Mark_neutral.wav" --config "Audio2Face-3D-Samples/scripts/audio2face_3d_microservices_interaction_app/config/config_james.yml"

# Test Piper TTS path
python -m backend.audio2face.test_tts --text "Hello, this is a test." --config "Audio2Face-3D-Samples/scripts/audio2face_3d_microservices_interaction_app/config/config_james.yml"

# Regenerate test animation data for the avatar viewer
python -m backend.audio2face.save_animation --wav "Audio2Face-3D-Samples/example_audio/Mark_neutral.wav" --config "Audio2Face-3D-Samples/scripts/audio2face_3d_microservices_interaction_app/config/config_james.yml" --out "frontend/assets/test_animation.json"

# View the avatar (from D:\A2F\frontend)
cd D:\A2F\frontend
python -m http.server 8080
# open http://localhost:8080/avatar_viewer.html, click once on page for audio
```

---

## 5. Suggested order for next session
1. Confirm mouth animation is actually visible (2.1) — quick visual check, do this first
2. Rotate the 4 API keys (2.3) — 5 minutes, clear it off the list
3. Rewrite `receiver.py` for streaming (2.4.1) — unblocks everything else
4. Build the WebSocket bridge server (2.4.2)
5. Switch viewer to live mode and test (2.4.3)
6. Only after live mode works end-to-end: begin integration into main VRA/ARIA app (2.5)
