# ARIA — Real-Time Voice-Driven 3D Avatar System

**ARIA** is a real-time, voice-driven 3D avatar system combining lip-sync animation, emotional expressiveness, and RAG-based (Retrieval-Augmented Generation) question answering. It streams speech into NVIDIA's Audio2Face-3D (A2F) pipeline to drive a live, emotionally expressive 3D avatar face, with a Three.js frontend for rendering.

> 📄 Patent filed (Indian Patent Office) for the underlying RAG Voice Knowledge Assistant with LipSync approach.

---

## Features

- **Real-time lip sync** via NVIDIA Audio2Face-3D NIM (regression/v2.3 James network, diffusion stylization supported)
- **Text-to-Speech** using Piper (ONNX, `en_US-amy-medium`), single-voice with tunable prosody presets (`length_scale`, `noise_scale`, `noise_w`) mapped per emotion
- **Emotion-aware speech**: per-answer emotion classification (j-hartmann DistilRoBERTa) mapped to the 6-value A2F emotion vocabulary (`anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`), routed via gRPC to A2F
- **RAG mode**: Qdrant + fastembed for document-grounded Q&A (collection `a2f_documents`) — upload PDF/TXT/MD/DOCX in-browser
- **Web-search mode**: Groq `compound-mini` for live web-grounded answers with cited sources
- **Voice input**: hold-to-talk mic button → Groq Whisper STT → full ask pipeline
- **Document upload UI**: drag-and-drop document ingestion directly from the browser
- **WebSocket bridge server** (`backend/bridge_server.py`) streaming chunked audio + blendshape animation data to the frontend in real time
- **Stop button**: mid-speech interrupt with per-request ID tracking to prevent stale/cancelled responses leaking into new conversations
- **Single-file Three.js frontend** (`frontend/index.html`) — full chat UI, history sidebar, mode switcher, avatar viewer
- **Emotion persistence layer**: brow/eye emotion blendshapes are held across frames so lip-sync frames don't wipe facial expression on every tick

---

## Architecture Overview

```
User speech/text (mic or keyboard)
       │
       ▼
Browser (frontend/index.html) ── WebSocket ──►
       │
       ▼
Bridge Server (backend/bridge_server.py) — ws://localhost:8765
       │
       ├── RAG mode   ──► Qdrant (a2f_documents) + fastembed ──► Groq LLM (streaming)
       ├── Web mode   ──► Groq compound-mini (web search + answer)
       ├── Voice mode ──► Groq Whisper STT ──► RAG or Web pipeline
       │
       ▼
Emotion Classifier (j-hartmann/emotion-english-distilroberta-base)
       │
       ▼
Piper TTS (ONNX, en_US-amy-medium + per-emotion prosody presets)
       │
       ▼
Audio2Face-3D NIM (gRPC, port 52000) ──► 55 blendshapes + head rotation per frame
       │
       ▼
backend/audio2face/receiver_streaming.py ──► WebSocket frames ──►
       │
       ▼
frontend/index.html (Three.js + Web Audio API)
  • Decodes WAV, schedules playback via AudioContext
  • Drives blendshapes frame-by-frame in sync with audio timecodes
  • Procedural idle motion: head sway, eye saccades, blinking
```

---

## Project Structure

```
e:\A2F_Project\
├── backend/
│   ├── bridge_server.py          # Main WebSocket server — orchestrates everything
│   ├── websearch.py              # Groq compound-mini web-search integration
│   ├── audio2face/
│   │   ├── client.py             # Persistent gRPC channel to A2F NIM
│   │   ├── stream.py             # Sends PCM audio + emotion vectors to A2F
│   │   ├── receiver_streaming.py # Async generator: yields blendshape frames live
│   │   ├── receiver.py           # Batch variant (offline/test use)
│   │   ├── wav_sender.py         # YAML config loader + audio chunking
│   │   ├── save_animation.py     # Offline: saves test_animation.json for avatar viewer
│   │   ├── test_audio.py         # CLI test: WAV file → A2F → print blendshapes
│   │   └── test_tts.py           # CLI test: text → Piper → A2F → print blendshapes
│   ├── tts/
│   │   └── piper_streamer.py     # Piper ONNX TTS — single voice, per-emotion prosody
│   ├── rag/
│   │   ├── config.py             # RAG settings (Qdrant URL, collection name, etc.)
│   │   ├── ingest.py             # Document parsing + chunking + Qdrant upsert
│   │   ├── retrieve.py           # Hybrid search (dense + sparse via fastembed)
│   │   ├── generate.py           # Groq LLM streaming answer with [emotion] tag
│   │   ├── pipeline.py           # answer_query_stream() — ties retrieve + generate
│   │   └── stt.py                # Groq Whisper transcription
│   ├── emotion/
│   │   └── classifier.py         # DistilRoBERTa emotion classifier
│   └── utils/
│       ├── logger.py             # Centralised logger (stdout + bridge.log)
│       ├── config.py             # Env-var helpers (LOG_LEVEL, etc.)
│       └── exceptions.py         # Custom exception types
├── frontend/
│   ├── index.html                # Single-file Three.js avatar viewer + full chat UI
│   └── assets/
│       ├── avatar.glb            # ReadyPlayerMe avatar (Wolf3D_Head, 72 morph targets)
│       ├── test_animation.json   # Pre-baked blendshape frames for offline testing
│       └── test_animation_audio.wav
├── Audio2Face-3D-Samples/
│   ├── quick-start/              # docker-compose.yml for A2F NIM container
│   │   └── configs/
│   │       ├── james_stylization_config.yaml
│   │       └── james_diffusion_stylization_config.yaml
│   └── scripts/audio2face_3d_microservices_interaction_app/config/
│       └── config_james.yml      # A2F NIM config — blendshape multipliers, emotion weights
├── voices/
│   └── neutral/
│       ├── en_US-amy-medium.onnx
│       └── en_US-amy-medium.onnx.json
├── docker-compose.qdrant.yml     # Qdrant vector DB container
├── requirements.txt              # Core Python deps
├── requirements-rag.txt          # RAG-specific deps (Qdrant, fastembed, Groq, etc.)
├── diagnostic_emotion_test.py    # Standalone emotion diagnostic script
├── test_emotion.py               # Quick emotion classifier unit test
├── test_a2f_stream.py            # Quick A2F streaming smoke test
└── .env.example                  # Template for environment variables (copy → .env)
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **OS** | Windows 10/11 |
| **GPU** | NVIDIA RTX 4050 Laptop GPU (6 GB VRAM) or better |
| **GPU Drivers** | Recent NVIDIA drivers with CUDA support |
| **Docker Desktop** | WSL2 backend enabled — required for Qdrant + A2F NIM containers |
| **Python 3.10** | Via virtual environment at `E:\a2f_env` (or your own venv) |
| **Piper voice model** | `voices/neutral/en_US-amy-medium.onnx` + `.onnx.json` |
| **NGC API Key** | From [build.nvidia.com](https://build.nvidia.com) — for A2F NIM image pull |
| **Groq API Key** | From [console.groq.com](https://console.groq.com) — for LLM + STT + web search |

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your keys:

```env
GROQ_API_KEY=gsk_...
NVIDIA_API_KEY=nvapi-...
NGC_API_KEY=nvapi-...
A2F_GRPC_HOST=127.0.0.1
A2F_GRPC_PORT=52000
A2F_HTTP_HOST=127.0.0.1
A2F_HTTP_PORT=8000
LOG_LEVEL=INFO
```

> ⚠️ **Never commit your real `.env` file.** It is listed in `.gitignore`.

---

## Installation

### 1. Clone the repo

```powershell
git clone https://github.com/LINGESH-0511/pep_pc_a2f_diffusion.git
cd pep_pc_a2f_diffusion
```

### 2. Create Python virtual environment and install deps

```powershell
python -m venv E:\a2f_env
E:\a2f_env\Scripts\activate

pip install -r requirements.txt
pip install -r requirements-rag.txt
pip install piper-tts
```

### 3. Download Piper voice model

Download `en_US-amy-medium.onnx` and `en_US-amy-medium.onnx.json` from [Piper releases](https://github.com/rhasspy/piper/releases) and place them in:

```
voices/neutral/en_US-amy-medium.onnx
voices/neutral/en_US-amy-medium.onnx.json
```

### 4. Pull the Audio2Face-3D NIM Docker image

```powershell
docker login nvcr.io --username '$oauthtoken' --password <NGC_API_KEY>
cd Audio2Face-3D-Samples\quick-start
$env:NGC_API_KEY = "<your NGC key>"
$env:A2F_3D_MODEL_NAME = "james"
docker compose pull
```

---

## Running the Project

Run each block in its **own terminal**, **in order**. Terminals 1 and 2 must be healthy before Terminal 3 starts, since `bridge_server.py` warm-ups check both on launch.

### Terminal 1 — Qdrant (Docker)

```powershell
docker start qdrant
```

First-time setup (if container doesn't exist yet):

```powershell
docker compose -f docker-compose.qdrant.yml up -d
```

Verify it's healthy:

```powershell
curl -UseBasicParsing http://localhost:6333/collections
# Should return 200 with {"result":{"collections":[...]}}
```

### Terminal 2 — Audio2Face-3D NIM (Docker)

```powershell
docker start quick-start-a2f-3d-service-1
docker logs -f quick-start-a2f-3d-service-1
```

Wait until logs settle to:
```
[global] [info] Running...
```

Then `Ctrl+C` to detach from logs (container keeps running).

**First-time start** (or after a `docker compose down`):

```powershell
cd Audio2Face-3D-Samples\quick-start
$env:A2F_3D_MODEL_NAME = "james"
$env:NGC_API_KEY = "<your NGC key>"
docker compose up --force-recreate
```

> ⚠️ Do **not** run `docker start a2f-nim` — that is an old container that conflicts on port `52000`. Leave it stopped.

### Terminal 3 — Bridge Server (main backend)

```powershell
# Kill any stale process on port 8765 first:
netstat -ano | findstr :8765
# taskkill /PID <pid> /F   (if one shows up)

E:\a2f_env\Scripts\python.exe -m backend.bridge_server
```

Confirm in logs:
- `Warm-up: Piper voice loaded`
- `Warm-up: A2F real inference #1/#2` with frame counts
- `Warm-up: RAG embedding models loaded`
- No `RAG/Qdrant startup check FAILED` line
- `WebSocket bridge starting on ws://localhost:8765`

### Terminal 4 — Frontend

```powershell
cd e:\A2F_Project\frontend
python -m http.server 8080
```

Open in browser:
```
http://localhost:8080/
```

Click once on the page to unblock audio autoplay, then start talking or typing.

---

## End-to-End Test

1. Select **Document Mode** or **Web Search** from the mode dropdown
2. **(Document mode)** Click `+` to upload a PDF/TXT/MD/DOCX file
3. Type a question or hold the mic button to speak
4. Confirm:
   - Response text streams into the chat panel
   - Audio plays from the avatar
   - Avatar lip-syncs in real time
   - Facial expression reflects the detected emotion (joy → smile, sadness → brow-down, etc.)
   - Stop button correctly halts mid-speech and returns avatar to neutral

### Quick pipeline smoke tests (without browser)

```powershell
# Test WAV → A2F blendshapes
E:\a2f_env\Scripts\python.exe -m backend.audio2face.test_audio ^
  --wav "Audio2Face-3D-Samples/example_audio/Mark_neutral.wav" ^
  --config "Audio2Face-3D-Samples/scripts/audio2face_3d_microservices_interaction_app/config/config_james.yml"

# Test Piper TTS → A2F blendshapes
E:\a2f_env\Scripts\python.exe -m backend.audio2face.test_tts ^
  --text "Hello, this is a test of the avatar pipeline." ^
  --config "Audio2Face-3D-Samples/scripts/audio2face_3d_microservices_interaction_app/config/config_james.yml"

# Regenerate offline test animation data for the avatar viewer
E:\a2f_env\Scripts\python.exe -m backend.audio2face.save_animation ^
  --wav "Audio2Face-3D-Samples/example_audio/Mark_neutral.wav" ^
  --config "Audio2Face-3D-Samples/scripts/audio2face_3d_microservices_interaction_app/config/config_james.yml" ^
  --out "frontend/assets/test_animation.json"
```

---

## Key Design Decisions

### TTS Stack History
`CosyVoice2` (RTF 4.68–8.72, too slow) → `Chatterbox-Turbo` → **Piper ONNX** (current, lowest latency). Multi-voice switching (Amy/Lessac/Kristin/Danny) was tested and reverted — a single consistent voice with per-emotion prosody presets sounds more natural.

### Emotion Pipeline
- **LLM tags** (`[friendly]`, `[caring]`, `[professional]`, `[neutral]`) drive TTS prosody preset and remap onto A2F's 6-emotion facial vocabulary
- **Classifier fallback** (DistilRoBERTa) runs on sentences with no explicit tag (used in the `speak` path)
- **One emotion per answer** (RAG + web search): a single leading `[emotion]` tag is parsed from the full answer, applied uniformly to every chunk — consistent facial expression across the whole response
- **A2F facial vocabulary**: `anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness` — always full 6-key vector, never partial dict

### Blendshape Tuning (config_james.yml)
Key multipliers tuned from diagnostic testing:
- `MouthSmileLeft/Right: 1.3` (raised from 1.0 — small signal needs boost)
- `MouthFrownLeft/Right: 1.2`
- `BrowInnerUp: 1.8`, `BrowDownLeft/Right: 1.6` (strongest emotion signal)

### Frontend Emotion Persistence Layer
Brow/eye emotion shapes are captured into `_currentEmotionValues` and re-applied every render tick *after* lip-sync shapes. This prevents lip-sync frames (which carry near-zero brow values) from wiping the facial expression on every tick.

### Docker Networking Fix
`network_mode: "host"` in `docker-compose.yml` **does not work on Docker Desktop for Windows** — host networking binds to the WSL2 VM, not `127.0.0.1`. The fix is explicit port mapping: `52000:52000`, `50000:50000`, `51000:51000`. If A2F connection errors recur, check this first.

---

## Known Issues / Pending Items

- [ ] Apply `inference_type: diffusion` in `james_diffusion_stylization_config.yaml` (highest-priority pending config change — files present, not yet hot-switched)
- [ ] Wire `HeadRoll`/`HeadPitch`/`HeadYaw` to the avatar's head bone (cosmetic, not blocking)
- [ ] Rotate API keys (`GROQ_API_KEY`, `NVIDIA_API_KEY`, `NGC_API_KEY`) — flagged as exposed in prior session
- [ ] RAG-mode prompt emotion tags in `generate.py` should be verified against `_VALID_EMOTIONS` in `bridge_server.py` to avoid silent fallback to `neutral`

---

## Common Issues

| Symptom | Likely Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'riva'` | `bridge_server.py` imports old `riva_streamer` | Import is now `piper_streamer` — pull latest |
| `Cannot import 'piper.PiperVoice'` | Wrong Python env used | Run with `E:\a2f_env\Scripts\python.exe` explicitly |
| `Connection refused` on port 52000 | A2F NIM not running or host-networking bug | Start container; verify `ports:` mapping, not `network_mode: host` |
| `conda : not recognized` | Conda not in PATH for this shell | Use venv directly: `E:\a2f_env\Scripts\activate` |
| 404 on `http://localhost:8080/avatar_viewer.html` | File renamed to `index.html` | Navigate to `http://localhost:8080/` |
| TTS/A2F fails on first chunk | Wrong Python env | Use `E:\a2f_env\Scripts\python.exe -m backend.bridge_server` |

---

## License / Patent

This project includes technology covered by a filed **Indian Patent Application** — *RAG Voice Knowledge Assistant with LipSync*. Contact the authors before external reuse or distribution.