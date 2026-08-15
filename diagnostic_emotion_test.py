"""
diagnostic_emotion_test.py

Run this from D:\\A2F to test the entire emotion pipeline end-to-end
WITHOUT needing the browser/frontend at all.

It sends a real audio chunk to A2F with a specific emotion and prints
exactly what blendshape values come back — so you can see in the terminal
whether the emotion is actually changing the face or not.

Usage:
    conda activate a2f
    cd D:\\A2F
    python diagnostic_emotion_test.py

What it tests (in order):
    1. Can we connect to A2F gRPC at all?
    2. Does neutral emotion produce blendshape frames?
    3. Does joy emotion produce DIFFERENT blendshape values than neutral?
    4. Prints a side-by-side comparison of key emotion blendshapes
       (BrowInnerUp, MouthSmileLeft, MouthSmileRight, CheekSquintLeft etc.)
       so you can see with your own eyes whether A2F is responding to emotion.
"""

import asyncio
import os
import sys
import numpy as np

if os.name == "nt":
    os.add_dll_directory(r"D:\Users\linge\miniconda3\envs\a2f\Lib\site-packages\nvidia\cudnn\bin")
    os.add_dll_directory(r"D:\Users\linge\miniconda3\envs\a2f\Lib\site-packages\nvidia\cublas\bin")

# --------------------------------------------------------------------------
# CONFIG — adjust if your paths differ
# --------------------------------------------------------------------------
CONFIG_PATH = (
    "Audio2Face-3D-Samples/scripts/"
    "audio2face_3d_microservices_interaction_app/config/config_james.yml"
)
SAMPLE_RATE = 22050   # Piper's output rate
DURATION_S  = 2       # seconds of synthetic audio to send
# --------------------------------------------------------------------------

# Emotion-specific blendshapes we care about for the visual check.
# These are the ones that SHOULD differ between emotions.
EMOTION_BLENDSHAPES = [
    # Joy / smile
    "MouthSmileLeft", "MouthSmileRight",
    "CheekSquintLeft", "CheekSquintRight",
    "BrowInnerUp",
    # Sadness
    "MouthFrownLeft", "MouthFrownRight",
    "BrowDownLeft", "BrowDownRight",
    # Anger
    "NoseSneerLeft", "NoseSneerRight",
    # All emotions affect jaw for speaking
    "JawOpen",
]


def make_sine_pcm(duration_s: int, sample_rate: int, freq_hz: float = 220.0) -> bytes:
    """
    Generate a simple sine-wave tone as int16 PCM.
    This gives A2F real audio to process (silence produces no lip movement
    and no emotion inference — we need actual audio energy).
    """
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    wave = (np.sin(2 * np.pi * freq_hz * t) * 16000).astype(np.int16)
    return wave.tobytes()


async def run_emotion_test(emotion_label: str, preferred_emotion_dict, config: dict):
    """
    Send audio to A2F with a given preferred_emotion, collect all blendshape
    frames, and return averaged values for the emotion-relevant shapes.
    """
    from backend.audio2face.client import Audio2FaceClient
    from backend.audio2face.stream import send_audio

    pcm_bytes = make_sine_pcm(DURATION_S, SAMPLE_RATE)
    audio_data = np.frombuffer(pcm_bytes, dtype=np.int16)

    print(f"\n{'='*60}")
    print(f"  TESTING EMOTION: {emotion_label.upper()}")
    print(f"  preferred_emotion dict sent: {preferred_emotion_dict}")
    print(f"{'='*60}")

    frame_values = {k: [] for k in EMOTION_BLENDSHAPES}
    total_frames = 0
    bs_names = []

    import grpc
    from nvidia_ace.animation_data.v1_pb2 import AnimationData, AnimationDataStreamHeader

    async with Audio2FaceClient() as client:
        stream = client.get_stream()

        send_task = asyncio.create_task(
            send_audio(stream, SAMPLE_RATE, audio_data, config,
                       preferred_emotion=preferred_emotion_dict)
        )

        while True:
            message = await stream.read()
            if message == grpc.aio.EOF:
                break

            if message.HasField("animation_data_stream_header"):
                header = message.animation_data_stream_header
                bs_names = list(header.skel_animation_header.blend_shapes)
                print(f"  [HEADER] A2F returned {len(bs_names)} blendshapes")
                if not bs_names:
                    print("  [ERROR] EMPTY blendshape list in header — A2F config issue!")

            elif message.HasField("animation_data"):
                animation_data = message.animation_data
                for bsw in animation_data.skel_animation.blend_shape_weights:
                    bs_dict = dict(zip(bs_names, bsw.values))
                    total_frames += 1
                    for key in EMOTION_BLENDSHAPES:
                        # Try both exact case and lowercase match
                        val = bs_dict.get(key) or bs_dict.get(key.lower()) or 0.0
                        frame_values[key].append(val)

            elif message.HasField("status"):
                status = message.status
                code_names = {0: "SUCCESS", 1: "INFO", 2: "WARNING", 3: "ERROR"}
                print(f"  [STATUS] {code_names.get(status.code, status.code)}: {status.message}")

        try:
            await send_task
        except Exception as e:
            print(f"  [WARN] send_audio task error: {e}")

    print(f"  [FRAMES] Total frames received: {total_frames}")

    # Average each blendshape across all frames
    averages = {}
    for key in EMOTION_BLENDSHAPES:
        vals = frame_values[key]
        averages[key] = sum(vals) / len(vals) if vals else 0.0

    return averages, total_frames, bs_names


def print_comparison(neutral_avg: dict, joy_avg: dict, anger_avg: dict, sadness_avg: dict):
    """
    Print a side-by-side table of averaged blendshape values per emotion.
    If emotion is working, joy should have higher MouthSmile/CheekSquint,
    sadness higher MouthFrown/BrowDown, anger higher NoseSneer vs neutral.
    """
    print("\n")
    print("=" * 80)
    print("  BLENDSHAPE COMPARISON (averaged across all frames)")
    print("  If facial emotion is working, values MUST differ between columns.")
    print("=" * 80)
    print(f"  {'Blendshape':<22} {'NEUTRAL':>10} {'JOY':>10} {'SADNESS':>10} {'ANGER':>10}  VERDICT")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*10}  {'-'*20}")

    emotion_working = False

    for key in EMOTION_BLENDSHAPES:
        n = neutral_avg.get(key, 0.0)
        j = joy_avg.get(key, 0.0)
        s = sadness_avg.get(key, 0.0)
        a = anger_avg.get(key, 0.0)

        # Check if any emotion differs meaningfully from neutral
        max_diff = max(abs(j - n), abs(s - n), abs(a - n))
        if max_diff > 0.01:
            verdict = "✅ CHANGES WITH EMOTION"
            emotion_working = True
        else:
            verdict = "❌ flat (no change)"

        print(f"  {key:<22} {n:>10.4f} {j:>10.4f} {s:>10.4f} {a:>10.4f}  {verdict}")

    print("=" * 80)

    if emotion_working:
        print("\n  ✅ RESULT: Facial emotion IS working — blendshapes change with emotion.")
        print("     If the avatar still looks the same in the browser, the problem is")
        print("     in the frontend (avatar_viewer.html) not applying these channels.")
    else:
        print("\n  ❌ RESULT: Facial emotion is NOT working — all values identical.")
        print("     A2F is ignoring the preferred_emotion override entirely.")
        print("     Root cause is one of:")
        print("       1. config_james.yml: preferred_emotion_strength too low vs emotion_strength")
        print("       2. gRPC proto mismatch (emotion dict shape not what A2F expects)")
        print("       3. enable_preferred_emotion: false in the loaded config")
        print("\n  CHECK: Look above for the 'A2F EMOTION VECTOR (full 6-key)' log line.")
        print("         If you don't see it, stream.py changes didn't save/reload correctly.")

    print()


async def main():
    print("\n" + "="*60)
    print("  ARIA — A2F Facial Emotion Diagnostic")
    print("="*60)

    # --- Step 1: Load config
    print("\n[1] Loading A2F config...")
    try:
        from backend.audio2face.wav_sender import load_config
        config = load_config(CONFIG_PATH)
        pp = config.get("post_processing_parameters", {})
        print(f"    enable_preferred_emotion : {pp.get('enable_preferred_emotion')}")
        print(f"    preferred_emotion_strength: {pp.get('preferred_emotion_strength')}")
        print(f"    emotion_strength          : {pp.get('emotion_strength')}")

        if not pp.get("enable_preferred_emotion"):
            print("\n  ❌ CRITICAL: enable_preferred_emotion is False or missing!")
            print("     Fix config_james.yml first, then re-run this script.")
            sys.exit(1)

        pref_strength = float(pp.get("preferred_emotion_strength", 0))
        emo_strength  = float(pp.get("emotion_strength", 0))
        if pref_strength <= emo_strength:
            print(f"\n  ⚠️  WARNING: preferred_emotion_strength ({pref_strength}) <= "
                  f"emotion_strength ({emo_strength})")
            print("     A2E's own audio inference may dominate. Recommend:")
            print("     preferred_emotion_strength: 0.9")
            print("     emotion_strength: 0.3")
        else:
            print(f"    ✅ Strength values look good ({pref_strength} > {emo_strength})")

    except Exception as e:
        print(f"  ❌ Failed to load config: {e}")
        sys.exit(1)

    # --- Step 2: Test gRPC connection with neutral
    print("\n[2] Testing A2F gRPC connection (neutral)...")
    try:
        neutral_avg, neutral_frames, bs_names = await run_emotion_test(
            "neutral", None, config
        )
        if neutral_frames == 0:
            print("  ❌ CRITICAL: No frames received from A2F at all!")
            print("     Check that A2F NIM is running (Terminal 2) on port 52000.")
            sys.exit(1)
        print(f"  ✅ Connection OK — {neutral_frames} frames received")
    except Exception as e:
        print(f"  ❌ A2F gRPC connection failed: {e}")
        print("     Is A2F NIM running? Check Terminal 2.")
        sys.exit(1)

    # --- Step 3: Check blendshape name coverage
    print("\n[3] Checking blendshape name coverage...")
    bs_names_lower = {n.lower() for n in bs_names}
    missing = []
    for key in EMOTION_BLENDSHAPES:
        if key.lower() not in bs_names_lower:
            missing.append(key)
    if missing:
        print(f"  ⚠️  These emotion blendshapes are NOT in A2F's output: {missing}")
        print("     They will show as 0.0 in the comparison below.")
    else:
        print(f"  ✅ All {len(EMOTION_BLENDSHAPES)} emotion blendshapes present in A2F output")

    # --- Step 4: Test joy
    print("\n[4] Testing JOY emotion...")
    joy_emotion = {
        "anger": 0.0, "disgust": 0.0, "fear": 0.0,
        "joy": 1.0, "neutral": 0.0, "sadness": 0.0
    }
    try:
        joy_avg, joy_frames, _ = await run_emotion_test("joy", joy_emotion, config)
        print(f"  ✅ Joy test done — {joy_frames} frames")
    except Exception as e:
        print(f"  ❌ Joy test failed: {e}")
        joy_avg = {k: 0.0 for k in EMOTION_BLENDSHAPES}

    # --- Step 5: Test sadness
    print("\n[5] Testing SADNESS emotion...")
    sadness_emotion = {
        "anger": 0.0, "disgust": 0.0, "fear": 0.0,
        "joy": 0.0, "neutral": 0.0, "sadness": 1.0
    }
    try:
        sadness_avg, sadness_frames, _ = await run_emotion_test("sadness", sadness_emotion, config)
        print(f"  ✅ Sadness test done — {sadness_frames} frames")
    except Exception as e:
        print(f"  ❌ Sadness test failed: {e}")
        sadness_avg = {k: 0.0 for k in EMOTION_BLENDSHAPES}

    # --- Step 6: Test anger
    print("\n[6] Testing ANGER emotion...")
    anger_emotion = {
        "anger": 1.0, "disgust": 0.0, "fear": 0.0,
        "joy": 0.0, "neutral": 0.0, "sadness": 0.0
    }
    try:
        anger_avg, anger_frames, _ = await run_emotion_test("anger", anger_emotion, config)
        print(f"  ✅ Anger test done — {anger_frames} frames")
    except Exception as e:
        print(f"  ❌ Anger test failed: {e}")
        anger_avg = {k: 0.0 for k in EMOTION_BLENDSHAPES}

    # --- Step 7: Print comparison
    print_comparison(neutral_avg, joy_avg, anger_avg, sadness_avg)


if __name__ == "__main__":
    asyncio.run(main())