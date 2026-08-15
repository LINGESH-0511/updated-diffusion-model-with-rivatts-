import asyncio
import websockets
import json

async def test():
    try:
        async with websockets.connect("ws://localhost:8765") as ws:
            req = {"type": "ask", "mode": "web", "question": "test"}
            print(f"Sending: {req}")
            await ws.send(json.dumps(req))
            
            while True:
                msg = await ws.recv()
                print(f"Received: {msg}")
                try:
                    data = json.loads(msg)
                    if data.get("type") in ("error", "all_done"):
                        if data.get("message") == "Web search answer could not be spoken (TTS/A2F failed on first chunk).":
                            break
                except Exception:
                    pass
    except Exception as e:
        print(f"WS Error: {e}")

asyncio.run(test())
