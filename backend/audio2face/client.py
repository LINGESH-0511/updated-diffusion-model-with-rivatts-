"""
backend/audio2face/client.py

Phase 1: Open the gRPC channel to the local Audio2Face NIM and expose
the bidirectional ProcessAudioStream() stream.

Responsibilities:
    - Open an insecure gRPC channel to the local Audio2Face NIM
    - Create an A2FControllerServiceStub
    - Expose ProcessAudioStream() via get_stream()
    - Usable as an async context manager (async with Audio2FaceClient())

Does NOT:
    - Build/send audio or config (stream.py's job)
    - Read animation data back (receiver.py's job)

--- PERSISTENT CHANNEL (perf fix) ---------------------------------
Opening a gRPC channel is expensive: TCP connect + HTTP/2 handshake +
the NIM's own session setup. The original version of this file opened
a brand new channel every time `async with Audio2FaceClient()` was
used — and bridge_server.py calls that once PER SENTENCE CHUNK, so
every chunk was paying full channel-setup cost from scratch.

Fix: a module-level shared channel is opened once (lazily, on first
use) and reused for the lifetime of the process. `Audio2FaceClient()`
now hands out a stream on that shared channel by default instead of
creating a new channel. Opening a new *stream* on an already-open
channel is cheap — that's the normal, fast gRPC usage pattern.

Old call sites (`async with Audio2FaceClient() as client: ...`) do not
need to change — the persistent behavior is the default.
--------------------------------------------------------------------
"""

import asyncio

import grpc
from nvidia_ace.services.a2f_controller.v1_pb2_grpc import A2FControllerServiceStub
from backend.utils.config import GRPC_TARGET
from backend.utils.logger import get_logger
from backend.utils.exceptions import Audio2FaceConnectionError

logger = get_logger(__name__)

# ── Shared/persistent channel state (module-level singleton) ──────────
_shared_channel: "grpc.aio.Channel | None" = None
_shared_stub: "A2FControllerServiceStub | None" = None
_channel_lock = asyncio.Lock()


async def get_shared_stub(target: str = GRPC_TARGET) -> A2FControllerServiceStub:
    """
    Returns the process-wide A2FControllerServiceStub, opening the
    underlying gRPC channel on first call only. Every subsequent call
    (i.e. every chunk, every request) reuses the same warm channel.
    """
    global _shared_channel, _shared_stub
    if _shared_stub is not None:
        return _shared_stub

    async with _channel_lock:
        # Re-check inside the lock — another task may have won the race.
        if _shared_stub is None:
            try:
                logger.info(f"Opening PERSISTENT gRPC channel to {target}")
                _shared_channel = grpc.aio.insecure_channel(target)
                _shared_stub = A2FControllerServiceStub(_shared_channel)
            except Exception as e:
                raise Audio2FaceConnectionError(
                    f"Failed to open gRPC channel to {target}: {e}"
                ) from e
    return _shared_stub


async def close_shared_channel():
    """Call this once, on clean server shutdown, to release the channel."""
    global _shared_channel, _shared_stub
    if _shared_channel is not None:
        logger.info("Closing persistent gRPC channel")
        await _shared_channel.close()
        _shared_channel = None
        _shared_stub = None


class Audio2FaceClient:
    """
    Async context manager exposing the ProcessAudioStream() stub call.

    By default (persistent=True) this reuses the process-wide shared
    channel — entering/exiting the context is then near-instant, since
    no new channel is opened or closed. Pass persistent=False to get
    the old behavior (a fresh channel per use) — useful for one-off
    scripts (test_audio.py, save_animation.py) where you don't want a
    lingering channel after the script exits.

    Usage:
        async with Audio2FaceClient() as client:
            stream = client.get_stream()
            ...
    """

    def __init__(self, target: str = GRPC_TARGET, persistent: bool = True):
        self._target = target
        self._persistent = persistent
        self._channel = None
        self._stub = None

    async def __aenter__(self) -> "Audio2FaceClient":
        if self._persistent:
            self._stub = await get_shared_stub(self._target)
            return self

        # Non-persistent path — original per-use channel behavior.
        try:
            logger.info(f"Opening gRPC channel to {self._target}")
            self._channel = grpc.aio.insecure_channel(self._target)
            self._stub = A2FControllerServiceStub(self._channel)
            return self
        except Exception as e:
            raise Audio2FaceConnectionError(
                f"Failed to open gRPC channel to {self._target}: {e}"
            ) from e

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._persistent:
            # Shared channel outlives this context — nothing to close.
            return
        if self._channel is not None:
            logger.info("Closing gRPC channel")
            await self._channel.close()

    def get_stream(self):
        """
        Returns the bidirectional ProcessAudioStream() stream.
        Must be called after entering the async context manager.
        """
        if self._stub is None:
            raise Audio2FaceConnectionError(
                "Client is not connected. Use 'async with Audio2FaceClient()'."
            )
        return self._stub.ProcessAudioStream()