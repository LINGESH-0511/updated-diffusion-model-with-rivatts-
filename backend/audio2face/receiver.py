"""
backend/audio2face/receiver.py

Phase 3: Read the response side of an open ProcessAudioStream() stream.

Responsibilities:
    - Receive AnimationDataStreamHeader (blendshape names, audio header)
    - Receive AnimationData (blendshape weights per frame, audio buffer, emotion metadata)
    - Receive status messages (SUCCESS / INFO / WARNING / ERROR)
    - Return structured results to the caller

Does NOT:
    - Open the gRPC channel (client.py's job)
    - Send audio (stream.py's job)
    - Save data to disk / wav / csv (test_audio.py's job, Phase 5)
"""

from dataclasses import dataclass, field

import grpc

from nvidia_ace.animation_data.v1_pb2 import AnimationData, AnimationDataStreamHeader
from nvidia_ace.audio.v1_pb2 import AudioHeader
from nvidia_ace.emotion_aggregate.v1_pb2 import EmotionAggregate

from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AnimationResult:
    """Structured result of a fully-consumed Audio2Face response stream."""
    bs_names: list = field(default_factory=list)
    animation_key_frames: list = field(default_factory=list)
    audio_buffer: bytes = b""
    audio_header: AudioHeader = None
    emotion_key_frames: dict = field(default_factory=lambda: {
        "input": [],
        "a2e_output": [],
        "a2f_smoothed_output": [],
    })
    status_code: int = None
    status_message: str = ""


def _parse_emotion_data(animation_data: AnimationData, emotion_key_frames: dict):
    """
    Unpacks the EmotionAggregate from AnimationData.metadata and appends
    input / a2e_output / a2f_smoothed_output key frames.
    """
    emotion_aggregate = EmotionAggregate()
    metadata = animation_data.metadata.get("emotion_aggregate")
    if metadata and metadata.Unpack(emotion_aggregate):
        for e in emotion_aggregate.a2e_output:
            emotion_key_frames["a2e_output"].append({
                "time_code": e.time_code,
                "emotion_values": dict(e.emotion),
            })
        for e in emotion_aggregate.input_emotions:
            emotion_key_frames["input"].append({
                "time_code": e.time_code,
                "emotion_values": dict(e.emotion),
            })
        for e in emotion_aggregate.a2f_smoothed_output:
            emotion_key_frames["a2f_smoothed_output"].append({
                "time_code": e.time_code,
                "emotion_values": dict(e.emotion),
            })


async def read_animation_stream(stream) -> AnimationResult:
    """
    Reads an open ProcessAudioStream() stream until EOF and returns the
    fully assembled AnimationResult.

    Args:
        stream: the bidirectional stream returned by
                Audio2FaceClient.get_stream() (client.py)

    Returns:
        AnimationResult with blendshape frames, audio buffer, emotion
        key frames, and the final status code/message.
    """
    result = AnimationResult()
    packet_count = 0

    while True:
        message = await stream.read()

        if message == grpc.aio.EOF:
            logger.info(f"Stream ended after {packet_count} packets")
            return result

        if message.HasField("animation_data_stream_header"):
            header: AnimationDataStreamHeader = message.animation_data_stream_header
            result.bs_names = list(header.skel_animation_header.blend_shapes)
            result.audio_header = header.audio_header
            logger.info(f"Received animation header ({len(result.bs_names)} blendshapes)")

        elif message.HasField("animation_data"):
            animation_data: AnimationData = message.animation_data
            _parse_emotion_data(animation_data, result.emotion_key_frames)

            for blendshapes in animation_data.skel_animation.blend_shape_weights:
                bs_values = dict(zip(result.bs_names, blendshapes.values))
                result.animation_key_frames.append({
                    "timeCode": blendshapes.time_code,
                    "blendShapes": bs_values,
                })

            result.audio_buffer += animation_data.audio.audio_buffer
            packet_count += 1

        elif message.HasField("status"):
            status = message.status
            result.status_code = status.code
            result.status_message = status.message
            status_names = {0: "SUCCESS", 1: "INFO", 2: "WARNING", 3: "ERROR"}
            logger.info(f"Status: {status_names.get(status.code, status.code)} - {status.message}")