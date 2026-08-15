"""
Health check module for NVIDIA Audio2Face NIM.
"""

import socket
import requests

from backend.utils import (
    logger,
    HTTP_BASE_URL,
    A2F_GRPC_HOST,
    A2F_GRPC_PORT,
)


class Audio2FaceHealth:
    """
    Performs health checks for the Audio2Face HTTP and gRPC services.
    """

    @staticmethod
    def check_http(timeout: int = 3) -> bool:
        """
        Check whether the HTTP server is reachable.

        Even if the server returns 404, it is still considered
        reachable because the server responded.
        """

        try:
            response = requests.get(
                HTTP_BASE_URL,
                timeout=timeout,
            )

            logger.info(
                f"HTTP Server Reachable (Status {response.status_code})"
            )

            return True

        except requests.RequestException as e:
            logger.error(f"HTTP Connection Failed: {e}")
            return False

    @staticmethod
    def check_grpc(timeout: int = 3) -> bool:
        """
        Check whether the gRPC port is reachable.
        """

        try:
            with socket.create_connection(
                (A2F_GRPC_HOST, A2F_GRPC_PORT),
                timeout=timeout,
            ):
                logger.info("gRPC Server Reachable")
                return True

        except OSError as e:
            logger.error(f"gRPC Connection Failed: {e}")
            return False

    @staticmethod
    def check_all() -> bool:
        """
        Perform all health checks.
        """

        logger.info("Checking Audio2Face Services...")

        http_ok = Audio2FaceHealth.check_http()
        grpc_ok = Audio2FaceHealth.check_grpc()

        if http_ok and grpc_ok:
            logger.info("Audio2Face is READY")
            return True

        logger.error("Audio2Face is NOT READY")
        return False