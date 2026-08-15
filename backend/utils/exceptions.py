class Audio2FaceConnectionError(Exception):
    """Raised when the gRPC channel to Audio2Face cannot be opened."""
    pass


class Audio2FaceStreamError(Exception):
    """Raised when sending/receiving on the Audio2Face stream fails."""
    pass