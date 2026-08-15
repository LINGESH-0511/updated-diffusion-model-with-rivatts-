from utils import logger
from utils import HTTP_BASE_URL
from utils import GRPC_TARGET

logger.info("Configuration Loaded Successfully")

print()
print("HTTP :", HTTP_BASE_URL)
print("gRPC :", GRPC_TARGET)