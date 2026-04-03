"""Logging configuration for hand-to-tex project."""

import sys

from loguru import logger

# Remove default handler
logger.remove()

# Add custom handler with format
logger.add(
    sys.stderr,
    format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)

# Optional: Add file logging
# logger.add(
#     "logs/hand_to_tex.log",
#     format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
#     level="DEBUG",
#     rotation="500 MB",
#     retention="10 days",
# )

__all__ = ["logger"]
