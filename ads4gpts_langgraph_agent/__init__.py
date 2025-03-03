import logging

# Configure package-level logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Stream handler for logging
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

from .agent import make_ads4gpts_langgraph_agent

__all__ = ["make_ads4gpts_langgraph_agent"]
