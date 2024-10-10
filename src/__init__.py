"""**src**

Template for Python Repositories
"""
__version__ = "0.0.1"

import logging
from dotenv import load_dotenv
from pathlib import Path

logger = logging.getLogger(__name__)


potential_dotenv = Path(__file__).parent / ".env"
if potential_dotenv.exists():
    logger.info(f"Loading environment variables from {potential_dotenv}")
    load_dotenv(potential_dotenv)
