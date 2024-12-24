# Simplified shared infrastructure
import logging

# Create a simple logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_info(message):
    logger.info(message)
    
def log_error(message):
    logger.error(message)
