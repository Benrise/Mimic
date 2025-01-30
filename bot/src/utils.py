import time
import logging

logger = logging.getLogger(__name__)

def humanSleep(message: str):
    delay_time = len(message) * 0.01
    logger.info(f"Applying delay: {delay_time} seconds")
    
    time.sleep(delay_time) 
    