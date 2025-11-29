import logging
import os
from datetime import datetime

#logs directory exists
os.makedirs("logs", exist_ok=True)

#format: logs/agent_run_20231025_120000.log
LOG_FILE = f'logs/agent_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

def log_event(step: str, message: str) -> list[str]:
    """
    Logs an event to console and file, returns it for the State.
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] [{step.upper()}] {message}"
    
    # 1. Console (Visual feedback)
    print(f"\n🔹 {formatted_msg}")
    
    # 2. File (Permanent record)
    logging.info(message)
    
    # 3. State (Reasoning trace)
    return [formatted_msg]