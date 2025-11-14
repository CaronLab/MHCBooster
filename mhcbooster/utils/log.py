
from pathlib import Path
from datetime import datetime


def append_log(message, log_path, include_timestamp=True):
    log_entry = message
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"

    # Create directory if it doesn't exist
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Append message to file
    with log_path.open('a', encoding='utf-8') as f:
        f.write(log_entry + '\n')

def read_from_log(key, log_path):
    message = None
    with open(log_path, 'r') as f:
        for line in f:
            if key in line:
                return line.split(key)[-1].strip()
    return message
