from pathlib import Path

# Ensure one or more directories exist. 
def ensure_dirs(*paths: str):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
