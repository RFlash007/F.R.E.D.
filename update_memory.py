import Episodic
import Semantic
import Dreaming
import os
from pathlib import Path

# Create synthetic_conversations directory if it doesn't exist
synthetic_dir = Path("synthetic_conversations")
if not synthetic_dir.exists():
    synthetic_dir.mkdir(exist_ok=True)
    print(f"Created {synthetic_dir} directory")

if __name__ == "__main__":
    # Standard memory consolidation
    Semantic.consolidate_semantic()
    Episodic.consolidate_episodic()
    Dreaming.consolidate_dreams()
    # Process dreams including synthetic conversations
    print("Memory update complete - real and synthetic conversations processed")
    

