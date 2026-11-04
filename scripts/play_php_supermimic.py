"""Source-checkout entry point; also available as supermimic-play."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blinkmimic.php.play import main

if __name__ == "__main__":
    main()
