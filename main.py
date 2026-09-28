#!/usr/bin/env python3
import sys
from pathlib import Path

# Add src to sys.path when running unpackaged directly from repository
sys.path.insert(0, str(Path(__file__).parent / "src"))

from moji.main import main

if __name__ == "__main__":
    main()
