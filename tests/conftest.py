# Ensure `src/` is on sys.path so tests can import `event_feed_app` without requiring editable install
import os
import sys

HERE = os.path.dirname(__file__)
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
if os.path.isdir(SRC) and SRC not in sys.path:
    sys.path.insert(0, SRC)
