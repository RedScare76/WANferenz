import os
from pathlib import Path
import sys

REPOSITORY_HOME = str(Path(__file__).resolve().parent)
if REPOSITORY_HOME not in sys.path:
    sys.path.insert(0, REPOSITORY_HOME)
search_paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
if REPOSITORY_HOME not in search_paths:
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [REPOSITORY_HOME, *filter(None, search_paths)]
    )
os.environ.setdefault("OMP_NUM_THREADS", "1")
