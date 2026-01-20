from __future__ import annotations

import os
import sys

from rich.traceback import install

# Enable readable tracebacks in development / test environments.
# Can be disabled with PYTEST_RICH=0
if os.getenv("PYTEST_RICH", "1") == "1":
    install(
        show_locals=True,  # show local variables for each frame
        width=None,  # use terminal width
        word_wrap=True,  # wrap long lines
        extra_lines=1,  # some context around lines
        suppress=["/usr/lib/python3", "site-packages"],  # hide "noisy" third-party frames
    )


# Ensure `src` is on sys.path so imports like `from rag2f.core...` resolve during tests.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if os.path.isdir(SRC):
    sys.path.insert(0, SRC)
sys.path.insert(0, ROOT)
