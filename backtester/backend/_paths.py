
import os
import sys

BACKEND = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BACKEND)
STRATEGIES = os.path.join(ROOT, 'strategies')

for _p in (ROOT, BACKEND, STRATEGIES):
    if _p not in sys.path:
        sys.path.insert(0, _p)
