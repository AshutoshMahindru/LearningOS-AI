"""Mission-specific repository tests discoverable by unittest.

When unittest uses ``tests`` as its discovery root, this package is imported as
``missions``. Extend its package path to the repository's mission artifacts so
tests can import mission-local implementation modules without duplicating them.
"""

from pathlib import Path


_SOURCE_MISSIONS = Path(__file__).resolve().parents[2] / "missions"
if str(_SOURCE_MISSIONS) not in __path__:
    __path__.append(str(_SOURCE_MISSIONS))
