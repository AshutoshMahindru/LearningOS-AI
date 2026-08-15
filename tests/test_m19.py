"""Discovery bridge for the mission-local M19 unittest suite.

The repository's standard ``unittest discover -s tests`` command does not
recurse into namespace directories, so this mission-specific bridge exposes
the substantive tests kept under ``tests/missions``.
"""

from tests.missions.test_m19 import *  # noqa: F401,F403
