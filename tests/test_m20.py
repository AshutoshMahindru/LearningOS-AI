"""Discovery bridge for the mission-local M20 unittest suite.

The repository's standard ``unittest discover -s tests`` command does not recurse
into namespace directories, so this bridge exposes the substantive tests kept under
``tests/missions``. The M20 tests use only the standard library in the bare workflow;
notebook dependencies are exercised separately in the full mission environment.
"""

from tests.missions.test_m20 import *  # noqa: F401,F403
