"""Optional G3 seed helper. Curriculum identity is owned by the worker lane."""


def seed_database() -> None:
    """No-op. Do not insert dummy curriculum or git_commit_sha='HEAD'."""
    return None
