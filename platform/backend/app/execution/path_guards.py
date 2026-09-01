"""Path and import guards for in-process learner exec (aligned with 31B).

Writes are confined to the job workdir. Git worktree paths, $LEARNINGOS_HOME
database files (including SQLite WAL/SHM), and banned imports are refused.
"""

from __future__ import annotations

import builtins as builtins_mod
import io as io_mod
import os
import types
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

ALLOWED_MODULES = frozenset(
    {
        "abc",
        "array",
        "base64",
        "binascii",
        "bisect",
        "calendar",
        "cmath",
        "collections",
        "collections.abc",
        "contextlib",
        "copy",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "fractions",
        "functools",
        "hashlib",
        "heapq",
        "hmac",
        "io",
        "itertools",
        "json",
        "math",
        "numbers",
        "operator",
        "pprint",
        "random",
        "re",
        "statistics",
        "string",
        "struct",
        "textwrap",
        "time",
        "types",
        "typing",
        "unicodedata",
        "warnings",
    }
)

BANNED_MODULES = frozenset(
    {
        "_thread",
        "_posixsubprocess",
        "asyncio",
        "builtins",
        "code",
        "codeop",
        "concurrent",
        "ctypes",
        "fcntl",
        "ftplib",
        "http",
        "importlib",
        "inspect",
        "mmap",
        "multiprocessing",
        "os",
        "pathlib",
        "pickle",
        "pkgutil",
        "posix",
        "pty",
        "requests",
        "resource",
        "runpy",
        "shelve",
        "shutil",
        "signal",
        "site",
        "smtplib",
        "socket",
        "sqlite3",
        "ssl",
        "subprocess",
        "sys",
        "syslog",
        "telnetlib",
        "termios",
        "threading",
        "tty",
        "urllib",
        "webbrowser",
        "xmlrpc",
        "_io",
    }
)

ALLOWED_BUILTIN_NAMES = frozenset(
    {
        "ArithmeticError",
        "AssertionError",
        "AttributeError",
        "BaseException",
        "EOFError",
        "Ellipsis",
        "Exception",
        "False",
        "FileNotFoundError",
        "FloatingPointError",
        "GeneratorExit",
        "ImportError",
        "IndentationError",
        "IndexError",
        "IsADirectoryError",
        "KeyError",
        "LookupError",
        "MemoryError",
        "NameError",
        "None",
        "NotADirectoryError",
        "NotImplemented",
        "NotImplementedError",
        "OSError",
        "OverflowError",
        "PermissionError",
        "RuntimeError",
        "StopIteration",
        "SyntaxError",
        "SystemExit",
        "TimeoutError",
        "True",
        "TypeError",
        "UnboundLocalError",
        "UnicodeError",
        "ValueError",
        "ZeroDivisionError",
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "bytearray",
        "bytes",
        "callable",
        "chr",
        "classmethod",
        "complex",
        "dict",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "getattr",
        "hasattr",
        "hash",
        "hex",
        "id",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "object",
        "oct",
        "ord",
        "pow",
        "print",
        "property",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "setattr",
        "slice",
        "sorted",
        "staticmethod",
        "str",
        "sum",
        "super",
        "tuple",
        "type",
        "zip",
    }
)

_REAL_OPEN = builtins_mod.open
_REAL_IMPORT = builtins_mod.__import__
_REAL_FILEIO = io_mod.FileIO
_REAL_OPEN_CODE = getattr(io_mod, "open_code", None)
_IO_WRITE_CTORS = frozenset({"open", "FileIO", "open_code"})
_DB_NAME = "learningos.db"
_DB_SIDECARS = (".db-wal", ".db-shm", ".db-journal", "-wal", "-shm")


class SandboxViolation(PermissionError):
    """Learner code attempted a blocked path or import."""


def detect_repo_root(start: Path | None = None) -> Path | None:
    origin = start if start is not None else Path(__file__).resolve()
    for candidate in origin.parents:
        has_platform = (candidate / "platform" / "backend" / "app").is_dir()
        has_git = (candidate / ".git").exists() or (candidate / ".git").is_file()
        has_arch = (candidate / "architecture" / "learningos-v3").is_dir()
        if has_platform and (has_git or has_arch):
            return candidate
    return None


def data_home_from_env(raw: str | None = None) -> Path | None:
    text = raw if raw is not None else os.environ.get("LEARNINGOS_HOME")
    if not text:
        return None
    return Path(text).expanduser().resolve()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def _is_protected_db_path(resolved: Path, data_home: Path | None) -> bool:
    name = resolved.name
    if name == _DB_NAME or name.startswith(_DB_NAME + "-"):
        return True
    lower = name.lower()
    if data_home is None:
        return False
    try:
        home = data_home.resolve()
    except OSError:
        return False
    if resolved.parent != home:
        return False
    if lower.endswith(".db") or any(lower.endswith(suffix) for suffix in _DB_SIDECARS):
        return True
    return False


def assert_allowed_path(
    raw: str | os.PathLike[str],
    *,
    workdir: Path,
    repo_root: Path | None,
    data_home: Path | None,
    writing: bool,
) -> Path:
    text = os.fspath(raw)
    if not text:
        raise SandboxViolation("empty path is not allowed")
    candidate = Path(text)
    if ".." in candidate.parts:
        raise SandboxViolation("path traversal rejected")
    try:
        resolved = candidate.resolve() if candidate.is_absolute() else (workdir / candidate).resolve()
    except OSError as exc:
        raise SandboxViolation(f"path resolve failed: {exc}") from exc
    if _is_protected_db_path(resolved, data_home):
        raise SandboxViolation("refusing LEARNINGOS_HOME database path")
    if writing and repo_root is not None and _is_within(resolved, repo_root):
        raise SandboxViolation("refusing write into the Git worktree")
    if not _is_within(resolved, workdir):
        raise SandboxViolation("path is outside the job workdir")
    return resolved


def _mode_writes(mode: object) -> bool:
    return any(flag in str(mode) for flag in ("w", "a", "x", "+"))


class PathGuards:
    """Session-scoped open/import policy. Does not leak process-global patches."""

    def __init__(
        self,
        *,
        workdir: Path,
        repo_root: Path | None,
        data_home: Path | None,
    ) -> None:
        self.workdir = workdir.resolve()
        self.repo_root = repo_root.resolve() if repo_root is not None else None
        self.data_home = data_home.resolve() if data_home is not None else None
        self.safe_open = self._make_safe_open()
        self.safe_fileio = self._make_safe_fileio()
        self.safe_open_code = self._make_safe_open_code()
        self.safe_import = self._make_safe_import()

    def _make_safe_open(self) -> Callable[..., Any]:
        workdir = self.workdir
        repo_root = self.repo_root
        data_home = self.data_home

        def safe_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            if isinstance(file, int):
                raise SandboxViolation("opening raw file descriptors is not allowed")
            assert_allowed_path(
                file,
                workdir=workdir,
                repo_root=repo_root,
                data_home=data_home,
                writing=_mode_writes(mode),
            )
            return _REAL_OPEN(file, mode, *args, **kwargs)

        safe_open.__name__ = "open"
        return safe_open

    def _guard_file_ctor(self, file: Any, mode: object) -> None:
        if isinstance(file, int):
            raise SandboxViolation("opening raw file descriptors is not allowed")
        assert_allowed_path(
            file,
            workdir=self.workdir,
            repo_root=self.repo_root,
            data_home=self.data_home,
            writing=_mode_writes(mode),
        )

    def _make_safe_fileio(self) -> type:
        guards = self

        class GuardedFileIO:
            """Path-checked stand-in for io.FileIO. Does not subclass (C __new__ opens first)."""

            def __new__(cls, file: Any, mode: str = "r", closefd: bool = True, opener: Any = None) -> Any:
                if opener is not None:
                    raise SandboxViolation("custom FileIO opener is not allowed")
                guards._guard_file_ctor(file, mode)
                return _REAL_FILEIO(file, mode, closefd=closefd, opener=None)

        GuardedFileIO.__name__ = "FileIO"
        GuardedFileIO.__qualname__ = "FileIO"
        GuardedFileIO.__module__ = "io"
        return GuardedFileIO

    def _make_safe_open_code(self) -> Callable[..., Any] | None:
        if _REAL_OPEN_CODE is None:
            return None

        def safe_open_code(path: Any) -> Any:
            self._guard_file_ctor(path, "r")
            return _REAL_OPEN_CODE(path)

        safe_open_code.__name__ = "open_code"
        return safe_open_code

    def _io_shim(self) -> Any:
        shim = types.ModuleType("io")
        for name in dir(io_mod):
            if name in _IO_WRITE_CTORS:
                continue
            try:
                setattr(shim, name, getattr(io_mod, name))
            except Exception:
                continue
        shim.open = self.safe_open  # type: ignore[method-assign]
        shim.FileIO = self.safe_fileio  # type: ignore[attr-defined]
        if self.safe_open_code is not None:
            shim.open_code = self.safe_open_code  # type: ignore[attr-defined]
        return shim

    def _make_safe_import(self) -> Callable[..., Any]:
        def safe_import(
            name: str,
            globals: Any = None,
            locals: Any = None,
            fromlist: Any = (),
            level: int = 0,
        ) -> Any:
            if level != 0:
                raise SandboxViolation("relative imports are not allowed")
            root = str(name).split(".", 1)[0]
            if root in BANNED_MODULES or root not in ALLOWED_MODULES:
                raise SandboxViolation(f"import of {name!r} is blocked")
            module = _REAL_IMPORT(name, globals, locals, fromlist, level)
            if root == "io":
                return self._io_shim()
            return module

        safe_import.__name__ = "__import__"
        return safe_import

    def restricted_builtins(self) -> dict[str, Any]:
        restricted: dict[str, Any] = {"__import__": self.safe_import, "open": self.safe_open}
        file_builtin = getattr(builtins_mod, "file", None)
        if file_builtin is not None:
            restricted["file"] = self.safe_open
        for name in ALLOWED_BUILTIN_NAMES:
            if hasattr(builtins_mod, name):
                restricted[name] = getattr(builtins_mod, name)
        return restricted

    @contextmanager
    def patch_process_opens(self) -> Iterator[None]:
        """Wrap process-global open/io.open/io.FileIO for the duration of learner exec."""
        previous_builtin = builtins_mod.open
        previous_io = io_mod.open
        previous_fileio = io_mod.FileIO
        previous_open_code = getattr(io_mod, "open_code", None)
        previous_file = getattr(builtins_mod, "file", None)
        builtins_mod.open = self.safe_open  # type: ignore[assignment]
        io_mod.open = self.safe_open  # type: ignore[method-assign]
        io_mod.FileIO = self.safe_fileio  # type: ignore[misc]
        if self.safe_open_code is not None:
            io_mod.open_code = self.safe_open_code  # type: ignore[attr-defined]
        if previous_file is not None:
            builtins_mod.file = self.safe_open  # type: ignore[attr-defined]
        try:
            yield
        finally:
            builtins_mod.open = previous_builtin
            io_mod.open = previous_io
            io_mod.FileIO = previous_fileio
            if previous_open_code is not None:
                io_mod.open_code = previous_open_code
            if previous_file is not None:
                builtins_mod.file = previous_file
