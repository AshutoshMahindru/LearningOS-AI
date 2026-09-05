#!/usr/bin/env python3
"""CLI for LearningOS provider-secret policy and SHA256SUMS verification.

Secrets stay server-side: macOS keychain when available, otherwise
``$LEARNINGOS_HOME/secrets/<NAME>`` mode 0600. Never writes into the Git
worktree, frontend, or VITE_ bindings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "platform" / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.secrets import (  # noqa: E402
    IntegrityError,
    SecretPolicyError,
    delete_secret,
    persist_secret,
    resolve_secret,
    secret_policy,
    verify_package_checksums,
)


def _read_value(explicit: str | None) -> str:
    if explicit is not None:
        return explicit
    if sys.stdin.isatty():
        raise SecretPolicyError("Pass --value or pipe the secret on stdin", code="EMPTY_SECRET")
    return sys.stdin.read().rstrip("\n")


def cmd_policy(_args: argparse.Namespace) -> int:
    json.dump(secret_policy(), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def cmd_get(args: argparse.Namespace) -> int:
    value = resolve_secret(args.name)
    if not value:
        print(f"secret {args.name} is not set", file=sys.stderr)
        return 1
    sys.stdout.write(value)
    if not value.endswith("\n"):
        sys.stdout.write("\n")
    return 0


def cmd_set(args: argparse.Namespace) -> int:
    value = _read_value(args.value)
    location = persist_secret(args.name, value)
    print(f"stored {args.name} in {location}")
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    delete_secret(args.name)
    print(f"deleted {args.name}")
    return 0


def cmd_verify_package(args: argparse.Namespace) -> int:
    listed = verify_package_checksums(args.package, required=True)
    print(f"SHA256SUMS ok ({len(listed)} files) under {Path(args.package).resolve()}")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("policy", help="print allowed and forbidden secret locations").set_defaults(
        func=cmd_policy
    )

    get_p = sub.add_parser("get", help="print a provider secret from env, keychain, or HOME file")
    get_p.add_argument("name")
    get_p.set_defaults(func=cmd_get)

    set_p = sub.add_parser("set", help="store a provider secret (keychain, else 0600 HOME file)")
    set_p.add_argument("name")
    set_p.add_argument("--value", help="secret value (otherwise read stdin)")
    set_p.set_defaults(func=cmd_set)

    del_p = sub.add_parser("delete", help="remove a persisted provider secret")
    del_p.add_argument("name")
    del_p.set_defaults(func=cmd_delete)

    verify_p = sub.add_parser("verify-package", help="fail closed if SHA256SUMS is missing or mismatches")
    verify_p.add_argument("package", type=Path, help="curriculum package directory")
    verify_p.set_defaults(func=cmd_verify_package)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return int(args.func(args))
    except IntegrityError as exc:
        print(f"integrity: {exc}", file=sys.stderr)
        return 1
    except SecretPolicyError as exc:
        print(f"secret policy: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
