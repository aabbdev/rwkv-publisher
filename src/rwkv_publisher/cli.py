from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .build import build_release
from .hub import publish_release


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rwkv-publisher",
        description="Build and publish auditable native RWKV model repositories.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser(
        "build", help="Resolve, convert, and build one complete RWKV release"
    )
    build.add_argument(
        "source", help="Local .pth, registered Hub file, URL, or repository"
    )
    build.add_argument("--output", type=Path, default=Path("dist"))
    build.add_argument(
        "--dtype",
        choices=("preserve", "float32", "float16", "bfloat16"),
        default="preserve",
    )
    build.add_argument("--max-shard-size", default="5GB")
    build.add_argument("--source-ref")
    build.add_argument("--offline", action="store_true")
    build.add_argument("--dry-run", action="store_true")
    build.add_argument("--json", action="store_true")

    publish = commands.add_parser(
        "publish", help="Publish and verify one staged release directory"
    )
    publish.add_argument("release", type=Path)
    publish.add_argument("--repo")
    publish.add_argument("--private", action="store_true")
    publish.add_argument("--revision", default="main")
    publish.add_argument("--dry-run", action="store_true")
    publish.add_argument("--json", action="store_true")
    return parser


def _print(result: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if result["dry_run"]:
        print("Validation plan complete; no release or Hub state was changed.")
    elif "commit" in result:
        print(f"Published and verified: {result['url']}")
    else:
        print(f"Built release: {result['release']}")


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "build":
        result = build_release(
            args.source,
            output=args.output,
            dtype=args.dtype,
            max_shard_size=args.max_shard_size,
            source_ref=args.source_ref,
            offline=args.offline,
            dry_run=args.dry_run,
        )
    else:
        result = publish_release(
            args.release,
            repo_id=args.repo,
            private=args.private,
            revision=args.revision,
            dry_run=args.dry_run,
        )
    _print(result, as_json=args.json)


__all__ = ["main"]
