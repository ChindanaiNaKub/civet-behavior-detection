"""Portable blind-review entry point; packaged as start_review.py.

Windows: python start_review.py
Fedora/Linux: python3 start_review.py
No third-party Python packages are required.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def review_arguments(root: Path, port: int | None, no_browser: bool) -> list[str]:
    """Absolute paths work on either OS and from any current directory."""
    root = root.resolve()
    required = [root / name for name in (
        "scripts/review.py", "scripts/review_page.html", "scripts/review_i18n.json", "scripts/labels.py",
        "data/clips.csv", "data/gold-clips.txt", "data/raw",
    )]
    missing = [str(path.relative_to(root)) for path in required if not path.exists()]
    if missing:
        raise ValueError("Review folder is incomplete: " + ", ".join(missing))
    args = [
        "--clips-file", str(root / "data/clips.csv"), "--raw", str(root / "data/raw"),
        "--timeline", str(root / "data/no-machine-proposals.csv"),
        "--only-file", str(root / "data/gold-clips.txt"),
        "--blind-file", str(root / "data/gold-clips.txt"),
        "--out", str(root / "data/annotations-second.csv"),
        "--protect-file", str(root / "data/annotations.csv"),
        "--protect-file", str(root.parent.parent / "data/annotations.csv"),
        "--port", str(port if port is not None else 8000),
    ]
    if port is None:
        args.append("--auto-port")
    if not no_browser:
        args.append("--open-browser")
    return args


def main(argv: list[str] | None = None, root: Path | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start the private blind reviewer on Windows or Fedora/Linux.")
    parser.add_argument("--port", type=int, help="Choose a port; 0 asks the OS for a free port")
    parser.add_argument("--no-browser", action="store_true", help="Print the URL without opening a browser (headless/SSH)")
    parser.add_argument("--check", action="store_true", help="Check packet files without starting a server or modifying records")
    args = parser.parse_args(argv)
    if args.port is not None and not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535")
    root = (root or Path(__file__).resolve().parent).resolve()
    try:
        review_args = review_arguments(root, args.port, args.no_browser)
        sys.path.insert(0, str(root / "scripts"))
        import review
        clips = review.load_clips(root / "data/clips.csv", root / "data/raw")
        gold = review.id_set([], root / "data/gold-clips.txt")
        if not gold or gold != {clip.clip_id for clip in clips}:
            raise ValueError("Packet clip list and gold list must match and must not be empty")
        missing = [clip.clip_id for clip in clips if not clip.path.is_file()]
        if missing:
            raise ValueError("Missing video files: " + ", ".join(missing))
        if args.check:
            print(f"Ready: {len(clips)} blind clips. Records were not changed.")
            return 0
        return review.main(review_args)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Could not start reviewer: {exc}\nKeep the review folder intact; use --port 0 if a port is busy.\n")


if __name__ == "__main__":
    raise SystemExit(main())
