#!/usr/bin/env python3
"""CLI: resolve a keyframe reference against a canonical keyframe artifact.

Given a canonical keyframe artifact directory (``frames.ndjson`` plus an
optional ``manifest.json`` / ``images/`` collection), resolve a keyframe
reference such as ``L25_V085/000123.jpg`` or ``L25_V085 123`` to its stored
metadata and, when requested with ``--window-sec``, derive the annotation GT
frame range ``gt_frame_start`` / ``gt_frame_end``. Stored metadata is never
modified. The logic lives in ``aic2026.frame_lookup``; this script is a thin
wrapper.

Example:
    python scripts/frame_lookup.py \\
        --artifact-dir data/keyframes/adaptive_sbd_v1 "L25_V085/000123.jpg"
    python scripts/frame_lookup.py \\
        --artifact-dir another/keyframe_artifact --window-sec 2.0 "L25_V085 123"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aic2026.frame_lookup import (
    FrameLookupError,
    KeyframeArtifact,
    compute_gt_range,
    format_record,
    load_frame_lookup_config,
    parse_keyframe_reference,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="frame_lookup",
        description=(
            "Resolve a keyframe reference against a canonical keyframe "
            "artifact and optionally derive an annotation GT frame range."
        ),
    )
    parser.add_argument("reference", help="keyframe reference, e.g. 'L25_V085/000123.jpg' or 'L25_V085 123'")
    parser.add_argument(
        "--artifact-dir",
        required=True,
        help="path to a canonical keyframe artifact directory (contains frames.ndjson)",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=None,
        help="total GT window duration in seconds; overrides the config default",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="optional YAML config; its 'frame_lookup.window_sec' is used when --window-sec is absent",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the frame lookup CLI."""
    args = build_argument_parser().parse_args(argv)
    try:
        config = load_frame_lookup_config(Path(args.config) if args.config else None)
        window_sec = args.window_sec if args.window_sec is not None else config.get("window_sec")
        if window_sec is not None and not isinstance(window_sec, (int, float)):
            raise FrameLookupError("config 'frame_lookup.window_sec' must be a number")
        video_id, keyframe_ordinal = parse_keyframe_reference(args.reference)
        artifact = KeyframeArtifact(args.artifact_dir)
        record = artifact.lookup(video_id, keyframe_ordinal)
        gt_range = compute_gt_range(record, float(window_sec)) if window_sec is not None else None
        image_path = artifact.image_path(video_id, keyframe_ordinal)
    except FrameLookupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_record(record, image_path, gt_range))
    return 0


if __name__ == "__main__":
    sys.exit(main())