"""Look up a keyframe's real frame index and suggested evaluation range.

Usage examples:
    python -m scripts.frame_lookup "L21_V001/042.jpg"
    python -m scripts.frame_lookup "L21_V001 42" --mapping-dir data/map-keyframes
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path, PurePosixPath
from typing import Sequence

from src.utils.config import load_config


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for a keyframe lookup.

    Expected input: ``arguments`` is an optional sequence of CLI argument strings;
    when omitted, arguments are read from the command line.
    Returns: Parsed keyframe reference, configuration path, and mapping-directory override.
    Raises: SystemExit if the supplied CLI arguments are invalid.
    """
    parser = argparse.ArgumentParser(
        description="Look up a keyframe's real frame index and evaluation range."
    )
    parser.add_argument(
        "keyframe",
        help='Keyframe reference: "L21_V001/042.jpg" or "L21_V001 42".',
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--mapping-dir",
        type=Path,
        default=None,
        help="Optional directory containing per-video map-keyframes CSV files.",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=3.0,
        help="Total width of the suggested evaluation window in seconds (default: 3.0).",
    )
    return parser.parse_args(arguments)


def parse_keyframe_reference(reference: str) -> tuple[str, int]:
    """Parse a supported keyframe reference into a video ID and one-based number.

    Expected input: ``reference`` is ``<video_id>/<zero-padded n>.jpg`` or
    ``<video_id> <n>``.
    Returns: A ``(video_id, n)`` tuple, where ``n`` is an integer.
    Raises: ValueError if the reference does not match either supported format.
    """
    normalized = reference.strip()
    parts = normalized.split()
    if len(parts) == 2:
        video_id, number_text = parts
        if "/" in video_id or "\\" in video_id or not number_text.isdigit():
            raise ValueError(
                'Invalid keyframe reference. Use "L21_V001/042.jpg" or "L21_V001 42".'
            )
        return video_id, int(number_text)

    path = PurePosixPath(normalized)
    if len(path.parts) != 2 or path.suffix.lower() != ".jpg" or not path.stem.isdigit():
        raise ValueError(
            'Invalid keyframe reference. Use "L21_V001/042.jpg" or "L21_V001 42".'
        )
    return path.parent.name, int(path.stem)


def load_frame_mapping(mapping_path: Path) -> dict[int, tuple[int, float]]:
    """Load keyframe-number to source-frame mappings from one video CSV.

    Expected input: ``mapping_path`` is a UTF-8 CSV with integer ``n`` and ``frame_idx``
    columns plus a numeric ``fps`` column.
    Returns: A dictionary mapping one-based keyframe numbers to ``(frame_idx, fps)`` tuples.
    Raises: FileNotFoundError if the CSV is missing; ValueError for invalid columns, values,
        or duplicate keyframe numbers.
    """
    with mapping_path.open("r", encoding="utf-8", newline="") as mapping_file:
        reader = csv.DictReader(mapping_file)
        if reader.fieldnames is None or not {"n", "frame_idx", "fps"}.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                f"Mapping CSV '{mapping_path}' must contain 'n', 'frame_idx', and 'fps' "
                "columns."
            )

        mapping: dict[int, tuple[int, float]] = {}
        for row_number, row in enumerate(reader, start=2):
            try:
                keyframe_number = int(row["n"])
                frame_index = int(row["frame_idx"])
                fps = float(row["fps"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid n, frame_idx, or fps value in '{mapping_path}' at CSV row "
                    f"{row_number}."
                ) from error
            if keyframe_number in mapping:
                raise ValueError(
                    f"Duplicate keyframe n={keyframe_number} in '{mapping_path}'."
                )
            mapping[keyframe_number] = (frame_index, fps)
    return mapping


def lookup_frame_range(
    mapping_dir: Path, video_id: str, keyframe_number: int, window_sec: float
) -> tuple[int, float, int, int, int]:
    """Find a keyframe's source frame and suggested inclusive evaluation range.

    Expected input: ``mapping_dir`` contains ``<video_id>.csv`` mapping files;
    ``keyframe_number`` is the requested one-based CSV ``n`` value; ``window_sec`` is a
    non-negative total time-window width in seconds.
    Returns: A ``(frame_idx, fps, half_frames, gt_frame_start, gt_frame_end)`` tuple.
    Raises: FileNotFoundError if the video mapping CSV is missing; ValueError if the mapping
        is empty, the keyframe number is out of range, its row is missing, or the window is
        negative.
    """
    mapping_path = mapping_dir / f"{video_id}.csv"
    if not mapping_path.is_file():
        raise FileNotFoundError(
            f"Mapping CSV not found for video_id '{video_id}': {mapping_path}. "
            "Check that the video_id is correct and the mapping file exists."
        )

    mapping = load_frame_mapping(mapping_path)
    if not mapping:
        raise ValueError(f"Mapping CSV for video_id '{video_id}' is empty.")

    total_keyframes = len(mapping)
    if keyframe_number < 1 or keyframe_number > total_keyframes:
        raise ValueError(
            f"Keyframe n={keyframe_number} is out of range for video_id '{video_id}'. "
            f"Valid range is 1..{total_keyframes}."
        )
    if keyframe_number not in mapping:
        raise ValueError(
            f"Keyframe n={keyframe_number} is missing from the mapping CSV for "
            f"video_id '{video_id}'."
        )
    if window_sec < 0:
        raise ValueError("window-sec must be non-negative.")

    frame_index, fps = mapping[keyframe_number]
    half_frames = round((window_sec / 2) * fps)
    frame_start = max(0, frame_index - half_frames)
    frame_end = frame_index + half_frames
    return frame_index, fps, half_frames, frame_start, frame_end


def main(arguments: Sequence[str] | None = None) -> None:
    """Print copy-paste-ready annotation fields for a requested keyframe.

    Expected input: ``arguments`` is an optional sequence accepted by ``parse_args``;
    configuration contains ``data.mapping_dir`` unless ``--mapping-dir`` is supplied.
    Returns: None.
    Raises: SystemExit with a clear message for invalid references, mappings, or keyframe
        numbers.
    """
    args = parse_args(arguments)
    try:
        video_id, keyframe_number = parse_keyframe_reference(args.keyframe)
        mapping_dir = args.mapping_dir or load_config(args.config)["data"]["mapping_dir"]
        frame_index, fps, half_frames, frame_start, frame_end = lookup_frame_range(
            mapping_dir, video_id, keyframe_number, args.window_sec
        )
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error

    file_name = f"{keyframe_number:03d}.jpg"
    print(f"video_id      : {video_id}")
    print(f"keyframe n    : {keyframe_number}  (file {file_name})")
    print(f"frame_idx     : {frame_index}")
    print(f"fps           : {fps}")
    print(f"window        : {args.window_sec}s total (+/-{half_frames} frames)")
    print(f"-> gt_frame_start = {frame_start}")
    print(f"-> gt_frame_end   = {frame_end}")


if __name__ == "__main__":
    main()
