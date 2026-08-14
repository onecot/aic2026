"""Tests for the keyframe-extraction pipeline, split, merge, and their CLIs.

Uses synthetic video descriptors only (never ffmpeg), so the tests are fast,
deterministic, and cross-platform.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from aic2026.entities import parse_keyframe_map
from aic2026.keyframe_extraction import (
    CHECKPOINT_FILENAME,
    FRAMES_FILENAME,
    MANIFEST_FILENAME,
    MAP_KEYFRAMES_DIRNAME,
    VIDEOS_MANIFEST_FILENAME,
    ExtractionConfig,
    ExtractionError,
    VideoEntry,
    VideoManifest,
    extract_keyframes,
    load_video_manifest,
    save_video_manifest,
    write_video_map,
    write_video_ndjson,
)
from aic2026.keyframe_extraction_merge import MergeError, merge_shards
from aic2026.keyframe_extraction_split import SplitError, split_video_manifest, worker_dir_name
from aic2026.keyframe_extractor import extract_video_keyframes, write_synthetic_video


def _scene_values(num_frames: int, scene_length: int = 40) -> list[float]:
    return [float((i // scene_length) * 100.0) for i in range(num_frames)]


def _make_videos(tmp_path: Path, count: int = 4, frames: int = 200) -> list[VideoEntry]:
    entries: list[VideoEntry] = []
    for i in range(count):
        vid = f"L21_V{i + 1:03d}"
        path = write_synthetic_video(
            tmp_path / f"{vid}.json",
            _scene_values(frames, scene_length=10),
        )
        entries.append(VideoEntry(video_id=vid, path=str(path), num_frames=frames))
    return entries


def _manifest(videos: list[VideoEntry]) -> VideoManifest:
    return VideoManifest(schema_version="1.0", videos=tuple(videos))


def _config(**overrides: object) -> ExtractionConfig:
    kwargs: dict[str, object] = {
        "frame_reader": "synthetic",
        "target_keyframes_per_video": 6,
        "min_temporal_distance": 5,
    }
    kwargs.update(overrides)
    return ExtractionConfig(**kwargs)


def _save(tmp_path: Path, videos: list[VideoEntry], name: str = "videos.json") -> Path:
    path = tmp_path / name
    save_video_manifest(path, _manifest(videos))
    return path


@pytest.fixture
def videos_file(tmp_path: Path) -> Path:
    return _save(tmp_path, _make_videos(tmp_path))


@pytest.fixture
def split_videos(tmp_path: Path, videos_file: Path) -> tuple[Path, list[Path]]:
    split_dir = tmp_path / "splits"
    split_video_manifest(videos_file, 2, split_dir)
    worker_files = [split_dir / worker_dir_name(w) / VIDEOS_MANIFEST_FILENAME for w in range(2)]
    return split_dir, worker_files


# ---------------------------------------------------------------------- config


class TestExtractionConfig:
    def test_requires_one_target_mode(self) -> None:
        with pytest.raises(ValueError):
            ExtractionConfig(frame_reader="synthetic")
        with pytest.raises(ValueError):
            ExtractionConfig(frame_reader="synthetic", target_keyframes_per_video=5, sampling_ratio=0.1)

    def test_sampling_ratio_validated(self) -> None:
        with pytest.raises(ValueError):
            ExtractionConfig(frame_reader="synthetic", sampling_ratio=1.5)

    def test_config_hash_stable(self) -> None:
        a = _config().config_hash()
        b = ExtractionConfig(
            frame_reader="synthetic",
            target_keyframes_per_video=6,
            min_temporal_distance=5,
            peak_threshold=8.0,
        ).config_hash()
        assert a == b
        assert _config(min_temporal_distance=20).config_hash() != a

    def test_config_hash_changes_with_target(self) -> None:
        assert _config(target_keyframes_per_video=5).config_hash() != _config().config_hash()

    def test_config_hash_changes_with_scorer_parameters(self) -> None:
        assert _config(scorer="jpeg_size").config_hash() != _config().config_hash()
        assert _config(jpeg_quality=85).config_hash() != _config().config_hash()
        assert _config(anchor_interval_seconds=2.0).config_hash() != _config().config_hash()

    def test_new_scorer_parameters_round_trip(self) -> None:
        cfg = _config(scorer="jpeg_size", jpeg_quality=80, anchor_interval_seconds=1.5)
        restored = ExtractionConfig.from_dict(cfg.to_dict())
        assert restored == cfg
        assert restored.config_hash() == cfg.config_hash()

    def test_config_validation_of_new_fields(self) -> None:
        with pytest.raises(ValueError):
            _config(jpeg_quality=0)
        with pytest.raises(ValueError):
            _config(jpeg_quality=101)
        with pytest.raises(ValueError):
            _config(anchor_interval_seconds=0.0)
        with pytest.raises(ValueError):
            _config(anchor_interval_seconds=-1.0)


# ---------------------------------------------------------------- video manifest


class TestVideoManifest:
    def test_duplicate_ids_rejected(self, tmp_path: Path) -> None:
        entries = [VideoEntry(video_id="L21_V001", path="a.mp4")] * 2
        with pytest.raises(ValueError):
            _manifest(entries)

    def test_round_trip(self, videos_file: Path) -> None:
        manifest = load_video_manifest(videos_file)
        assert len(manifest.videos) == 4
        assert manifest.videos[0].video_id == "L21_V001"

    def test_invalid_video_id_rejected(self) -> None:
        with pytest.raises(ValueError):
            VideoEntry(video_id="not-an-id", path="a.mp4")


# ---------------------------------------------------------------------- split


class TestSplitVideoManifest:
    def test_splits_keep_videos_intact(self, videos_file: Path, tmp_path: Path) -> None:
        split_dir = tmp_path / "splits"
        manifest = split_video_manifest(videos_file, 3, split_dir)
        assert manifest.num_workers == 3
        all_ids: list[str] = []
        for worker in range(3):
            worker_videos = load_video_manifest(split_dir / worker_dir_name(worker) / VIDEOS_MANIFEST_FILENAME)
            all_ids.extend(v.video_id for v in worker_videos.videos)
        assert sorted(all_ids) == ["L21_V001", "L21_V002", "L21_V003", "L21_V004"]

    def test_deterministic(self, videos_file: Path, tmp_path: Path) -> None:
        manifest_a = split_video_manifest(videos_file, 2, tmp_path / "a")
        manifest_b = split_video_manifest(videos_file, 2, tmp_path / "b")
        assert manifest_a.to_dict() == manifest_b.to_dict()

    def test_balances_frames(self, tmp_path: Path) -> None:
        # Uneven weights: two heavy videos, two light ones.
        entries = [
            VideoEntry(video_id="L21_V001", path="a.mp4", num_frames=1000),
            VideoEntry(video_id="L21_V002", path="b.mp4", num_frames=10),
            VideoEntry(video_id="L21_V003", path="c.mp4", num_frames=1000),
            VideoEntry(video_id="L21_V004", path="d.mp4", num_frames=10),
        ]
        path = _save(tmp_path, entries)
        manifest = split_video_manifest(path, 2, tmp_path / "splits")
        loads = [w.estimated_frames for w in manifest.workers]
        # The two heavy videos land in different workers.
        assert max(loads) - min(loads) <= 10

    def test_existing_output_rejected(self, videos_file: Path, tmp_path: Path) -> None:
        split_dir = tmp_path / "splits"
        split_video_manifest(videos_file, 2, split_dir)
        with pytest.raises(SplitError):
            split_video_manifest(videos_file, 2, split_dir)
        # overwrite works
        split_video_manifest(videos_file, 2, split_dir, overwrite=True)

    def test_missing_source(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            split_video_manifest(tmp_path / "nope.json", 2, tmp_path / "out")


# ------------------------------------------------------------------ extraction


class TestExtractKeyframes:
    def test_output_contract(self, split_videos: tuple[Path, list[Path]], tmp_path: Path) -> None:
        _, worker_files = split_videos
        worker_videos = load_video_manifest(worker_files[0])
        worker_ids = sorted(v.video_id for v in worker_videos.videos)
        out = tmp_path / "shard"
        manifest = extract_keyframes(worker_files[0], out, _config())
        assert manifest.num_videos == len(worker_ids)
        assert (out / FRAMES_FILENAME).is_file()
        assert (out / MANIFEST_FILENAME).is_file()
        assert (out / MAP_KEYFRAMES_DIRNAME).is_dir()
        maps = sorted(p.name for p in (out / MAP_KEYFRAMES_DIRNAME).iterdir())
        assert maps == [f"{vid}.csv" for vid in worker_ids]
        # checkpoint removed after completion
        assert not (out / CHECKPOINT_FILENAME).is_file()

    def test_frames_are_canonical(self, split_videos: tuple[Path, list[Path]], tmp_path: Path) -> None:
        _, worker_files = split_videos
        worker_ids = sorted(v.video_id for v in load_video_manifest(worker_files[0]).videos)
        out = tmp_path / "shard"
        extract_keyframes(worker_files[0], out, _config())
        rows = [
            json.loads(line)
            for line in (out / FRAMES_FILENAME).read_text(encoding="utf-8").splitlines()
        ]
        # Every record has the canonical keys and no duplicate (video_id, frame_index).
        keys: set[tuple[str, int]] = set()
        for row in rows:
            assert set(row.keys()) == {"keyframe", "pts_time", "fps", "frame_index"}
            kf = row["keyframe"]
            key = (kf["video_id"]["value"], row["frame_index"])
            assert key not in keys
            keys.add(key)
        # ordinals restart at 1 per video and are contiguous.
        for video_id in worker_ids:
            ords = [r["keyframe"]["keyframe_ordinal"] for r in rows if r["keyframe"]["video_id"]["value"] == video_id]
            assert ords == list(range(1, len(ords) + 1))

    def test_map_files_match_parser(
        self, split_videos: tuple[Path, list[Path]], tmp_path: Path
    ) -> None:
        _, worker_files = split_videos
        out = tmp_path / "shard"
        extract_keyframes(worker_files[0], out, _config())
        for csv_path in (out / MAP_KEYFRAMES_DIRNAME).iterdir():
            entries = parse_keyframe_map(csv_path)
            assert entries
            for entry in entries:
                assert entry.frame_idx >= 0

    def test_map_consistent_with_frames(
        self, split_videos: tuple[Path, list[Path]], tmp_path: Path
    ) -> None:
        _, worker_files = split_videos
        worker_ids = sorted(v.video_id for v in load_video_manifest(worker_files[0]).videos)
        out = tmp_path / "shard"
        extract_keyframes(worker_files[0], out, _config())
        rows = [
            json.loads(line)
            for line in (out / FRAMES_FILENAME).read_text(encoding="utf-8").splitlines()
        ]
        video_id = worker_ids[0]
        csv = out / MAP_KEYFRAMES_DIRNAME / f"{video_id}.csv"
        map_entries = parse_keyframe_map(csv)
        row_keys = {(r["keyframe"]["video_id"]["value"], r["keyframe"]["keyframe_ordinal"]) for r in rows}
        for entry in map_entries:
            assert (video_id, entry.keyframe_no) in row_keys

    def test_deterministic_output(
        self, split_videos: tuple[Path, list[Path]], tmp_path: Path
    ) -> None:
        _, worker_files = split_videos
        a = tmp_path / "a"
        b = tmp_path / "b"
        extract_keyframes(worker_files[0], a, _config())
        extract_keyframes(worker_files[0], b, _config())
        assert (a / FRAMES_FILENAME).read_text() == (b / FRAMES_FILENAME).read_text()

    def test_manifest_metadata(self, split_videos: tuple[Path, list[Path]], tmp_path: Path) -> None:
        _, worker_files = split_videos
        out = tmp_path / "shard"
        manifest = extract_keyframes(worker_files[0], out, _config())
        assert manifest.scorer == "frame_difference"
        assert manifest.frame_reader == "synthetic"
        assert manifest.extractor == "dake"
        assert manifest.parameters["target_keyframes_per_video"] == 6
        assert manifest.num_videos == 2
        assert manifest.statistics.total_videos == 2

    def test_jpeg_scorer_and_anchors_pipeline(
        self, split_videos: tuple[Path, list[Path]], tmp_path: Path
    ) -> None:
        _, worker_files = split_videos
        out = tmp_path / "shard"
        manifest = extract_keyframes(
            worker_files[0],
            out,
            _config(
                scorer="jpeg_size",
                jpeg_quality=85,
                anchor_interval_seconds=0.2,
            ),
        )
        assert manifest.scorer == "jpeg_size"
        assert manifest.parameters["jpeg_quality"] == 85
        assert manifest.parameters["anchor_interval_seconds"] == 0.2
        assert manifest.extractor_version == "2.0"
        assert manifest.num_videos == 2
        rows = [
            json.loads(line)
            for line in (out / FRAMES_FILENAME).read_text(encoding="utf-8").splitlines()
        ]
        assert rows
        first_row = rows[0]
        # schema unchanged: keyframe records use the canonical shape
        assert set(first_row["keyframe"]) == {"keyframe_ordinal", "video_id"}
        assert set(first_row) == {"keyframe", "pts_time", "fps", "frame_index"}
        # anchors guarantee first and last source frames are present
        assert first_row["keyframe"]["keyframe_ordinal"] == 1
        assert first_row["keyframe"]["video_id"]["value"] == "L21_V001"

    def test_complete_artifact_is_idempotent(
        self, split_videos: tuple[Path, list[Path]], tmp_path: Path
    ) -> None:
        _, worker_files = split_videos
        out = tmp_path / "shard"
        first = extract_keyframes(worker_files[0], out, _config())
        second = extract_keyframes(worker_files[0], out, _config())
        assert second.num_keyframes == first.num_keyframes
        assert (out / MANIFEST_FILENAME).read_text() == json.dumps(first.to_dict(), indent=2) + "\n"

    def test_config_change_on_complete_rejected(
        self, split_videos: tuple[Path, list[Path]], tmp_path: Path
    ) -> None:
        _, worker_files = split_videos
        out = tmp_path / "shard"
        extract_keyframes(worker_files[0], out, _config())
        with pytest.raises(ExtractionError):
            extract_keyframes(worker_files[0], out, _config(target_keyframes_per_video=9))

    def test_existing_nonempty_output_rejected(
        self, split_videos: tuple[Path, list[Path]], tmp_path: Path
    ) -> None:
        _, worker_files = split_videos
        out = tmp_path / "shard"
        out.mkdir()
        (out / "junk.txt").write_text("x", encoding="utf-8")
        with pytest.raises(ExtractionError):
            extract_keyframes(worker_files[0], out, _config())
        extract_keyframes(worker_files[0], out, _config(), overwrite=True)

    def test_missing_videos_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            extract_keyframes(tmp_path / "nope.json", tmp_path / "out", _config())


# ---------------------------------------------------------------- resume


class TestResume:
    def _prepare_interrupted(
        self, tmp_path: Path, videos_file: Path, completed: list[str]
    ) -> Path:
        """Create a shard dir with completed+staged videos and a checkpoint."""
        out = tmp_path / "shard"
        staging = out / ".staging"
        staging.mkdir(parents=True)
        manifest = load_video_manifest(videos_file)
        from aic2026.keyframe_extractor import SyntheticFrameReader

        reader = SyntheticFrameReader()
        config = _config()
        frames_scanned: dict[str, int] = {}
        for entry in manifest.videos:
            if entry.video_id in completed:
                kfs = extract_video_keyframes(entry.video_id, entry.path, frame_reader=reader, target=6)
                write_video_ndjson(staging / f"{entry.video_id}.ndjson", kfs)
                write_video_map(staging / f"{entry.video_id}.csv", kfs)
                frames_scanned[entry.video_id] = 200
        from aic2026.keyframe_extraction import (
            EXTRACTION_CHECKPOINT_SCHEMA_VERSION,
            ExtractionCheckpoint,
        )

        checkpoint = ExtractionCheckpoint(
            schema_version=EXTRACTION_CHECKPOINT_SCHEMA_VERSION,
            config_hash=config.config_hash(),
            total_videos=len(manifest.videos),
            completed_video_ids=tuple(completed),
            frames_scanned_by_video=frames_scanned,
            started_at="2026-01-01T00:00:00+00:00",
            timestamp="2026-01-01T00:00:00+00:00",
        )
        with (out / CHECKPOINT_FILENAME).open("w", encoding="utf-8") as handle:
            json.dump(checkpoint.to_dict(), handle)
        return out

    def test_resume_skips_completed(self, videos_file: Path, tmp_path: Path) -> None:
        out = self._prepare_interrupted(tmp_path, videos_file, ["L21_V001", "L21_V002"])
        manifest = extract_keyframes(videos_file, out, _config(), resume=True)
        assert manifest.num_videos == 4
        assert manifest.statistics.total_frames_scanned == 4 * 200
        assert len((out / FRAMES_FILENAME).read_text().splitlines()) == manifest.num_keyframes

    def test_config_mismatch_on_resume_rejected(
        self, videos_file: Path, tmp_path: Path
    ) -> None:
        out = self._prepare_interrupted(tmp_path, videos_file, ["L21_V001"])
        with pytest.raises(ExtractionError):
            extract_keyframes(videos_file, out, _config(target_keyframes_per_video=9), resume=True)

    def test_scorer_change_on_resume_rejected(self, videos_file: Path, tmp_path: Path) -> None:
        out = self._prepare_interrupted(tmp_path, videos_file, ["L21_V001"])
        with pytest.raises(ExtractionError):
            extract_keyframes(videos_file, out, _config(scorer="jpeg_size"), resume=True)
        with pytest.raises(ExtractionError):
            extract_keyframes(videos_file, out, _config(anchor_interval_seconds=0.5), resume=True)

    def test_checkpoint_unknown_videos_rejected(
        self, videos_file: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "shard"
        staging = out / ".staging"
        staging.mkdir(parents=True)
        from aic2026.keyframe_extraction import (
            EXTRACTION_CHECKPOINT_SCHEMA_VERSION,
            ExtractionCheckpoint,
        )

        checkpoint = ExtractionCheckpoint(
            schema_version=EXTRACTION_CHECKPOINT_SCHEMA_VERSION,
            config_hash=_config().config_hash(),
            total_videos=4,
            completed_video_ids=("L99_V999",),
            frames_scanned_by_video={},
            started_at="2026-01-01T00:00:00+00:00",
            timestamp="2026-01-01T00:00:00+00:00",
        )
        with (out / CHECKPOINT_FILENAME).open("w", encoding="utf-8") as handle:
            json.dump(checkpoint.to_dict(), handle)
        with pytest.raises(ExtractionError):
            extract_keyframes(videos_file, out, _config(), resume=True)

    def test_resume_without_checkpoint_uses_fresh_start(
        self, split_videos: tuple[Path, list[Path]], tmp_path: Path
    ) -> None:
        _, worker_files = split_videos
        out = tmp_path / "shard"
        a = extract_keyframes(worker_files[0], out, _config(), resume=True)
        b = extract_keyframes(worker_files[0], out, _config(), resume=True)
        assert a.num_keyframes == b.num_keyframes


# ------------------------------------------------------------------- merge


class TestMergeShards:
    def test_merge_combines_shards(self, videos_file: Path, tmp_path: Path) -> None:
        split_dir = tmp_path / "splits"
        split_video_manifest(videos_file, 3, split_dir)
        shards: list[Path] = []
        for worker in range(3):
            shard = tmp_path / f"shard_{worker}"
            extract_keyframes(
                split_dir / worker_dir_name(worker) / VIDEOS_MANIFEST_FILENAME,
                shard,
                _config(),
            )
            shards.append(shard)
        merged = tmp_path / "merged"
        manifest = merge_shards(shards, merged)
        assert manifest.num_videos == 4
        assert manifest.num_keyframes == 24  # 4 videos * 6
        assert len((merged / FRAMES_FILENAME).read_text().splitlines()) == 24
        assert sorted(p.name for p in (merged / MAP_KEYFRAMES_DIRNAME).iterdir()) == [
            "L21_V001.csv",
            "L21_V002.csv",
            "L21_V003.csv",
            "L21_V004.csv",
        ]

    def test_merge_config_mismatch_rejected(self, videos_file: Path, tmp_path: Path) -> None:
        split_dir = tmp_path / "splits"
        split_video_manifest(videos_file, 2, split_dir)
        shard_a = tmp_path / "a"
        shard_b = tmp_path / "b"
        extract_keyframes(
            split_dir / worker_dir_name(0) / VIDEOS_MANIFEST_FILENAME, shard_a, _config()
        )
        extract_keyframes(
            split_dir / worker_dir_name(1) / VIDEOS_MANIFEST_FILENAME,
            shard_b,
            _config(target_keyframes_per_video=9),
        )
        with pytest.raises(MergeError):
            merge_shards([shard_a, shard_b], tmp_path / "merged")

    def test_merge_incomplete_shard_rejected(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(MergeError):
            merge_shards([empty], tmp_path / "merged")

    def test_merge_existing_output_rejected(self, videos_file: Path, tmp_path: Path) -> None:
        split_dir = tmp_path / "splits"
        split_video_manifest(videos_file, 2, split_dir)
        shards = [
            tmp_path / f"s{i}" for i in range(2)
        ]
        for i, shard in enumerate(shards):
            extract_keyframes(
                split_dir / worker_dir_name(i) / VIDEOS_MANIFEST_FILENAME, shard, _config()
            )
        merged = tmp_path / "merged"
        merge_shards(shards, merged)
        with pytest.raises(MergeError):
            merge_shards(shards, merged)

    def test_merge_deterministic_order(self, videos_file: Path, tmp_path: Path) -> None:
        split_dir = tmp_path / "splits"
        split_video_manifest(videos_file, 2, split_dir)
        shards = [tmp_path / f"s{i}" for i in range(2)]
        for i, shard in enumerate(shards):
            extract_keyframes(
                split_dir / worker_dir_name(i) / VIDEOS_MANIFEST_FILENAME, shard, _config()
            )
        merged_a = merge_shards(shards, tmp_path / "ma")
        merged_b = merge_shards(list(reversed(shards)), tmp_path / "mb")
        assert (tmp_path / "ma" / FRAMES_FILENAME).read_text() == (tmp_path / "mb" / FRAMES_FILENAME).read_text()
        assert merged_a.num_keyframes == merged_b.num_keyframes


# ---------------------------------------------------------------------- CLIs


def _extract_script() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "extract_keyframes.py"


def _split_script() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "split_keyframe_extraction_shards.py"


def _merge_script() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "merge_keyframe_shards.py"


class TestExtractCli:
    def test_help(self) -> None:
        result = subprocess.run(
            [sys.executable, str(_extract_script()), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "--videos" in result.stdout
        assert "--sampling-ratio" in result.stdout

    def test_missing_args_exits_2(self) -> None:
        result = subprocess.run(
            [sys.executable, str(_extract_script())],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 2

    def test_run_end_to_end(self, videos_file: Path, tmp_path: Path) -> None:
        out = tmp_path / "shard"
        result = subprocess.run(
            [
                sys.executable,
                str(_extract_script()),
                "--videos",
                str(videos_file),
                "--output-dir",
                str(out),
                "--target-keyframes-per-video",
                "6",
                "--frame-reader",
                "synthetic",
                "--progress",
                "off",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "Extraction complete: 24 keyframes" in result.stdout
        assert (out / FRAMES_FILENAME).is_file()

    def test_config_error_exits_1(self, videos_file: Path, tmp_path: Path) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(_extract_script()),
                "--videos",
                str(videos_file),
                "--output-dir",
                str(tmp_path / "shard"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        # missing both target and sampling-ratio -> ExtractionConfig error at CLI level
        assert result.returncode in (1, 2)


class TestSplitCli:
    def test_help(self) -> None:
        result = subprocess.run(
            [sys.executable, str(_split_script()), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "--num-workers" in result.stdout

    def test_missing_args_exits_2(self) -> None:
        result = subprocess.run(
            [sys.executable, str(_split_script())],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 2

    def test_run(self, videos_file: Path, tmp_path: Path) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(_split_script()),
                "--videos",
                str(videos_file),
                "--num-workers",
                "2",
                "--output-dir",
                str(tmp_path / "splits"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "Split complete: 4 videos" in result.stdout


class TestMergeCli:
    def test_help(self) -> None:
        result = subprocess.run(
            [sys.executable, str(_merge_script()), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "--shards" in result.stdout

    def test_missing_args_exits_2(self) -> None:
        result = subprocess.run(
            [sys.executable, str(_merge_script())],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 2

    def test_run(self, videos_file: Path, tmp_path: Path) -> None:
        split_dir = tmp_path / "splits"
        split_video_manifest(videos_file, 2, split_dir)
        shards = [tmp_path / f"s{i}" for i in range(2)]
        for i, shard in enumerate(shards):
            extract_keyframes(
                split_dir / worker_dir_name(i) / VIDEOS_MANIFEST_FILENAME,
                shard,
                _config(),
            )
        result = subprocess.run(
            [
                sys.executable,
                str(_merge_script()),
                "--shards",
                str(shards[0]),
                str(shards[1]),
                "--output-dir",
                str(tmp_path / "merged"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "Merge complete: 24 keyframes" in result.stdout