"""Tests for the media-info lexical retrieval branch.

These tests exercise document loading, BM25 scoring determinism, missing /
corrupt media-info handling, query output formatting, serialization, and the
CLI entrypoint. They never touch CLIP retrieval, scoring, or evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aic2026.media_info_retrieval import (
    MEDIA_INFO_INDEX_SCHEMA_VERSION,
    MediaInfoDocument,
    MediaInfoHit,
    MediaInfoIndex,
    MediaInfoRetrievalConfig,
    tokenize,
)

_MEDIA_BASE = {
    "author": "Test Author",
    "channel_id": "UCHANNEL",
    "channel_url": "https://www.youtube.com/channel/UCHANNEL",
    "description": "test",
    "keywords": ["k1"],
    "length": 42,
    "publish_date": "01/01/2024",
    "thumbnail_url": "https://example.com/thumb.jpg",
    "title": "Test video",
    "watch_url": "https://youtube.com/watch?v=TEST",
}


def _document(
    video_id: str,
    title: str = "",
    keywords: tuple[str, ...] = (),
    description: str = "",
) -> MediaInfoDocument:
    return MediaInfoDocument(
        video_id=video_id,
        title=title,
        keywords=keywords,
        description=description,
    )


def _write_media_info(
    directory: Path,
    video_id: str,
    title: str = "Test video",
    keywords: list[str] | None = None,
    description: str = "test",
    **overrides: object,
) -> Path:
    payload = {
        **_MEDIA_BASE,
        "title": title,
        "keywords": keywords if keywords is not None else ["k1"],
        "description": description,
        **overrides,
    }
    path = directory / f"{video_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def test_tokenize_is_deterministic_and_normalized() -> None:
    assert tokenize("Thuyền Chở du khách") == ("thuyền", "chở", "du", "khách")
    assert tokenize("  a  b, c! ") == ("a", "b", "c")
    assert tokenize("HTV-Tin-tức") == ("htv", "tin", "tức")
    assert tokenize("") == ()
    assert tokenize("!!!...") == ()


def test_tokenize_preserves_unicode_letters() -> None:
    tokens = tokenize("Hàu xào bông cải đầm")
    assert "hàu" in tokens
    assert "bông" in tokens
    assert "đầm" in tokens


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


def test_config_validation_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="k1"):
        MediaInfoRetrievalConfig(k1=0)
    with pytest.raises(ValueError, match="k1"):
        MediaInfoRetrievalConfig(k1=-1.0)
    with pytest.raises(ValueError, match="b"):
        MediaInfoRetrievalConfig(b=-0.1)
    with pytest.raises(ValueError, match="b"):
        MediaInfoRetrievalConfig(b=1.5)
    with pytest.raises(ValueError, match="title_weight"):
        MediaInfoRetrievalConfig(title_weight=-1)
    with pytest.raises(ValueError, match="at least one field weight"):
        MediaInfoRetrievalConfig(title_weight=0, keywords_weight=0, description_weight=0)


# ---------------------------------------------------------------------------
# Building from documents
# ---------------------------------------------------------------------------


def test_build_orders_documents_by_video_id() -> None:
    index = MediaInfoIndex.build(
        [
            _document("L22_V002", title="second"),
            _document("L21_V001", title="first"),
        ]
    )
    assert index.video_ids == ("L21_V001", "L22_V002")
    assert index.num_documents == 2


def test_build_rejects_duplicate_video_ids() -> None:
    with pytest.raises(ValueError, match="duplicate video id"):
        MediaInfoIndex.build(
            [
                _document("L21_V001", title="a"),
                _document("L21_V001", title="b"),
            ]
        )


def test_build_rejects_non_document_values() -> None:
    with pytest.raises(ValueError, match="MediaInfoDocument"):
        MediaInfoIndex.build([object()])  # type: ignore[list-item]


def test_document_validation_rejects_empty_video_id() -> None:
    with pytest.raises(ValueError, match="video_id"):
        MediaInfoDocument(video_id="", title="t", keywords=(), description="d")


def test_query_on_empty_index_returns_empty() -> None:
    index = MediaInfoIndex.build([])
    assert index.query("anything") == []
    assert index.num_documents == 0


# ---------------------------------------------------------------------------
# Scoring and deterministic ranking
# ---------------------------------------------------------------------------


def test_query_returns_ranked_hits_in_score_order() -> None:
    index = MediaInfoIndex.build(
        [
            _document("L21_V001", title="thuyền du khách", description="thuyền"),
            _document("L22_V002", title="món ngon", description="ẩm thực"),
        ]
    )
    hits = index.query("thuyền")
    assert len(hits) == 1
    hit = hits[0]
    assert isinstance(hit, MediaInfoHit)
    assert hit.rank == 1
    assert hit.video_id == "L21_V001"
    assert hit.score > 0.0


def test_document_with_more_query_terms_ranks_higher() -> None:
    index = MediaInfoIndex.build(
        [
            _document("L21_V001", title="thuyền", description=""),
            _document("L22_V002", title="thuyền chở khách", description=""),
        ]
    )
    hits = index.query("thuyền chở khách")
    assert [hit.video_id for hit in hits] == ["L22_V002", "L21_V001"]
    assert hits[0].score > hits[1].score


def test_field_weighting_ranks_title_match_higher(tmp_path: Path) -> None:
    # Same term in title vs description: title-weighted doc wins with defaults.
    index = MediaInfoIndex.build(
        [
            _document("L21_V001", title="thuyền", description=""),
            _document("L22_V002", title="", description="thuyền"),
        ]
    )
    hits = index.query("thuyền")
    assert [hit.video_id for hit in hits] == ["L21_V001", "L22_V002"]


def test_keywords_field_is_indexed() -> None:
    index = MediaInfoIndex.build(
        [
            _document("L21_V001", title="", keywords=("ẩm thực",), description=""),
            _document("L22_V002", title="", keywords=("thể thao",), description=""),
        ]
    )
    hits = index.query("ẩm thực")
    assert [hit.video_id for hit in hits] == ["L21_V001"]


def test_query_ranking_is_deterministic() -> None:
    documents = [
        _document("L21_V001", title="thuyền chở khách", keywords=("du lịch",), description="trên sông"),
        _document("L22_V002", title="món ngon", keywords=("ẩm thực",), description="công thức"),
        _document("L23_V003", title="thuyền thúng", keywords=("biển",), description="ngư dân"),
    ]
    first = MediaInfoIndex.build(documents)
    second = MediaInfoIndex.build(documents)
    assert first.query("thuyền") == second.query("thuyền")
    assert [(h.rank, h.video_id) for h in first.query("thuyền")] == [
        (h.rank, h.video_id) for h in second.query("thuyền")
    ]


def test_ties_are_resolved_by_video_id() -> None:
    # Two documents with identical weighted token counts: deterministic order.
    index = MediaInfoIndex.build(
        [
            _document("L22_V002", title="thuyền"),
            _document("L21_V001", title="thuyền"),
        ]
    )
    hits = index.query("thuyền")
    assert [hit.video_id for hit in hits] == ["L21_V001", "L22_V002"]
    assert hits[0].score == pytest.approx(hits[1].score)


def test_no_match_query_returns_empty() -> None:
    index = MediaInfoIndex.build([_document("L21_V001", title="thuyền")])
    assert index.query("xe tải") == []


def test_top_k_limits_results() -> None:
    index = MediaInfoIndex.build(
        [
            _document(f"L21_V{i:03d}", title="thuyền")
            for i in range(1, 6)
        ]
    )
    hits = index.query("thuyền", top_k=3)
    assert [hit.rank for hit in hits] == [1, 2, 3]
    assert len(hits) == 3


def test_top_k_exceeding_documents_returns_all() -> None:
    index = MediaInfoIndex.build([_document("L21_V001", title="thuyền")])
    hits = index.query("thuyền", top_k=50)
    assert len(hits) == 1


def test_query_rejects_invalid_top_k() -> None:
    index = MediaInfoIndex.build([_document("L21_V001", title="thuyền")])
    with pytest.raises(ValueError, match="top_k"):
        index.query("thuyền", top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        index.query("thuyền", top_k=-2)
    with pytest.raises(ValueError, match="top_k"):
        index.query("thuyền", top_k=1.5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Loading from a media-info directory
# ---------------------------------------------------------------------------


def test_build_from_dir_loads_all_valid_files(tmp_path: Path) -> None:
    directory = tmp_path / "media-info"
    directory.mkdir()
    _write_media_info(directory, "L21_V001", title="thuyền du khách")
    _write_media_info(directory, "L22_V002", title="món ngon")
    _write_media_info(directory, "L23_V003", title="thể thao")

    index = MediaInfoIndex.build_from_dir(directory)

    assert index.num_documents == 3
    assert index.video_ids == ("L21_V001", "L22_V002", "L23_V003")
    assert index.problems == ()
    hits = index.query("thuyền")
    assert [hit.video_id for hit in hits] == ["L21_V001"]


def test_build_from_dir_ignores_non_json_files(tmp_path: Path) -> None:
    directory = tmp_path / "media-info"
    directory.mkdir()
    _write_media_info(directory, "L21_V001", title="thuyền")
    (directory / ".DS_Store").write_bytes(b"")
    (directory / "notes.txt").write_text("not media")

    index = MediaInfoIndex.build_from_dir(directory)

    assert index.num_documents == 1
    assert index.problems == ()


def test_build_from_dir_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="media-info directory"):
        MediaInfoIndex.build_from_dir(tmp_path / "absent")


def test_build_from_dir_strict_reports_malformed_json(tmp_path: Path) -> None:
    directory = tmp_path / "media-info"
    directory.mkdir()
    _write_media_info(directory, "L21_V001", title="thuyền")
    (directory / "L22_V002.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="L22_V002.json"):
        MediaInfoIndex.build_from_dir(directory)


def test_build_from_dir_strict_reports_missing_keys(tmp_path: Path) -> None:
    directory = tmp_path / "media-info"
    directory.mkdir()
    _write_media_info(directory, "L21_V001", title="thuyền")
    (directory / "L22_V002.json").write_text(json.dumps({"title": "only"}), encoding="utf-8")

    with pytest.raises(ValueError, match="L22_V002.json"):
        MediaInfoIndex.build_from_dir(directory)


def test_build_from_dir_strict_reports_unexpected_filenames(tmp_path: Path) -> None:
    directory = tmp_path / "media-info"
    directory.mkdir()
    _write_media_info(directory, "L21_V001", title="thuyền")
    (directory / "not-a-video.json").write_text(json.dumps(_MEDIA_BASE), encoding="utf-8")

    with pytest.raises(ValueError, match="not-a-video.json"):
        MediaInfoIndex.build_from_dir(directory)


def test_build_from_dir_lenient_skips_problem_files(tmp_path: Path) -> None:
    directory = tmp_path / "media-info"
    directory.mkdir()
    _write_media_info(directory, "L21_V001", title="thuyền")
    _write_media_info(directory, "L23_V003", title="món ngon")
    (directory / "L22_V002.json").write_text("{not json", encoding="utf-8")

    index = MediaInfoIndex.build_from_dir(directory, strict=False)

    assert index.num_documents == 2
    assert index.video_ids == ("L21_V001", "L23_V003")
    assert any(name == "L22_V002.json" for name, _message in index.problems)


def test_build_from_dir_missing_and_valid_mix_is_deterministic(tmp_path: Path) -> None:
    directory = tmp_path / "media-info"
    directory.mkdir()
    _write_media_info(directory, "L23_V003", title="món ngon")
    (directory / "L22_V002.json").write_text("{broken", encoding="utf-8")
    _write_media_info(directory, "L21_V001", title="thuyền")

    first = MediaInfoIndex.build_from_dir(directory, strict=False)
    second = MediaInfoIndex.build_from_dir(directory, strict=False)

    assert first.video_ids == second.video_ids == ("L21_V001", "L23_V003")
    assert first.query("thuyền") == second.query("thuyền")


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    index = MediaInfoIndex.build(
        [
            _document("L21_V001", title="thuyền", keywords=("du lịch",), description="sông"),
            _document("L22_V002", title="món ngon", description="ẩm thực"),
        ],
        config=MediaInfoRetrievalConfig(title_weight=5.0),
    )
    path = tmp_path / "index.json"

    index.save(path)

    assert path.is_file()
    loaded = MediaInfoIndex.load(path)
    assert loaded.config == index.config
    assert loaded.num_documents == index.num_documents
    assert loaded.video_ids == index.video_ids
    assert loaded.query("thuyền") == index.query("thuyền")
    assert loaded.query("ẩm thực") == index.query("ẩm thực")


def test_serialized_artifact_carries_schema_version(tmp_path: Path) -> None:
    index = MediaInfoIndex.build([_document("L21_V001", title="thuyền")])
    path = tmp_path / "index.json"
    index.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == MEDIA_INFO_INDEX_SCHEMA_VERSION
    assert payload["num_documents"] == 1


def test_load_rejects_unknown_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "index.json"
    path.write_text(json.dumps({"schema_version": 99, "config": {}, "documents": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema version"):
        MediaInfoIndex.load(path)


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        MediaInfoIndex.load(tmp_path / "absent.json")


def test_save_and_load_from_directory_round_trip(tmp_path: Path) -> None:
    directory = tmp_path / "media-info"
    directory.mkdir()
    _write_media_info(directory, "L21_V001", title="thuyền du khách")
    _write_media_info(directory, "L22_V002", title="món ngon")
    index = MediaInfoIndex.build_from_dir(directory)
    path = tmp_path / "index.json"
    index.save(path)

    loaded = MediaInfoIndex.load(path)

    assert loaded.query("thuyền") == index.query("thuyền")
    assert loaded.video_ids == index.video_ids


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_media_info_search_end_to_end(tmp_path: Path) -> None:
    from aic2026.cli import main

    directory = tmp_path / "media-info"
    directory.mkdir()
    _write_media_info(directory, "L21_V001", title="thuyền du khách")
    _write_media_info(directory, "L22_V002", title="món ngon")
    output_path = tmp_path / "hits.json"

    argv = [
        "media-info-search",
        "--media-info-dir",
        str(directory),
        "--query",
        "thuyền",
        "--top-k",
        "2",
        "--output",
        str(output_path),
    ]
    exit_code = main(argv)

    assert exit_code == 0
    assert output_path.is_file()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["queries"]["thuyền"][0]["video_id"] == "L21_V001"
    assert payload["queries"]["thuyền"][0]["rank"] == 1
    assert payload["queries"]["thuyền"][0]["score"] > 0.0


def test_cli_media_info_search_custom_config(tmp_path: Path) -> None:
    from aic2026.cli import main

    directory = tmp_path / "media-info"
    directory.mkdir()
    _write_media_info(directory, "L21_V001", title="thuyền")
    _write_media_info(directory, "L22_V002", title="", description="thuyền")

    argv = [
        "media-info-search",
        "--media-info-dir",
        str(directory),
        "--query",
        "thuyền",
        "--top-k",
        "2",
        "--title-weight",
        "1.0",
        "--description-weight",
        "1.0",
    ]
    exit_code = main(argv)

    assert exit_code == 0


def test_cli_media_info_search_missing_directory(tmp_path: Path) -> None:
    from aic2026.cli import main

    exit_code = main(
        [
            "media-info-search",
            "--media-info-dir",
            str(tmp_path / "absent"),
            "--query",
            "thuyền",
        ]
    )

    assert exit_code == 1


def test_cli_media_info_search_requires_query(tmp_path: Path) -> None:
    from aic2026.cli import main

    directory = tmp_path / "media-info"
    directory.mkdir()
    _write_media_info(directory, "L21_V001", title="thuyền")

    exit_code = main(
        [
            "media-info-search",
            "--media-info-dir",
            str(directory),
        ]
    )

    assert exit_code == 1


def test_cli_media_info_search_invalid_k1(tmp_path: Path) -> None:
    from aic2026.cli import main

    directory = tmp_path / "media-info"
    directory.mkdir()
    _write_media_info(directory, "L21_V001", title="thuyền")

    exit_code = main(
        [
            "media-info-search",
            "--media-info-dir",
            str(directory),
            "--query",
            "thuyền",
            "--k1",
            "-1",
        ]
    )

    assert exit_code == 1


# ---------------------------------------------------------------------------
# Real dataset smoke test
# ---------------------------------------------------------------------------


def test_real_media_info_directory_builds_and_queries() -> None:
    path = Path(__file__).resolve().parents[1] / "data/raw/aic2026_batch1/media-info"
    if not path.is_dir():
        pytest.skip("raw media-info directory not available")
    index = MediaInfoIndex.build_from_dir(path)
    assert index.num_documents == 873
    assert index.problems == ()
    hits = index.query("thuyền chở nhiều du khách", top_k=5)
    assert len(hits) <= 5
    assert all(hit.rank == expected for hit, expected in zip(hits, range(1, len(hits) + 1)))
    assert all(hit.video_id in index.video_ids for hit in hits)
