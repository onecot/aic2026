"""BTC experimental query-file contract.

The AIC2026 BTC experimental queries are plain ``query-<name>-<task>.txt``
files (``task`` one of ``kis`` / ``qa`` / ``trake``) whose content is the
exact query text. This module is the single shared contract for reading them:

- :func:`parse_query_filename` / :func:`enumerate_query_files` for filename
  parsing and discovery;
- :func:`read_query_text` for text extraction;
- :func:`normalize_query_text` for whitespace-normalized comparison;
- :class:`BtcQuery` / :func:`load_btc_queries` for typed, ready-to-run
  query records.

Both the inference adapter (:mod:`aic2026.btc_inference`) and the submission
exporter consume this module so that the text stored in prediction artifacts
is byte-identical to the text matched during submission generation. It
contains no retrieval and no artifact logic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_QUERY_FILENAME_PATTERN = re.compile(
    r"^query-(?P<name>[a-zA-Z0-9-]+)-(?P<task>kis|qa|trake)\.txt$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QueryFileInfo:
    """Parsed information from a BTC query filename.

    Expected input:
        - path: the original file path.
        - name: the query name extracted from the filename (e.g. ``p1-1``).
        - task: the task type (``kis``, ``qa``, or ``trake``).

    Returns:
        - A validated query file descriptor.

    Raises:
        - ValueError: if name is empty or task is not kis/qa/trake.
    """

    path: Path
    name: str
    task: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or self.name.strip() == "":
            raise ValueError("name must be a non-empty string")
        if self.task not in ("kis", "qa", "trake"):
            raise ValueError(f"task must be kis, qa, or trake; got {self.task!r}")

    @property
    def csv_name(self) -> str:
        """The BTC CSV filename derived from the query filename."""
        return f"query-{self.name}-{self.task}.csv"


def parse_query_filename(path: Path) -> QueryFileInfo:
    """Extract query name and task from a BTC query filename.

    Expected input:
        - path: a ``query-<name>-<task>.txt`` compatible filename.

    Returns:
        - A ``QueryFileInfo`` with name and task.

    Raises:
        - ValueError: if the filename does not match the expected pattern.
    """
    match = _QUERY_FILENAME_PATTERN.match(path.name)
    if match is None:
        raise ValueError(
            f"query filename {path.name!r} does not match pattern "
            "'query-<name>-<task>.txt'"
        )
    return QueryFileInfo(
        path=path,
        name=match.group("name"),
        task=match.group("task").lower(),
    )


def enumerate_query_files(query_dir: Path) -> list[QueryFileInfo]:
    """Enumerate actual ``query-*-<task>.txt`` files in a directory.

    Expected input:
        - query_dir: directory containing BTC query files.

    Returns:
        - Sorted ``QueryFileInfo`` values for every file matching the query
          filename pattern.

    Raises:
        - FileNotFoundError: if the directory does not exist.
        - ValueError: if no matching query files are found.
    """
    if not query_dir.is_dir():
        raise FileNotFoundError(f"query directory not found: {query_dir}")
    infos: list[QueryFileInfo] = []
    for path in sorted(query_dir.iterdir()):
        if not path.is_file() or not _QUERY_FILENAME_PATTERN.match(path.name):
            continue
        infos.append(parse_query_filename(path))
    if not infos:
        raise ValueError(
            f"no 'query-<name>-<task>.txt' files found in {query_dir}"
        )
    return infos


def read_query_text(path: Path) -> str:
    """Read the exact query text from a BTC query file.

    Expected input:
        - path: path to a ``.txt`` query file.

    Returns:
        - The stripped query text content.

    Raises:
        - FileNotFoundError: if the file does not exist.
        - ValueError: if the file is empty.
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"query file {path.name!r} is empty")
    return text


def normalize_query_text(text: str) -> str:
    """Collapse every whitespace run (including newlines) to one space.

    Expected input:
        - text: raw query text.

    Returns:
        - The normalized text with outer whitespace stripped.
    """
    return " ".join(text.split())


@dataclass(frozen=True)
class BtcQuery:
    """One BTC experimental query ready for prediction.

    Expected input:
        - info: parsed filename descriptor (source of truth for identity).
        - query_id: stable artifact identifier derived from the filename.
        - query_text: the exact stripped file content.

    Returns:
        - A validated BTC query record.

    Raises:
        - ValueError: if any field violates its contract.
    """

    info: QueryFileInfo
    query_id: str
    query_text: str

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or self.query_id.strip() == "":
            raise ValueError("query_id must be a non-empty string")
        if not isinstance(self.query_text, str) or self.query_text.strip() == "":
            raise ValueError("query_text must be a non-empty string")


def load_btc_queries(query_dir: Path) -> tuple[BtcQuery, ...]:
    """Load every BTC query file in a directory as typed query records.

    Query ids follow ``btc-<name>-<task>`` so artifacts remain traceable to
    both the source file and its task. Files are returned in filename order;
    the query text is preserved verbatim (stripped) so downstream strict
    matching against artifacts is exact.

    Expected input:
        - query_dir: directory containing ``query-<name>-<task>.txt`` files.

    Returns:
        - One :class:`BtcQuery` per query file, in filename order.

    Raises:
        - FileNotFoundError: if the directory does not exist.
        - ValueError: if no query files are found or any file is empty.
    """
    return tuple(
        BtcQuery(
            info=info,
            query_id=f"btc-{info.name}-{info.task}",
            query_text=read_query_text(info.path),
        )
        for info in enumerate_query_files(query_dir)
    )
