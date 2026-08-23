"""Small shared JSON/NDJSON helpers for artifact modules.

These are the low-level primitives behind the embedding and index artifact
contracts (manifests are pretty-printed JSON; frame rows are NDJSON). The
artifact modules keep their typed wrappers; only the duplicated byte-level
behavior lives here. Deliberately boring: no registry, no schema framework.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON object pretty-printed with a trailing newline.

    Expected input:
        - path: target file path.
        - payload: JSON-safe dictionary.

    Returns:
        - None.

    Raises:
        - OSError: if writing fails.
    """
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def read_json_file(path: Path) -> dict[str, Any]:
    """Read one JSON object from disk.

    Expected input:
        - path: source file path.

    Returns:
        - The parsed dictionary.

    Raises:
        - FileNotFoundError: if the file does not exist.
        - json.JSONDecodeError: if the JSON is malformed.
    """
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_ndjson(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Write records as NDJSON, one compact JSON object per line.

    Expected input:
        - path: target file path.
        - records: iterable of JSON-safe dictionaries in row order.

    Returns:
        - Number of lines written.

    Raises:
        - OSError: if writing fails.
    """
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
            count += 1
    return count


def iter_ndjson(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from an NDJSON file, skipping blank lines.

    Expected input:
        - path: source file path.

    Returns:
        - Iterator of parsed dictionaries in file (row) order.

    Raises:
        - FileNotFoundError: if the file does not exist.
        - json.JSONDecodeError: if a line is malformed.
    """
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip() == "":
                continue
            yield json.loads(line)
