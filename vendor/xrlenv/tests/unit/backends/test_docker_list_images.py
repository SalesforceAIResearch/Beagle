"""Unit tests for ``DockerBackend.list_images`` layer-share accounting.

The backend pairs ``client.images.list()`` (which reports each image's
total ``Size`` including shared layers) with ``client.df()`` (which
reports per-image ``SharedSize`` — bytes shared with other tagged
images on the same node) so callers can derive the **unique**
incremental footprint of caching one more image when its base layers
are already present from a sibling.

``xrlenv build calibrate`` consumes ``shared_size_bytes`` to write
the bin-packer-relevant ``unique = size - shared`` into plan YAMLs;
the legacy ``size_bytes`` path over-counts shared layers and inflates
FFD reservation in plans where many images share a common base.

The ``client.df()`` probe is **opt-in**: ``list_images`` only calls it
when ``include_shared_size=True`` (it walks the whole layer graph and is
expensive under load — the node's hot-path stats refresh skips it). These
tests therefore pass the flag explicitly to exercise the SharedSize path.

These tests stub the Docker SDK so the helper logic can be exercised
without a live daemon.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

from xrlenv.backends.docker import DockerBackend


def _backend_with_client(client: Any) -> DockerBackend:
    """Construct a DockerBackend with a stubbed docker.DockerClient.

    ``list_images`` only touches ``self._client``; the heavyweight
    config + mount roots aren't read on this code path. Bypass
    ``__init__`` and patch the one attribute the method needs.
    """
    backend = DockerBackend.__new__(DockerBackend)
    backend._client = client  # type: ignore[attr-defined]
    return backend


def _image(
    *, image_id: str, tag: str | None, size: int,
) -> Any:
    img = MagicMock()
    img.id = image_id
    img.tags = [tag] if tag else []
    img.attrs = {
        "Size": size,
        "RepoDigests": [],
        "Config": {"Labels": {}},
    }
    return img


def test_list_images_populates_shared_size_from_system_df() -> None:
    """When ``client.df()`` returns a SharedSize-bearing payload,
    every image's ``shared_size_bytes`` reflects the matching id."""
    client = MagicMock()
    client.images.list.return_value = [
        _image(image_id="sha256:a", tag="my/a:1", size=1_500_000_000),
        _image(image_id="sha256:b", tag="my/b:1", size=2_000_000_000),
    ]
    client.df.return_value = {
        "Images": [
            {"Id": "sha256:a", "Size": 1_500_000_000,
             "SharedSize": 1_000_000_000},
            {"Id": "sha256:b", "Size": 2_000_000_000,
             "SharedSize":   800_000_000},
        ],
    }
    backend = _backend_with_client(client)
    records = asyncio.new_event_loop().run_until_complete(
        backend.list_images(include_shared_size=True),
    )
    by_name = {r.name: r for r in records}
    assert by_name["my/a:1"].size_bytes == 1_500_000_000
    assert by_name["my/a:1"].shared_size_bytes == 1_000_000_000
    assert by_name["my/b:1"].size_bytes == 2_000_000_000
    assert by_name["my/b:1"].shared_size_bytes == 800_000_000


def test_list_images_falls_back_when_df_unavailable() -> None:
    """Older daemons (or transient df failures) leave
    ``shared_size_bytes`` as ``None``; ``size_bytes`` is still
    populated from ``client.images.list()`` as before. Calibrate
    sees ``shared_size_bytes=None`` and reverts to the legacy
    ``size_bytes``-only accounting (no behavior break)."""
    client = MagicMock()
    client.images.list.return_value = [
        _image(image_id="sha256:a", tag="my/a:1", size=1_500_000_000),
    ]
    client.df.side_effect = RuntimeError("df not supported")
    backend = _backend_with_client(client)
    records = asyncio.new_event_loop().run_until_complete(
        backend.list_images(include_shared_size=True),
    )
    assert len(records) == 1
    assert records[0].size_bytes == 1_500_000_000
    assert records[0].shared_size_bytes is None


def test_list_images_handles_image_missing_from_df_response() -> None:
    """``client.df()`` may omit an image (race between the
    ``images.list`` call and the ``df`` call). Those images get
    ``shared_size_bytes=None`` (consistent with the no-df case);
    images present in df get their ``SharedSize`` as expected."""
    client = MagicMock()
    client.images.list.return_value = [
        _image(image_id="sha256:a", tag="my/a:1", size=1_000_000_000),
        _image(image_id="sha256:b", tag="my/b:1", size=2_000_000_000),
    ]
    # Only my/a:1 is in the df response.
    client.df.return_value = {
        "Images": [
            {"Id": "sha256:a", "Size": 1_000_000_000,
             "SharedSize": 500_000_000},
        ],
    }
    backend = _backend_with_client(client)
    records = asyncio.new_event_loop().run_until_complete(
        backend.list_images(include_shared_size=True),
    )
    by_name = {r.name: r for r in records}
    assert by_name["my/a:1"].shared_size_bytes == 500_000_000
    assert by_name["my/b:1"].shared_size_bytes is None


def test_list_images_tolerates_malformed_df_entries() -> None:
    """``df()`` entries with missing/wrong-typed Id or SharedSize
    fields are skipped silently. Surviving entries still propagate
    correctly. Defensive against Docker API drift across versions."""
    client = MagicMock()
    client.images.list.return_value = [
        _image(image_id="sha256:a", tag="my/a:1", size=1_000_000_000),
    ]
    client.df.return_value = {
        "Images": [
            {"Id": "sha256:a", "Size": 1_000_000_000,
             "SharedSize": 300_000_000},
            # Missing Id — skipped.
            {"Size": 999, "SharedSize": 100},
            # Negative SharedSize (shouldn't happen, defensive) — skipped.
            {"Id": "sha256:weird", "SharedSize": -1},
            # Wrong type for SharedSize — skipped.
            {"Id": "sha256:other", "SharedSize": "lots"},
        ],
    }
    backend = _backend_with_client(client)
    records = asyncio.new_event_loop().run_until_complete(
        backend.list_images(include_shared_size=True),
    )
    by_name = {r.name: r for r in records}
    assert by_name["my/a:1"].shared_size_bytes == 300_000_000
