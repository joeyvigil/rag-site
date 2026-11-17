"""File storage for uploaded originals.

Phase 1 uses a local directory (a Docker volume shared by the API and the
worker). The interface is deliberately tiny so it can be swapped for S3 /
R2 / GCS later without touching callers.
"""

from pathlib import Path

import anyio

from app.config import settings


class LocalStorage:
    def __init__(self, root: str) -> None:
        self.root = Path(root).resolve()

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root):  # defence against "../" keys
            raise ValueError(f"Invalid storage key: {key}")
        return path

    async def save(self, key: str, data: bytes) -> None:
        def _write() -> None:
            path = self._path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

        await anyio.to_thread.run_sync(_write)

    async def read(self, key: str) -> bytes:
        return await anyio.to_thread.run_sync(self._path(key).read_bytes)

    async def delete(self, key: str) -> None:
        await anyio.to_thread.run_sync(lambda: self._path(key).unlink(missing_ok=True))


storage = LocalStorage(settings.storage_dir)
