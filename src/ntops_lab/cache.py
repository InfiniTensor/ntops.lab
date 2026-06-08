import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CacheStats:
    path: Path
    files: int
    bytes: int


def ninetoothed_cache_path() -> Path:
    return Path.home() / ".ninetoothed"


def triton_cache_path() -> Path:
    configured = os.environ.get("TRITON_CACHE_DIR")
    return Path(configured).expanduser() if configured else Path.home() / ".triton" / "cache"


def cache_stats(path: Path) -> CacheStats:
    files = 0
    size = 0
    if path.exists():
        for item in path.rglob("*"):
            if item.is_file():
                files += 1
                size += item.stat().st_size
    return CacheStats(path=path, files=files, bytes=size)


def format_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")
