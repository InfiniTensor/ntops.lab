from ntops_lab.cache import cache_stats, format_size


def test_cache_stats(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "source.py").write_bytes(b"123")
    (tmp_path / "nested" / "kernel.bin").write_bytes(b"12345")

    stats = cache_stats(tmp_path)

    assert stats.path == tmp_path
    assert stats.files == 2
    assert stats.bytes == 8


def test_format_size():
    assert format_size(512) == "512.0 B"
    assert format_size(1536) == "1.5 KiB"
