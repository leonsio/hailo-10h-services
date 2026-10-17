"""Focused tests for cgroup memory accounting used by the monitoring UI."""

from hailo_services.diagnostics.monitoring import _read_cgroup_memory


def test_cgroup_memory_working_set_subtracts_inactive_file(tmp_path):
    (tmp_path / "memory.current").write_text("5947101184\n", encoding="utf-8")
    (tmp_path / "memory.max").write_text("max\n", encoding="utf-8")
    (tmp_path / "memory.stat").write_text(
        "anon 1562378240\n"
        "file 4356300800\n"
        "inactive_file 3268616192\n"
        "active_file 1045745664\n"
        "file_mapped 997720064\n",
        encoding="utf-8",
    )

    result = _read_cgroup_memory(tmp_path)

    assert result["current_bytes"] == 5947101184
    assert result["working_set_bytes"] == 5947101184 - 3268616192
    assert result["limit_bytes"] is None
    assert result["stat_bytes"]["anon"] == 1562378240
    assert result["stat_bytes"]["file_mapped"] == 997720064


def test_cgroup_memory_working_set_falls_back_without_inactive_file(tmp_path):
    (tmp_path / "memory.current").write_text("1024\n", encoding="utf-8")
    (tmp_path / "memory.max").write_text("4096\n", encoding="utf-8")
    (tmp_path / "memory.stat").write_text("anon 768\nmalformed value\n", encoding="utf-8")

    result = _read_cgroup_memory(tmp_path)

    assert result["working_set_bytes"] == 1024
    assert result["limit_bytes"] == 4096
    assert result["stat_bytes"] == {"anon": 768}
