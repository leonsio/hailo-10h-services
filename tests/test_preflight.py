import errno
from pathlib import Path

import pytest

from hailo_services.preflight import check_writable_directory


def test_checks_actual_write_and_rename_without_leaving_files(tmp_path):
    check_writable_directory(tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("error", [errno.EROFS, errno.EACCES])
def test_explains_mount_and_permission_failures(monkeypatch, error):
    def fail(**kwargs):
        raise OSError(
            error, "read-only filesystem" if error == errno.EROFS else "permission denied"
        )

    monkeypatch.setattr("hailo_services.preflight.tempfile.NamedTemporaryFile", fail)
    with pytest.raises(RuntimeError, match="ReadWritePaths") as result:
        check_writable_directory(Path("/model-store"))
    assert result.value.__cause__.errno == error
