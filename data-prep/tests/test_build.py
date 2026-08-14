import numpy as np

from data_prep import build


def test_valid_output_checks_shape_dtype_and_truncation(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "CSFD_NPIX", 4)
    destination = tmp_path / "csfd_ebv.npy"
    np.save(destination, np.zeros(4, dtype=np.float32), allow_pickle=False)

    assert build._valid_output(destination)

    with destination.open("r+b") as file:
        file.truncate(destination.stat().st_size - 1)
    assert not build._valid_output(destination)


def test_valid_output_rejects_wrong_dtype(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "CSFD_NPIX", 4)
    destination = tmp_path / "csfd_ebv.npy"
    np.save(destination, np.zeros(4, dtype="<f8"), allow_pickle=False)

    assert not build._valid_output(destination)
