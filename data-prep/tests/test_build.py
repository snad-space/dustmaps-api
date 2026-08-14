import numpy as np

from data_prep import bayestar, csfd


def test_valid_output_checks_shape_dtype_and_truncation(tmp_path, monkeypatch):
    monkeypatch.setattr(csfd, "CSFD_NPIX", 4)
    destination = tmp_path / "csfd_ebv.npy"
    np.save(destination, np.zeros(4, dtype=np.float32), allow_pickle=False)

    assert csfd._valid_output(destination)

    with destination.open("r+b") as file:
        file.truncate(destination.stat().st_size - 1)
    assert not csfd._valid_output(destination)


def test_valid_output_rejects_wrong_dtype(tmp_path, monkeypatch):
    monkeypatch.setattr(csfd, "CSFD_NPIX", 4)
    destination = tmp_path / "csfd_ebv.npy"
    np.save(destination, np.zeros(4, dtype="<f8"), allow_pickle=False)

    assert not csfd._valid_output(destination)


def test_build_lookup_fills_parents_then_overwrites_with_finer_pixels():
    pixel_info = np.array(
        [(1, 0), (2, 0)],
        dtype=[("nside", "u4"), ("healpix_index", "u8")],
    )
    old_npix = bayestar.BAYESTAR_NPIX
    old_nside = bayestar.BAYESTAR_NSIDE
    try:
        bayestar.BAYESTAR_NPIX = 16
        bayestar.BAYESTAR_NSIDE = 2
        lookup = bayestar._build_lookup(pixel_info)
    finally:
        bayestar.BAYESTAR_NPIX = old_npix
        bayestar.BAYESTAR_NSIDE = old_nside

    expected = np.full(16, bayestar.BAYESTAR_MISSING, dtype=np.uint32)
    expected[:4] = 0
    expected[0] = 1
    assert np.array_equal(lookup, expected)


def test_read_bayestar_rejects_wrong_distance_grid(tmp_path):
    h5py = __import__("h5py")
    path = tmp_path / "bad.h5"
    pixel_info = np.array([(1, 0)], dtype=[("nside", "u4"), ("healpix_index", "u8")])
    with h5py.File(path, "w") as file:
        dataset = file.create_dataset("pixel_info", data=pixel_info)
        dataset.attrs["DM_bin_edges"] = np.arange(bayestar.BAYESTAR_N_DIST, dtype=float)
        file.create_dataset(
            "best_fit", data=np.zeros((1, bayestar.BAYESTAR_N_DIST), dtype=np.float32)
        )

    try:
        bayestar._read(path)
    except RuntimeError as error:
        assert "DM_bin_edges" in str(error)
    else:
        raise AssertionError("wrong DM grid must be rejected")
