from __future__ import annotations

import os

import numpy as np

from integration_tests.utils import api_values, coordinates, dustmaps_values


def test_csfd_matches_dustmaps() -> None:
    server_url = os.environ["DUSTMAPS_API_URL"]
    map_path = os.environ["CSFD_FITS"]
    sample = coordinates()
    expected = dustmaps_values(map_path, sample)
    actual = np.asarray(api_values(server_url, sample))

    assert np.all(np.isfinite(actual))
    assert np.allclose(actual, expected, rtol=1e-6, atol=1e-9)
