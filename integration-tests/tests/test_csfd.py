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

    tolerance = np.maximum(1e-9, np.abs(expected) * 1e-6)
    assert np.all(np.isfinite(actual))
    mismatches = np.flatnonzero(np.abs(actual - expected) > tolerance)
    assert len(mismatches) == 0, [
        {
            "ra": sample[index][0],
            "dec": sample[index][1],
            "actual": float(actual[index]),
            "dustmaps": float(expected[index]),
            "tolerance": float(tolerance[index]),
        }
        for index in mismatches[:10]
    ]
