from __future__ import annotations

import os

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from dustmaps.bayestar import BayestarQuery

from integration_tests.utils import api_bayestar_values


EBV_FACTOR = 0.884
SAMPLE = (
    (0.0, 0.0),
    (266.4051, -28.936175),
    (83.6331, 22.0145),
    (180.0, 45.0),
    (359.9, -89.0),
    (194.7545309478567, -18.2512760863485),
    (184.24879007626018, 25.47694717342729),
    (331.7934220050504, -35.695893147402366),
)
DISTANCES_PC = (1.0, 10.0, 100.0, 10_000.0, 59_566.214, 60_000.0)
RANDOM_COUNT = 1_000
RANDOM_SEED = 20260813
RANDOM_DISTANCE_MIN_PC = 1.0
RANDOM_DISTANCE_MAX_PC = 60_000.0


def test_bayestar_matches_dustmaps() -> None:
    server_url = os.environ["DUSTMAPS_API_URL"]
    map_path = os.environ["BAYESTAR_H5"]

    rng = np.random.default_rng(RANDOM_SEED)
    random_ra = rng.uniform(0.0, 360.0, RANDOM_COUNT)
    random_dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, RANDOM_COUNT)))
    coordinates = list(SAMPLE) + list(zip(random_ra.tolist(), random_dec.tolist()))
    sample = [
        (ra, dec, distance_pc)
        for ra, dec in coordinates
        for distance_pc in DISTANCES_PC
    ]
    random_distances = 10.0 ** rng.uniform(
        np.log10(RANDOM_DISTANCE_MIN_PC),
        np.log10(RANDOM_DISTANCE_MAX_PC),
        RANDOM_COUNT,
    )
    sample.extend(
        (ra, dec, distance_pc)
        for ra, dec, distance_pc in zip(
            random_ra.tolist(), random_dec.tolist(), random_distances.tolist()
        )
    )
    sky = SkyCoord(
        [ra for ra, _, _ in sample] * u.deg,
        [dec for _, dec, _ in sample] * u.deg,
        [distance_pc for _, _, distance_pc in sample] * u.pc,
        frame="icrs",
    )
    expected = (
        np.asarray(
            BayestarQuery(map_path, max_samples=0)(sky, mode="best"), dtype=np.float64
        )
        * EBV_FACTOR
    )
    actual = np.asarray(api_bayestar_values(server_url, sample), dtype=np.float64)

    expected_missing = ~np.isfinite(expected)
    actual_missing = ~np.isfinite(actual)
    assert np.array_equal(actual_missing, expected_missing)

    finite = ~expected_missing
    tolerance = np.maximum(1e-9, np.abs(expected) * 1e-6)
    assert np.all(np.abs(actual[finite] - expected[finite]) <= tolerance[finite])
