"""Shared helpers for real-data API integration tests."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode
from urllib.request import urlopen

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from dustmaps.healpix_map import HEALPixQuery

EDGE_COORDINATES = (
    (0.0, 0.0),
    (266.4051, -28.936175),
    (83.6331, 22.0145),
    (180.0, 45.0),
    (359.9, -89.0),
)
RANDOM_COUNT = 1_000
RANDOM_SEED = 20260813


def coordinates() -> list[tuple[float, float]]:
    """Return edge cases plus a reproducible uniform-on-sphere sample."""

    rng = np.random.default_rng(RANDOM_SEED)
    ra = rng.uniform(0.0, 360.0, RANDOM_COUNT)
    sin_b = rng.uniform(-1.0, 1.0, RANDOM_COUNT)
    dec = np.degrees(np.arcsin(sin_b))
    return list(EDGE_COORDINATES) + list(zip(ra.tolist(), dec.tolist()))


def dustmaps_values(map_path, sample: list[tuple[float, float]]) -> np.ndarray:
    sky = SkyCoord(
        [ra for ra, _ in sample] * u.deg,
        [dec for _, dec in sample] * u.deg,
        frame="icrs",
    )
    with fits.open(map_path, memmap=True) as hdul:
        values = hdul["xtension"].data["T"].flatten()
        return HEALPixQuery(values, False, "galactic")(sky)


def api_values(server_url: str, sample: list[tuple[float, float]]) -> list[float]:
    def query(item: tuple[float, float]) -> float:
        ra, dec = item
        params = urlencode({"ra": ra, "dec": dec})
        with urlopen(
            f"{server_url.rstrip('/')}/api/v1/csfd?{params}", timeout=10
        ) as response:
            return json.load(response)["ebv"]

    with ThreadPoolExecutor(max_workers=32) as executor:
        return list(executor.map(query, sample))
