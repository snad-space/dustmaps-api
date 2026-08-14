"""Prepare the Bayestar19 best-fit map for the Rust service."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import astropy.units as u
import h5py
from astropy.coordinates import Latitude, Longitude
from cdshealpix.nested import healpix_to_lonlat, lonlat_to_healpix
from dustmaps.fetch_utils import download_and_verify

BAYESTAR_URL = "https://sai.snad.space/tmp/viewer-files/bayestar2019-bestfit.h5"
BAYESTAR_MD5 = "4dd35460f1da9bb4f4e535f25eb0c530"
BAYESTAR_NSIDE = 1024
BAYESTAR_NPIX = 12 * BAYESTAR_NSIDE**2
BAYESTAR_N_DIST = 120
BAYESTAR_LOOKUP_NAME = "bayestar_lookup.npy"
BAYESTAR_BESTFIT_NAME = "bayestar_bestfit.npy"
BAYESTAR_MISSING = np.uint32(0xFFFFFFFF)
DM_BIN_EDGES = np.arange(4.0, 19.0, 0.125, dtype=np.float64)


def _valid_array(path: Path, shape: tuple[int, ...], dtype: np.dtype[Any]) -> bool:
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        return (
            array.shape == shape
            and array.dtype == dtype
            and array.flags.c_contiguous
            and array.offset + array.nbytes == path.stat().st_size
        )
    except (OSError, ValueError, EOFError):
        return False


def _download_source(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    source = raw_dir / "bayestar2019-bestfit.h5"
    download_and_verify(BAYESTAR_URL, BAYESTAR_MD5, str(source))
    return source


def _read(source: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(source, "r") as file:
        for name in ("/pixel_info", "/best_fit"):
            if name not in file:
                raise RuntimeError(f"Bayestar file is missing {name}")
        pixel_info = file["/pixel_info"][:]
        raw_best_fit = file["/best_fit"][:]
        best_fit = np.asarray(raw_best_fit, dtype=np.float32)
        edges = np.asarray(file["/pixel_info"].attrs["DM_bin_edges"], dtype=np.float64)

    required = {"nside", "healpix_index"}
    missing = required - set(pixel_info.dtype.names or ())
    if missing:
        raise RuntimeError(f"Bayestar pixel_info is missing fields: {sorted(missing)}")
    if raw_best_fit.dtype != np.dtype(np.float32):
        raise RuntimeError(f"expected float32 best_fit, got {raw_best_fit.dtype}")
    if best_fit.ndim != 2 or best_fit.shape[1] != BAYESTAR_N_DIST:
        raise RuntimeError(
            f"expected best_fit shape (n_pix, {BAYESTAR_N_DIST}), got {best_fit.shape}"
        )
    if edges.shape != (BAYESTAR_N_DIST,) or not np.array_equal(edges, DM_BIN_EDGES):
        raise RuntimeError(
            "Bayestar DM_bin_edges differ from the pinned Bayestar19 grid"
        )

    nsides = np.unique(pixel_info["nside"])
    if (
        nsides.size == 0
        or nsides.max() != BAYESTAR_NSIDE
        or np.any(nsides < 1)
        or np.any(nsides > BAYESTAR_NSIDE)
    ):
        raise RuntimeError(f"unsupported Bayestar nside levels: {nsides.tolist()}")
    if np.any(BAYESTAR_NSIDE % nsides != 0) or np.any((nsides & (nsides - 1)) != 0):
        raise RuntimeError(
            f"Bayestar nside levels must be powers of two: {nsides.tolist()}"
        )
    if pixel_info.size != best_fit.shape[0]:
        raise RuntimeError("Bayestar pixel_info and best_fit row counts differ")
    npix_at_level = 12 * nsides.astype(np.uint64) ** 2
    level_indices = np.searchsorted(nsides, pixel_info["nside"])
    if np.any(pixel_info["healpix_index"] >= npix_at_level[level_indices]):
        raise RuntimeError("Bayestar healpix_index is out of range for its nside")
    return pixel_info, best_fit, edges


def _build_lookup(pixel_info: np.ndarray) -> np.ndarray:
    lookup = np.full(BAYESTAR_NPIX, BAYESTAR_MISSING, dtype=np.uint32)
    for nside in np.unique(pixel_info["nside"]):
        shift = 2 * int(np.log2(BAYESTAR_NSIDE // int(nside)))
        mask = pixel_info["nside"] == nside
        pixels = pixel_info["healpix_index"][mask]
        rows = np.flatnonzero(mask)
        span = 1 << shift
        for pixel, row in zip(pixels, rows):
            start = int(pixel) << shift
            lookup[start : start + span] = row
    return lookup


def _table_lookup(
    lon_deg: np.ndarray, lat_deg: np.ndarray, lookup: np.ndarray
) -> np.ndarray:
    pixels = np.asarray(
        lonlat_to_healpix(
            Longitude(lon_deg, u.deg),
            Latitude(lat_deg, u.deg),
            int(np.log2(BAYESTAR_NSIDE)),
        ),
        dtype=np.int64,
    )
    return lookup[pixels]


def _self_check_lookup(
    pixel_info: np.ndarray,
    lookup: np.ndarray,
    reference: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    seed: int = 20260813,
    random_count: int = 20_000,
) -> None:
    rng = np.random.default_rng(seed)
    lon = rng.uniform(0.0, 360.0, random_count)
    lat = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, random_count)))
    expected = np.asarray(reference(lon, lat), dtype=np.int64)
    actual = _table_lookup(lon, lat, lookup).astype(np.int64)
    expected = np.where(expected < 0, -1, expected)
    if not np.array_equal(actual, expected):
        bad = np.flatnonzero(actual != expected)[0]
        raise RuntimeError(f"Bayestar lookup mismatch at random sample {bad}")

    for nside in np.unique(pixel_info["nside"]):
        rows = np.flatnonzero(pixel_info["nside"] == nside)
        lon_center, lat_center = healpix_to_lonlat(
            pixel_info["healpix_index"][rows], int(np.log2(int(nside)))
        )
        expected = np.asarray(
            reference(lon_center.to_value(u.deg), lat_center.to_value(u.deg)),
            dtype=np.int64,
        )
        actual = _table_lookup(
            lon_center.to_value(u.deg), lat_center.to_value(u.deg), lookup
        ).astype(np.int64)
        if not np.array_equal(actual, expected):
            raise RuntimeError(
                f"Bayestar lookup mismatch at nside {nside} pixel center"
            )


def build(out_dir: Path) -> tuple[Path, Path]:
    bestfit_path = out_dir / BAYESTAR_BESTFIT_NAME
    source = _download_source(out_dir / "raw")
    pixel_info, best_fit, _ = _read(source)
    lookup_path = out_dir / BAYESTAR_LOOKUP_NAME
    expected_shape = (best_fit.shape[0], BAYESTAR_N_DIST)

    if not _valid_array(bestfit_path, expected_shape, np.dtype(np.float32)):
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(bestfit_path, best_fit, allow_pickle=False)
    if not _valid_array(lookup_path, (BAYESTAR_NPIX,), np.dtype(np.uint32)):
        out_dir.mkdir(parents=True, exist_ok=True)
        lookup = _build_lookup(pixel_info)
        np.save(lookup_path, lookup, allow_pickle=False)
    else:
        lookup = np.load(lookup_path, mmap_mode="r", allow_pickle=False)

    from dustmaps.bayestar import BayestarQuery

    query = BayestarQuery(str(source), max_samples=0)
    _self_check_lookup(pixel_info, lookup, query._find_data_idx)
    return lookup_path, bestfit_path
