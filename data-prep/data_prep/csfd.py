"""Prepare the CSFD map for the Rust service."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from dustmaps.fetch_utils import download_and_verify

CSFD_URL = "https://zenodo.org/records/8207175/files/csfd_ebv.fits?download=1"
CSFD_MD5 = "31cd2eec51bcb5f106af84a610ced53c"
CSFD_NSIDE = 2048
CSFD_NPIX = 12 * CSFD_NSIDE**2
OUTPUT_NAME = "csfd_ebv.npy"


def _valid_output(path: Path) -> bool:
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        return (
            array.shape == (CSFD_NPIX,)
            and array.dtype == np.dtype(np.float32)
            and array.offset + array.nbytes == path.stat().st_size
        )
    except (OSError, ValueError, EOFError):
        return False


def _download_source(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    source = raw_dir / "csfd_ebv.fits"
    download_and_verify(CSFD_URL, CSFD_MD5, str(source))
    return source


def _read(source: Path) -> np.ndarray:
    with fits.open(source, memmap=True) as hdul:
        try:
            values = hdul["xtension"].data["T"].flatten()
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("CSFD FITS is missing xtension.T") from error
        if values.size != CSFD_NPIX:
            raise RuntimeError(
                f"expected {CSFD_NPIX} CSFD pixels for nside {CSFD_NSIDE}, "
                f"got {values.size}"
            )
        return np.asarray(values, dtype=np.float32)


def build(out_dir: Path) -> Path:
    destination = out_dir / OUTPUT_NAME
    if _valid_output(destination):
        return destination

    values = _read(_download_source(out_dir / "raw"))
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(destination, values, allow_pickle=False)
    if not _valid_output(destination):
        raise RuntimeError(f"wrote an invalid NPY file: {destination}")
    return destination
