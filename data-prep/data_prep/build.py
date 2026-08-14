"""Download the CSFD map and convert it to the service's flat ``.npy`` file.

This module intentionally keeps the raw FITS dependency in the prep environment;
the Rust service only needs the resulting NumPy array.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from dustmaps.fetch_utils import download_and_verify

CSFD_URL = "https://zenodo.org/records/8207175/files/csfd_ebv.fits?download=1"
CSFD_MD5 = "31cd2eec51bcb5f106af84a610ced53c"
CSFD_NSIDE = 4096
CSFD_NPIX = 12 * CSFD_NSIDE**2
OUTPUT_NAME = "csfd_ebv.npy"


def _valid_output(path: Path) -> bool:
    """Return whether *path* is an exact, readable CSFD f32 NPY array."""

    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        valid = (
            array.shape == (CSFD_NPIX,)
            and array.dtype == np.dtype(np.float32)
            and array.offset + array.nbytes == path.stat().st_size
        )
        return valid
    except (OSError, ValueError, EOFError):
        return False


def _download_source(raw_dir: Path) -> Path:
    """Fetch the source through dustmaps' URL/checksum helper."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    source = raw_dir / "csfd_ebv.fits"
    download_and_verify(CSFD_URL, CSFD_MD5, str(source))
    return source


def _read_csfd(source: Path) -> np.ndarray:
    """Read and validate the FITS extension used by ``dustmaps.csfd``."""

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

        converted = np.asarray(values, dtype=np.float32)

    return converted


def build(out_dir: Path) -> Path:
    """Build and return ``out_dir/csfd_ebv.npy``.

    A valid existing output is reused, making Docker rebuilds and local reruns
    idempotent. An invalid output is replaced only after conversion succeeds.
    """

    destination = out_dir / OUTPUT_NAME
    if _valid_output(destination):
        return destination

    source = _download_source(out_dir / "raw")
    values = _read_csfd(source)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(destination, values, allow_pickle=False)

    if not _valid_output(destination):
        raise RuntimeError(f"wrote an invalid NPY file: {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="output data directory")
    args = parser.parse_args()
    output = build(args.out)
    print(f"ready: {output}")


if __name__ == "__main__":
    main()
