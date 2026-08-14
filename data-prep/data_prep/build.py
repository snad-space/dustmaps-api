"""Build all data files consumed by the dustmaps-api service."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import bayestar, csfd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="output data directory")
    args = parser.parse_args()

    csfd_path = csfd.build(args.out)
    lookup_path, bestfit_path = bayestar.build(args.out)
    for path in (csfd_path, lookup_path, bestfit_path):
        print(f"ready: {path}")


if __name__ == "__main__":
    main()
