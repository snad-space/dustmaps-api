#!/bin/sh
set -eu

DUSTMAPS_DATA_DIR=/data \
DUSTMAPS_LISTEN_ADDR=127.0.0.1:8080 \
/usr/local/bin/dustmaps-api >/tmp/dustmaps-api.log 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT

attempt=0
until uv run --no-sync python -c \
  'from urllib.request import urlopen; urlopen("http://127.0.0.1:8080/api/v1/health")' \
  >/dev/null 2>&1; do
    attempt=$((attempt + 1))
    if [ "$attempt" -ge 30 ]; then
        cat /tmp/dustmaps-api.log
        exit 1
    fi
    sleep 1
done

DUSTMAPS_API_URL=http://127.0.0.1:8080 \
CSFD_FITS=/data/raw/csfd_ebv.fits \
BAYESTAR_H5=/data/raw/bayestar2019-bestfit.h5 \
uv run --no-sync pytest -q tests
