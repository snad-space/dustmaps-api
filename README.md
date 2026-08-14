# dustmaps-api

A small, fast HTTP service that answers dust-extinction queries: given a sky
position (and, for Bayestar19, a distance), it returns E(B−V), matching the
reference Python `dustmaps` package.

## Layout

- `app/`: the Rust/axum server. Mmaps the prebuilt `.npy` maps and serves the
- ve
  endpoints.
- `data-prep/`: a `uv`-run Python package that downloads the raw CSFD and
  Bayestar19 releases and converts them to the flat `.npy` layout the server
  reads. Not a runtime dependency of the server.
- `integration-tests/`: golden tests that compare the running server's
  answers against the reference Python `dustmaps` package.

## Running

```console
$ docker compose up --build
```

Builds the maps, compiles the server, and serves on port 80 (see
`docker-compose.yml` / `Dockerfile`). There is no configuration beyond
`RUST_LOG`; data path and listen address are fixed for this single
deployment.

For local dev, `docker compose -f docker-compose-dev.yml up --build` serves
on `localhost:8080`.

## Testing

```console
$ cargo test --manifest-path app/Cargo.toml          # unit + HTTP tests, no map data needed
$ uv run --project data-prep --group test pytest -q data-prep/tests
$ docker build --target golden-tests -t dustmaps-api:golden-tests . \
    && docker run --rm dustmaps-api:golden-tests      # full pipeline vs. Python dustmaps
```

## Attribution

- CSFD: Chiang (2023).
- Bayestar19: Green et al. (2019).
- Both accessed during data prep via the [`dustmaps`](https://dustmaps.readthedocs.io)
  package (Green 2018).

## License

[MIT](LICENSE).
