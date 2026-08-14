# Build the maps in a disposable Python stage.  Keeping the raw downloads in
# this stage means they never enter the runtime image.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS prep

WORKDIR /build/data-prep
COPY data-prep/pyproject.toml data-prep/uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY data-prep/data_prep ./data_prep
RUN uv run --no-sync python -m data_prep.build --out /data

# Compile without adding a Rust toolchain to the runtime image.
FROM rust:trixie AS build

WORKDIR /build
COPY app/Cargo.toml app/Cargo.lock ./app/
COPY app/src ./app/src
RUN cargo build --manifest-path app/Cargo.toml --release

# This target is intentionally separate from the small runtime image. It
# retains the raw sources and Python dependencies needed for golden tests.
FROM prep AS golden-tests

WORKDIR /build/integration-tests
COPY integration-tests/pyproject.toml integration-tests/uv.lock ./
RUN uv sync --locked --no-install-project
COPY integration-tests/tests ./tests
COPY integration-tests/integration_tests ./integration_tests
RUN uv sync --locked
COPY .ci/run-golden-tests.sh /usr/local/bin/run-golden-tests
COPY --from=build /build/app/target/release/dustmaps-api /usr/local/bin/dustmaps-api
RUN chmod 755 /usr/local/bin/run-golden-tests
CMD ["/usr/local/bin/run-golden-tests"]

FROM debian:trixie-slim AS runtime

RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --create-home dustmaps
COPY --from=build /build/app/target/release/dustmaps-api /usr/local/bin/dustmaps-api
COPY --from=prep /data /data
USER dustmaps
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1/api/v1/health || exit 1
ENTRYPOINT ["/usr/local/bin/dustmaps-api"]
