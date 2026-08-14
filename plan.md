# dustmaps-api — development plan

A small, fast HTTP service that answers dust-map queries for
[ztf.snad.space](https://ztf.snad.space). It replaces the in-process Python `dustmaps`
queries in `/Users/hombit/projects/supernovaAD/ztf/web`, which today force every viewer
worker to hold ~GB-scale maps in RAM.

Home: `snad-space/dustmaps-api`.
Status: plan only, nothing implemented. Baseline: empty repo (`Cargo.toml` +
`src/main.rs` stub).

## TODO

- [x] **M0** axum skeleton: `/api/v1/health`, config, tracing, fmt/clippy/test green
- [ ] **M1** geometry: ICRS→Galactic + `ang2pix` RING/NESTED, golden-tested vs astropy/healpy
- [ ] **M2** `prep`: download + convert CSFD → `csfd_ebv.npy` (f32)
- [ ] **M3** CSFD endpoint: mmap reader, `spawn_blocking`, golden test green
- [ ] **M4** `prep`: Bayestar19 → dense nested lookup table + `best_fit` f32, equivalence self-check
- [ ] **M5** Bayestar endpoint: DM interpolation (3 branches), footprint handling, golden test green
- [ ] **M6** Dockerfile: multi-stage (prep → build → slim runtime), healthcheck, docker golden-test CI job
- [ ] **M7** perf: criterion + load benches, `madvise` tuning, numbers in README
- [ ] **M8** viewer integration PR in `ztf/web`: HTTP clients replace `dustmaps`, drop healpy/h5py/data

Details in §9; the same list with context lives there.

---

## 1. Objective and scope

**Objective.** One Rust/axum service, one Docker image, serving exactly the two dust-map
queries the ZTF viewer makes, with:

- no full load of map data into RAM (mmap + OS page cache),
- results matching Python `dustmaps` to ≲1e-6 relative, verified by tests,
- no query ever blocking the async runtime.

**Scope — only what ztf.snad.space uses.** From `ztf_viewer/catalogs/extinction/`:

| Map | Python call site | What the viewer needs |
|---|---|---|
| CSFD (Chiang 2023) | `CSFDQuery()(coord)` — `csfd.py` | E(B–V) at (ra, dec) |
| Bayestar19 | `BayestarQuery(max_samples=0)(coord, mode='best')` — `bayestar.py` | best-fit reddening at (ra, dec, distance) |

Nothing else. No SFD, no other Bayestar modes (`samples`, `median`, `percentile`,
`random_sample`), no `max_samples > 0`, no distance-free Bayestar profile queries, no
generic "dust map" trait to unify the two. **Explicitly allowed and expected: the two maps
get different storage layouts, different code paths, and different API shapes.** They are
different data structures; pretending otherwise costs more than it saves.

Scope is tight by decision, not by accident. Also **out** of v1, each because the viewer
does not use it today:

- **QA flags.** No CSFD `mask.fits`, no Bayestar `converged` / `DM_reliable_min|max`.
  The viewer ignores them; we do not store or serve them. (Adding them later means a new
  data-format version and an image rebuild — accepted.)
- **Batch endpoints.** The viewer queries one coordinate at a time. Single-coordinate GETs
  only; a POST batch is a later addition if a caller ever needs one.
- **The band-extinction arithmetic** (`_BaseExtinctionQuery.__call__`: `R_V = 3.1`, the
  `af2av` ZTF filter table). Three multiplications and a ZTF-specific constant table —
  stays in the viewer.

The service is for ztf.snad.space and nothing else, so the API speaks the viewer's
convention directly: both endpoints return **E(B–V)**, with the Bayestar `0.884` factor
(`bayestar.py:24`) already applied server-side.

---

## 2. What "same as Python" means, precisely

Both queries are simple enough to reimplement exactly. The reference semantics, read off
`dustmaps` master:

### CSFD (`csfd.py` + `healpix_map.py`)

1. `coords.transform_to('galactic')` → `(l, b)` in degrees.
2. `hp.ang2pix(nside, theta=90°−b, phi=l, nest=False)` — **RING ordering**, nearest
   pixel, no interpolation.
3. `ebv = pix_val[ipix]`, from `csfd_ebv.fits['xtension'].data['T'].flatten()` (float64).

(`mask.fits` and the `return_flags` path are out of scope — see §1.)

Deterministic, no interpolation: the only difference from Python is that we store the map
as float32 (§3), so our answer is Python's answer rounded to f32 — ≤6e-8 relative, well
inside the 1e-6 budget. The *pixel index* is still expected to match exactly.

### Bayestar19, `mode='best'` (`bayestar.py`)

1. `(l, b, d_kpc)` in Galactic.
2. Multi-resolution pixel lookup: for each `nside` level present in `pixel_info`,
   `ipix = ang2pix(nside, l, b, nest=True)`, binary-search the sorted `healpix_index`
   list for that level; **later (larger) nside levels overwrite earlier matches** — the
   loop in `_find_data_idx` iterates `np.unique(nside)` ascending and assigns on match,
   so the finest matching level wins. No match at any level → `NaN` (outside the PS1
   footprint, dec < −30°).
3. `dm = 5·(log10(d_kpc) + 2)`; `bin_idx_ceil = searchsorted(DM_bin_edges, dm)`.
4. Three cases against `best_fit[pix, :]` (float32, `n_dist` = `len(DM_bin_edges)` = 120):
   - `bin_idx_ceil == 0`: `10^(0.2·(dm − DM_bin_edges[0])) · best_fit[pix, 0]`
   - `bin_idx_ceil == n_dist`: `best_fit[pix, -1]`
   - otherwise: linear-in-DM interpolation,
     `a = (DM_ceil − dm)/(DM_ceil − DM_floor)`, `(1−a)·v[ceil] + a·v[ceil−1]`
5. The viewer then multiplies by `0.884` to get E(B–V) (`bayestar.py:24`).

Two subtleties the tests must pin down:

- **float32 accumulation.** Python computes the interpolation in float32 (`ret` is
  `dtype='f4'`, `a` is float64 → the product promotes, then stores back to f4). We will
  compute in f64 and compare with tolerance 1e-6 *relative*, which comfortably covers the
  f4 storage rounding of the final value. Document this; do not chase bit-equality here.
- **`searchsorted` side.** Default `side='left'`: `dm` exactly equal to a bin edge picks
  that edge as the ceiling. Must be reproduced exactly (Rust `partition_point`).

### Frame conversion

Both maps are Galactic; the viewer hands ICRS `SkyCoord`s. We implement the ICRS→Galactic
rotation as the fixed 3×3 matrix astropy uses (Hipparcos pole/zero-point constants). This
is exact to ~1e-12 rad and is itself golden-tested against astropy, because a frame bug
would silently shift pixels near boundaries.

**Tolerance policy.** Value agreement ≤1e-6 relative (or ≤1e-9 absolute for tiny E(B–V)).
Pixel-index agreement must be *exact* — a mismatched pixel is not a rounding error, it is
a bug, and it shows up as a large value difference. Tests assert both.

---

## 3. On-disk format (built at image build time)

Design rule: **the file layout is the index**. Every query is O(1) address arithmetic plus
one or two page faults, straight out of an mmap — no decoding, no chunk lookup, no HDF5 or
FITS in the Rust binary.

**Format: `.npy`.** Three files, each a single `numpy` array:

```
/data/
  csfd_ebv.npy            # (201 326 592,) f32, RING order          (~805 MB)
  bayestar_lookup.npy     # (12·1024²,)   u32, NESTED, best_fit row or 0xFFFFFFFF (~50 MB)
  bayestar_bestfit.npy    # (n_pix, 120)  f32, C order
```

`.npy` is raw little-endian C-order data behind a ~128-byte header, with the data start
64-byte aligned. So it *is* the flat-array plan — same bytes, same page behaviour,
`mmap + offset` still works — and in exchange we get:

- **self-description.** dtype, shape and byte order are in the file. A wrong-dtype or
  truncated file is caught by the header, not inferred from `file_len` arithmetic.
- **a free debugging and testing path.** `np.load(path, mmap_mode='r')` opens these in one
  line, so the prep self-checks, the golden generator, and any future "what does the map
  say here?" poke at the *shipped* files rather than a re-derivation of them.
- **no meaningful cost.** Rust reads it with an existing crate — `ndarray-npy`'s
  `ViewNpyExt::view_npy` over the mmap'd bytes, which is zero-copy and exists precisely
  for this. We do **not** hand-roll a header parser: the header is a Python `repr` dict,
  and parsing Python literals by hand is exactly the kind of wheel that gets subtly wrong.
  (`npyz` is the alternative if we ever want to drop the `ndarray` dependency; both are
  maintained and both support mmap'd input.)

**Alternatives considered.** The access pattern is narrow — integer index in, one or two
scalars out, one reader process, files written and read by the same image — so most format
machinery solves problems we do not have:

- **Raw flat binary.** The closest call. Zero parsing anywhere, but dtype and shape then
  live only in Rust constants, and correctness rests entirely on file-length assertions.
  `.npy` is this plus a self-check, for one dependency.
- **Arrow IPC** — cross-language columnar interop, schema evolution, chunked record
  batches. None of it load-bearing here, and it costs `arrow-rs` plus an indirection
  between pixel index and byte offset.
- **FITS** — astronomy-native and CSFD ships as FITS, but it puts `libcfitsio` back in the
  runtime image (dropping it from the viewer's image is part of the win), the mature Rust
  binding is a cfitsio wrapper, it is big-endian, and our Bayestar lookup table has no
  natural FITS representation. Interop would matter if anyone else read these files;
  nobody does.
- **safetensors** — one file with all three named arrays, JSON header, Rust-native crate,
  mmap-designed. Loses on the Python side: its numpy loader is eager, so prep self-checks
  would pull 805 MB into RAM instead of memmapping it.
- **HDF5** — `libhdf5` back in the runtime image, and chunking/compression breaks the
  direct-offset property this whole design rests on.
- **Zarr/N5, Parquet** — chunked and/or compressed, i.e. a decode per read. Actively
  hostile to random single-element access.

`.npy` is the smallest thing that is still a real format.

Shapes are still **constants in the Rust source**, because we pin two specific data
releases: `CSFD_NSIDE = 4096`, `BAYESTAR_NSIDE = 1024`, `N_DIST = 120`, and the 120
`DM_BIN_EDGES` values. `n_pix` is the one free number, read from the `bestfit` header.

Both halves assert against those constants:

- the **prep script** checks the values it read from the raw upstream files against the
  Rust constants and fails the build on mismatch — if upstream ever reships a different
  nside or DM grid, the image does not build;
- the **server** checks each `.npy` header's dtype and shape at startup and exits
  otherwise, so a truncated or half-written file can never become silent NaNs.

**CSFD.** Stored as float32 (decision: f32 is enough). The FITS file holds float64, and the
`f64 → f32` round trip costs ≤6e-8 relative — two orders of magnitude inside the 1e-6
requirement, and far below CSFD's own uncertainty. In exchange the array halves to ~805 MB
(nside 4096 → 201 326 592 pixels; verify at prep time), which twice as much of fits in page
cache and makes the image meaningfully smaller. The server reads the f32 and widens to f64
for the response. Values are converted with plain `astype(np.float32)` (round-to-nearest-
even), and the golden test compares against Python's f64 answers at the 1e-6 tolerance, so
the rounding is measured rather than assumed.

**Bayestar lookup: a flattened quadtree, no traversal at runtime.**

Bayestar is multi-resolution: pixels are stored at several `nside` levels (64…1024 for
bayestar2019 — verify at prep time), and a query must find which stored pixel, at whatever
level, contains the coordinate. That is a tree descent — but we never perform one, because
**the NESTED HEALPix index already *is* the quadtree path**: the nested index at nside
1024, shifted right by 2 bits, is the nested index of its parent at nside 512, and so on.
The hierarchy is address arithmetic, not pointers.

So prep flattens the whole structure to its leaf level: a dense array of
12·1024² = 12 582 912 `u32` slots (~50 MB), each holding the `best_fit` row index of
whichever stored pixel covers that leaf, or `0xFFFFFFFF` for "outside the footprint".
A coarse pixel at nside 64 simply writes its row index into all (1024/64)² = 256 leaves it
covers. Filled **coarsest level first, finest last**, so the overwrite order reproduces
Python's "finest matching level wins" semantics by construction.

Runtime cost per query: one `ang2pix` (analytic, no lookup) and one array index. No tree,
no binary search, no branch on level. That is as simple as this gets, and it is why the
50 MB is worth spending.

Alternatives, and when they'd win:

- **Per-level sorted arrays + binary search** — what Python's `_find_data_idx` does: one
  binary search per nside level, ~5 searches of ~22 steps each per query. Smaller
  (~12 MB) but an order of magnitude more scattered memory touches, and materially more
  code to keep bug-free. Rejected.
- **Single sorted array of leaf ranges** (`(start_leaf, row)` pairs, MOC/NUNIQ-flavoured):
  because every stored pixel covers a *contiguous* run of leaves, one binary search over
  ~`n_pix` pairs answers the query — roughly half the size of the dense table, at ~22
  probes instead of 1. **This becomes the better choice if the finest nside turns out to
  be 2048 rather than 1024**, since the dense table would then be ~200 MB. Prep measures
  the real value in M4; the plan switches on that number.
- **Hash map / minimal perfect hash over stored pixels** — O(1) like the dense table, but
  either built in RAM at startup (violates the never-fully-load rule) or a static MPH
  structure that is real complexity for no gain over a 50 MB array. Rejected.

The build script must *verify* the flattening, not assume it: after building, it queries
Python's `_find_data_idx` and the table on a large random sample plus every level's pixel
centers, and fails the build on any disagreement.

**How much is the pre-index actually worth?** Honestly: not much, and it should not be
oversold. Warm, the dense table saves ~100 ns per query over a binary search — invisible
next to ~10–50 µs of HTTP overhead. Cold, it is one page fault versus ~8–10, so ~100 µs
versus ~1 ms, at a traffic level of a handful of queries per viewer page view. The table
is chosen because `lookup[ipix]` is the simplest correct thing, not because the search was
a bottleneck. It costs 50 MB and stays reversible: it is confined to prep plus one line in
the server.

Note also that **the only binary search left at runtime is over `DM_bin_edges`** — 120
f64 values, 960 bytes, permanently in L1, ~7 iterations. That one is unavoidable in any
design.

**Why not SQLite (or any embedded KV store)?** Because our keys *are* array offsets. A
HEALPix index is a dense integer in `[0, 12·nside²)` by construction; asking a B-tree
"what is stored at position *i*?" pays a tree descent to rediscover *i*.

- **CSFD would get strictly worse.** One addressed read becomes a ~4-level B-tree descent
  (4 page reads) plus record-header and varint decoding, and storage grows from 4 bytes
  per pixel to ~10–15 with per-row overhead: **~2–3 GB instead of 805 MB**, for 201 M rows
  whose key is a counter.
- **`best_fit` would gain nothing** — 120 floats per row becomes a BLOB, i.e. raw bytes
  and our own offset arithmetic, now behind a B-tree.
- **The Bayestar pixel lookup is the one defensible case** (a range query over sparse-ish
  keys), but that is the binary search above with more overhead per probe.
- And it adds `rusqlite`, a connection pool, and the same `spawn_blocking` requirement.

SQLite earns its keep with sparse keys, variable-size rows, unpredictable queries, or
mutation. We have dense integer keys, fixed-size values, two query shapes known at build
time, and read-only data. If the scope ever grew to arbitrary cone searches, or to many
maps queried by name, this answer flips.

**CSFD needs no index at all.** It is a full-sky dense map at a single nside — `ang2pix` in
RING ordering gives the array index directly.

**`best_fit`.** `n_pix × 120 × f32`, row-major, so the two distance bins needed by one
query are adjacent — a single page fault. NaN rows are not stored; missing pixels are
signalled by `0xFFFFFFFF` in the lookup table.

**Endianness/alignment.** `.npy` written with the native little-endian dtypes (`<f4`,
`<u4`); the server rejects a big-endian descriptor rather than silently byte-swapping. The
64-byte-aligned data start satisfies f32/u32 alignment for zero-copy views. The files are
built and consumed in the same image, by the same commit, so there is no cross-version
compatibility problem to design for and no layout-version field.

---

## 4. Data preparation (Python, `uv`)

`prep/` — a tiny `uv`-run package, *not* a shipped dependency of the server.

**One environment**, including `dustmaps`: `cdshealpix`, `astropy`, `h5py`, `numpy`,
`dustmaps`. Run as `uv run --project prep python -m prep.build --out /data`. Using
`dustmaps` here is worth it — it owns the download URLs and checksums, and it gives us
`BayestarQuery._find_data_idx`, which is the reference our flattened lookup table must
agree with. Its transitive `healpy` costs nothing at runtime: the prep stage is a
*discarded* Docker build stage, and only `/data` is copied into the final image.

**We still never call `healpy` ourselves.** Every HEALPix computation we write uses
`cdshealpix` (`cdshealpix.nested.lonlat_to_healpix` /
`cdshealpix.ring.lonlat_to_healpix`, depth = log2(nside)) — the Python binding of the
*same* Rust library the server uses. Identical implementation on both sides removes a
whole class of pixel-boundary disagreement between prep and runtime: they can only
disagree with each other if they disagree with themselves. `healpy` appears solely inside
`dustmaps`, as the reference we are checking against. (`mocpy` is the fallback if we ever
need MOC algebra; plain index conversion does not.)
- Sources, fetched with `dustmaps.fetch_utils.download_and_verify` — the same URLs and
  checksums the viewer's Dockerfile uses today:
  - Bayestar: `https://sai.snad.space/tmp/viewer-files/bayestar2019-bestfit.h5`
    (md5 `4dd35460f1da9bb4f4e535f25eb0c530`) — the best-fit-only copy SNAD hosts, because
    `dustmaps.bayestar.fetch()` is blocked by the Harvard Dataverse WAF
    (gregreen/dustmaps#54). This file already lacks `samples`, which is exactly our scope.
  - CSFD: Zenodo record 8207175, `csfd_ebv.fits` only (md5
    `31cd2eec51bcb5f106af84a610ced53c`). We skip `mask.fits` — no flags in scope.
- Steps: download + md5-verify → read (`astropy.io.fits`, `h5py`) → assert
  shapes/`DM_bin_edges` against the Rust constants (§3) → build the flattened lookup with
  `cdshealpix` → `np.save` the three arrays → reopen with `np.load(mmap_mode='r')` and
  self-check. Reading back the *shipped* files, rather than the in-memory arrays, is the
  main practical reason for `.npy`.
- The flattening self-check runs **in the build**, against `dustmaps` itself: the
  flattened table is compared with `BayestarQuery._find_data_idx` over a large random
  sample plus every stored pixel's own center. It is authoritative (it is the reference
  implementation) and it is exactly where a `cdshealpix` vs `healpy` boundary
  disagreement would surface. A mismatch fails the image build rather than shipping.
- Every executable entry point sits behind `if __name__ == "__main__": main()`.
- Idempotent: skip work when the output file exists with the expected header and size.

---

## 5. Server

**Stack.** `axum` + `tokio`, `memmap2` + `ndarray-npy` (`ViewNpyExt`, zero-copy `.npy`
views over the mmap — an existing crate, no hand-rolled header parsing),
`cdshealpix` (`healpix` crate) for `ang2pix` in
both NESTED and RING, `serde`/`serde_json`, `tracing`, `clap` for config. No
astropy-equivalent dependency — the ICRS→Galactic matrix is ~20 lines. (No `rayon` in v1;
there are no batch queries to parallelise.)

**API** — everything under `/api/v1/`, matching the other SNAD services the viewer already
calls (`ztf_viewer/catalogs/conesearch/*.py`, `model_fit.py`). Single-coordinate, and
deliberately *not* unified between the two maps:

```
GET  /api/v1/csfd?ra=<deg>&dec=<deg>
     → {"ebv": 0.0312}

GET  /api/v1/bayestar2019?ra=<deg>&dec=<deg>&distance=<pc>
     → {"ebv": 0.2395}          # 0.884 × best-fit map value, the viewer's convention
     → {"ebv": null}            # outside the PS1 footprint

GET  /api/v1/health  → 200 once the files are mapped and size-checked
```

`ra`/`dec` are ICRS degrees, `distance` is parsecs — matching the viewer's `SkyCoord`s.
No `frame` parameter: the viewer always has ICRS.

No `/version` endpoint. The deployed image tag is the version; nothing in the API needs to
report it, and a service that answers questions about itself is exactly the kind of
surface this plan is trying not to grow. `/api/v1/health` exists only because Docker's
`HEALTHCHECK` needs a target.

Out-of-footprint Bayestar is `"ebv": null` with HTTP 200, not an error — that is normal
sky, and the viewer already handles "no value" (`CatalogUnavailable` / `NotFound` paths).
Bad input (non-finite, |dec| > 90, distance ≤ 0) → 400 with a message.

**Never blocking.** A single query is a handful of arithmetic ops plus 1–2 potential page
faults. A page fault on a cold mmap is a *synchronous disk read* that stalls the whole
tokio worker thread, so:

- every map access runs inside `tokio::task::spawn_blocking`, so a cold-page read stalls a
  blocking-pool thread and never a runtime worker;
- `madvise(MADV_RANDOM)` on the big arrays; the 50 MB Bayestar lookup table is
  `MADV_WILLNEED`-hinted (and optionally `mlock`ed, config flag) since it is hit by every
  single query;
- graceful shutdown, request timeout, and a concurrency limit (`tower` layers) so a cold
  cache degrades latency instead of exhausting the blocking pool.

Alternative considered and rejected for v1: a dedicated thread pool with io_uring/direct
reads. mmap + `spawn_blocking` is enough at ztf.snad.space's traffic, and the page cache
does the right thing once warm.

**Config: none.** The data lives at `/data` and the server listens on `0.0.0.0:80` — both
fixed by our own Dockerfile, in our own image, for a service with one deployment. Making
them configurable would add env vars nobody ever sets. They are constants in the source.

The one exception is `RUST_LOG`, which we get for free from
`tracing_subscriber::EnvFilter` and did not invent. Tests point at their own data
directory through a test-only argument, not an env var.

If a benchmark later shows blocking-pool sizing or `mlock` actually matters, those become
flags *then*, backed by a number.

**Startup.** mmap the three `.npy` files, validate each header (dtype, shape, byte order)
against the constants, serve. Fail fast and loudly on any mismatch — a truncated or
wrong-dtype data file must not become silent NaNs.

---

## 6. Tests

The correctness claim is the whole point of the service, so tests come before polish.

1. **Golden fixtures** (committed, small — a few hundred KB). `prep/golden.py` runs the
   real Python `dustmaps` over:
   - ~5 000 uniform-on-sphere coords (fixed seed),
   - the viewer's actual usage pattern: a set of real ZTF object coordinates,
   - edge cases: poles, `l = 0/360` seam, `b = ±90`, coords exactly on HEALPix pixel
     boundaries, Bayestar footprint edge (dec ≈ −30°), distances below the first DM bin,
     above the last, and exactly on bin edges, plus known out-of-footprint coords.

   Output: JSON with inputs and Python's answers (plus intermediate `l`, `b`, `pix_idx`,
   so a failure says *which stage* diverged).

2. **Rust integration tests** (`tests/golden.rs`) load the fixtures and open the map files
   from `/data`, asserting exact pixel indices and ≤1e-6 relative values. Skipped with a
   clear message when `/data` is absent, so `cargo test` still works on a laptop; inside
   the image, where the tests actually run against real data, `/data` is simply there.

3. **Unit tests, no big data needed** — small committed fixtures:
   - ICRS→Galactic against astropy goldens (≤1e-9 deg),
   - `ang2pix` NESTED and RING against healpy goldens at all relevant nsides,
   - the DM-interpolation function against a hand-computed table covering all three
     branches.

4. **HTTP-level tests**: malformed input → 400, out-of-footprint → `"ebv": null` with 200,
   `/api/v1/health`.

5. **Test jobs**, whatever CI system we settle on (§7): a fast job (units + HTTP) that
   needs no map data, and a slow job that builds the image and runs the golden tests
   against real data — the latter downloads several GB, so it belongs on a nightly/tag
   trigger, not every push.

6. **Bench** (`criterion` + a `wrk`/`oha` script): cold-cache and warm-cache p50/p99 for
   both endpoints under concurrency. Recorded in the README as the baseline; the target is
   warm p99 well under a millisecond.

---

## 7. Docker and deployment

Deployed at **`dustmaps.snad.space`**, exactly like the other SNAD APIs
(`fit.lc.snad.space`, `ogle3.snad.space`, `tns.snad.space`, `periodic.ztf.snad.space`): an
`nginx-proxy` + `acme-companion` front end discovers the container on the **external
`proxy` network** through `VIRTUAL_HOST` / `HTTPS_METHOD` / `DYNDNS_HOST` /
`LETSENCRYPT_HOST` / `LETSENCRYPT_EMAIL`.

### Dockerfile

Multi-stage, one self-contained image:

1. `python` stage with `uv` — downloads raw maps and runs `prep` → `/data`.
2. `rust` stage — `cargo build --release` (musl or `debian:slim`-compatible glibc;
   decide once, prefer a slim glibc image over musl since mmap-heavy workloads are
   happier with glibc's allocator).
3. Final stage: `debian:*-slim` (or distroless) + the binary + `/data`.

`HEALTHCHECK` hits `/api/v1/health`. Non-root user. `EXPOSE 80` to match the viewer's convention.

**Decided: the data is baked into the image** (~2–3 GB, to be measured). Reproducible, no
runtime downloads, no volumes to manage — the image *is* the deployable unit. The raw
downloads and the conversion live in cached early layers, so source edits do not re-fetch
gigabytes; keeping the download and the convert steps as separate `RUN`s matters here.

If registry/pull time later becomes a real problem, the conversion is deterministic, so
publishing pre-converted artifacts and fetching them at first start remains available as a
pure optimisation. Not now.

**Undecided: where the image is built and published.** The Dockerfile stays
registry-agnostic — no hard-coded image names, no CI-specific assumptions — so that either
"build and push from CI" or "build in place on the deploy host" remains possible without
touching it. Same for the test jobs in §6: they are plain commands first, workflow files
only once we pick a home.

### `docker-compose.yml`

Modelled on the most recently maintained single-service SNAD repos — `ztf-reference`'s
`app` service and `web-light-curve-features` — not on the older `ports:`-carrying ones:

```yaml
services:
  app:
    build: .
    environment:
      VIRTUAL_HOST: dustmaps.snad.space
      HTTPS_METHOD: noredirect
      DYNDNS_HOST: dustmaps.snad.space
      LETSENCRYPT_HOST: dustmaps.snad.space
      LETSENCRYPT_EMAIL: letsencrypt@snad.space
    networks:
      - proxy
    restart: always

networks:
  proxy:
    external: true
```

**No `ports:`.** nginx-proxy reaches the container over the shared `proxy` network on the
container's own port 80, which is its default, so there is no `VIRTUAL_PORT` either.
Publishing a host port would only create a second, unencrypted way in that bypasses TLS
termination. (`model-fit-api` and `snad-ogle3` still carry `ports: - "80"`; `ztf-reference`
and `web-light-curve-features`, the two most recently touched, do not. We follow the
latter.)

No internal `app` network either — that exists in the other repos to isolate a Postgres
container from the proxy, and we have no database.

### `docker-compose-dev.yml`

SNAD's dev convention is a `dev.` subdomain through the *same* proxy, not a loopback port
(`ztf-reference/docker-compose-dev.yml`). Overriding only the three host variables:

```yaml
services:
  app:
    environment:
      VIRTUAL_HOST: dev.dustmaps.snad.space
      DYNDNS_HOST: dev.dustmaps.snad.space
      LETSENCRYPT_HOST: dev.dustmaps.snad.space
```

The production file's environment is *only* the five proxy variables, so the dev override
is only the three that name the host.

For laptop iteration a further local override can bind-mount a pre-built `/data` (cf.
`ztf-reference`, which host-mounts `/srv/data/ztf-reference/...`), so rebuilding the server
does not re-download and re-convert several GB each time.

**Memory limits.** `web-light-curve-features` sets `mem_limit: 2g`. We should *not* copy
that blindly: mmap'd page cache counts toward a cgroup's memory limit, so a tight limit
would fight the page cache this design depends on. It is reclaimable, so a limit is not
fatal — but if we set one at all it should be generous and chosen after M7's benchmarks
show the real warm working set.

---

## 8. Integration with ztf.snad.space (follow-up, separate PR in the `web` repo)

A full replacement, not a fallback arrangement. The viewer stops knowing what a dust map
is; it calls an API like it calls every other SNAD API.

- `DUSTMAPS_API_URL: http://dustmaps.snad.space` in the viewer's `docker-compose.yml`
  environment, next to `OGLE_III_API_URL` and `TNS_API_URL`, with the matching entry in
  `ztf_viewer/config.py`.
- `ztf_viewer/catalogs/extinction/` becomes two small HTTP clients keeping today's
  `ebv(coord)` / `__call__(coord)` surface and raising `CatalogUnavailable` when the
  service is unreachable — so `viewer.py` (lines ~911, ~1508, ~1517) does not change. The
  `af2av`/`R_V = 3.1` arithmetic in `_base.py` stays; it was never dust-map code.
  `_BaseLocalExtinctionQuery` and its `new_local_query`/lazy-load machinery go away
  entirely — there is no local query any more.
- **`NO_LOCAL_3D_DUST_MAP` is deleted**, from `config.py` and from every deployment. It
  existed to let a machine skip loading a multi-GB map; nothing loads a map now.
- **`dustmaps`, `healpy` and `h5py` come out of `pyproject.toml`/`uv.lock`**, and
  `libhdf5-dev` + `libcfitsio-dev`, the `/dustmapsrc` config, `DUSTMAPS_CONFIG_FNAME`, the
  `BAYESTAR_URL` build arg and both map-download `RUN` steps come out of the viewer's
  `Dockerfile`. That image loses several GB and its two slowest build steps.
- The viewer's existing tests keep the answers honest across the switch: same coordinates
  in, same numbers out.

---

## 9. Milestones

- [ ] **M0 — skeleton.** axum app, `/api/v1/health`, config, tracing, fmt/clippy/test. No
      data.
- [ ] **M1 — geometry.** ICRS→Galactic + `ang2pix` (RING/NESTED) with astropy/healpy
      golden unit tests. This is the foundation both maps stand on.
- [ ] **M2 — prep for CSFD.** Download, convert to `csfd_ebv.npy`, confirm nside 4096.
- [ ] **M3 — CSFD endpoint.** mmap reader, `spawn_blocking`, golden test green.
- [ ] **M4 — prep for Bayestar.** Lookup-table build + the `_find_data_idx` equivalence
      self-check; confirm finest nside and `n_pix`.
- [ ] **M5 — Bayestar endpoint.** DM interpolation with all three branches, footprint
      handling, golden test green.
- [ ] **M6 — Docker.** Multi-stage image, healthcheck, docker-based golden test job.
- [ ] **M7 — perf.** Benchmarks, `madvise` tuning, documented numbers.
- [ ] **M8 — viewer integration.** PR in `ztf/web` (see §8), deployed behind the existing
      `pr<N>.ztf.snad.space` preview before merge.

---

## 10. Decisions and remaining questions

**Decided** (Konstantin, 2026-08-13):

- Scope is exactly what the viewer already does for itself — no QA flags, no batch, no
  band-extinction math, no `frame` parameter, no second consumer to design for.
- Data is baked into the image (§7).
- The SNAD-hosted `bayestar2019-bestfit.h5` is permanent; `prep` depends on it directly.
- Storage is mmap'd flat arrays, not SQLite or an embedded KV store: HEALPix indices are
  already dense array offsets, so a B-tree would add indirection and roughly triple CSFD's
  size (§3).
- Data files are `.npy` — self-describing, still mmap-able at a fixed offset — read on the
  Rust side with the existing `ndarray-npy` crate, not a hand-written header parser.
  Alternatives (raw, Arrow, FITS, safetensors, HDF5, Zarr, Parquet) weighed in §3. No
  sidecar metadata files: shapes stay Rust constants, checked both ways.
- CSFD is stored as `f32` (~805 MB), not `f64` — the rounding is ≤6e-8 relative (§3).
  Both map arrays are therefore f32, and all arithmetic happens in f64.
- Repo home is `snad-space/dustmaps-api`.
- Attribution (Chiang 2023 for CSFD, Green et al. 2019 for Bayestar19, Green 2018 for
  `dustmaps` itself) goes in the README, with the exact data provenance. Not in the API —
  the viewer already handles attribution toward its own users.

**Still open:**

1. Build/publish home — CI + a registry (e.g. ghcr.io) vs. building on the deploy host.
   Left out of the plan deliberately; §7 keeps the Dockerfile registry-agnostic so this
   can be decided late.
2. Where does the service run — its own container in the viewer's `docker-compose.yml`
   (so it is only reachable inside the compose network), or exposed under a public path on
   ztf.snad.space? The plan assumes internal-only.
2. Does the viewer need a timeout/fallback story when the service is down, beyond raising
   `CatalogUnavailable` as it does today?
