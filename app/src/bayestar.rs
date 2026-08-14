//! Memory-mapped reader for the Bayestar19 best-fit map.

use std::{
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
};

use memmap2::{Advice, Mmap};
use ndarray::{ArrayView1, ArrayView2};
use ndarray_npy::ViewNpyExt;
use thiserror::Error;

use crate::geometry::ang2pix_nested;

pub const BAYESTAR_NSIDE: u32 = 1024;
pub const BAYESTAR_NPIX: usize = 12 * (BAYESTAR_NSIDE as usize).pow(2);
pub const BAYESTAR_N_DIST: usize = 120;
pub const BAYESTAR_MISSING: u32 = u32::MAX;
pub const BAYESTAR_EBV_FACTOR: f32 = 0.884;

const DISTANCE_PC_PER_KPC: f32 = 1000.0;
const DISTANCE_MODULUS_SCALE: f32 = 5.0;
const DISTANCE_MODULUS_OFFSET: f32 = 2.0;
const DM_BIN_START: f32 = 4.0;
const DM_BIN_STEP: f32 = 0.125;
const DISTANCE_SCALE_BASE: f32 = 10.0;
const DISTANCE_SCALE_EXPONENT: f32 = 0.2;

#[derive(Debug, Error)]
pub enum BayestarError {
    #[error("failed to open {path}: {source}")]
    Open {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to mmap {path}: {source}")]
    Mmap {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid Bayestar NPY {path}: {source}")]
    InvalidNpy {
        path: PathBuf,
        #[source]
        source: ndarray_npy::ViewNpyError,
    },
    #[error("Bayestar lookup has {actual} pixels, expected {expected}")]
    WrongLookupShape { actual: usize, expected: usize },
    #[error("Bayestar best-fit has shape ({rows}, {cols}), expected ({rows}, {expected_cols})")]
    WrongBestfitShape {
        rows: usize,
        cols: usize,
        expected_cols: usize,
    },
    #[error("Bayestar lookup row {row} is out of range for {rows} best-fit rows")]
    RowOutOfRange { row: u32, rows: usize },
}

#[derive(Debug)]
pub struct BayestarMap {
    lookup_mmap: Mmap,
    bestfit_mmap: Mmap,
    lookup_path: PathBuf,
    bestfit_path: PathBuf,
    n_rows: usize,
}

impl BayestarMap {
    pub fn open(
        lookup_path: impl AsRef<Path>,
        bestfit_path: impl AsRef<Path>,
    ) -> Result<Arc<Self>, BayestarError> {
        let lookup_path = lookup_path.as_ref().to_owned();
        let bestfit_path = bestfit_path.as_ref().to_owned();
        let lookup_mmap = map_file(&lookup_path)?;
        let bestfit_mmap = map_file(&bestfit_path)?;
        // Queries land on scattered pixel/row offsets, so sequential
        // readahead would waste I/O and page-cache space.
        if let Err(source) = lookup_mmap.advise(Advice::Random) {
            tracing::warn!(path = %lookup_path.display(), %source, "madvise(MADV_RANDOM) failed");
        }
        if let Err(source) = bestfit_mmap.advise(Advice::Random) {
            tracing::warn!(path = %bestfit_path.display(), %source, "madvise(MADV_RANDOM) failed");
        }
        // The lookup table is hit by every single query, so warm it into
        // page cache proactively instead of paying cold faults on first use.
        if let Err(source) = lookup_mmap.advise(Advice::WillNeed) {
            tracing::warn!(path = %lookup_path.display(), %source, "madvise(MADV_WILLNEED) failed");
        }
        let map = Self {
            lookup_mmap,
            bestfit_mmap,
            lookup_path,
            bestfit_path,
            n_rows: 0,
        };

        let lookup = map.lookup_view()?;
        if lookup.len() != BAYESTAR_NPIX {
            return Err(BayestarError::WrongLookupShape {
                actual: lookup.len(),
                expected: BAYESTAR_NPIX,
            });
        }
        let bestfit = map.bestfit_view()?;
        let (rows, cols) = bestfit.dim();
        if cols != BAYESTAR_N_DIST {
            return Err(BayestarError::WrongBestfitShape {
                rows,
                cols,
                expected_cols: BAYESTAR_N_DIST,
            });
        }
        Ok(Arc::new(Self {
            n_rows: rows,
            ..map
        }))
    }

    fn lookup_view(&self) -> Result<ArrayView1<'_, u32>, BayestarError> {
        ArrayView1::<u32>::view_npy(&self.lookup_mmap).map_err(|source| BayestarError::InvalidNpy {
            path: self.lookup_path.clone(),
            source,
        })
    }

    fn bestfit_view(&self) -> Result<ArrayView2<'_, f32>, BayestarError> {
        ArrayView2::<f32>::view_npy(&self.bestfit_mmap).map_err(|source| {
            BayestarError::InvalidNpy {
                path: self.bestfit_path.clone(),
                source,
            }
        })
    }

    /// Query E(B-V) for an ICRS coordinate and distance in parsecs.
    /// `None` means the coordinate is outside the Bayestar footprint.
    pub fn query(
        &self,
        ra_deg: f64,
        dec_deg: f64,
        distance_pc: f64,
    ) -> Result<Option<f32>, BayestarError> {
        let galactic = crate::geometry::icrs_to_galactic(ra_deg, dec_deg);
        let pixel = ang2pix_nested(BAYESTAR_NSIDE, galactic.l_deg, galactic.b_deg) as usize;
        let row = self
            .lookup_view()?
            .get(pixel)
            .copied()
            .unwrap_or(BAYESTAR_MISSING);
        if row == BAYESTAR_MISSING {
            return Ok(None);
        }
        if row as usize >= self.n_rows {
            return Err(BayestarError::RowOutOfRange {
                row,
                rows: self.n_rows,
            });
        }

        let dm = DISTANCE_MODULUS_SCALE
            * (f32::log10(distance_pc as f32 / DISTANCE_PC_PER_KPC) + DISTANCE_MODULUS_OFFSET);
        let bestfit = self.bestfit_view()?;
        let values = bestfit.row(row as usize);
        Ok(Some(BAYESTAR_EBV_FACTOR * interpolate(values, dm)))
    }
}

fn map_file(path: &Path) -> Result<Mmap, BayestarError> {
    let file = File::open(path).map_err(|source| BayestarError::Open {
        path: path.to_owned(),
        source,
    })?;
    // The mapping is read-only and owns the file's contents for its lifetime.
    unsafe { Mmap::map(&file) }.map_err(|source| BayestarError::Mmap {
        path: path.to_owned(),
        source,
    })
}

/// Match numpy.searchsorted(edges, dm, side="left") and Bayestar's three branches.
fn interpolate(values: ArrayView1<'_, f32>, dm: f32) -> f32 {
    let mut edges = [0.0_f32; BAYESTAR_N_DIST];
    for (index, edge) in edges.iter_mut().enumerate() {
        *edge = DM_BIN_START + index as f32 * DM_BIN_STEP;
    }
    // The slice method directly finds the first edge >= dm, matching
    // searchsorted(side="left").
    let ceil = edges.partition_point(|&edge| edge < dm);
    if ceil == 0 {
        f32::powf(
            DISTANCE_SCALE_BASE,
            DISTANCE_SCALE_EXPONENT * (dm - edges[0]),
        ) * values[0]
    } else if ceil == BAYESTAR_N_DIST {
        values[BAYESTAR_N_DIST - 1]
    } else {
        let upper = edges[ceil];
        let lower = edges[ceil - 1];
        let a = (upper - dm) / (upper - lower);
        (1.0_f32 - a) * values[ceil] + a * values[ceil - 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn interpolation_covers_all_branches_and_left_search() {
        let mut values = vec![0.0_f32; BAYESTAR_N_DIST];
        values[0] = 2.0;
        values[1] = 4.0;
        values[BAYESTAR_N_DIST - 1] = 9.0;
        let values = ndarray::Array1::from(values);
        assert!(f32::abs(interpolate(values.view(), 3.0) - 2.0 * f32::powf(10.0, -0.2)) < 1e-6);
        assert_eq!(interpolate(values.view(), 4.125), 4.0);
        assert_eq!(interpolate(values.view(), 4.0), 2.0);
        assert_eq!(interpolate(values.view(), 19.0), 9.0);
    }

    #[test]
    fn open_rejects_wrong_lookup_shape() {
        let dir =
            std::env::temp_dir().join(format!("dustmaps-api-bayestar-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let lookup = dir.join("lookup.npy");
        let bestfit = dir.join("bestfit.npy");
        ndarray_npy::write_npy(&lookup, &array![0_u32]).unwrap();
        ndarray_npy::write_npy(
            &bestfit,
            &ndarray::Array2::<f32>::zeros((1, BAYESTAR_N_DIST)),
        )
        .unwrap();
        assert!(matches!(
            BayestarMap::open(&lookup, &bestfit),
            Err(BayestarError::WrongLookupShape { .. })
        ));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
