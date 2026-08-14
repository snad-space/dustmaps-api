//! Memory-mapped reader for the CSFD E(B-V) map.

use std::{
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
};

use memmap2::{Advice, Mmap};
use ndarray::ArrayView1;
use ndarray_npy::ViewNpyExt;
use thiserror::Error;

use crate::geometry::ang2pix_ring;

pub const CSFD_NSIDE: u32 = 2048;
pub const CSFD_NPIX: usize = 12 * (CSFD_NSIDE as usize).pow(2);

#[derive(Debug, Error)]
pub enum CsfdError {
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
    #[error("invalid CSFD NPY {path}: {source}")]
    InvalidNpy {
        path: PathBuf,
        #[source]
        source: ndarray_npy::ViewNpyError,
    },
    #[error("CSFD NPY has {actual} pixels, expected {expected}")]
    WrongShape { actual: usize, expected: usize },
    #[error("CSFD pixel {0} is out of range")]
    PixelOutOfRange(u64),
}

#[derive(Debug)]
pub struct CsfdMap {
    mmap: Mmap,
    path: PathBuf,
}

impl CsfdMap {
    /// Open and validate a memory-mapped CSFD NPY file.
    pub fn open(path: impl AsRef<Path>) -> Result<Arc<Self>, CsfdError> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|source| CsfdError::Open {
            path: path.to_owned(),
            source,
        })?;
        // The file is read-only and remains open through the lifetime of the mapping.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|source| CsfdError::Mmap {
            path: path.to_owned(),
            source,
        })?;
        let map = Self {
            mmap,
            path: path.to_owned(),
        };
        // Queries land on scattered pixel offsets, so sequential readahead
        // would waste I/O and page-cache space.
        if let Err(source) = map.mmap.advise(Advice::Random) {
            tracing::warn!(path = %map.path.display(), %source, "madvise(MADV_RANDOM) failed");
        }
        let view = map.view().map_err(|source| CsfdError::InvalidNpy {
            path: path.to_owned(),
            source,
        })?;
        if view.len() != CSFD_NPIX {
            return Err(CsfdError::WrongShape {
                actual: view.len(),
                expected: CSFD_NPIX,
            });
        }
        Ok(Arc::new(map))
    }

    fn view(&self) -> Result<ArrayView1<'_, f32>, ndarray_npy::ViewNpyError> {
        ArrayView1::<f32>::view_npy(&self.mmap)
    }

    pub fn value(&self, pixel: u64) -> Result<f32, CsfdError> {
        // `open` validates the shape and dtype before this map is put in app state.
        let view = self.view().map_err(|source| CsfdError::InvalidNpy {
            path: self.path.clone(),
            source,
        })?;
        view.get(pixel as usize)
            .copied()
            .ok_or(CsfdError::PixelOutOfRange(pixel))
    }

    pub fn query(&self, ra_deg: f64, dec_deg: f64) -> Result<f32, CsfdError> {
        let galactic = crate::geometry::icrs_to_galactic(ra_deg, dec_deg);
        let pixel = ang2pix_ring(CSFD_NSIDE, galactic.l_deg, galactic.b_deg);
        self.value(pixel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn open_rejects_wrong_shape_with_typed_error() {
        let path =
            std::env::temp_dir().join(format!("dustmaps-api-csfd-test-{}.npy", std::process::id()));
        ndarray_npy::write_npy(&path, &array![1.0_f32, 2.0]).expect("write fixture");

        let error = CsfdMap::open(&path).expect_err("wrong shape must fail");
        assert!(matches!(
            &error,
            CsfdError::WrongShape {
                actual: 2,
                expected: CSFD_NPIX
            }
        ));
        assert!(error.to_string().contains("expected"));
        std::fs::remove_file(path).expect("remove fixture");
    }

    #[test]
    fn open_reports_missing_file_as_typed_error() {
        let path = std::env::temp_dir().join(format!(
            "dustmaps-api-csfd-missing-{}.npy",
            std::process::id()
        ));
        let error = CsfdMap::open(&path).expect_err("missing file must fail");
        assert!(matches!(&error, CsfdError::Open { .. }));
    }
}
