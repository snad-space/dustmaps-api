//! Coordinate and HEALPix geometry used by the dust-map readers.
//!
//! Angles exposed by this module are in degrees where noted; `cdshealpix`
//! receives radians, as required by its API.

use cdshealpix::nested::map::astrometry::{gal::Galactic as CdsGalactic, math::Coo};

/// Galactic longitude and latitude in degrees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Galactic {
    pub l_deg: f64,
    pub b_deg: f64,
}

/// Convert an ICRS position, given in degrees, to Galactic coordinates.
pub fn icrs_to_galactic(ra_deg: f64, dec_deg: f64) -> Galactic {
    let frame = CdsGalactic::new_for_fk5_j2000_and_icrs();
    let (l_deg, b_deg) = frame.coo_eq2gal(&Coo::from_deg(ra_deg, dec_deg)).to_deg();
    Galactic { l_deg, b_deg }
}

/// Return the RING-ordered HEALPix pixel containing Galactic `(l, b)`.
pub fn ang2pix_ring(nside: u32, l_deg: f64, b_deg: f64) -> u64 {
    cdshealpix::ring::hash(nside, l_deg.to_radians(), b_deg.to_radians())
}

/// Return the NESTED-ordered HEALPix pixel containing Galactic `(l, b)`.
pub fn ang2pix_nested(nside: u32, l_deg: f64, b_deg: f64) -> u64 {
    cdshealpix::nested::hash(
        cdshealpix::depth(nside),
        l_deg.to_radians(),
        b_deg.to_radians(),
    )
}

/// Convert ICRS degrees directly to a RING pixel.
pub fn icrs_to_ring_pixel(nside: u32, ra_deg: f64, dec_deg: f64) -> u64 {
    let galactic = icrs_to_galactic(ra_deg, dec_deg);
    ang2pix_ring(nside, galactic.l_deg, galactic.b_deg)
}

/// Convert ICRS degrees directly to a NESTED pixel.
pub fn icrs_to_nested_pixel(nside: u32, ra_deg: f64, dec_deg: f64) -> u64 {
    let galactic = icrs_to_galactic(ra_deg, dec_deg);
    ang2pix_nested(nside, galactic.l_deg, galactic.b_deg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn galactic_center_is_at_zero_zero() {
        // The standard J2000 coordinates of the Galactic center, rounded to
        // the precision commonly published for this reference position.
        let galactic = icrs_to_galactic(266.4051, -28.936175);
        assert!(galactic.l_deg < 0.001 || galactic.l_deg > 359.999);
        assert!(galactic.b_deg.abs() < 0.001);
    }

    #[test]
    fn longitude_is_normalized() {
        let a = icrs_to_galactic(0.0, 0.0);
        let b = icrs_to_galactic(360.0, 0.0);
        assert!((a.l_deg - b.l_deg).abs() < 1e-12);
        assert!((0.0..360.0).contains(&a.l_deg));
    }

    #[test]
    fn cdshealpix_ordering_wrappers_use_radians() {
        assert_eq!(ang2pix_ring(1, 40.0, 90.0), 0);
        assert_eq!(ang2pix_ring(1, 40.0, -90.0), 8);
        assert_eq!(ang2pix_nested(1, 45.0, 90.0), 0);
        assert!(ang2pix_nested(1, 45.0, -90.0) < 12);
    }
}
