//! Coordinate and HEALPix geometry used by the dust-map readers.
//!
//! Angles exposed by this module are in degrees where noted; `cdshealpix`
//! receives radians, as required by its API.

/// Galactic longitude and latitude in degrees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Galactic {
    pub l_deg: f64,
    pub b_deg: f64,
}

/// Convert an ICRS position, given in degrees, to Galactic coordinates.
pub fn icrs_to_galactic(ra_deg: f64, dec_deg: f64) -> Galactic {
    // ICRS/J2000 -> Galactic rotation used by Astropy/ERFA.
    #[allow(clippy::excessive_precision)]
    const ROTATION: [[f64; 3]; 3] = [
        [
            -0.05487565771259168,
            -0.8734370519556163,
            -0.4838350736167155,
        ],
        [0.4941094371927275, -0.44482972122329517, 0.7469821839866676],
        [
            -0.8676661375596587,
            -0.1980763372730008,
            0.45598381368730173,
        ],
    ];
    let ra = f64::to_radians(ra_deg);
    let dec = f64::to_radians(dec_deg);
    let equatorial = [
        f64::cos(dec) * f64::cos(ra),
        f64::cos(dec) * f64::sin(ra),
        f64::sin(dec),
    ];
    let galactic = [
        ROTATION[0][0] * equatorial[0]
            + ROTATION[0][1] * equatorial[1]
            + ROTATION[0][2] * equatorial[2],
        ROTATION[1][0] * equatorial[0]
            + ROTATION[1][1] * equatorial[1]
            + ROTATION[1][2] * equatorial[2],
        ROTATION[2][0] * equatorial[0]
            + ROTATION[2][1] * equatorial[1]
            + ROTATION[2][2] * equatorial[2],
    ];
    let l_deg = f64::rem_euclid(f64::to_degrees(f64::atan2(galactic[1], galactic[0])), 360.0);
    let b_deg = f64::to_degrees(f64::asin(galactic[2]));
    Galactic { l_deg, b_deg }
}

/// Return the RING-ordered HEALPix pixel containing Galactic `(l, b)`.
pub fn ang2pix_ring(nside: u32, l_deg: f64, b_deg: f64) -> u64 {
    cdshealpix::ring::hash(nside, f64::to_radians(l_deg), f64::to_radians(b_deg))
}

/// Return the NESTED-ordered HEALPix pixel containing Galactic `(l, b)`.
pub fn ang2pix_nested(nside: u32, l_deg: f64, b_deg: f64) -> u64 {
    cdshealpix::nested::hash(
        cdshealpix::depth(nside),
        f64::to_radians(l_deg),
        f64::to_radians(b_deg),
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
        assert!(f64::abs(galactic.b_deg) < 0.001);
    }

    #[test]
    fn longitude_is_normalized() {
        let a = icrs_to_galactic(0.0, 0.0);
        let b = icrs_to_galactic(360.0, 0.0);
        assert!(f64::abs(a.l_deg - b.l_deg) < 1e-12);
        assert!((0.0..360.0).contains(&a.l_deg));
    }

    #[test]
    fn cdshealpix_ordering_wrappers_use_radians() {
        assert_eq!(ang2pix_ring(1, 40.0, 90.0), 0);
        assert_eq!(ang2pix_ring(1, 40.0, -90.0), 8);
        assert_eq!(ang2pix_nested(1, 45.0, 90.0), 0);
        assert!(ang2pix_nested(1, 45.0, -90.0) < 12);
    }

    #[test]
    fn rotation_matches_astropy_goldens() {
        let goldens = [
            (
                194.7545309478567,
                -18.2512760863485,
                305.4591565365893,
                44.58327212072334,
            ),
            (
                184.24879007626018,
                25.47694717342729,
                223.07344679015384,
                82.10800905739632,
            ),
            (
                331.7934220050504,
                -35.695893147402366,
                8.702988164490204,
                -54.19396087242497,
            ),
        ];
        for (ra, dec, expected_l, expected_b) in goldens {
            let galactic = icrs_to_galactic(ra, dec);
            assert!(
                f64::abs(galactic.l_deg - expected_l) < 1e-9,
                "l {} != {}",
                galactic.l_deg,
                expected_l
            );
            assert!(
                f64::abs(galactic.b_deg - expected_b) < 1e-9,
                "b {} != {}",
                galactic.b_deg,
                expected_b
            );
        }
    }
}
