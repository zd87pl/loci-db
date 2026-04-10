//! Multi-resolution Hilbert curve encoding for N-dimensional coordinates.
//!
//! Implements the Skilling (2004) algorithm for converting between
//! N-dimensional integer coordinates and 1D Hilbert distances.
//! Supports 3D (x, y, z) and 4D (x, y, z, t) — the temporal dimension
//! extension is LOCI's novel contribution to spatiotemporal indexing.

#[cfg(feature = "python")]
use numpy::{PyArray1, PyReadonlyArray2};
#[cfg(feature = "python")]
use pyo3::prelude::*;
use rayon::prelude::*;

/// Clamp an integer coordinate to [0, max_val].
#[inline(always)]
fn clamp_coord(val: i64, max_val: i64) -> u32 {
    val.max(0).min(max_val) as u32
}

/// Quantise a normalised [0, 1] float to an integer grid coordinate.
#[inline(always)]
fn quantise(v: f64, side: u32) -> u32 {
    clamp_coord((v * side as f64).round() as i64, side as i64)
}

// ---------------------------------------------------------------------------
// Skilling's algorithm: AxesToTranspose / TransposeToAxes
// ---------------------------------------------------------------------------

/// Convert N-dimensional coordinates to Hilbert transpose representation.
///
/// This is the "AxesToTranspose" procedure from Skilling (2004).
/// Operates in-place on the coordinate array.
fn axes_to_transpose(x: &mut [u32], order: u32) {
    let n = x.len();
    let m: u32 = 1 << (order - 1);

    // Inverse undo
    let mut q = m;
    while q > 1 {
        let p = q - 1;
        for i in 0..n {
            if x[i] & q != 0 {
                x[0] ^= p; // invert
            } else {
                let t = (x[0] ^ x[i]) & p;
                x[0] ^= t;
                x[i] ^= t; // exchange
            }
        }
        q >>= 1;
    }

    // Gray encode
    for i in 1..n {
        x[i] ^= x[i - 1];
    }
    let mut t2: u32 = 0;
    let mut q = m;
    while q > 1 {
        if x[n - 1] & q != 0 {
            t2 ^= q - 1;
        }
        q >>= 1;
    }
    for xi in x.iter_mut() {
        *xi ^= t2;
    }
}

/// Convert Hilbert transpose representation back to N-dimensional coordinates.
///
/// This is the "TransposeToAxes" procedure from Skilling (2004).
/// Operates in-place on the coordinate array.
fn transpose_to_axes(x: &mut [u32], order: u32) {
    let n = x.len();
    let big_n: u32 = 2 << (order - 1);

    // Gray decode
    let t = x[n - 1] >> 1;
    for i in (1..n).rev() {
        x[i] ^= x[i - 1];
    }
    x[0] ^= t;

    // Undo excess work
    let mut m: u32 = 2;
    while m != big_n {
        let p = m - 1;
        for i in (0..n).rev() {
            if x[i] & m != 0 {
                x[0] ^= p;
            } else {
                let t = (x[0] ^ x[i]) & p;
                x[0] ^= t;
                x[i] ^= t;
            }
        }
        m <<= 1;
    }
}

/// Interleave coordinate bits into a single Hilbert distance integer.
///
/// For N dimensions and order p, the result has N*p bits.
/// Bit layout: MSB is bit (p-1) of x[0], then bit (p-1) of x[1], ...,
/// down to bit 0 of x[N-1].
fn transpose_to_distance(x: &[u32], order: u32) -> u64 {
    let mut d: u64 = 0;
    for bit in (0..order).rev() {
        for xi in x.iter() {
            d <<= 1;
            if xi & (1 << bit) != 0 {
                d |= 1;
            }
        }
    }
    d
}

/// De-interleave a Hilbert distance integer back to coordinate transpose.
fn distance_to_transpose(d: u64, n: u32, order: u32) -> Vec<u32> {
    let mut x = vec![0u32; n as usize];
    let mut bit_pos = 0u32;
    for bit in 0..order {
        for dim in (0..n).rev() {
            if d & (1u64 << bit_pos) != 0 {
                x[dim as usize] |= 1 << bit;
            }
            bit_pos += 1;
        }
    }
    x
}

// ---------------------------------------------------------------------------
// Public Rust API
// ---------------------------------------------------------------------------

/// Encode 3D normalised coordinates to a Hilbert index.
pub fn encode_3d(x: f64, y: f64, z: f64, order: u32) -> u64 {
    let side = (1u32 << order) - 1;
    let mut coords = [quantise(x, side), quantise(y, side), quantise(z, side)];
    axes_to_transpose(&mut coords, order);
    transpose_to_distance(&coords, order)
}

/// Encode 4D normalised coordinates to a Hilbert index.
///
/// The temporal dimension (t) is LOCI's novel contribution —
/// extending the standard 3D Hilbert curve to include time as a
/// fourth spatial dimension for spatiotemporal locality preservation.
pub fn encode_4d(x: f64, y: f64, z: f64, t: f64, order: u32) -> u64 {
    let side = (1u32 << order) - 1;
    let mut coords = [
        quantise(x, side),
        quantise(y, side),
        quantise(z, side),
        quantise(t, side),
    ];
    axes_to_transpose(&mut coords, order);
    transpose_to_distance(&coords, order)
}

/// Decode a 3D Hilbert index back to normalised coordinates.
pub fn decode_3d(h: u64, order: u32) -> (f64, f64, f64) {
    let side = (1u32 << order) - 1;
    let mut coords = distance_to_transpose(h, 3, order);
    transpose_to_axes(&mut coords, order);
    (
        coords[0] as f64 / side as f64,
        coords[1] as f64 / side as f64,
        coords[2] as f64 / side as f64,
    )
}

/// Decode a 4D Hilbert index back to normalised coordinates.
pub fn decode_4d(h: u64, order: u32) -> (f64, f64, f64, f64) {
    let side = (1u32 << order) - 1;
    let mut coords = distance_to_transpose(h, 4, order);
    transpose_to_axes(&mut coords, order);
    (
        coords[0] as f64 / side as f64,
        coords[1] as f64 / side as f64,
        coords[2] as f64 / side as f64,
        coords[3] as f64 / side as f64,
    )
}

/// Compute Hilbert bucket IDs overlapping a 3D bounding box.
///
/// Expands the bounding box by `overlap_factor`, then enumerates all
/// grid points within the expanded box and returns their Hilbert IDs.
pub fn spatial_bounds_to_buckets(
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    z_min: f64,
    z_max: f64,
    order: u32,
    overlap_factor: f64,
) -> Vec<u64> {
    let side = (1u32 << order) - 1;
    let min_pad = 1.0 / side.max(1) as f64;

    let expand = |lo: f64, hi: f64| -> (f64, f64) {
        let span = hi - lo;
        let pad = (span * (overlap_factor - 1.0) / 2.0).max(min_pad);
        ((lo - pad).max(0.0), (hi + pad).min(1.0))
    };

    let (xl, xh) = expand(x_min, x_max);
    let (yl, yh) = expand(y_min, y_max);
    let (zl, zh) = expand(z_min, z_max);

    let ix_lo = clamp_coord((xl * side as f64).floor() as i64, side as i64);
    let ix_hi = clamp_coord((xh * side as f64).ceil() as i64, side as i64);
    let iy_lo = clamp_coord((yl * side as f64).floor() as i64, side as i64);
    let iy_hi = clamp_coord((yh * side as f64).ceil() as i64, side as i64);
    let iz_lo = clamp_coord((zl * side as f64).floor() as i64, side as i64);
    let iz_hi = clamp_coord((zh * side as f64).ceil() as i64, side as i64);

    let mut ids = std::collections::BTreeSet::new();
    for ix in ix_lo..=ix_hi {
        for iy in iy_lo..=iy_hi {
            for iz in iz_lo..=iz_hi {
                let mut coords = [ix, iy, iz];
                axes_to_transpose(&mut coords, order);
                ids.insert(transpose_to_distance(&coords, order));
            }
        }
    }
    ids.into_iter().collect()
}

/// Batch encode 3D coordinates (pure Rust, for benchmarks).
pub fn batch_encode_3d_vec(coords: &[(f64, f64, f64)], order: u32) -> Vec<u64> {
    coords
        .par_iter()
        .map(|&(x, y, z)| encode_3d(x, y, z, order))
        .collect()
}

/// Batch encode 4D coordinates (pure Rust, for benchmarks).
pub fn batch_encode_4d_vec(coords: &[(f64, f64, f64, f64)], order: u32) -> Vec<u64> {
    coords
        .par_iter()
        .map(|&(x, y, z, t)| encode_4d(x, y, z, t, order))
        .collect()
}

// ---------------------------------------------------------------------------
// PyO3 exports
// ---------------------------------------------------------------------------

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "encode_hilbert_3d")]
pub fn py_encode_hilbert_3d(x: f64, y: f64, z: f64, order: u32) -> u64 {
    encode_3d(x, y, z, order)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "encode_hilbert_4d")]
pub fn py_encode_hilbert_4d(x: f64, y: f64, z: f64, t: f64, order: u32) -> u64 {
    encode_4d(x, y, z, t, order)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "decode_hilbert_3d")]
pub fn py_decode_hilbert_3d(h: u64, order: u32) -> (f64, f64, f64) {
    decode_3d(h, order)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "batch_encode_hilbert_3d")]
pub fn py_batch_encode_hilbert_3d<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    order: u32,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    let arr = coords.as_array();
    let n = arr.nrows();
    if arr.ncols() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "coords must have shape (N, 3)",
        ));
    }

    let rows: Vec<(f64, f64, f64)> = (0..n)
        .map(|i| (arr[[i, 0]], arr[[i, 1]], arr[[i, 2]]))
        .collect();

    let results = batch_encode_3d_vec(&rows, order);
    Ok(PyArray1::from_vec(py, results))
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "batch_encode_hilbert_4d")]
pub fn py_batch_encode_hilbert_4d<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    order: u32,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    let arr = coords.as_array();
    let n = arr.nrows();
    if arr.ncols() != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "coords must have shape (N, 4)",
        ));
    }

    let rows: Vec<(f64, f64, f64, f64)> = (0..n)
        .map(|i| (arr[[i, 0]], arr[[i, 1]], arr[[i, 2]], arr[[i, 3]]))
        .collect();

    let results = batch_encode_4d_vec(&rows, order);
    Ok(PyArray1::from_vec(py, results))
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "spatial_bounds_to_hilbert_buckets")]
pub fn py_spatial_bounds_to_hilbert_buckets(
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    z_min: f64,
    z_max: f64,
    order: u32,
    overlap_factor: f64,
) -> Vec<u64> {
    spatial_bounds_to_buckets(x_min, x_max, y_min, y_max, z_min, z_max, order, overlap_factor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_3d_roundtrip() {
        for order in [4, 8, 12] {
            let h = encode_3d(0.5, 0.3, 0.8, order);
            let (x, y, z) = decode_3d(h, order);
            let side = (1u32 << order) - 1;
            let tol = 1.0 / side as f64;
            assert!((x - 0.5).abs() <= tol, "x mismatch at order {order}");
            assert!((y - 0.3).abs() <= tol, "y mismatch at order {order}");
            assert!((z - 0.8).abs() <= tol, "z mismatch at order {order}");
        }
    }

    #[test]
    fn test_encode_decode_4d_roundtrip() {
        for order in [4, 8] {
            let h = encode_4d(0.5, 0.3, 0.8, 0.6, order);
            let (x, y, z, t) = decode_4d(h, order);
            let side = (1u32 << order) - 1;
            let tol = 1.0 / side as f64;
            assert!((x - 0.5).abs() <= tol, "x mismatch");
            assert!((y - 0.3).abs() <= tol, "y mismatch");
            assert!((z - 0.8).abs() <= tol, "z mismatch");
            assert!((t - 0.6).abs() <= tol, "t mismatch");
        }
    }

    #[test]
    fn test_spatial_locality() {
        let h1 = encode_3d(0.5, 0.5, 0.5, 8);
        let h2 = encode_3d(0.51, 0.51, 0.51, 8);
        let h3 = encode_3d(0.0, 0.0, 0.0, 8);
        let d_near = (h1 as i64 - h2 as i64).unsigned_abs();
        let d_far = (h1 as i64 - h3 as i64).unsigned_abs();
        assert!(d_near < d_far, "spatial locality violated");
    }

    #[test]
    fn test_zero_coords() {
        let h = encode_3d(0.0, 0.0, 0.0, 4);
        assert_eq!(h, 0);
    }

    #[test]
    fn test_buckets_nonempty() {
        let buckets = spatial_bounds_to_buckets(0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 4, 1.2);
        assert!(!buckets.is_empty());
    }
}
