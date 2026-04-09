//! Spatial utility functions: distance, bounding box checks, adaptive resolution.

#[cfg(feature = "python")]
use numpy::{PyArray1, PyReadonlyArray2};
#[cfg(feature = "python")]
use pyo3::prelude::*;
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Public Rust API
// ---------------------------------------------------------------------------

/// Euclidean distance between two 3D points.
#[inline]
pub fn distance_3d(x1: f64, y1: f64, z1: f64, x2: f64, y2: f64, z2: f64) -> f64 {
    let dx = x1 - x2;
    let dy = y1 - y2;
    let dz = z1 - z2;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Check if a point is within an axis-aligned bounding box.
#[inline]
pub fn point_in_bounds(
    x: f64,
    y: f64,
    z: f64,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    z_min: f64,
    z_max: f64,
) -> bool {
    x >= x_min && x <= x_max && y >= y_min && y <= y_max && z >= z_min && z <= z_max
}

/// Adaptive Hilbert resolution selection based on region density.
///
/// Returns 4 for sparse regions, 8 for moderate density, 12 for dense regions.
pub fn adaptive_hilbert_order(density: f64) -> u32 {
    if density < 100.0 {
        4
    } else if density < 10_000.0 {
        8
    } else {
        12
    }
}

/// Batch distance computation (pure Rust, for benchmarks).
pub fn batch_distances_3d_vec(
    points: &[(f64, f64, f64)],
    qx: f64,
    qy: f64,
    qz: f64,
) -> Vec<f64> {
    points
        .par_iter()
        .map(|&(x, y, z)| distance_3d(x, y, z, qx, qy, qz))
        .collect()
}

// ---------------------------------------------------------------------------
// PyO3 exports
// ---------------------------------------------------------------------------

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "distance_3d")]
pub fn py_distance_3d(x1: f64, y1: f64, z1: f64, x2: f64, y2: f64, z2: f64) -> f64 {
    distance_3d(x1, y1, z1, x2, y2, z2)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "point_in_bounds")]
pub fn py_point_in_bounds(
    x: f64,
    y: f64,
    z: f64,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    z_min: f64,
    z_max: f64,
) -> bool {
    point_in_bounds(x, y, z, x_min, x_max, y_min, y_max, z_min, z_max)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "batch_distances_3d")]
pub fn py_batch_distances_3d<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    query_x: f64,
    query_y: f64,
    query_z: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let arr = points.as_array();
    let n = arr.nrows();
    if arr.ncols() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "points must have shape (N, 3)",
        ));
    }

    let rows: Vec<(f64, f64, f64)> = (0..n)
        .map(|i| (arr[[i, 0]], arr[[i, 1]], arr[[i, 2]]))
        .collect();

    let results = batch_distances_3d_vec(&rows, query_x, query_y, query_z);
    Ok(PyArray1::from_vec(py, results))
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "adaptive_hilbert_order")]
pub fn py_adaptive_hilbert_order(density: f64) -> u32 {
    adaptive_hilbert_order(density)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_3d() {
        let d = distance_3d(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        assert!((d - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_distance_3d_diagonal() {
        let d = distance_3d(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
        assert!((d - 3.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_point_in_bounds() {
        assert!(point_in_bounds(0.5, 0.5, 0.5, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));
        assert!(!point_in_bounds(1.5, 0.5, 0.5, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));
    }

    #[test]
    fn test_adaptive_order() {
        assert_eq!(adaptive_hilbert_order(10.0), 4);
        assert_eq!(adaptive_hilbert_order(500.0), 8);
        assert_eq!(adaptive_hilbert_order(50_000.0), 12);
    }
}
