//! Batch processing utilities for WorldState insertions.
//!
//! The primary hot path: takes parallel arrays of coordinates and timestamps,
//! returns computed hilbert_3d, hilbert_4d, and epoch_id values.

#[cfg(feature = "python")]
use numpy::{PyArray1, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::hilbert;
use crate::temporal;

/// Batch prepare WorldState data (pure Rust).
/// Returns Vec of (hilbert_3d, hilbert_4d, epoch_id) tuples.
pub fn prepare_world_states(
    xs: &[f64],
    ys: &[f64],
    zs: &[f64],
    timestamps_ms: &[u64],
    epoch_size_ms: u64,
    hilbert_order: u32,
) -> Vec<(u64, u64, u64)> {
    (0..xs.len())
        .into_par_iter()
        .map(|i| {
            let h3 = hilbert::encode_3d(xs[i], ys[i], zs[i], hilbert_order);
            let epoch_id = temporal::compute_epoch(timestamps_ms[i], epoch_size_ms);
            let t_norm = temporal::normalise_in_epoch(timestamps_ms[i], epoch_id, epoch_size_ms);
            let h4 = hilbert::encode_4d(xs[i], ys[i], zs[i], t_norm, hilbert_order);
            (h3, h4, epoch_id)
        })
        .collect()
}

/// Batch prepare WorldState data: compute hilbert_3d, hilbert_4d, epoch_id.
///
/// Returns (hilbert_3d_array, hilbert_4d_array, epoch_id_array).
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "batch_prepare_world_states")]
pub fn py_batch_prepare_world_states<'py>(
    py: Python<'py>,
    xs: PyReadonlyArray1<'py, f64>,
    ys: PyReadonlyArray1<'py, f64>,
    zs: PyReadonlyArray1<'py, f64>,
    timestamps_ms: PyReadonlyArray1<'py, u64>,
    epoch_size_ms: u64,
    hilbert_order: u32,
) -> PyResult<(
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
)> {
    let xs_arr = xs.as_array();
    let ys_arr = ys.as_array();
    let zs_arr = zs.as_array();
    let ts_arr = timestamps_ms.as_array();
    let n = xs_arr.len();

    if ys_arr.len() != n || zs_arr.len() != n || ts_arr.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "All input arrays must have the same length",
        ));
    }

    let xs_vec: Vec<f64> = xs_arr.iter().copied().collect();
    let ys_vec: Vec<f64> = ys_arr.iter().copied().collect();
    let zs_vec: Vec<f64> = zs_arr.iter().copied().collect();
    let ts_vec: Vec<u64> = ts_arr.iter().copied().collect();

    let results = prepare_world_states(&xs_vec, &ys_vec, &zs_vec, &ts_vec, epoch_size_ms, hilbert_order);

    let mut h3_vec = Vec::with_capacity(n);
    let mut h4_vec = Vec::with_capacity(n);
    let mut epoch_vec = Vec::with_capacity(n);

    for (h3, h4, epoch) in results {
        h3_vec.push(h3);
        h4_vec.push(h4);
        epoch_vec.push(epoch);
    }

    Ok((
        PyArray1::from_vec(py, h3_vec),
        PyArray1::from_vec(py, h4_vec),
        PyArray1::from_vec(py, epoch_vec),
    ))
}
