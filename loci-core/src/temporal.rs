//! Epoch sharding and temporal utilities.
//!
//! Vectors are routed to per-epoch Qdrant collections named `loci_{epoch_id}`.
//! An epoch is a fixed-width time window (default 5000 ms).

#[cfg(feature = "python")]
use numpy::{PyArray1, PyReadonlyArray1};
#[cfg(feature = "python")]
use pyo3::prelude::*;
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Public Rust API
// ---------------------------------------------------------------------------

/// Compute epoch_id from timestamp and epoch size.
#[inline]
pub fn compute_epoch(timestamp_ms: u64, epoch_size_ms: u64) -> u64 {
    timestamp_ms / epoch_size_ms
}

/// Return the Qdrant collection name for an epoch.
pub fn collection_name(epoch_id: u64) -> String {
    format!("loci_{epoch_id}")
}

/// Return all epoch IDs overlapping a time window (inclusive).
pub fn epochs_for_window(start_ms: u64, end_ms: u64, epoch_size_ms: u64) -> Vec<u64> {
    let first = compute_epoch(start_ms, epoch_size_ms);
    let last = compute_epoch(end_ms, epoch_size_ms);
    (first..=last).collect()
}

/// Normalise a timestamp within its epoch to [0.0, 1.0].
#[inline]
pub fn normalise_in_epoch(timestamp_ms: u64, epoch_id: u64, epoch_size_ms: u64) -> f64 {
    let epoch_start = epoch_id * epoch_size_ms;
    let offset = timestamp_ms.saturating_sub(epoch_start);
    (offset as f64 / epoch_size_ms as f64).clamp(0.0, 1.0)
}

/// Batch compute epoch IDs (pure Rust, for benchmarks).
pub fn batch_compute_epochs(timestamps: &[u64], epoch_size_ms: u64) -> Vec<u64> {
    timestamps
        .par_iter()
        .map(|&ts| compute_epoch(ts, epoch_size_ms))
        .collect()
}

// ---------------------------------------------------------------------------
// PyO3 exports
// ---------------------------------------------------------------------------

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "compute_epoch_id")]
pub fn py_compute_epoch_id(timestamp_ms: u64, epoch_size_ms: u64) -> u64 {
    compute_epoch(timestamp_ms, epoch_size_ms)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "epoch_collection_name")]
pub fn py_epoch_collection_name(epoch_id: u64) -> String {
    collection_name(epoch_id)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "epochs_for_time_window")]
pub fn py_epochs_for_time_window(start_ms: u64, end_ms: u64, epoch_size_ms: u64) -> Vec<u64> {
    epochs_for_window(start_ms, end_ms, epoch_size_ms)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "normalise_timestamp_in_epoch")]
pub fn py_normalise_timestamp_in_epoch(
    timestamp_ms: u64,
    epoch_id: u64,
    epoch_size_ms: u64,
) -> f64 {
    normalise_in_epoch(timestamp_ms, epoch_id, epoch_size_ms)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "batch_compute_epoch_ids")]
pub fn py_batch_compute_epoch_ids<'py>(
    py: Python<'py>,
    timestamps_ms: PyReadonlyArray1<'py, u64>,
    epoch_size_ms: u64,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    let arr = timestamps_ms.as_array();
    let timestamps: Vec<u64> = arr.iter().copied().collect();
    let results = batch_compute_epochs(&timestamps, epoch_size_ms);
    Ok(PyArray1::from_vec(py, results))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epoch_id() {
        assert_eq!(compute_epoch(10_000, 5_000), 2);
        assert_eq!(compute_epoch(4_999, 5_000), 0);
        assert_eq!(compute_epoch(5_000, 5_000), 1);
    }

    #[test]
    fn test_collection_name() {
        assert_eq!(collection_name(42), "loci_42");
    }

    #[test]
    fn test_epochs_for_window() {
        let epochs = epochs_for_window(4_000, 12_000, 5_000);
        assert_eq!(epochs, vec![0, 1, 2]);
    }

    #[test]
    fn test_normalise_in_epoch() {
        let t = normalise_in_epoch(7_500, 1, 5_000);
        assert!((t - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_normalise_clamped() {
        let t = normalise_in_epoch(0, 1, 5_000);
        assert_eq!(t, 0.0);
    }
}
