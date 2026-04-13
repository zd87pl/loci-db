//! loci_core — High-performance native primitives for LOCI-DB.
//!
//! Provides Rust implementations of LOCI's compute-intensive operations
//! exposed to Python via PyO3 bindings:
//!
//! - Multi-resolution Hilbert curve encoding (3D and 4D)
//! - Temporal epoch sharding
//! - Spatial distance and bounding box utilities
//! - Novelty scoring for predict-then-retrieve
//! - Batch WorldState preparation (the primary hot path)

#[cfg(feature = "python")]
use pyo3::prelude::*;

pub mod batch;
pub mod hilbert;
pub mod hilbert_experiments;
pub mod novelty;
pub mod spatial;
pub mod temporal;

/// The loci_core Python module.
#[cfg(feature = "python")]
#[pymodule]
fn loci_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Hilbert encoding
    m.add_function(wrap_pyfunction!(hilbert::py_encode_hilbert_3d, m)?)?;
    m.add_function(wrap_pyfunction!(hilbert::py_encode_hilbert_4d, m)?)?;
    m.add_function(wrap_pyfunction!(hilbert::py_decode_hilbert_3d, m)?)?;
    m.add_function(wrap_pyfunction!(hilbert::py_batch_encode_hilbert_3d, m)?)?;
    m.add_function(wrap_pyfunction!(hilbert::py_batch_encode_hilbert_4d, m)?)?;
    m.add_function(wrap_pyfunction!(
        hilbert::py_spatial_bounds_to_hilbert_buckets,
        m
    )?)?;

    // Temporal sharding
    m.add_function(wrap_pyfunction!(temporal::py_compute_epoch_id, m)?)?;
    m.add_function(wrap_pyfunction!(temporal::py_epoch_collection_name, m)?)?;
    m.add_function(wrap_pyfunction!(temporal::py_epochs_for_time_window, m)?)?;
    m.add_function(wrap_pyfunction!(
        temporal::py_normalise_timestamp_in_epoch,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(temporal::py_batch_compute_epoch_ids, m)?)?;

    // Spatial utilities
    m.add_function(wrap_pyfunction!(spatial::py_distance_3d, m)?)?;
    m.add_function(wrap_pyfunction!(spatial::py_point_in_bounds, m)?)?;
    m.add_function(wrap_pyfunction!(spatial::py_batch_distances_3d, m)?)?;
    m.add_function(wrap_pyfunction!(spatial::py_adaptive_hilbert_order, m)?)?;

    // Novelty scoring
    m.add_function(wrap_pyfunction!(novelty::py_cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(novelty::py_compute_novelty_score, m)?)?;
    m.add_function(wrap_pyfunction!(novelty::py_batch_novelty_scores, m)?)?;
    m.add_function(wrap_pyfunction!(novelty::py_temporal_decay_weight, m)?)?;

    // Batch processing
    m.add_function(wrap_pyfunction!(batch::py_batch_prepare_world_states, m)?)?;

    Ok(())
}
