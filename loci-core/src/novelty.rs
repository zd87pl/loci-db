//! Novelty scoring primitives for predict-then-retrieve.
//!
//! Novelty measures how different a predicted observation is from the
//! closest known observations:
//! - 0.0 = predicted state has been seen before
//! - 1.0 = completely new territory

#[cfg(feature = "python")]
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Public Rust API
// ---------------------------------------------------------------------------

/// Cosine similarity between two f32 slices.
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// Compute novelty score between a predicted vector and k nearest retrieved vectors.
///
/// novelty = 1.0 - max(cosine_similarity(predicted, retrieved_i))
pub fn novelty_score(predicted: &[f32], retrieved: &[&[f32]]) -> f32 {
    if retrieved.is_empty() {
        return 1.0;
    }
    let max_sim = retrieved
        .iter()
        .map(|r| cosine_sim(predicted, r))
        .fold(f32::NEG_INFINITY, f32::max);
    (1.0 - max_sim).clamp(0.0, 1.0)
}

/// Temporal decay weight.
///
/// More recent observations are weighted higher. Returns a weight in [0, 1].
/// decay_factor controls the rate: 0.0 = no decay, higher = faster decay.
pub fn temporal_decay_weight(observation_ms: u64, query_ms: u64, decay_factor: f64) -> f64 {
    if decay_factor == 0.0 {
        return 1.0;
    }
    let age_ms = query_ms.saturating_sub(observation_ms) as f64;
    (-decay_factor * age_ms).exp()
}

// ---------------------------------------------------------------------------
// PyO3 exports
// ---------------------------------------------------------------------------

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "cosine_similarity")]
pub fn py_cosine_similarity(a: PyReadonlyArray1<'_, f32>, b: PyReadonlyArray1<'_, f32>) -> f32 {
    let a = a.as_slice().expect("contiguous array");
    let b = b.as_slice().expect("contiguous array");
    cosine_sim(a, b)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "compute_novelty_score")]
pub fn py_compute_novelty_score(
    predicted: PyReadonlyArray1<'_, f32>,
    retrieved: PyReadonlyArray2<'_, f32>,
) -> f32 {
    let pred = predicted.as_slice().expect("contiguous array");
    let ret_arr = retrieved.as_array();
    let k = ret_arr.nrows();
    if k == 0 {
        return 1.0;
    }
    let mut max_sim = f32::NEG_INFINITY;
    for i in 0..k {
        let row: Vec<f32> = (0..ret_arr.ncols()).map(|j| ret_arr[[i, j]]).collect();
        let sim = cosine_sim(pred, &row);
        if sim > max_sim {
            max_sim = sim;
        }
    }
    (1.0 - max_sim).clamp(0.0, 1.0)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "batch_novelty_scores")]
pub fn py_batch_novelty_scores<'py>(
    py: Python<'py>,
    predicted: PyReadonlyArray2<'py, f32>,
    retrieved: PyReadonlyArray3<'py, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let pred_arr = predicted.as_array();
    let ret_arr = retrieved.as_array();
    let m = pred_arr.nrows();
    let d = pred_arr.ncols();
    let k = ret_arr.shape()[1];

    if ret_arr.shape()[0] != m || ret_arr.shape()[2] != d {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Shape mismatch: predicted (M, D), retrieved (M, K, D)",
        ));
    }

    let pred_rows: Vec<Vec<f32>> = (0..m)
        .map(|i| (0..d).map(|j| pred_arr[[i, j]]).collect())
        .collect();

    let ret_rows: Vec<Vec<Vec<f32>>> = (0..m)
        .map(|i| {
            (0..k)
                .map(|ki| (0..d).map(|j| ret_arr[[i, ki, j]]).collect())
                .collect()
        })
        .collect();

    let results: Vec<f32> = (0..m)
        .into_par_iter()
        .map(|i| {
            let pred = &pred_rows[i];
            let refs: Vec<&[f32]> = ret_rows[i].iter().map(|v| v.as_slice()).collect();
            novelty_score(pred, &refs)
        })
        .collect();

    Ok(PyArray1::from_vec(py, results))
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "temporal_decay_weight")]
pub fn py_temporal_decay_weight(observation_ms: u64, query_ms: u64, decay_factor: f64) -> f64 {
    temporal_decay_weight(observation_ms, query_ms, decay_factor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0f32, 2.0, 3.0];
        let sim = cosine_sim(&a, &a);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        let sim = cosine_sim(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_novelty_identical() {
        let pred = vec![1.0f32, 0.0, 0.0];
        let ret = vec![1.0f32, 0.0, 0.0];
        let score = novelty_score(&pred, &[&ret]);
        assert!(score.abs() < 1e-6);
    }

    #[test]
    fn test_novelty_orthogonal() {
        let pred = vec![1.0f32, 0.0, 0.0];
        let ret = vec![0.0f32, 1.0, 0.0];
        let score = novelty_score(&pred, &[&ret]);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_novelty_empty() {
        let pred = vec![1.0f32, 0.0];
        let score = novelty_score(&pred, &[]);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_temporal_decay_no_decay() {
        let w = temporal_decay_weight(1000, 2000, 0.0);
        assert_eq!(w, 1.0);
    }

    #[test]
    fn test_temporal_decay_recent() {
        let w = temporal_decay_weight(1000, 1000, 0.1);
        assert!((w - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_temporal_decay_old() {
        let w = temporal_decay_weight(0, 10_000, 0.001);
        assert!(w < 1.0);
        assert!(w > 0.0);
    }
}
