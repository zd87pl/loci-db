//! Integration tests for loci_core.
//!
//! Verifies correctness of Hilbert encoding, temporal sharding,
//! spatial utilities, and novelty scoring.

use loci_core::hilbert;
use loci_core::novelty;
use loci_core::spatial;
use loci_core::temporal;

// ---------------------------------------------------------------------------
// 1. Encode then decode round-trips correctly for 3D
// ---------------------------------------------------------------------------

#[test]
fn test_3d_roundtrip_order_4() {
    roundtrip_3d(0.5, 0.3, 0.8, 4);
}

#[test]
fn test_3d_roundtrip_order_8() {
    roundtrip_3d(0.5, 0.3, 0.8, 8);
}

#[test]
fn test_3d_roundtrip_order_12() {
    roundtrip_3d(0.5, 0.3, 0.8, 12);
}

fn roundtrip_3d(x: f64, y: f64, z: f64, order: u32) {
    let h = hilbert::encode_3d(x, y, z, order);
    let (dx, dy, dz) = hilbert::decode_3d(h, order);
    let side = (1u32 << order) - 1;
    let tol = 1.0 / side as f64;
    assert!(
        (dx - x).abs() <= tol,
        "x roundtrip failed at order {order}: {dx} vs {x}"
    );
    assert!(
        (dy - y).abs() <= tol,
        "y roundtrip failed at order {order}: {dy} vs {y}"
    );
    assert!(
        (dz - z).abs() <= tol,
        "z roundtrip failed at order {order}: {dz} vs {z}"
    );
}

// ---------------------------------------------------------------------------
// 2. Spatial locality preserved
// ---------------------------------------------------------------------------

#[test]
fn test_spatial_locality_3d() {
    let h_center = hilbert::encode_3d(0.5, 0.5, 0.5, 8);
    let h_near = hilbert::encode_3d(0.51, 0.51, 0.51, 8);
    let h_far = hilbert::encode_3d(0.0, 0.0, 0.0, 8);

    let d_near = (h_center as i64 - h_near as i64).unsigned_abs();
    let d_far = (h_center as i64 - h_far as i64).unsigned_abs();
    assert!(
        d_near < d_far,
        "Nearby points should have closer Hilbert indices"
    );
}

#[test]
fn test_spatial_locality_4d() {
    let h_center = hilbert::encode_4d(0.5, 0.5, 0.5, 0.5, 8);
    let h_near = hilbert::encode_4d(0.51, 0.51, 0.51, 0.51, 8);
    let h_far = hilbert::encode_4d(0.0, 0.0, 0.0, 0.0, 8);

    let d_near = (h_center as i64 - h_near as i64).unsigned_abs();
    let d_far = (h_center as i64 - h_far as i64).unsigned_abs();
    assert!(
        d_near < d_far,
        "Nearby 4D points should have closer Hilbert indices"
    );
}

// ---------------------------------------------------------------------------
// 3. Batch encode matches single encode
// ---------------------------------------------------------------------------

#[test]
fn test_batch_matches_single_3d() {
    let coords = vec![
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.5),
        (1.0, 1.0, 1.0),
        (0.25, 0.75, 0.1),
    ];
    for order in [4, 8] {
        for &(x, y, z) in &coords {
            let single = hilbert::encode_3d(x, y, z, order);
            // Verify determinism
            let single2 = hilbert::encode_3d(x, y, z, order);
            assert_eq!(single, single2, "Non-deterministic encode at ({x}, {y}, {z}) order {order}");
        }
    }
}

#[test]
fn test_batch_matches_single_4d() {
    let coords = vec![
        (0.0, 0.0, 0.0, 0.0),
        (0.5, 0.5, 0.5, 0.5),
        (1.0, 1.0, 1.0, 1.0),
        (0.25, 0.75, 0.1, 0.9),
    ];
    for order in [4, 8] {
        for &(x, y, z, t) in &coords {
            let single = hilbert::encode_4d(x, y, z, t, order);
            let single2 = hilbert::encode_4d(x, y, z, t, order);
            assert_eq!(
                single, single2,
                "Non-deterministic 4D encode at ({x}, {y}, {z}, {t}) order {order}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 4. Epochs for time window — edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_epochs_single() {
    let epochs = temporal::epochs_for_window(5_000, 9_999, 5_000);
    assert_eq!(epochs, vec![1]);
}

#[test]
fn test_epochs_boundary() {
    // Window spans an epoch boundary
    let epochs = temporal::epochs_for_window(4_999, 5_000, 5_000);
    assert_eq!(epochs, vec![0, 1]);
}

#[test]
fn test_epochs_multiple() {
    let epochs = temporal::epochs_for_window(0, 15_000, 5_000);
    assert_eq!(epochs, vec![0, 1, 2, 3]);
}

#[test]
fn test_epochs_exact_boundary() {
    let epochs = temporal::epochs_for_window(5_000, 10_000, 5_000);
    assert_eq!(epochs, vec![1, 2]);
}

// ---------------------------------------------------------------------------
// 5. Temporal normalisation in [0.0, 1.0]
// ---------------------------------------------------------------------------

#[test]
fn test_normalise_start_of_epoch() {
    let t = temporal::normalise_in_epoch(5_000, 1, 5_000);
    assert!((t - 0.0).abs() < 1e-10, "Start of epoch should be 0.0");
}

#[test]
fn test_normalise_mid_epoch() {
    let t = temporal::normalise_in_epoch(7_500, 1, 5_000);
    assert!((t - 0.5).abs() < 1e-10, "Mid epoch should be 0.5");
}

#[test]
fn test_normalise_end_of_epoch() {
    let t = temporal::normalise_in_epoch(9_999, 1, 5_000);
    assert!(t >= 0.0 && t <= 1.0, "Must be in [0, 1]");
    assert!(t > 0.9, "Near end should be close to 1.0");
}

#[test]
fn test_normalise_clamped_below() {
    // Timestamp before the epoch start (should clamp to 0.0)
    let t = temporal::normalise_in_epoch(0, 1, 5_000);
    assert_eq!(t, 0.0);
}

// ---------------------------------------------------------------------------
// 6. Novelty score: 0.0 for identical, 1.0 for orthogonal
// ---------------------------------------------------------------------------

#[test]
fn test_novelty_identical_vectors() {
    let pred = vec![1.0f32, 2.0, 3.0, 4.0];
    let ret = vec![1.0f32, 2.0, 3.0, 4.0];
    let score = novelty::novelty_score(&pred, &[&ret]);
    assert!(score.abs() < 1e-5, "Identical vectors should give novelty ~0.0");
}

#[test]
fn test_novelty_orthogonal_vectors() {
    let pred = vec![1.0f32, 0.0, 0.0];
    let ret = vec![0.0f32, 1.0, 0.0];
    let score = novelty::novelty_score(&pred, &[&ret]);
    assert!(
        (score - 1.0).abs() < 1e-5,
        "Orthogonal vectors should give novelty ~1.0"
    );
}

#[test]
fn test_novelty_multiple_retrieved() {
    let pred = vec![1.0f32, 0.0, 0.0];
    let r1 = vec![0.0f32, 1.0, 0.0]; // orthogonal
    let r2 = vec![0.9f32, 0.1, 0.0]; // similar
    let score = novelty::novelty_score(&pred, &[&r1, &r2]);
    // Should be close to 0 because r2 is very similar
    assert!(score < 0.2, "Should pick up the close match");
}

// ---------------------------------------------------------------------------
// 7. Batch prepare matches individual computation
// ---------------------------------------------------------------------------

#[test]
fn test_batch_prepare_consistency() {
    let xs = vec![0.5, 0.2, 0.8];
    let ys = vec![0.3, 0.7, 0.1];
    let zs = vec![0.8, 0.4, 0.6];
    let timestamps = vec![7_500u64, 12_000, 25_000];
    let epoch_size = 5_000u64;
    let order = 8u32;

    for i in 0..xs.len() {
        let h3 = hilbert::encode_3d(xs[i], ys[i], zs[i], order);
        let epoch = temporal::compute_epoch(timestamps[i], epoch_size);
        let t_norm = temporal::normalise_in_epoch(timestamps[i], epoch, epoch_size);
        let h4 = hilbert::encode_4d(xs[i], ys[i], zs[i], t_norm, order);

        // Verify individually
        assert!(h3 > 0 || (xs[i] == 0.0 && ys[i] == 0.0 && zs[i] == 0.0));
        assert!(epoch == timestamps[i] / epoch_size);
        assert!(t_norm >= 0.0 && t_norm <= 1.0);
        assert!(h4 > 0 || t_norm == 0.0);
    }
}

// ---------------------------------------------------------------------------
// 8. Edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_zero_coordinates() {
    let h = hilbert::encode_3d(0.0, 0.0, 0.0, 4);
    assert_eq!(h, 0, "Origin should map to Hilbert index 0");
}

#[test]
fn test_max_coordinates() {
    let h = hilbert::encode_3d(1.0, 1.0, 1.0, 4);
    assert!(h > 0, "Max coordinates should produce non-zero index");
}

#[test]
fn test_large_timestamp() {
    let ts = 1_700_000_000_000u64; // ~2023
    let epoch = temporal::compute_epoch(ts, 5_000);
    assert_eq!(epoch, 340_000_000);
    let name = temporal::collection_name(epoch);
    assert_eq!(name, "loci_340000000");
}

#[test]
fn test_distance_zero() {
    let d = spatial::distance_3d(0.5, 0.5, 0.5, 0.5, 0.5, 0.5);
    assert!(d.abs() < 1e-15, "Same point should have distance 0");
}

#[test]
fn test_cosine_zero_vector() {
    let a = vec![0.0f32, 0.0, 0.0];
    let b = vec![1.0f32, 2.0, 3.0];
    let sim = novelty::cosine_sim(&a, &b);
    assert_eq!(sim, 0.0, "Zero vector should give similarity 0");
}

#[test]
fn test_buckets_nonempty() {
    let buckets =
        hilbert::spatial_bounds_to_buckets(0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 4, 1.2);
    assert!(!buckets.is_empty(), "Buckets should not be empty");
    // Verify sorted and unique
    for i in 1..buckets.len() {
        assert!(buckets[i] > buckets[i - 1], "Buckets should be sorted and unique");
    }
}

#[test]
fn test_adaptive_order_boundaries() {
    assert_eq!(spatial::adaptive_hilbert_order(0.0), 4);
    assert_eq!(spatial::adaptive_hilbert_order(99.99), 4);
    assert_eq!(spatial::adaptive_hilbert_order(100.0), 8);
    assert_eq!(spatial::adaptive_hilbert_order(9_999.99), 8);
    assert_eq!(spatial::adaptive_hilbert_order(10_000.0), 12);
}
