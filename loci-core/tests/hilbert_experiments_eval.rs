//! Evaluation tests for IDD-58 experimental Hilbert-bucket search improvements.
//!
//! Measures correctness (recall vs exhaustive baseline), bucket count,
//! range compression ratio, and other quality metrics.

use loci_core::hilbert;
use loci_core::hilbert_experiments;
use std::collections::BTreeSet;

/// Test parameters for evaluation.
struct EvalScenario {
    label: &'static str,
    x_min: f64, x_max: f64,
    y_min: f64, y_max: f64,
    z_min: f64, z_max: f64,
    order: u32,
    overlap_factor: f64,
}

fn eval_scenarios() -> Vec<EvalScenario> {
    vec![
        EvalScenario { label: "narrow_p4", x_min: 0.4, x_max: 0.6, y_min: 0.4, y_max: 0.6, z_min: 0.4, z_max: 0.6, order: 4, overlap_factor: 1.2 },
        EvalScenario { label: "medium_p4", x_min: 0.3, x_max: 0.7, y_min: 0.3, y_max: 0.7, z_min: 0.3, z_max: 0.7, order: 4, overlap_factor: 1.2 },
        EvalScenario { label: "wide_p4",   x_min: 0.2, x_max: 0.8, y_min: 0.2, y_max: 0.8, z_min: 0.2, z_max: 0.8, order: 4, overlap_factor: 1.2 },
        EvalScenario { label: "narrow_p6", x_min: 0.4, x_max: 0.6, y_min: 0.4, y_max: 0.6, z_min: 0.4, z_max: 0.6, order: 6, overlap_factor: 1.2 },
        EvalScenario { label: "medium_p6", x_min: 0.3, x_max: 0.7, y_min: 0.3, y_max: 0.7, z_min: 0.3, z_max: 0.7, order: 6, overlap_factor: 1.2 },
        EvalScenario { label: "narrow_p8", x_min: 0.4, x_max: 0.6, y_min: 0.4, y_max: 0.6, z_min: 0.4, z_max: 0.6, order: 8, overlap_factor: 1.2 },
    ]
}

#[test]
fn eval_hypothesis_a_4d_correctness() {
    println!("\n=== Hypothesis A: 4D Bucket Enumeration ===");
    println!("{:<15} {:>10} {:>10} {:>10}", "Scenario", "3D_count", "4D_serial", "4D_par");

    for s in eval_scenarios() {
        let baseline_3d = hilbert::spatial_bounds_to_buckets(
            s.x_min, s.x_max, s.y_min, s.y_max, s.z_min, s.z_max, s.order, s.overlap_factor,
        );
        let serial_4d = hilbert_experiments::spatial_bounds_to_buckets_4d(
            s.x_min, s.x_max, s.y_min, s.y_max, s.z_min, s.z_max, 0.0, 1.0,
            s.order, s.overlap_factor,
        );
        let parallel_4d = hilbert_experiments::spatial_bounds_to_buckets_4d_parallel(
            s.x_min, s.x_max, s.y_min, s.y_max, s.z_min, s.z_max, 0.0, 1.0,
            s.order, s.overlap_factor,
        );

        // Parallel must exactly match serial
        assert_eq!(serial_4d, parallel_4d, "Parallel mismatch for {}", s.label);

        println!("{:<15} {:>10} {:>10} {:>10}",
            s.label, baseline_3d.len(), serial_4d.len(), parallel_4d.len());
    }
}

#[test]
fn eval_hypothesis_b_range_compression() {
    println!("\n=== Hypothesis B: Range Clustering Compression ===");
    println!("{:<15} {:>10} {:>10} {:>10} {:>10}",
        "Scenario", "Bucket_IDs", "Ranges", "Ratio", "Avg_range");

    for s in eval_scenarios() {
        let ids = hilbert_experiments::spatial_bounds_to_buckets_4d(
            s.x_min, s.x_max, s.y_min, s.y_max, s.z_min, s.z_max, 0.4, 0.6,
            s.order, s.overlap_factor,
        );
        let ranges = hilbert_experiments::cluster_into_ranges(&ids);

        let num_ids = ids.len();
        let num_ranges = ranges.len();
        let ratio = if num_ranges > 0 { num_ids as f64 / num_ranges as f64 } else { 0.0 };
        let avg_range_len: f64 = if num_ranges > 0 {
            ranges.iter().map(|r| r.len() as f64).sum::<f64>() / num_ranges as f64
        } else {
            0.0
        };

        println!("{:<15} {:>10} {:>10} {:>10.1}x {:>10.1}",
            s.label, num_ids, num_ranges, ratio, avg_range_len);
    }
}

#[test]
fn eval_hypothesis_c_hierarchical_recall() {
    println!("\n=== Hypothesis C: Hierarchical Coarse-to-Fine Recall ===");
    println!("{:<15} {:>12} {:>12} {:>10} {:>10}",
        "Scenario", "Direct_fine", "Hierarchical", "Recall%", "Extra%");

    // Test 3D hierarchical at coarse=4, fine=6
    for s in eval_scenarios().iter().filter(|s| s.order <= 6) {
        let direct_fine = hilbert::spatial_bounds_to_buckets(
            s.x_min, s.x_max, s.y_min, s.y_max, s.z_min, s.z_max,
            s.order.max(6), s.overlap_factor,
        );
        let hierarchical = hilbert_experiments::hierarchical_buckets_3d(
            s.x_min, s.x_max, s.y_min, s.y_max, s.z_min, s.z_max,
            4, s.order.max(6), s.overlap_factor,
        );

        let direct_set: BTreeSet<u64> = direct_fine.iter().copied().collect();
        let hier_set: BTreeSet<u64> = hierarchical.iter().copied().collect();

        let covered = direct_set.intersection(&hier_set).count();
        let extra = hier_set.difference(&direct_set).count();
        let recall = if direct_set.is_empty() { 100.0 } else { covered as f64 / direct_set.len() as f64 * 100.0 };
        let extra_pct = if hier_set.is_empty() { 0.0 } else { extra as f64 / hier_set.len() as f64 * 100.0 };

        println!("{:<15} {:>12} {:>12} {:>9.1}% {:>9.1}%",
            s.label, direct_fine.len(), hierarchical.len(), recall, extra_pct);
    }
}

#[test]
fn eval_hypothesis_d_sampling_recall() {
    println!("\n=== Hypothesis D: Sampling-Based Approximation Recall ===");
    println!("{:<15} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Scenario", "Total", "N=100", "N=500", "N=1000", "N=5000");

    for s in eval_scenarios().iter().filter(|s| s.order <= 6) {
        let baseline = hilbert::spatial_bounds_to_buckets(
            s.x_min, s.x_max, s.y_min, s.y_max, s.z_min, s.z_max, s.order, s.overlap_factor,
        );
        let baseline_set: BTreeSet<u64> = baseline.iter().copied().collect();

        let mut recalls = Vec::new();
        for n in [100, 500, 1000, 5000] {
            let (sampled, _total) = hilbert_experiments::sampled_buckets_3d(
                s.x_min, s.x_max, s.y_min, s.y_max, s.z_min, s.z_max,
                s.order, s.overlap_factor, n, 42,
            );
            let sampled_set: BTreeSet<u64> = sampled.iter().copied().collect();
            let covered = baseline_set.intersection(&sampled_set).count();
            let recall = if baseline_set.is_empty() { 100.0 } else { covered as f64 / baseline_set.len() as f64 * 100.0 };
            recalls.push(recall);
        }

        println!("{:<15} {:>8} {:>7.1}% {:>7.1}% {:>7.1}% {:>7.1}%",
            s.label, baseline.len(), recalls[0], recalls[1], recalls[2], recalls[3]);
    }
}
