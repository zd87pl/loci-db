//! Criterion benchmarks for IDD-58 experimental Hilbert-bucket search improvements.
//!
//! Compares four hypotheses against the baseline:
//! - Baseline: existing 3D spatial_bounds_to_buckets (Rust)
//! - Hypothesis A: 4D serial and parallel bucket enumeration
//! - Hypothesis B: Range clustering (post-processing)
//! - Hypothesis C: Hierarchical coarse-to-fine
//! - Hypothesis D: Sampling-based approximation

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use loci_core::hilbert;
use loci_core::hilbert_experiments;

// ---------------------------------------------------------------------------
// Test scenarios: (x_min, x_max, y_min, y_max, z_min, z_max, label)
// ---------------------------------------------------------------------------

struct Scenario3D {
    label: &'static str,
    x_min: f64, x_max: f64,
    y_min: f64, y_max: f64,
    z_min: f64, z_max: f64,
}

struct Scenario4D {
    label: &'static str,
    x_min: f64, x_max: f64,
    y_min: f64, y_max: f64,
    z_min: f64, z_max: f64,
    t_min: f64, t_max: f64,
}

fn scenarios_3d() -> Vec<Scenario3D> {
    vec![
        Scenario3D { label: "narrow_0.2", x_min: 0.4, x_max: 0.6, y_min: 0.4, y_max: 0.6, z_min: 0.4, z_max: 0.6 },
        Scenario3D { label: "medium_0.4", x_min: 0.3, x_max: 0.7, y_min: 0.3, y_max: 0.7, z_min: 0.3, z_max: 0.7 },
        Scenario3D { label: "wide_0.6",   x_min: 0.2, x_max: 0.8, y_min: 0.2, y_max: 0.8, z_min: 0.2, z_max: 0.8 },
    ]
}

fn scenarios_4d() -> Vec<Scenario4D> {
    vec![
        Scenario4D { label: "narrow_spatial+narrow_t", x_min: 0.4, x_max: 0.6, y_min: 0.4, y_max: 0.6, z_min: 0.4, z_max: 0.6, t_min: 0.4, t_max: 0.6 },
        Scenario4D { label: "narrow_spatial+full_t",   x_min: 0.4, x_max: 0.6, y_min: 0.4, y_max: 0.6, z_min: 0.4, z_max: 0.6, t_min: 0.0, t_max: 1.0 },
        Scenario4D { label: "medium_spatial+narrow_t", x_min: 0.3, x_max: 0.7, y_min: 0.3, y_max: 0.7, z_min: 0.3, z_max: 0.7, t_min: 0.3, t_max: 0.7 },
        Scenario4D { label: "wide_all",                x_min: 0.2, x_max: 0.8, y_min: 0.2, y_max: 0.8, z_min: 0.2, z_max: 0.8, t_min: 0.2, t_max: 0.8 },
    ]
}

// ---------------------------------------------------------------------------
// Baseline: existing 3D spatial_bounds_to_buckets
// ---------------------------------------------------------------------------

fn bench_baseline_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("baseline_3d");
    for s in scenarios_3d() {
        for order in [4u32, 6, 8] {
            group.bench_with_input(
                BenchmarkId::new(s.label, format!("p{order}")),
                &order,
                |b, &order| {
                    b.iter(|| {
                        hilbert::spatial_bounds_to_buckets(
                            black_box(s.x_min), black_box(s.x_max),
                            black_box(s.y_min), black_box(s.y_max),
                            black_box(s.z_min), black_box(s.z_max),
                            black_box(order), black_box(1.2),
                        )
                    })
                },
            );
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Hypothesis A: 4D serial and parallel
// ---------------------------------------------------------------------------

fn bench_hyp_a_4d_serial(c: &mut Criterion) {
    let mut group = c.benchmark_group("hyp_a_4d_serial");
    for s in scenarios_4d() {
        for order in [4u32, 6] {
            group.bench_with_input(
                BenchmarkId::new(s.label, format!("p{order}")),
                &order,
                |b, &order| {
                    b.iter(|| {
                        hilbert_experiments::spatial_bounds_to_buckets_4d(
                            black_box(s.x_min), black_box(s.x_max),
                            black_box(s.y_min), black_box(s.y_max),
                            black_box(s.z_min), black_box(s.z_max),
                            black_box(s.t_min), black_box(s.t_max),
                            black_box(order), black_box(1.2),
                        )
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_hyp_a_4d_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("hyp_a_4d_parallel");
    for s in scenarios_4d() {
        for order in [4u32, 6] {
            group.bench_with_input(
                BenchmarkId::new(s.label, format!("p{order}")),
                &order,
                |b, &order| {
                    b.iter(|| {
                        hilbert_experiments::spatial_bounds_to_buckets_4d_parallel(
                            black_box(s.x_min), black_box(s.x_max),
                            black_box(s.y_min), black_box(s.y_max),
                            black_box(s.z_min), black_box(s.z_max),
                            black_box(s.t_min), black_box(s.t_max),
                            black_box(order), black_box(1.2),
                        )
                    })
                },
            );
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Hypothesis B: Range clustering
// ---------------------------------------------------------------------------

fn bench_hyp_b_range_clustering(c: &mut Criterion) {
    let mut group = c.benchmark_group("hyp_b_range_clustering");
    for s in scenarios_4d() {
        for order in [4u32, 6] {
            // Pre-compute bucket IDs (not part of the benchmark)
            let ids = hilbert_experiments::spatial_bounds_to_buckets_4d(
                s.x_min, s.x_max, s.y_min, s.y_max,
                s.z_min, s.z_max, s.t_min, s.t_max,
                order, 1.2,
            );
            group.bench_with_input(
                BenchmarkId::new(s.label, format!("p{order}_n{}", ids.len())),
                &ids,
                |b, ids| {
                    b.iter(|| hilbert_experiments::cluster_into_ranges(black_box(ids)))
                },
            );
        }
    }
    group.finish();
}

fn bench_hyp_b_end_to_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("hyp_b_end_to_end");
    for s in scenarios_4d() {
        for order in [4u32, 6] {
            group.bench_with_input(
                BenchmarkId::new(s.label, format!("p{order}")),
                &order,
                |b, &order| {
                    b.iter(|| {
                        hilbert_experiments::spatial_bounds_to_bucket_ranges_4d(
                            black_box(s.x_min), black_box(s.x_max),
                            black_box(s.y_min), black_box(s.y_max),
                            black_box(s.z_min), black_box(s.z_max),
                            black_box(s.t_min), black_box(s.t_max),
                            black_box(order), black_box(1.2),
                        )
                    })
                },
            );
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Hypothesis C: Hierarchical coarse-to-fine
// ---------------------------------------------------------------------------

fn bench_hyp_c_hierarchical_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("hyp_c_hierarchical_3d");
    for s in scenarios_3d() {
        for (coarse, fine) in [(4u32, 6u32), (4, 8)] {
            group.bench_with_input(
                BenchmarkId::new(s.label, format!("c{coarse}_f{fine}")),
                &(coarse, fine),
                |b, &(coarse, fine)| {
                    b.iter(|| {
                        hilbert_experiments::hierarchical_buckets_3d(
                            black_box(s.x_min), black_box(s.x_max),
                            black_box(s.y_min), black_box(s.y_max),
                            black_box(s.z_min), black_box(s.z_max),
                            black_box(coarse), black_box(fine),
                            black_box(1.2),
                        )
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_hyp_c_hierarchical_4d(c: &mut Criterion) {
    let mut group = c.benchmark_group("hyp_c_hierarchical_4d");
    for s in scenarios_4d() {
        for (coarse, fine) in [(4u32, 6u32)] {
            group.bench_with_input(
                BenchmarkId::new(s.label, format!("c{coarse}_f{fine}")),
                &(coarse, fine),
                |b, &(coarse, fine)| {
                    b.iter(|| {
                        hilbert_experiments::hierarchical_buckets_4d(
                            black_box(s.x_min), black_box(s.x_max),
                            black_box(s.y_min), black_box(s.y_max),
                            black_box(s.z_min), black_box(s.z_max),
                            black_box(s.t_min), black_box(s.t_max),
                            black_box(coarse), black_box(fine),
                            black_box(1.2),
                        )
                    })
                },
            );
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Hypothesis D: Sampling-based approximation
// ---------------------------------------------------------------------------

fn bench_hyp_d_sampling_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("hyp_d_sampling_3d");
    for s in scenarios_3d() {
        for num_samples in [100u32, 500, 1000, 5000] {
            group.bench_with_input(
                BenchmarkId::new(s.label, format!("n{num_samples}_p4")),
                &num_samples,
                |b, &n| {
                    b.iter(|| {
                        hilbert_experiments::sampled_buckets_3d(
                            black_box(s.x_min), black_box(s.x_max),
                            black_box(s.y_min), black_box(s.y_max),
                            black_box(s.z_min), black_box(s.z_max),
                            black_box(4), black_box(1.2),
                            black_box(n), black_box(42),
                        )
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_hyp_d_sampling_4d(c: &mut Criterion) {
    let mut group = c.benchmark_group("hyp_d_sampling_4d");
    for s in scenarios_4d() {
        for num_samples in [500u32, 1000, 5000] {
            group.bench_with_input(
                BenchmarkId::new(s.label, format!("n{num_samples}_p4")),
                &num_samples,
                |b, &n| {
                    b.iter(|| {
                        hilbert_experiments::sampled_buckets_4d(
                            black_box(s.x_min), black_box(s.x_max),
                            black_box(s.y_min), black_box(s.y_max),
                            black_box(s.z_min), black_box(s.z_max),
                            black_box(s.t_min), black_box(s.t_max),
                            black_box(4), black_box(1.2),
                            black_box(n), black_box(42),
                        )
                    })
                },
            );
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_baseline_3d,
    bench_hyp_a_4d_serial,
    bench_hyp_a_4d_parallel,
    bench_hyp_b_range_clustering,
    bench_hyp_b_end_to_end,
    bench_hyp_c_hierarchical_3d,
    bench_hyp_c_hierarchical_4d,
    bench_hyp_d_sampling_3d,
    bench_hyp_d_sampling_4d,
);
criterion_main!(benches);
