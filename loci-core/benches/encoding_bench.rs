//! Criterion benchmarks for loci_core encoding primitives.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use loci_core::hilbert;
use loci_core::novelty;
use loci_core::spatial;
use loci_core::temporal;

fn bench_encode_hilbert_3d(c: &mut Criterion) {
    c.bench_function("encode_hilbert_3d/single", |b| {
        b.iter(|| hilbert::encode_3d(black_box(0.5), black_box(0.3), black_box(0.8), black_box(8)))
    });
}

fn bench_encode_hilbert_4d(c: &mut Criterion) {
    c.bench_function("encode_hilbert_4d/single", |b| {
        b.iter(|| {
            hilbert::encode_4d(
                black_box(0.5),
                black_box(0.3),
                black_box(0.8),
                black_box(0.6),
                black_box(8),
            )
        })
    });
}

fn bench_batch_encode_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_encode_hilbert_3d");
    for size in [100, 1_000, 10_000, 100_000] {
        let coords: Vec<(f64, f64, f64)> = (0..size)
            .map(|i| {
                let f = i as f64 / size as f64;
                (f, (f * 1.5) % 1.0, (f * 2.7) % 1.0)
            })
            .collect();

        group.bench_with_input(BenchmarkId::new("N", size), &coords, |b, coords| {
            b.iter(|| {
                for &(x, y, z) in coords {
                    black_box(hilbert::encode_3d(x, y, z, 8));
                }
            })
        });
    }
    group.finish();
}

fn bench_batch_encode_4d(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_encode_hilbert_4d");
    for size in [100, 1_000, 10_000, 100_000] {
        let coords: Vec<(f64, f64, f64, f64)> = (0..size)
            .map(|i| {
                let f = i as f64 / size as f64;
                (f, (f * 1.5) % 1.0, (f * 2.7) % 1.0, (f * 3.1) % 1.0)
            })
            .collect();

        group.bench_with_input(BenchmarkId::new("N", size), &coords, |b, coords| {
            b.iter(|| {
                for &(x, y, z, t) in coords {
                    black_box(hilbert::encode_4d(x, y, z, t, 8));
                }
            })
        });
    }
    group.finish();
}

fn bench_batch_prepare(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_prepare_world_states");
    for size in [100, 1_000, 10_000] {
        let xs: Vec<f64> = (0..size).map(|i| (i as f64 / size as f64)).collect();
        let ys: Vec<f64> = (0..size)
            .map(|i| ((i as f64 * 1.5) / size as f64) % 1.0)
            .collect();
        let zs: Vec<f64> = (0..size)
            .map(|i| ((i as f64 * 2.7) / size as f64) % 1.0)
            .collect();
        let timestamps: Vec<u64> = (0..size).map(|i| (i as u64) * 100).collect();

        group.bench_with_input(
            BenchmarkId::new("N", size),
            &(xs, ys, zs, timestamps),
            |b, (xs, ys, zs, ts)| {
                b.iter(|| {
                    for i in 0..xs.len() {
                        let h3 = hilbert::encode_3d(xs[i], ys[i], zs[i], 8);
                        let epoch = temporal::compute_epoch(ts[i], 5_000);
                        let t_norm = temporal::normalise_in_epoch(ts[i], epoch, 5_000);
                        let h4 = hilbert::encode_4d(xs[i], ys[i], zs[i], t_norm, 8);
                        black_box((h3, h4, epoch));
                    }
                })
            },
        );
    }
    group.finish();
}

fn bench_spatial_bounds(c: &mut Criterion) {
    c.bench_function("spatial_bounds_to_hilbert_buckets", |b| {
        b.iter(|| {
            hilbert::spatial_bounds_to_buckets(
                black_box(0.4),
                black_box(0.6),
                black_box(0.4),
                black_box(0.6),
                black_box(0.4),
                black_box(0.6),
                black_box(4),
                black_box(1.2),
            )
        })
    });
}

fn bench_cosine_similarity(c: &mut Criterion) {
    let dim = 512;
    let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();
    let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.02).cos()).collect();

    c.bench_function("cosine_similarity/512d", |bench| {
        bench.iter(|| novelty::cosine_sim(black_box(&a), black_box(&b)))
    });
}

fn bench_novelty_score(c: &mut Criterion) {
    let dim = 512;
    let k = 10;
    let predicted: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();
    let retrieved: Vec<Vec<f32>> = (0..k)
        .map(|j| {
            (0..dim)
                .map(|i| ((i + j * 37) as f32 * 0.02).cos())
                .collect()
        })
        .collect();
    let refs: Vec<&[f32]> = retrieved.iter().map(|v| v.as_slice()).collect();

    c.bench_function("compute_novelty_score/512d_k10", |bench| {
        bench.iter(|| novelty::novelty_score(black_box(&predicted), black_box(&refs)))
    });
}

fn bench_distance_3d(c: &mut Criterion) {
    c.bench_function("distance_3d/single", |b| {
        b.iter(|| {
            spatial::distance_3d(
                black_box(0.1),
                black_box(0.2),
                black_box(0.3),
                black_box(0.9),
                black_box(0.8),
                black_box(0.7),
            )
        })
    });
}

criterion_group!(
    benches,
    bench_encode_hilbert_3d,
    bench_encode_hilbert_4d,
    bench_batch_encode_3d,
    bench_batch_encode_4d,
    bench_batch_prepare,
    bench_spatial_bounds,
    bench_cosine_similarity,
    bench_novelty_score,
    bench_distance_3d,
);
criterion_main!(benches);
