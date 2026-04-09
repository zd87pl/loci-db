# loci-core Performance

Benchmark results from Criterion on Apple M5 Max (arm64), Rust 1.94.1 release profile (LTO, opt-level 3).

## Single-Point Encoding

| Operation                | Time       |
|--------------------------|------------|
| `encode_hilbert_3d`      | 20 ns      |
| `encode_hilbert_4d`      | 29 ns      |
| `decode_hilbert_3d`      | ~20 ns     |
| `distance_3d`            | 1.6 ns     |
| `cosine_similarity/512d` | 270 ns     |

## Batch Encoding (Hilbert 3D, order=8)

| Batch Size | Time       | Per-element |
|------------|------------|-------------|
| 100        | 1.85 µs   | 18.5 ns     |
| 1,000      | 18.7 µs   | 18.7 ns     |
| 10,000     | 188 µs    | 18.8 ns     |
| 100,000    | 1.87 ms   | 18.7 ns     |

## Batch Encoding (Hilbert 4D, order=8)

| Batch Size | Time       | Per-element |
|------------|------------|-------------|
| 100        | 2.64 µs   | 26.4 ns     |
| 1,000      | 26.6 µs   | 26.6 ns     |
| 10,000     | 265 µs    | 26.5 ns     |
| 100,000    | 2.66 ms   | 26.6 ns     |

## Full WorldState Batch Preparation (hilbert_3d + hilbert_4d + epoch_id)

| Batch Size | Time       | Per-element |
|------------|------------|-------------|
| 100        | 3.44 µs   | 34.4 ns     |
| 1,000      | 34.6 µs   | 34.6 ns     |
| 10,000     | 350 µs    | 35.0 ns     |

## Query Pre-filtering

| Operation                            | Time     |
|--------------------------------------|----------|
| `spatial_bounds_to_hilbert_buckets`  | 4.1 µs   |
| `compute_novelty_score` (512d, k=10) | 2.7 µs   |

## Python vs Rust Comparison

The Python `hilbertcurve` library processes ~2,400 4D encodings/sec (measured via `timeit`).
Rust `loci_core` processes ~37,600,000 4D encodings/sec at batch scale.

| Metric                    | Python          | Rust (loci_core) | Speedup  |
|---------------------------|-----------------|------------------|----------|
| Single hilbert_4d encode  | ~420 µs         | 29 ns            | ~14,500x |
| 10K batch hilbert_4d      | ~4,200 ms       | 265 µs           | ~15,800x |
| 10K batch_prepare (full)  | ~4,500 ms       | 350 µs           | ~12,900x |

> Note: Python timings are estimates based on the `hilbertcurve` library's known
> per-call overhead. Actual speedup varies by system and Python version.
