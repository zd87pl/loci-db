# IDD-58: Research Results — Hilbert-Bucket Vector Search Optimization

## Executive Summary

We evaluated 4 hypotheses for improving Hilbert-bucket vector search performance
in LOCI-DB. **Two approaches are recommended for production adoption:**

1. **Hypothesis A (Rust 4D Parallel Enumeration)**: 10-150x faster than the current
   Python path. Drop-in replacement. Score: 22/25.
2. **Hypothesis B (Hilbert Range Clustering)**: Lossless compression of bucket ID
   sets into contiguous ranges. 2.9x-1.2M:1 compression ratio. Score: 23/25.

Two approaches were rejected:
- **Hypothesis C (Hierarchical Coarse-to-Fine)**: 1.5-11x slower than direct enumeration.
- **Hypothesis D (Sampling-Based)**: Recall too low (<60% at p=6 even with 5000 samples).

## Detailed Results

### Performance Comparison (p=4, narrow spatial bounds 0.4-0.6)

| Approach | Time | vs Python (~5ms) | vs Rust 3D (4.5us) |
|----------|------|-------------------|---------------------|
| Python itertools (4D, p=4, LUT) | ~5 ms | 1x | 1,111x slower |
| **Rust 4D Serial (Hyp A)** | **34 us** | **147x faster** | 7.6x slower |
| Rust 4D Parallel (Hyp A) | 173 us | 29x faster | 38x slower (overhead) |
| Rust 4D + Range Cluster (Hyp B) | 35 us | 143x faster | 7.8x slower |
| Hierarchical c4->f6 (Hyp C) | 2.57 ms | 1.9x faster | 571x slower |
| Sampling N=1000 (Hyp D) | 19 us | 263x faster | 4.2x slower |

### Correctness

| Approach | Recall (p=4) | Recall (p=6) | Production-safe? |
|----------|-------------|-------------|------------------|
| Hyp A (Rust 4D) | 100% | 100% | Yes |
| Hyp B (Range Cluster) | 100% | 100% | Yes |
| Hyp C (Hierarchical) | 100% (+ 21-70% extra) | 100% (+ 61% extra) | Yes (wasteful) |
| Hyp D (Sampling N=5000) | 94.7-100% | 14.2-57.4% | No |

### Range Clustering Compression (Hypothesis B)

The standout finding: at high resolutions, Hilbert bucket IDs are highly clustered.

| Resolution | Bucket IDs | Ranges | Compression |
|-----------|-----------|--------|-------------|
| p=4 narrow | 1,296 | 446 | 2.9x |
| p=4 wide | 10,368 | 842 | 12.3x |
| p=6 medium | 589,824 | 15,744 | 37.5x |
| **p=8 narrow** | **16,777,216** | **14** | **1,198,373x** |

This means a query that would send 16.7M individual IDs to Qdrant's MatchAny
filter (far exceeding the 10K limit) can instead be expressed as just 14
contiguous Range filters. This unlocks high-resolution queries that were
previously impossible.

## Code Artifacts

All experimental code lives in the repository:

- **Rust implementations**: `loci-core/src/hilbert_experiments.rs`
  - `spatial_bounds_to_buckets_4d()` — Serial 4D enumeration
  - `spatial_bounds_to_buckets_4d_parallel()` — Parallel 4D enumeration
  - `cluster_into_ranges()` — Range clustering
  - `spatial_bounds_to_bucket_ranges_4d()` — Combined enumeration + clustering
  - `hierarchical_buckets_3d/4d()` — Hierarchical approach
  - `sampled_buckets_3d/4d()` — Sampling approach

- **Benchmarks**: `loci-core/benches/hilbert_experiments_bench.rs`
  - 70+ individual benchmarks across all hypotheses, resolutions, and scenarios

- **Evaluation tests**: `loci-core/tests/hilbert_experiments_eval.rs`
  - Correctness, recall, compression ratio measurements

- **Research docs**: `experimental/IDD-58-hilbert-bucket-search/`
  - `THESIS.md` — Research design and hypotheses
  - `JUDGE_EVALUATION.md` — Blind judge scoring
  - `RESULTS.md` — This report

## Recommended Next Steps

### Immediate (Hypothesis A integration)
1. Add PyO3 binding for `spatial_bounds_to_buckets_4d()`
2. Call from `HilbertIndex.query_buckets()` when `loci_core` is available
3. Remove Python `itertools.product` fallback for resolutions with Rust support

### Short-term (Hypothesis B integration)
1. Add PyO3 binding for `cluster_into_ranges()`
2. When bucket count > threshold, use ranges instead of individual IDs
3. Prototype Qdrant filter: `should: [Range(field, gte=start, lte=end) for (start, end) in ranges]`
4. Benchmark Qdrant Range filter vs MatchAny performance

### Future exploration
- Combine A+B for end-to-end 4D → range pipeline
- Investigate Qdrant native Hilbert indexing plugin
- Consider pre-computed range tables for common query shapes
