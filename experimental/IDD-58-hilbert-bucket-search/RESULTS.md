# IDD-58: Research Results — Hilbert-Bucket Vector Search Optimization

## Executive Summary

We evaluated 4 hypotheses for improving Hilbert-bucket vector search performance
in LOCI-DB. **Two approaches are recommended for production adoption:**

1. **Hypothesis A (Rust 4D Enumeration)**: the **serial** implementation is
   10-150x faster than the current Python path. Drop-in replacement. Score: 22/25.
   (The rayon-parallel variant *loses* to serial by ~5x on small workloads — 173us
   vs 34us at p=4 narrow — and gains only 1.3-1.6x on large ones; serial should be
   the default.)
2. **Hypothesis B (Hilbert Range Clustering)**: Lossless compression of bucket ID
   sets into contiguous ranges. 2.9x-37.5x on typical bounds; the 1.2M:1 headline
   figure is a grid-aligned special case (see caveat below). Score: 23/25.

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

Hilbert bucket IDs for contiguous boxes cluster into far fewer contiguous
ranges than individual IDs, with the degree of clustering depending strongly
on how the query bounds align to the Hilbert grid.

| Resolution | Bucket IDs | Ranges | Compression |
|-----------|-----------|--------|-------------|
| p=4 narrow | 1,296 | 446 | 2.9x |
| p=4 wide | 10,368 | 842 | 12.3x |
| p=6 medium | 589,824 | 15,744 | 37.5x |
| **p=8 narrow** | **16,777,216** | **14** | **1,198,373x** |

In the p=8 narrow scenario, a query that would send 16.7M individual IDs to
Qdrant's MatchAny filter (far exceeding the 10K limit) can instead be expressed
as 14 contiguous Range filters — but see the caveat below before generalizing.

#### Caveat: the p=8 result is a grid-alignment artifact, not a general property

The 1.2M:1 compression at p=8 is specific to the single scenario tested and
does **not** generalize:

- **Why it happens**: the narrow bounds (0.4-0.6), after padding, quantize to
  cells [96, 159] on each axis at p=8. Both edges are multiples of 32
  (96 = 3x32, 160 = 5x32), so the query box decomposes exactly into 16
  order-5 subcubes (2 per axis in 4D, each 32^4 cells). Every complete
  subcube of a Hilbert curve is a single contiguous ID range, so the box
  collapses to at most 16 ranges (14 after merging adjacent ones). This is a
  power-of-2 grid-alignment coincidence of these particular bounds.
- **Non-aligned bounds do not reproduce it**: at p=6, the narrow scenario
  (an 18-cell-per-axis box, not subcube-aligned) yields 104,976 IDs in
  11,726 ranges — only 9.0x, three orders of magnitude away from "a handful
  of ranges". Generic bounds at p=8 should be expected to behave like the
  p=4/p=6 rows (single-digit to double-digit compression), not like the
  aligned row.
- **Enumeration cost remains the binding constraint at p>=8**: ranges are
  computed *after* enumerating all bucket IDs, and the judge's own scalability
  scoring calls exhaustive enumeration intractable at p>=8 (19.5ms-517ms even
  for the 3D Rust baseline; 4D is worse). Range clustering reduces filter
  cardinality; it does not make p=8 enumeration affordable.

Net: range clustering is a solid, lossless cardinality reduction (roughly
3x-40x on measured non-aligned scenarios) that can bring some
previously-over-limit queries under Qdrant's 10K filter cap when bounds
happen to align well. It should not be read as "p=8 queries are now
unlocked" — that requires first solving enumeration cost (e.g. deriving
ranges directly from the box decomposition without per-cell enumeration,
which the aligned case suggests is a promising follow-up).

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
