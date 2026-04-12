# IDD-58: Improving Hilbert-Bucket Vector Search Performance

## Problem Statement

LOCI-DB uses Hilbert space-filling curves to map 4D spatiotemporal coordinates
(x, y, z, t) to 1D integer indices, enabling spatial pre-filtering via Qdrant's
`MatchAny` payload filter. The current implementation has performance bottlenecks
in bucket enumeration that limit query throughput at higher resolutions.

### Current Architecture

```
Query(vector, spatial_bounds, time_window)
  -> Temporal Sharding (epoch selection)
  -> Adaptive Resolution Selection
  -> Hilbert Bucket Enumeration  <-- BOTTLENECK
  -> Qdrant ANN Search (MatchAny filter)
  -> Exact Post-filtering
  -> Temporal Decay Scoring
  -> Top-k Selection
```

### Identified Bottlenecks

1. **4D bucket enumeration is Python-only**: The Rust `spatial_bounds_to_buckets`
   handles 3D only. The 4D `query_buckets` in Python uses `itertools.product`
   with per-point `HilbertCurve.distance_from_point()` calls — O(range^4) at
   ~420us per encode in pure Python.

2. **Exhaustive enumeration**: Every grid point in the expanded bounding box is
   individually encoded. At p=4 with typical bounds this is manageable (~500 points),
   but at p=8 it becomes intractable (millions of points).

3. **Large MatchAny filter sets**: The resulting bucket ID list can be very large,
   which degrades Qdrant filter performance (hard cap at 10K IDs).

4. **No Rust path for 4D bounds-to-buckets**: Forces fallback to slow Python path
   even when loci_core is available.

### Baseline Performance

| Operation | Time | Source |
|-----------|------|--------|
| Rust encode_3d (single) | 20 ns | Criterion bench |
| Rust encode_4d (single) | 29 ns | Criterion bench |
| Rust spatial_bounds_to_buckets (3D, p=4) | 4.1 us | Criterion bench |
| Python query_buckets (4D, p=4, LUT) | <5 ms | test_hilbert.py |
| Python query_buckets (4D, p=8, itertools) | ~47 ms | test_hilbert.py |

## Research Hypotheses

### Hypothesis A: Rust 4D Parallel Bucket Enumeration

**Thesis**: Moving the 4D bucket enumeration from Python to Rust with rayon
parallelization will achieve 100-1000x speedup over the Python itertools path
while maintaining correctness.

**Approach**: Implement `spatial_bounds_to_buckets_4d()` in Rust with the same
expand-quantize-enumerate algorithm but using rayon parallel iterators.

**Expected impact**: Eliminates the Python bottleneck entirely. At p=4, should
complete in <100us. At p=8 with typical bounds, should complete in <10ms.

### Hypothesis B: Hilbert Range Clustering

**Thesis**: The set of Hilbert IDs for a contiguous bounding box forms a small
number of contiguous ID ranges. Returning sorted ranges `[(lo, hi), ...]` instead
of individual IDs enables more efficient Qdrant filters (Range vs MatchAny) and
reduces data transfer.

**Approach**: After enumerating bucket IDs, merge adjacent IDs into contiguous
ranges. Measure compression ratio (num_ranges / num_ids) and evaluate whether
Qdrant range filters can exploit this.

**Expected impact**: 10-50x reduction in filter cardinality for typical queries.
May enable higher-resolution queries that were previously blocked by the 10K
bucket limit.

### Hypothesis C: Hierarchical Coarse-to-Fine with Early Termination

**Thesis**: A two-phase approach — enumerate at coarse resolution (p=4), then
selectively refine only those coarse buckets that overlap the query bounds at
fine resolution (p=8) — will reduce total encoding operations by 90%+ compared
to exhaustive fine-resolution enumeration.

**Approach**: 
1. Enumerate all p=4 buckets in the expanded bounds
2. For each p=4 bucket, decode its center point
3. If the center is within 2x the query bounds, enumerate its p=8 sub-buckets
4. Skip distant coarse buckets entirely

**Expected impact**: Makes p=8 queries feasible in sub-millisecond time by
pruning ~90% of the enumeration space.

### Hypothesis D: Sampling-Based Bucket Approximation

**Thesis**: For large bounding boxes where exhaustive enumeration is intractable,
random sampling of grid points within the bounds can approximate the full bucket
set with controllable precision, trading a small false-negative rate for dramatic
speedup.

**Approach**: Sample N random grid points uniformly within the expanded bounds,
encode each to Hilbert IDs, and return the unique set. Measure recall (fraction
of true buckets found) as a function of N.

**Expected impact**: O(N) time regardless of bounding box volume. With N=1000,
expect >95% recall for typical query shapes at p=4.

## Evaluation Criteria

Each hypothesis will be evaluated on:

1. **Throughput** (operations/sec): Bucket generation speed
2. **Correctness**: Recall vs exhaustive baseline (must be >=99% for production)
3. **Scalability**: Performance at p=4, p=8, p=12 with varying bounding box sizes
4. **Integration cost**: Complexity of integrating into the existing query pipeline
5. **Memory overhead**: Additional memory required

## Experimental Design

Each experiment:
1. Implements the approach as a Rust function in `loci-core`
2. Includes Criterion benchmarks comparing to baseline
3. Tests correctness against the exhaustive baseline
4. Measures at multiple resolutions and bounding box sizes
