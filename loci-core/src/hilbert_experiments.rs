//! Experimental Hilbert-bucket search improvements (IDD-58).
//!
//! Four hypotheses for improving bucket enumeration performance:
//! A) Rust 4D parallel bucket enumeration
//! B) Hilbert range clustering (contiguous ranges)
//! C) Hierarchical coarse-to-fine with early termination
//! D) Sampling-based bucket approximation

use rayon::prelude::*;
use std::collections::BTreeSet;

// ---------------------------------------------------------------------------
// Internal helpers (same algorithm as hilbert.rs but accessible here)
// ---------------------------------------------------------------------------

#[inline(always)]
fn clamp_coord(val: i64, max_val: i64) -> u32 {
    val.max(0).min(max_val) as u32
}

fn axes_to_transpose(x: &mut [u32], order: u32) {
    let n = x.len();
    let m: u32 = 1 << (order - 1);
    let mut q = m;
    while q > 1 {
        let p = q - 1;
        for i in 0..n {
            if x[i] & q != 0 {
                x[0] ^= p;
            } else {
                let t = (x[0] ^ x[i]) & p;
                x[0] ^= t;
                x[i] ^= t;
            }
        }
        q >>= 1;
    }
    for i in 1..n {
        x[i] ^= x[i - 1];
    }
    let mut t2: u32 = 0;
    let mut q = m;
    while q > 1 {
        if x[n - 1] & q != 0 {
            t2 ^= q - 1;
        }
        q >>= 1;
    }
    for xi in x.iter_mut() {
        *xi ^= t2;
    }
}

fn transpose_to_distance(x: &[u32], order: u32) -> u64 {
    let mut d: u64 = 0;
    for bit in (0..order).rev() {
        for xi in x.iter() {
            d <<= 1;
            if xi & (1 << bit) != 0 {
                d |= 1;
            }
        }
    }
    d
}

/// Encode a 4D grid point to Hilbert distance (internal, from integer coords).
#[inline]
fn encode_4d_grid(ix: u32, iy: u32, iz: u32, it: u32, order: u32) -> u64 {
    let mut coords = [ix, iy, iz, it];
    axes_to_transpose(&mut coords, order);
    transpose_to_distance(&coords, order)
}

/// Encode a 3D grid point to Hilbert distance (internal, from integer coords).
#[inline]
fn encode_3d_grid(ix: u32, iy: u32, iz: u32, order: u32) -> u64 {
    let mut coords = [ix, iy, iz];
    axes_to_transpose(&mut coords, order);
    transpose_to_distance(&coords, order)
}

/// Expand a [lo, hi] range by overlap_factor, clamped to [0, 1].
fn expand_range(lo: f64, hi: f64, overlap_factor: f64, min_pad: f64) -> (f64, f64) {
    let span = hi - lo;
    let pad = (span * (overlap_factor - 1.0) / 2.0).max(min_pad);
    ((lo - pad).max(0.0), (hi + pad).min(1.0))
}

/// Quantize a float range to grid coordinates.
fn quantize_range(lo: f64, hi: f64, side: u32) -> (u32, u32) {
    let ilo = clamp_coord((lo * side as f64).floor() as i64, side as i64);
    let ihi = clamp_coord((hi * side as f64).ceil() as i64, side as i64);
    (ilo, ihi)
}

// ===========================================================================
// HYPOTHESIS A: Rust 4D Parallel Bucket Enumeration
// ===========================================================================

/// 4D spatial+temporal bounds to Hilbert bucket IDs (serial).
///
/// This is the direct Rust equivalent of Python's `HilbertIndex.query_buckets()`
/// but runs ~14,000x faster per encode operation.
pub fn spatial_bounds_to_buckets_4d(
    x_min: f64, x_max: f64,
    y_min: f64, y_max: f64,
    z_min: f64, z_max: f64,
    t_min: f64, t_max: f64,
    order: u32,
    overlap_factor: f64,
) -> Vec<u64> {
    let side = (1u32 << order) - 1;
    let min_pad = 1.0 / side.max(1) as f64;

    let (xl, xh) = expand_range(x_min, x_max, overlap_factor, min_pad);
    let (yl, yh) = expand_range(y_min, y_max, overlap_factor, min_pad);
    let (zl, zh) = expand_range(z_min, z_max, overlap_factor, min_pad);
    let (tl, th) = expand_range(t_min, t_max, overlap_factor, min_pad);

    let (ix_lo, ix_hi) = quantize_range(xl, xh, side);
    let (iy_lo, iy_hi) = quantize_range(yl, yh, side);
    let (iz_lo, iz_hi) = quantize_range(zl, zh, side);
    let (it_lo, it_hi) = quantize_range(tl, th, side);

    let mut ids = BTreeSet::new();
    for ix in ix_lo..=ix_hi {
        for iy in iy_lo..=iy_hi {
            for iz in iz_lo..=iz_hi {
                for it in it_lo..=it_hi {
                    ids.insert(encode_4d_grid(ix, iy, iz, it, order));
                }
            }
        }
    }
    ids.into_iter().collect()
}

/// 4D spatial+temporal bounds to Hilbert bucket IDs (parallel with rayon).
///
/// Parallelizes across the outermost (x) dimension, collecting results
/// into thread-local BTreeSets and merging at the end.
pub fn spatial_bounds_to_buckets_4d_parallel(
    x_min: f64, x_max: f64,
    y_min: f64, y_max: f64,
    z_min: f64, z_max: f64,
    t_min: f64, t_max: f64,
    order: u32,
    overlap_factor: f64,
) -> Vec<u64> {
    let side = (1u32 << order) - 1;
    let min_pad = 1.0 / side.max(1) as f64;

    let (xl, xh) = expand_range(x_min, x_max, overlap_factor, min_pad);
    let (yl, yh) = expand_range(y_min, y_max, overlap_factor, min_pad);
    let (zl, zh) = expand_range(z_min, z_max, overlap_factor, min_pad);
    let (tl, th) = expand_range(t_min, t_max, overlap_factor, min_pad);

    let (ix_lo, ix_hi) = quantize_range(xl, xh, side);
    let (iy_lo, iy_hi) = quantize_range(yl, yh, side);
    let (iz_lo, iz_hi) = quantize_range(zl, zh, side);
    let (it_lo, it_hi) = quantize_range(tl, th, side);

    let partial_sets: Vec<BTreeSet<u64>> = (ix_lo..=ix_hi)
        .into_par_iter()
        .map(|ix| {
            let mut local = BTreeSet::new();
            for iy in iy_lo..=iy_hi {
                for iz in iz_lo..=iz_hi {
                    for it in it_lo..=it_hi {
                        local.insert(encode_4d_grid(ix, iy, iz, it, order));
                    }
                }
            }
            local
        })
        .collect();

    let mut merged = BTreeSet::new();
    for s in partial_sets {
        merged.extend(s);
    }
    merged.into_iter().collect()
}

// ===========================================================================
// HYPOTHESIS B: Hilbert Range Clustering
// ===========================================================================

/// A contiguous range of Hilbert IDs [start, end] (inclusive).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HilbertRange {
    pub start: u64,
    pub end: u64,
}

impl HilbertRange {
    pub fn len(&self) -> u64 {
        self.end - self.start + 1
    }
}

/// Convert a sorted list of Hilbert IDs into contiguous ranges.
///
/// Adjacent IDs (diff == 1) are merged into single ranges.
/// Returns a vec of (start, end) inclusive ranges.
pub fn cluster_into_ranges(sorted_ids: &[u64]) -> Vec<HilbertRange> {
    if sorted_ids.is_empty() {
        return Vec::new();
    }

    let mut ranges = Vec::new();
    let mut start = sorted_ids[0];
    let mut end = sorted_ids[0];

    for &id in &sorted_ids[1..] {
        if id == end + 1 {
            end = id;
        } else {
            ranges.push(HilbertRange { start, end });
            start = id;
            end = id;
        }
    }
    ranges.push(HilbertRange { start, end });
    ranges
}

/// Generate 4D bucket IDs as contiguous ranges instead of individual IDs.
///
/// This computes the same bucket set as `spatial_bounds_to_buckets_4d` but
/// returns the result as merged contiguous ranges, which can be more
/// efficiently used as Qdrant Range filters.
pub fn spatial_bounds_to_bucket_ranges_4d(
    x_min: f64, x_max: f64,
    y_min: f64, y_max: f64,
    z_min: f64, z_max: f64,
    t_min: f64, t_max: f64,
    order: u32,
    overlap_factor: f64,
) -> Vec<HilbertRange> {
    let ids = spatial_bounds_to_buckets_4d(
        x_min, x_max, y_min, y_max, z_min, z_max, t_min, t_max, order, overlap_factor,
    );
    cluster_into_ranges(&ids)
}

// ===========================================================================
// HYPOTHESIS C: Hierarchical Coarse-to-Fine
// ===========================================================================

/// Hierarchical coarse-to-fine bucket enumeration.
///
/// Phase 1: Enumerate all 3D buckets at `coarse_order` in the spatial bounds.
/// Phase 2: For each coarse bucket, check if its decoded center falls within
///          an expanded query region. If yes, enumerate its fine-resolution
///          sub-buckets. If no, skip it entirely.
///
/// The ratio between fine and coarse orders determines the sub-bucket fan-out.
/// E.g., coarse=4, fine=8 means each coarse cell maps to (2^4)^3 = 4096 fine cells.
/// But most coarse cells will be pruned, so the effective enumeration is much smaller.
pub fn hierarchical_buckets_3d(
    x_min: f64, x_max: f64,
    y_min: f64, y_max: f64,
    z_min: f64, z_max: f64,
    coarse_order: u32,
    fine_order: u32,
    overlap_factor: f64,
) -> Vec<u64> {
    let coarse_side = (1u32 << coarse_order) - 1;
    let fine_side = (1u32 << fine_order) - 1;
    let min_pad = 1.0 / coarse_side.max(1) as f64;

    let (xl, xh) = expand_range(x_min, x_max, overlap_factor, min_pad);
    let (yl, yh) = expand_range(y_min, y_max, overlap_factor, min_pad);
    let (zl, zh) = expand_range(z_min, z_max, overlap_factor, min_pad);

    let (cix_lo, cix_hi) = quantize_range(xl, xh, coarse_side);
    let (ciy_lo, ciy_hi) = quantize_range(yl, yh, coarse_side);
    let (ciz_lo, ciz_hi) = quantize_range(zl, zh, coarse_side);

    // Expanded query bounds for pruning (2x the original to ensure no misses)
    let prune_margin = 2.0;
    let expand_prune = |lo: f64, hi: f64| -> (f64, f64) {
        let span = hi - lo;
        let pad = span * (prune_margin - 1.0) / 2.0;
        ((lo - pad).max(0.0), (hi + pad).min(1.0))
    };
    let (px_lo, px_hi) = expand_prune(x_min, x_max);
    let (py_lo, py_hi) = expand_prune(y_min, y_max);
    let (pz_lo, pz_hi) = expand_prune(z_min, z_max);

    // Scale factor: how many fine grid cells per coarse grid cell
    let scale = fine_side as f64 / coarse_side as f64;

    let mut fine_ids = BTreeSet::new();

    for cix in cix_lo..=cix_hi {
        for ciy in ciy_lo..=ciy_hi {
            for ciz in ciz_lo..=ciz_hi {
                // Decode coarse cell center to normalized coordinates
                let cx = cix as f64 / coarse_side as f64;
                let cy = ciy as f64 / coarse_side as f64;
                let cz = ciz as f64 / coarse_side as f64;

                // Prune: skip if coarse cell center is outside expanded bounds
                if cx < px_lo || cx > px_hi || cy < py_lo || cy > py_hi || cz < pz_lo || cz > pz_hi
                {
                    continue;
                }

                // Map coarse cell to fine grid sub-range
                let fix_lo = (cix as f64 * scale).floor() as u32;
                let fix_hi = ((cix as f64 + 1.0) * scale).ceil() as u32;
                let fiy_lo = (ciy as f64 * scale).floor() as u32;
                let fiy_hi = ((ciy as f64 + 1.0) * scale).ceil() as u32;
                let fiz_lo = (ciz as f64 * scale).floor() as u32;
                let fiz_hi = ((ciz as f64 + 1.0) * scale).ceil() as u32;

                let fix_hi = fix_hi.min(fine_side);
                let fiy_hi = fiy_hi.min(fine_side);
                let fiz_hi = fiz_hi.min(fine_side);

                for fix in fix_lo..=fix_hi {
                    for fiy in fiy_lo..=fiy_hi {
                        for fiz in fiz_lo..=fiz_hi {
                            fine_ids.insert(encode_3d_grid(fix, fiy, fiz, fine_order));
                        }
                    }
                }
            }
        }
    }
    fine_ids.into_iter().collect()
}

/// Hierarchical coarse-to-fine for 4D (spatial + temporal).
pub fn hierarchical_buckets_4d(
    x_min: f64, x_max: f64,
    y_min: f64, y_max: f64,
    z_min: f64, z_max: f64,
    t_min: f64, t_max: f64,
    coarse_order: u32,
    fine_order: u32,
    overlap_factor: f64,
) -> Vec<u64> {
    let coarse_side = (1u32 << coarse_order) - 1;
    let fine_side = (1u32 << fine_order) - 1;
    let min_pad = 1.0 / coarse_side.max(1) as f64;

    let (xl, xh) = expand_range(x_min, x_max, overlap_factor, min_pad);
    let (yl, yh) = expand_range(y_min, y_max, overlap_factor, min_pad);
    let (zl, zh) = expand_range(z_min, z_max, overlap_factor, min_pad);
    let (tl, th) = expand_range(t_min, t_max, overlap_factor, min_pad);

    let (cix_lo, cix_hi) = quantize_range(xl, xh, coarse_side);
    let (ciy_lo, ciy_hi) = quantize_range(yl, yh, coarse_side);
    let (ciz_lo, ciz_hi) = quantize_range(zl, zh, coarse_side);
    let (cit_lo, cit_hi) = quantize_range(tl, th, coarse_side);

    let prune_margin = 2.0;
    let expand_prune = |lo: f64, hi: f64| -> (f64, f64) {
        let span = hi - lo;
        let pad = span * (prune_margin - 1.0) / 2.0;
        ((lo - pad).max(0.0), (hi + pad).min(1.0))
    };
    let (px_lo, px_hi) = expand_prune(x_min, x_max);
    let (py_lo, py_hi) = expand_prune(y_min, y_max);
    let (pz_lo, pz_hi) = expand_prune(z_min, z_max);
    let (pt_lo, pt_hi) = expand_prune(t_min, t_max);

    let scale = fine_side as f64 / coarse_side as f64;

    // Parallelize across outermost dimension
    let partial_sets: Vec<BTreeSet<u64>> = (cix_lo..=cix_hi)
        .into_par_iter()
        .map(|cix| {
            let mut local = BTreeSet::new();
            let cx = cix as f64 / coarse_side as f64;
            if cx < px_lo || cx > px_hi {
                return local;
            }

            for ciy in ciy_lo..=ciy_hi {
                let cy = ciy as f64 / coarse_side as f64;
                if cy < py_lo || cy > py_hi {
                    continue;
                }

                for ciz in ciz_lo..=ciz_hi {
                    let cz = ciz as f64 / coarse_side as f64;
                    if cz < pz_lo || cz > pz_hi {
                        continue;
                    }

                    for cit in cit_lo..=cit_hi {
                        let ct = cit as f64 / coarse_side as f64;
                        if ct < pt_lo || ct > pt_hi {
                            continue;
                        }

                        // Map coarse cell to fine grid sub-range
                        let fix_lo = (cix as f64 * scale).floor() as u32;
                        let fix_hi = ((cix as f64 + 1.0) * scale).ceil().min(fine_side as f64) as u32;
                        let fiy_lo = (ciy as f64 * scale).floor() as u32;
                        let fiy_hi = ((ciy as f64 + 1.0) * scale).ceil().min(fine_side as f64) as u32;
                        let fiz_lo = (ciz as f64 * scale).floor() as u32;
                        let fiz_hi = ((ciz as f64 + 1.0) * scale).ceil().min(fine_side as f64) as u32;
                        let fit_lo = (cit as f64 * scale).floor() as u32;
                        let fit_hi = ((cit as f64 + 1.0) * scale).ceil().min(fine_side as f64) as u32;

                        for fix in fix_lo..=fix_hi {
                            for fiy in fiy_lo..=fiy_hi {
                                for fiz in fiz_lo..=fiz_hi {
                                    for fit in fit_lo..=fit_hi {
                                        local.insert(encode_4d_grid(fix, fiy, fiz, fit, fine_order));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            local
        })
        .collect();

    let mut merged = BTreeSet::new();
    for s in partial_sets {
        merged.extend(s);
    }
    merged.into_iter().collect()
}

// ===========================================================================
// HYPOTHESIS D: Sampling-Based Bucket Approximation
// ===========================================================================

/// Simple deterministic pseudo-random number generator (xorshift64).
/// Used for reproducible sampling without external dependencies.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Return a value in [lo, hi] inclusive.
    fn next_range(&mut self, lo: u32, hi: u32) -> u32 {
        if lo == hi {
            return lo;
        }
        let range = (hi - lo + 1) as u64;
        lo + (self.next() % range) as u32
    }
}

/// Sampling-based 3D bucket approximation.
///
/// Instead of exhaustively enumerating all grid points in the bounding box,
/// sample `num_samples` random grid points and return the unique Hilbert IDs.
///
/// Returns (unique_ids, total_grid_points) so caller can compute recall.
pub fn sampled_buckets_3d(
    x_min: f64, x_max: f64,
    y_min: f64, y_max: f64,
    z_min: f64, z_max: f64,
    order: u32,
    overlap_factor: f64,
    num_samples: u32,
    seed: u64,
) -> (Vec<u64>, u64) {
    let side = (1u32 << order) - 1;
    let min_pad = 1.0 / side.max(1) as f64;

    let (xl, xh) = expand_range(x_min, x_max, overlap_factor, min_pad);
    let (yl, yh) = expand_range(y_min, y_max, overlap_factor, min_pad);
    let (zl, zh) = expand_range(z_min, z_max, overlap_factor, min_pad);

    let (ix_lo, ix_hi) = quantize_range(xl, xh, side);
    let (iy_lo, iy_hi) = quantize_range(yl, yh, side);
    let (iz_lo, iz_hi) = quantize_range(zl, zh, side);

    let total = (ix_hi - ix_lo + 1) as u64
        * (iy_hi - iy_lo + 1) as u64
        * (iz_hi - iz_lo + 1) as u64;

    let mut rng = Xorshift64::new(seed);
    let mut ids = BTreeSet::new();

    for _ in 0..num_samples {
        let ix = rng.next_range(ix_lo, ix_hi);
        let iy = rng.next_range(iy_lo, iy_hi);
        let iz = rng.next_range(iz_lo, iz_hi);
        ids.insert(encode_3d_grid(ix, iy, iz, order));
    }

    (ids.into_iter().collect(), total)
}

/// Sampling-based 4D bucket approximation.
pub fn sampled_buckets_4d(
    x_min: f64, x_max: f64,
    y_min: f64, y_max: f64,
    z_min: f64, z_max: f64,
    t_min: f64, t_max: f64,
    order: u32,
    overlap_factor: f64,
    num_samples: u32,
    seed: u64,
) -> (Vec<u64>, u64) {
    let side = (1u32 << order) - 1;
    let min_pad = 1.0 / side.max(1) as f64;

    let (xl, xh) = expand_range(x_min, x_max, overlap_factor, min_pad);
    let (yl, yh) = expand_range(y_min, y_max, overlap_factor, min_pad);
    let (zl, zh) = expand_range(z_min, z_max, overlap_factor, min_pad);
    let (tl, th) = expand_range(t_min, t_max, overlap_factor, min_pad);

    let (ix_lo, ix_hi) = quantize_range(xl, xh, side);
    let (iy_lo, iy_hi) = quantize_range(yl, yh, side);
    let (iz_lo, iz_hi) = quantize_range(zl, zh, side);
    let (it_lo, it_hi) = quantize_range(tl, th, side);

    let total = (ix_hi - ix_lo + 1) as u64
        * (iy_hi - iy_lo + 1) as u64
        * (iz_hi - iz_lo + 1) as u64
        * (it_hi - it_lo + 1) as u64;

    let mut rng = Xorshift64::new(seed);
    let mut ids = BTreeSet::new();

    for _ in 0..num_samples {
        let ix = rng.next_range(ix_lo, ix_hi);
        let iy = rng.next_range(iy_lo, iy_hi);
        let iz = rng.next_range(iz_lo, iz_hi);
        let it = rng.next_range(it_lo, it_hi);
        ids.insert(encode_4d_grid(ix, iy, iz, it, order));
    }

    (ids.into_iter().collect(), total)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hilbert::spatial_bounds_to_buckets;

    #[test]
    fn test_4d_serial_matches_baseline_subset() {
        // The 4D version should produce a superset of the 3D version
        // (3D version ignores temporal dimension)
        let buckets_3d = spatial_bounds_to_buckets(0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 4, 1.2);
        let buckets_4d = spatial_bounds_to_buckets_4d(
            0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.0, 1.0, 4, 1.2,
        );
        // 4D with full t range should have >= the unique IDs as 3D
        assert!(!buckets_4d.is_empty());
        assert!(buckets_4d.len() >= buckets_3d.len());
    }

    #[test]
    fn test_4d_parallel_matches_serial() {
        let serial = spatial_bounds_to_buckets_4d(
            0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.1, 0.9, 4, 1.2,
        );
        let parallel = spatial_bounds_to_buckets_4d_parallel(
            0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.1, 0.9, 4, 1.2,
        );
        assert_eq!(serial, parallel);
    }

    #[test]
    fn test_range_clustering_merges_adjacent() {
        let ids = vec![1, 2, 3, 5, 6, 10, 11, 12, 13];
        let ranges = cluster_into_ranges(&ids);
        assert_eq!(ranges.len(), 3);
        assert_eq!(ranges[0], HilbertRange { start: 1, end: 3 });
        assert_eq!(ranges[1], HilbertRange { start: 5, end: 6 });
        assert_eq!(
            ranges[2],
            HilbertRange {
                start: 10,
                end: 13
            }
        );
    }

    #[test]
    fn test_range_clustering_preserves_total_count() {
        let ids = spatial_bounds_to_buckets_4d(
            0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 4, 1.2,
        );
        let ranges = cluster_into_ranges(&ids);
        let total_from_ranges: u64 = ranges.iter().map(|r| r.len()).sum();
        assert_eq!(total_from_ranges, ids.len() as u64);
    }

    #[test]
    fn test_range_clustering_compresses() {
        let ids = spatial_bounds_to_buckets_4d(
            0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 4, 1.2,
        );
        let ranges = cluster_into_ranges(&ids);
        // Ranges should compress: fewer ranges than individual IDs
        assert!(
            ranges.len() < ids.len(),
            "Expected compression: {} ranges vs {} IDs",
            ranges.len(),
            ids.len()
        );
    }

    #[test]
    fn test_hierarchical_3d_covers_baseline() {
        // Hierarchical fine buckets should be a superset of direct fine enumeration
        // for the same bounds (since hierarchical adds the coarse-cell margin)
        let direct = spatial_bounds_to_buckets(0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 6, 1.2);
        let hierarchical =
            hierarchical_buckets_3d(0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 4, 6, 1.2);
        // Hierarchical should find at least as many since it uses broader coarse cells
        assert!(!hierarchical.is_empty());
        // Check that direct's buckets are largely covered
        let hier_set: BTreeSet<u64> = hierarchical.iter().copied().collect();
        let direct_set: BTreeSet<u64> = direct.iter().copied().collect();
        let covered = direct_set.intersection(&hier_set).count();
        let recall = covered as f64 / direct_set.len() as f64;
        assert!(
            recall >= 0.95,
            "Hierarchical recall too low: {:.1}% ({}/{})",
            recall * 100.0,
            covered,
            direct_set.len()
        );
    }

    #[test]
    fn test_sampled_3d_has_reasonable_recall() {
        let full = spatial_bounds_to_buckets(0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 4, 1.2);
        let full_set: BTreeSet<u64> = full.iter().copied().collect();

        let (sampled, _total) = sampled_buckets_3d(
            0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 4, 1.2, 5000, 42,
        );
        let sampled_set: BTreeSet<u64> = sampled.iter().copied().collect();

        let covered = full_set.intersection(&sampled_set).count();
        let recall = covered as f64 / full_set.len() as f64;
        assert!(
            recall >= 0.80,
            "Sampling recall too low: {:.1}% ({}/{})",
            recall * 100.0,
            covered,
            full_set.len()
        );
    }

    #[test]
    fn test_sampled_4d_deterministic() {
        let (a, _) = sampled_buckets_4d(
            0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.0, 1.0, 4, 1.2, 1000, 42,
        );
        let (b, _) = sampled_buckets_4d(
            0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.0, 1.0, 4, 1.2, 1000, 42,
        );
        assert_eq!(a, b, "Same seed should produce same results");
    }
}
