/**
 * Hilbert curve index, Skilling's transform: a port of `hilbertcurve`'s
 * `HilbertCurve(p, n).distance_from_point`, which `loci/spatial/hilbert.py` uses.
 * Quantisation matches the library: round half to even of u * (2^p - 1), clamped.
 * For p * n <= 52 the result is an exact JavaScript number.
 */

function roundHalfEven(v: number): number {
  const f = Math.floor(v);
  const diff = v - f;
  if (diff > 0.5) return f + 1;
  if (diff < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

export function quantise(u: number, order: number): number {
  const side = (1 << order) - 1;
  const q = roundHalfEven(Math.min(1, Math.max(0, u)) * side);
  return Math.min(side, Math.max(0, q));
}

export function hilbertDistance(point: number[], order: number): number {
  const n = point.length;
  if (order * n > 52) throw new RangeError("order * dims must be <= 52");
  const x = point.map((v) => v | 0);
  const m = 1 << (order - 1);

  // Inverse undo excess work.
  for (let q = m; q > 1; q >>= 1) {
    const p = q - 1;
    for (let i = 0; i < n; i++) {
      if (x[i]! & q) {
        x[0]! ^= p;
      } else {
        const t = (x[0]! ^ x[i]!) & p;
        x[0]! ^= t;
        x[i]! ^= t;
      }
    }
  }
  // Gray encode.
  for (let i = 1; i < n; i++) x[i]! ^= x[i - 1]!;
  let t = 0;
  for (let q = m; q > 1; q >>= 1) if (x[n - 1]! & q) t ^= q - 1;
  for (let i = 0; i < n; i++) x[i]! ^= t;

  // Interleave the transposed bits, most significant first.
  let h = 0;
  for (let bit = order - 1; bit >= 0; bit--) {
    for (let i = 0; i < n; i++) h = h * 2 + ((x[i]! >> bit) & 1);
  }
  return h;
}

/** 3-D index of a unit-cube position at the given order. */
export function hilbert3(u: [number, number, number], order: number): number {
  return hilbertDistance(u.map((v) => quantise(v, order)), order);
}

/** 4-D index of a unit-hypercube position (x, y, z, t within the epoch), as loci's HilbertIndex.encode. */
export function hilbert4(u: [number, number, number, number], order: number): number {
  return hilbertDistance(u.map((v) => quantise(v, order)), order);
}

/**
 * Index ranges that cover every order-`order` cell of a unit box in n dimensions, as inclusive [lo, hi]
 * pairs. The top n·p bits of an order-`order` index are the order-p index of the enclosing coarse cell,
 * so each coarse cell is one contiguous range. Picks the finest p whose merged cover stays within
 * `maxRanges` ranges (and `maxCells` cells to enumerate); the cover is a superset, so callers still
 * apply the exact bounds.
 */
export function hilbertRanges(lo: number[], hi: number[], order: number, maxRanges = 24, maxCells = 4096): [number, number][] {
  const n = lo.length;
  const qlo = lo.map((u) => quantise(u, order));
  const qhi = hi.map((u) => quantise(u, order));
  let best: [number, number][] = [[0, 2 ** (n * order) - 1]];
  for (let p = 1; p <= order; p++) {
    const shift = order - p;
    const clo = qlo.map((q) => q >> shift);
    const chi = qhi.map((q) => q >> shift);
    if (clo.reduce((m, c, k) => m * (chi[k]! - c + 1), 1) > maxCells) break;
    const cells: number[] = [];
    const point = [...clo];
    for (;;) {
      cells.push(hilbertDistance(point, p));
      let k = n - 1;
      while (k >= 0 && point[k] === chi[k]) point[k] = clo[k--]!;
      if (k < 0) break;
      point[k]!++;
    }
    cells.sort((a, b) => a - b);
    const span = 2 ** (n * shift);
    const ranges: [number, number][] = [];
    for (const c of cells) {
      const last = ranges[ranges.length - 1];
      if (last && last[1] + 1 === c * span) last[1] = (c + 1) * span - 1;
      else ranges.push([c * span, (c + 1) * span - 1]);
    }
    if (ranges.length > maxRanges) break;
    best = ranges;
  }
  return best;
}

export function hilbertRanges3(
  lo: [number, number, number], hi: [number, number, number], order: number, maxRanges = 24, maxCells = 4096,
): [number, number][] {
  return hilbertRanges(lo, hi, order, maxRanges, maxCells);
}
