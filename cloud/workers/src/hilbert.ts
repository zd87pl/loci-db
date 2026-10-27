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
