import { describe, expect, it } from "vitest";
import { hilbert3, hilbertDistance, hilbertRanges3, quantise } from "../src/hilbert";
import golden from "./hilbert-golden.json";

describe("hilbert parity with the Python hilbertcurve package used by loci", () => {
  it("matches distance_from_point for integer points (3-D and 4-D)", () => {
    for (const c of golden.cases) expect(hilbertDistance(c.point, c.order)).toBe(c.h);
  });
  it("matches LOCI's round-half-even quantisation of unit coordinates", () => {
    for (const c of golden.units) expect(hilbert3(c.u as [number, number, number], 8)).toBe(c.h);
  });
});

describe("hilbert range cover", () => {
  let seed = 42;
  const rand = () => ((seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31);
  const unit3 = () => [rand(), rand(), rand()] as [number, number, number];

  it("relies on coarse indices being prefixes of fine ones", () => {
    for (let i = 0; i < 500; i++) {
      const q = unit3().map((u) => quantise(u, 8));
      const h8 = hilbertDistance(q, 8);
      for (let p = 1; p <= 8; p++) expect(Math.floor(h8 / 8 ** (8 - p))).toBe(hilbertDistance(q.map((v) => v >> (8 - p)), p));
    }
  });

  it("covers every point in the box with few sorted, disjoint ranges", () => {
    for (let k = 0; k < 200; k++) {
      const a = unit3();
      const b = unit3();
      const lo = a.map((v, i) => Math.min(v, b[i]!)) as [number, number, number];
      const hi = a.map((v, i) => Math.max(v, b[i]!)) as [number, number, number];
      const ranges = hilbertRanges3(lo, hi, 8);
      expect(ranges.length).toBeLessThanOrEqual(24);
      ranges.forEach(([s, e], i) => {
        expect(s).toBeLessThanOrEqual(e);
        if (i > 0) expect(s).toBeGreaterThan(ranges[i - 1]![1] + 1);
      });
      for (let j = 0; j < 50; j++) {
        const u = lo.map((l, i) => l + rand() * (hi[i]! - l)) as [number, number, number];
        const h = hilbert3(u, 8);
        expect(ranges.some(([s, e]) => h >= s && h <= e)).toBe(true);
      }
      for (const corner of [lo, hi]) {
        const h = hilbert3(corner, 8);
        expect(ranges.some(([s, e]) => h >= s && h <= e)).toBe(true);
      }
    }
  });

  it("narrows a small box to a small part of the curve", () => {
    const ranges = hilbertRanges3([0.4, 0.4, 0.4], [0.45, 0.45, 0.45], 8);
    const covered = ranges.reduce((n, [s, e]) => n + e - s + 1, 0);
    expect(covered / 2 ** 24).toBeLessThan(0.01);
  });
});
