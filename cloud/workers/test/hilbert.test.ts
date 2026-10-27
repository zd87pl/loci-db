import { describe, expect, it } from "vitest";
import { hilbert3, hilbertDistance } from "../src/hilbert";
import golden from "./hilbert-golden.json";

describe("hilbert parity with the Python hilbertcurve package used by loci", () => {
  it("matches distance_from_point for integer points (3-D and 4-D)", () => {
    for (const c of golden.cases) expect(hilbertDistance(c.point, c.order)).toBe(c.h);
  });
  it("matches LOCI's round-half-even quantisation of unit coordinates", () => {
    for (const c of golden.units) expect(hilbert3(c.u as [number, number, number], 8)).toBe(c.h);
  });
});
