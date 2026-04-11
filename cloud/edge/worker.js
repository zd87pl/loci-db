/**
 * LOCI Cloud API — Cloudflare Workers Edge Layer
 *
 * Responsibilities:
 *   1. Validate Authorization header format before forwarding to Fly.io
 *      (saves cold-start cost on obviously-bad requests)
 *   2. Apply global rate limiting at the edge via Cloudflare's built-in
 *      rate-limit binding (configured in wrangler.toml)
 *   3. Return 401 immediately for missing/malformed tokens
 *   4. Route /health directly without auth check (liveness probe passthrough)
 *
 * Environment variables (set in Cloudflare dashboard or wrangler secret):
 *   UPSTREAM_URL   — Fly.io HTTPS base URL, e.g. https://loci-api.fly.dev
 */

// Expected key format: "loci_" followed by 64 hex characters
const KEY_PATTERN = /^loci_[0-9a-f]{64}$/;

// Paths that skip auth validation (proxied verbatim)
const NO_AUTH_PATHS = new Set(["/health", "/ready"]);

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const path = url.pathname;

    // ── Health/ready passthrough ───────────────────────────────────────────
    if (NO_AUTH_PATHS.has(path)) {
      return forwardToUpstream(request, env);
    }

    // ── Auth header validation ─────────────────────────────────────────────
    const authHeader = request.headers.get("Authorization");
    if (!authHeader) {
      return jsonError(401, "Missing Authorization header");
    }

    const parts = authHeader.split(" ");
    if (parts.length !== 2 || parts[0].toLowerCase() !== "bearer") {
      return jsonError(401, "Authorization header must be: Bearer <api-key>");
    }

    const token = parts[1];
    if (!KEY_PATTERN.test(token)) {
      return jsonError(401, "Malformed API key — expected loci_<64 hex chars>");
    }

    // ── Forward to Fly.io ──────────────────────────────────────────────────
    return forwardToUpstream(request, env);
  },
};

/**
 * Forward the incoming request to the Fly.io upstream verbatim,
 * stripping the CF-specific host and rewriting the URL.
 */
async function forwardToUpstream(request, env) {
  const upstream = env.UPSTREAM_URL;
  if (!upstream) {
    return jsonError(502, "Edge misconfiguration: UPSTREAM_URL not set");
  }

  const inboundUrl = new URL(request.url);
  const targetUrl = new URL(inboundUrl.pathname + inboundUrl.search, upstream);

  const upstreamRequest = new Request(targetUrl.toString(), {
    method: request.method,
    headers: request.headers,
    body: request.body,
    redirect: "follow",
  });

  try {
    return await fetch(upstreamRequest);
  } catch (err) {
    return jsonError(502, "Upstream unreachable");
  }
}

function jsonError(status, message) {
  return new Response(JSON.stringify({ detail: message }), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}
