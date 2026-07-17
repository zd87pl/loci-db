"""HTTP transport for the LOCI Cloud API.

When ``LociClient`` or ``AsyncLociClient`` is constructed with both ``base_url``
and ``api_key``, the client routes ``insert`` and ``query`` calls through the
managed HTTP API instead of talking to a local Qdrant directly. Everything
except the routing target (auth header, payload shape) is identical to the
local path from the caller's point of view.

Only ``insert`` and ``query`` are supported in cloud mode today (plus the
legacy ``predict_and_retrieve`` path, which is a plain query under the hood).
Other methods (``insert_batch``, ``query_scored``, ``get_trajectory``,
``get_causal_context``, ``funnel_query``, and the extended
``predict_and_retrieve`` path) raise :class:`CloudModeUnsupportedError` until
the cloud API exposes matching endpoints.
"""

from __future__ import annotations

from typing import Any

from loci.schema import WorldState


class CloudModeUnsupportedError(NotImplementedError):
    """Raised when a local-only client method is called in cloud mode."""


class _CloudError(RuntimeError):
    """Raised when the cloud API returns a non-2xx response."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"cloud API error {status_code}: {detail}")


def _insert_payload(state: WorldState) -> dict[str, Any]:
    payload = {
        "x": state.x,
        "y": state.y,
        "z": state.z,
        "timestamp_ms": state.timestamp_ms,
        "vector": state.vector,
        "scene_id": state.scene_id,
        "scale_level": state.scale_level,
        "confidence": state.confidence,
    }
    # Only send metadata when present: newer servers store it, older servers
    # ignore unknown fields, and omitting it keeps the wire format minimal.
    if state.metadata:
        payload["metadata"] = state.metadata
    return payload


def _query_payload(
    vector: list[float],
    spatial_bounds: dict | None,
    time_window_ms: tuple[int, int] | None,
    limit: int,
    overlap_factor: float,
    include_vectors: bool = True,
) -> dict[str, Any]:
    body: dict[str, Any] = {"vector": vector, "limit": limit, "overlap_factor": overlap_factor}
    if not include_vectors:
        body["include_vectors"] = False
    if spatial_bounds is not None:
        body.update(
            {
                "x_min": spatial_bounds.get("x_min", 0.0),
                "x_max": spatial_bounds.get("x_max", 1.0),
                "y_min": spatial_bounds.get("y_min", 0.0),
                "y_max": spatial_bounds.get("y_max", 1.0),
                "z_min": spatial_bounds.get("z_min", 0.0),
                "z_max": spatial_bounds.get("z_max", 1.0),
            }
        )
    if time_window_ms is not None:
        body["time_start_ms"] = time_window_ms[0]
        body["time_end_ms"] = time_window_ms[1]
    return body


def _parse_query_results(payload: dict[str, Any]) -> list[WorldState]:
    """Map cloud query result items to WorldStates.

    ``scale_level``, ``confidence``, and ``metadata`` are parsed tolerantly:
    older servers omit them, in which case the WorldState defaults apply.
    """
    results = []
    for r in payload.get("results", []):
        results.append(
            WorldState(
                x=r["x"],
                y=r["y"],
                z=r["z"],
                timestamp_ms=r["timestamp_ms"],
                vector=r.get("vector", []),
                scene_id=r.get("scene_id", ""),
                scale_level=r.get("scale_level", "patch"),
                confidence=r.get("confidence", 1.0),
                metadata=r.get("metadata") or {},
                id=str(r["id"]),
            )
        )
    return results


class CloudTransport:
    """Synchronous HTTP transport — uses :mod:`urllib.request` to avoid new deps."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    def _request(self, method: str, path: str, body: dict | None = None) -> dict[str, Any]:
        import json
        import urllib.error
        import urllib.request

        url = f"{self._base_url}{path}"
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"base_url must be http(s): {self._base_url!r}")
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)  # noqa: S310
        req.add_header("Authorization", f"Bearer {self._api_key}")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # nosec B310 — scheme validated above  # noqa: S310
                parsed: dict[str, Any] = json.loads(resp.read().decode() or "{}")
                return parsed
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace") if exc.fp else str(exc)
            raise _CloudError(exc.code, detail) from exc

    def insert(self, state: WorldState) -> str:
        resp = self._request("POST", "/insert", _insert_payload(state))
        return str(resp["id"])

    def query(
        self,
        vector: list[float],
        spatial_bounds: dict | None = None,
        time_window_ms: tuple[int, int] | None = None,
        limit: int = 10,
        overlap_factor: float = 1.2,
        include_vectors: bool = True,
    ) -> list[WorldState]:
        body = _query_payload(
            vector, spatial_bounds, time_window_ms, limit, overlap_factor, include_vectors
        )
        resp = self._request("POST", "/query", body)
        return _parse_query_results(resp)


class AsyncCloudTransport:
    """Async HTTP transport — uses :mod:`httpx` (already a dev dep; runtime optional)."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        # httpx.AsyncClient, but typed Any so httpx isn't required at import time
        self._client: Any = None

    async def _get_client(self):
        if self._client is None:
            try:
                import httpx
            except ImportError as exc:
                raise ImportError(
                    "AsyncLociClient cloud mode requires httpx. Install with: pip install httpx"
                ) from exc
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _request(self, method: str, path: str, body: dict | None = None) -> dict[str, Any]:
        client = await self._get_client()
        resp = await client.request(method, path, json=body)
        if resp.status_code >= 400:
            raise _CloudError(resp.status_code, resp.text)
        return resp.json() if resp.content else {}

    async def insert(self, state: WorldState) -> str:
        resp = await self._request("POST", "/insert", _insert_payload(state))
        return str(resp["id"])

    async def query(
        self,
        vector: list[float],
        spatial_bounds: dict | None = None,
        time_window_ms: tuple[int, int] | None = None,
        limit: int = 10,
        overlap_factor: float = 1.2,
        include_vectors: bool = True,
    ) -> list[WorldState]:
        body = _query_payload(
            vector, spatial_bounds, time_window_ms, limit, overlap_factor, include_vectors
        )
        resp = await self._request("POST", "/query", body)
        return _parse_query_results(resp)
