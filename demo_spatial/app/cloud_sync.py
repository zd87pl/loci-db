"""Mirror the demo host's spatial memory into a LOCI Cloud space (Phase 0, cloud/workers).

The ESP32 host, the handheld runtime and the browser demo all write into one
``SpatialMemory``; this module diffs ``memory.export_records()`` against what it has
already pushed and sends new observations and deletions to
``POST /p/<ref>/v1/spaces/<space>/observations`` in batches of 50. Writes carry the
host record id, so retries and restarts never duplicate data.

Environment (all optional; sync is off unless URL, project and space are set):

    LOCI_CLOUD_URL            e.g. http://127.0.0.1:8787 or https://loci-cloud.<acct>.workers.dev
    LOCI_CLOUD_PROJECT        project ref (12 lowercase alphanumerics)
    LOCI_CLOUD_SPACE          space slug, e.g. home
    LOCI_CLOUD_KEY            secret key (loci_sk_...), or instead:
    LOCI_CLOUD_DEVICE_ID      device id (dev_...) and
    LOCI_CLOUD_DEVICE_SECRET  device secret (loci_ds_...) from POST /devices/claim
    LOCI_CLOUD_SYNC_S         push interval in seconds (default 5)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

MAX_BATCH = 50
MAX_DELETES = 200
_FRAMES = {"room", "camera", "image", "map", "geo"}


class CloudError(RuntimeError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


@dataclass(frozen=True)
class CloudConfig:
    base_url: str
    project: str
    space: str
    secret_key: str | None = None
    device_id: str | None = None
    device_secret: str | None = None
    interval_s: float = 5.0
    timeout_s: float = 10.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> CloudConfig | None:
        e = os.environ if env is None else env
        url = e.get("LOCI_CLOUD_URL")
        project = e.get("LOCI_CLOUD_PROJECT")
        space = e.get("LOCI_CLOUD_SPACE")
        if not (url and project and space):
            return None
        if not url.startswith(("http://", "https://")):
            raise ValueError("LOCI_CLOUD_URL must be an http(s) URL")
        cfg = cls(
            base_url=url.rstrip("/"),
            project=project,
            space=space,
            secret_key=e.get("LOCI_CLOUD_KEY") or None,
            device_id=e.get("LOCI_CLOUD_DEVICE_ID") or None,
            device_secret=e.get("LOCI_CLOUD_DEVICE_SECRET") or None,
            interval_s=float(e.get("LOCI_CLOUD_SYNC_S", "5") or 5),
        )
        if not cfg.secret_key and not (cfg.device_id and cfg.device_secret):
            raise ValueError(
                "LOCI_CLOUD_KEY or LOCI_CLOUD_DEVICE_ID + LOCI_CLOUD_DEVICE_SECRET is required"
            )
        return cfg


class LociCloudClient:
    """Minimal stdlib client for the Phase 0 v1 API."""

    def __init__(self, cfg: CloudConfig, *, clock: Callable[[], float] = time.time) -> None:
        self.cfg = cfg
        self._clock = clock
        self._token: str | None = cfg.secret_key
        self._token_exp: float = float("inf") if cfg.secret_key else 0.0
        self._lock = threading.Lock()

    @property
    def project_url(self) -> str:
        return f"{self.cfg.base_url}/p/{self.cfg.project}/v1"

    def _request(
        self, method: str, path: str, body: Any | None = None, token: str | None = None
    ) -> dict:
        data = None if body is None else json.dumps(body).encode()
        url = f"{self.project_url}/{path}"  # scheme checked in CloudConfig.from_env
        req = urllib.request.Request(url, data=data, method=method)  # noqa: S310
        req.add_header("accept", "application/json")
        if data is not None:
            req.add_header("content-type", "application/json")
        if token:
            req.add_header("authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:  # noqa: S310 - configured URL
                return json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read() or b"{}").get("detail", exc.reason)
            except ValueError:
                detail = str(exc.reason)
            raise CloudError(exc.code, str(detail)) from None

    def _bearer(self) -> str:
        with self._lock:
            if self._token and self._clock() < self._token_exp - 60:
                return self._token
            res = self._request(
                "POST",
                "devices/token",
                {"device_id": self.cfg.device_id, "device_secret": self.cfg.device_secret},
            )
            self._token = str(res["token"])
            self._token_exp = float(res["expires_at"])
            return self._token

    def _authed(self, method: str, path: str, body: Any | None = None) -> dict:
        try:
            return self._request(method, path, body, self._bearer())
        except CloudError as exc:
            if exc.status == 401 and self.cfg.device_id:
                with self._lock:
                    self._token_exp = 0.0  # token superseded or expired: refresh once
                return self._request(method, path, body, self._bearer())
            raise

    def write(self, observations: list[dict], deletes: list[str] | None = None) -> dict:
        body: dict[str, Any] = {"observations": observations}
        if deletes:
            body["deletes"] = deletes
        return self._authed("POST", f"spaces/{self.cfg.space}/observations", body)

    def digest(self, since_rev: int = 0) -> dict:
        return self._authed("GET", f"spaces/{self.cfg.space}/digest?since_rev={int(since_rev)}")

    def ask(self, text: str) -> dict:
        return self._authed("POST", f"spaces/{self.cfg.space}/ask", {"text": text})

    def stats(self) -> dict:
        return self._authed("GET", f"spaces/{self.cfg.space}/stats")


def _triple(v: Any) -> tuple[float, float, float] | None:
    if isinstance(v, (list, tuple)) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v):
        return float(v[0]), float(v[1]), float(v[2])
    return None


def record_to_observation(rec: Mapping[str, Any]) -> dict | None:
    """Map a ``persistence.ObservationRecord`` onto a v1 observation (metres when known)."""
    meta = dict(rec.get("metadata") or {})
    frame = str(rec.get("frame") or "image")
    if frame not in _FRAMES:
        frame = "image"
    pos: tuple[float, float, float] | None
    if frame == "room":
        pos = _triple(meta.get("pos_m"))
    elif frame == "camera":
        pos = _triple(meta.get("xyz_cam_m"))
    else:
        pos = None
    if pos is None:
        # Image-frame (or legacy) records: unit-cube coordinates, stored as the image frame.
        frame = "image"
        pos = (float(rec.get("x", 0.0)), float(rec.get("y", 0.0)), float(rec.get("z", 0.0)))
    label = str(rec.get("label") or "").strip()
    rid = str(rec.get("id") or "")
    if not label or not rid:
        return None
    kept_keys = ("anchored", "anchor_confidence", "merge_count", "source", "scene_id")
    keep = {k: meta[k] for k in kept_keys if k in meta}
    keep.setdefault("scene_id", rec.get("scene_id"))
    obs: dict[str, Any] = {
        "id": f"h_{rid}"[:96],
        "label": label[:64],
        "frame": frame,
        "x": pos[0],
        "y": pos[1],
        "z": pos[2],
        "t": int(rec.get("timestamp_ms") or 0),
        "confidence": max(0.0, min(1.0, float(rec.get("confidence") or 0.0))),
        "metadata": {k: v for k, v in keep.items() if v is not None},
    }
    if rec.get("room"):
        obs["room"] = str(rec["room"])[:64]
    return obs


class _Memory(Protocol):
    def export_records(self) -> list[Any]: ...


class CloudSync:
    """Background thread pushing new records and deletions to one cloud space."""

    def __init__(
        self,
        memory: _Memory,
        client: LociCloudClient,
        *,
        state_path: Path | None = None,
        interval_s: float | None = None,
    ) -> None:
        self.memory = memory
        self.client = client
        self.interval_s = interval_s if interval_s is not None else client.cfg.interval_s
        self.state_path = state_path
        self._pushed: set[str] = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_ok_ms: int | None = None
        self.last_error: str | None = None
        self.last_rev: int | None = None
        self.pushed_total = 0
        self.deleted_total = 0
        self._load_state()

    def _state_key(self) -> str:
        c = self.client.cfg
        return f"{c.base_url}|{c.project}|{c.space}"

    def _load_state(self) -> None:
        if not self.state_path or not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text())
            if data.get("key") == self._state_key():
                self._pushed = set(map(str, data.get("pushed", [])))
        except (OSError, ValueError) as exc:
            logger.warning("cloud sync state unreadable (%s); starting fresh", exc)

    def _save_state(self) -> None:
        if not self.state_path:
            return
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"key": self._state_key(), "pushed": sorted(self._pushed)}))
        os.replace(tmp, self.state_path)

    def sync_once(self) -> dict:
        """One diff-and-push pass. Returns counts; raises CloudError on failure."""
        current: dict[str, dict] = {}
        for rec in self.memory.export_records():
            obs = record_to_observation(rec)
            if obs is not None:
                current[obs["id"]] = obs
        new_ids = [i for i in current if i not in self._pushed]
        gone = sorted(self._pushed - current.keys())
        new_ids.sort(key=lambda i: current[i]["t"])
        sent = deleted = 0
        while new_ids or gone:
            batch = [current[i] for i in new_ids[:MAX_BATCH]]
            dels = gone[:MAX_DELETES]
            res = self.client.write(batch, dels)
            self.last_rev = int(res.get("rev", self.last_rev or 0))
            for o in batch:
                self._pushed.add(o["id"])
            self._pushed.difference_update(dels)
            sent += len(batch)
            deleted += len(dels)
            new_ids = new_ids[MAX_BATCH:]
            gone = gone[MAX_DELETES:]
            self._save_state()
        self.pushed_total += sent
        self.deleted_total += deleted
        self.last_ok_ms = int(time.time() * 1000)
        self.last_error = None
        return {"sent": sent, "deleted": deleted, "rev": self.last_rev}

    def _run(self) -> None:
        backoff = self.interval_s
        while not self._stop.is_set():
            try:
                self.sync_once()
                backoff = self.interval_s
            except Exception as exc:  # noqa: BLE001 - network errors must never kill the host
                self.last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("cloud sync failed: %s", self.last_error)
                backoff = min(max(backoff * 2, self.interval_s), 300.0)
            self._stop.wait(backoff)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="loci-cloud-sync", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)

    def status(self) -> dict:
        c = self.client.cfg
        return {
            "enabled": True,
            "url": c.base_url,
            "project": c.project,
            "space": c.space,
            "auth": "secret_key" if c.secret_key else "device",
            "pushed": len(self._pushed),
            "pushed_total": self.pushed_total,
            "deleted_total": self.deleted_total,
            "last_rev": self.last_rev,
            "last_ok_ms": self.last_ok_ms,
            "last_error": self.last_error,
        }


def start_cloud_sync_from_env(
    memory: _Memory, data_dir: str | os.PathLike[str] | None
) -> CloudSync | None:
    cfg = CloudConfig.from_env()
    if cfg is None:
        return None
    state = Path(data_dir) / "cloud_sync.json" if data_dir else None
    if state:
        state.parent.mkdir(parents=True, exist_ok=True)
    sync = CloudSync(memory, LociCloudClient(cfg), state_path=state)
    sync.start()
    logger.info("cloud sync on: %s project %s space %s", cfg.base_url, cfg.project, cfg.space)
    return sync
