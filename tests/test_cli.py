"""Smoke tests for the `loci` CLI.

`loci cloud keys` is a thin wrapper over HTTP; we stub the HTTP layer and
verify argument parsing, auth resolution, and output formatting.
`loci migrate-layout` is exercised against an in-memory fake Qdrant client
that records every write.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from loci import cli as loci_cli


def test_create_calls_admin_keys_with_body(monkeypatch, capsys):
    monkeypatch.setenv("LOCI_API_KEY", "loci_admin_xxx")
    monkeypatch.setenv("LOCI_BASE_URL", "https://api.example.com")

    captured: dict = {}

    def fake_http(method, base_url, path, api_key, body=None):
        captured.update(method=method, base_url=base_url, path=path, api_key=api_key, body=body)
        return {
            "key_id": "kid-1",
            "raw_key": "loci_" + "a" * 64,
            "prefix": "loci_aaaaaaa",
            "tenant_id": "tid-1",
            "namespace": "myapp_prod",
            "is_admin": False,
        }

    monkeypatch.setattr(loci_cli, "_http", fake_http)

    loci_cli.main(
        [
            "cloud",
            "keys",
            "create",
            "--email",
            "a@b.c",
            "--namespace",
            "myapp_prod",
            "--label",
            "prod",
        ]
    )

    assert captured["method"] == "POST"
    assert captured["path"] == "/admin/keys"
    assert captured["base_url"] == "https://api.example.com"
    assert captured["api_key"] == "loci_admin_xxx"
    assert captured["body"]["tenant_email"] == "a@b.c"
    assert captured["body"]["namespace"] == "myapp_prod"
    assert captured["body"]["label"] == "prod"
    assert captured["body"]["is_admin"] is False

    out = capsys.readouterr().out
    assert "RAW KEY" in out
    assert "loci_" in out


def test_list_parses_filters(monkeypatch, capsys):
    monkeypatch.setenv("LOCI_API_KEY", "loci_admin_xxx")

    captured = {}

    def fake_http(method, base_url, path, api_key, body=None):
        captured.update(method=method, path=path)
        return {"keys": []}

    monkeypatch.setattr(loci_cli, "_http", fake_http)

    loci_cli.main(
        [
            "cloud",
            "keys",
            "list",
            "--tenant-id",
            "aaaa-bbbb",
            "--include-revoked",
        ]
    )
    assert captured["method"] == "GET"
    assert "tenant_id=aaaa-bbbb" in captured["path"]
    assert "include_revoked=true" in captured["path"]


def test_revoke_calls_delete(monkeypatch, capsys):
    monkeypatch.setenv("LOCI_API_KEY", "loci_admin_xxx")

    captured = {}

    def fake_http(method, base_url, path, api_key, body=None):
        captured.update(method=method, path=path)
        return {"key_id": "kid-42", "revoked": True}

    monkeypatch.setattr(loci_cli, "_http", fake_http)

    loci_cli.main(["cloud", "keys", "revoke", "kid-42"])
    assert captured["method"] == "DELETE"
    assert captured["path"] == "/admin/keys/kid-42"

    out = capsys.readouterr().out
    assert "revoked: kid-42" in out


def test_missing_api_key_exits(monkeypatch):
    monkeypatch.delenv("LOCI_API_KEY", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        loci_cli.main(["cloud", "keys", "list"])
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# `loci migrate-layout`
# ---------------------------------------------------------------------------


class FakeQdrant:
    """In-memory stand-in for qdrant_client.QdrantClient.

    Collections map name -> {"vectors": <config>, "points": {id: (vector,
    payload)}}.  Every write is recorded so tests can assert on (the absence
    of) side effects.
    """

    def __init__(self, collections: dict | None = None) -> None:
        self.collections = collections or {}
        self.created: list[str] = []
        self.index_calls: list[tuple[str, str]] = []
        self.upsert_calls: list[tuple[str, list]] = []
        self.deleted: list[str] = []

    def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in self.collections])

    def collection_exists(self, name):
        return name in self.collections

    def get_collection(self, collection_name):
        vectors = self.collections[collection_name]["vectors"]
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=vectors)))

    def create_collection(self, collection_name, vectors_config):
        self.created.append(collection_name)
        self.collections[collection_name] = {"vectors": vectors_config, "points": {}}

    def create_payload_index(self, collection_name, field_name, field_schema):
        self.index_calls.append((collection_name, field_name))

    def count(self, collection_name, exact=True):
        return SimpleNamespace(count=len(self.collections[collection_name]["points"]))

    def scroll(self, collection_name, limit, offset=None, with_payload=True, with_vectors=True):
        items = list(self.collections[collection_name]["points"].items())
        start = offset or 0
        chunk = items[start : start + limit]
        points = [
            SimpleNamespace(id=pid, vector=vec, payload=payload) for pid, (vec, payload) in chunk
        ]
        next_offset = start + limit if start + limit < len(items) else None
        return points, next_offset

    def retrieve(self, collection_name, ids, with_payload=False, with_vectors=False):
        store = self.collections.get(collection_name, {"points": {}})["points"]
        return [SimpleNamespace(id=pid) for pid in ids if pid in store]

    def upsert(self, collection_name, points, wait=True):
        self.upsert_calls.append((collection_name, [p.id for p in points]))
        store = self.collections[collection_name]["points"]
        for p in points:
            store[p.id] = (p.vector, p.payload)

    def delete_collection(self, collection_name):
        self.deleted.append(collection_name)
        del self.collections[collection_name]


class LossyQdrant(FakeQdrant):
    """Fake that silently drops the last point of every upsert batch."""

    def upsert(self, collection_name, points, wait=True):
        super().upsert(collection_name, points[:-1], wait=wait)


_VECTORS = SimpleNamespace(size=2, distance="Cosine")


def _legacy_points(epoch: int, n: int) -> dict:
    return {
        f"pt-{epoch}-{i}": (
            [float(i), float(epoch)],
            {"timestamp_ms": epoch * 5000 + i, "scene_id": "s1", "hilbert_r4": i},
        )
        for i in range(n)
    }


def _legacy_store() -> FakeQdrant:
    return FakeQdrant(
        {
            "loci_3": {"vectors": _VECTORS, "points": _legacy_points(3, 3)},
            "loci_7": {"vectors": _VECTORS, "points": _legacy_points(7, 5)},
            "loci_sum_0": {"vectors": _VECTORS, "points": _legacy_points(0, 2)},
            # Another tenant — must never be touched without its prefix.
            "t2_loci_9": {"vectors": _VECTORS, "points": _legacy_points(9, 1)},
        }
    )


def _run_migrate(monkeypatch, fake, *extra_args):
    monkeypatch.setattr(loci_cli, "_qdrant_client", lambda url: fake)
    loci_cli.main(["migrate-layout", "--qdrant-url", "http://localhost:6333", *extra_args])


def test_migrate_legacy_discovery_regex_and_prefix_isolation():
    names = [
        "loci_3",
        "loci_42",
        "loci_sum_1",
        "t2_loci_9",
        "t2_loci_sum_2",
        "loci_data",  # new layout — never a migration source
        "loci_summary",
        "loci_abc",  # non-numeric suffix — not legacy
        "other_loci_5",  # different tenant prefix
    ]
    raw, summaries = loci_cli._legacy_collections(names, "")
    assert raw == [("loci_3", 3), ("loci_42", 42)]
    assert summaries == [("loci_sum_1", 1)]

    raw, summaries = loci_cli._legacy_collections(names, "t2_")
    assert raw == [("t2_loci_9", 9)]
    assert summaries == [("t2_loci_sum_2", 2)]

    # Regex metacharacters in the prefix are escaped, not interpreted.
    raw, summaries = loci_cli._legacy_collections(["aXb_loci_1", "a.b_loci_2"], "a.b_")
    assert raw == [("a.b_loci_2", 2)]
    assert summaries == []


def test_migrate_dry_run_makes_no_writes(monkeypatch, capsys):
    fake = _legacy_store()
    _run_migrate(monkeypatch, fake, "--dry-run")

    assert fake.created == []
    assert fake.upsert_calls == []
    assert fake.deleted == []
    assert fake.index_calls == []

    out = capsys.readouterr().out
    assert "loci_3  ->  loci_data  (3 points)" in out
    assert "loci_sum_0  ->  loci_summary  (2 points)" in out
    assert "t2_loci_9" not in out  # other tenant excluded
    assert "Dry run — no changes made." in out


def test_migrate_copies_points_verbatim(monkeypatch, capsys):
    fake = _legacy_store()
    expected_data = {**_legacy_points(3, 3), **_legacy_points(7, 5)}
    expected_summary = dict(_legacy_points(0, 2))

    # batch-size 2 forces multi-page scrolls and multi-batch upserts.
    _run_migrate(monkeypatch, fake, "--batch-size", "2")

    data_points = {
        pid: (vec, payload)
        for pid, (vec, payload) in fake.collections["loci_data"]["points"].items()
    }
    assert data_points == expected_data  # same ids, vectors, payloads
    assert fake.collections["loci_summary"]["points"] == expected_summary

    # Data collection gets the full index set; summary skips Hilbert fields.
    data_fields = {f for col, f in fake.index_calls if col == "loci_data"}
    summary_fields = {f for col, f in fake.index_calls if col == "loci_summary"}
    assert data_fields == {
        "hilbert_r4",
        "hilbert_r8",
        "hilbert_r12",
        "timestamp_ms",
        "scene_id",
        "scale_level",
    }
    assert summary_fields == {"timestamp_ms", "scene_id", "scale_level"}

    # Without --delete-old the legacy collections survive, other tenants too.
    assert fake.deleted == []
    assert "loci_3" in fake.collections
    assert "t2_loci_9" in fake.collections
    assert fake.collections["t2_loci_9"]["points"] == _legacy_points(9, 1)

    out = capsys.readouterr().out
    assert "verified 3/3" in out
    assert "Done: 10 points copied" in out


def test_migrate_delete_old_drops_only_verified_legacy(monkeypatch):
    fake = _legacy_store()
    _run_migrate(monkeypatch, fake, "--delete-old")

    assert sorted(fake.deleted) == ["loci_3", "loci_7", "loci_sum_0"]
    assert "t2_loci_9" in fake.collections  # prefix isolation on delete too
    assert len(fake.collections["loci_data"]["points"]) == 8
    assert len(fake.collections["loci_summary"]["points"]) == 2


def test_migrate_verify_failure_blocks_delete_and_exits_nonzero(monkeypatch, capsys):
    fake = LossyQdrant(_legacy_store().collections)
    monkeypatch.setattr(loci_cli, "_qdrant_client", lambda url: fake)

    with pytest.raises(SystemExit) as excinfo:
        loci_cli.main(
            [
                "migrate-layout",
                "--qdrant-url",
                "http://localhost:6333",
                "--delete-old",
            ]
        )
    assert excinfo.value.code == 1
    assert fake.deleted == []  # nothing dropped without a verified copy
    assert "verification failed" in capsys.readouterr().err


def test_migrate_rerun_is_idempotent(monkeypatch, capsys):
    fake = _legacy_store()
    _run_migrate(monkeypatch, fake, "--batch-size", "2")
    first_upserts = len(fake.upsert_calls)
    first_data = dict(fake.collections["loci_data"]["points"])

    _run_migrate(monkeypatch, fake, "--batch-size", "2")

    # Second run skips every already-present point: no new writes, no dupes.
    assert len(fake.upsert_calls) == first_upserts
    assert fake.collections["loci_data"]["points"] == first_data
    out = capsys.readouterr().out
    assert "copied 0, skipped 3" in out
    assert "Done: 0 points copied, 10 already present" in out


def test_migrate_nothing_to_do(monkeypatch, capsys):
    fake = FakeQdrant({"loci_data": {"vectors": _VECTORS, "points": {}}})
    _run_migrate(monkeypatch, fake)
    assert "nothing to migrate" in capsys.readouterr().out
    assert fake.upsert_calls == []
