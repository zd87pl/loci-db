"""Smoke tests for the `loci cloud keys` CLI.

The CLI is a thin wrapper over HTTP; we stub the HTTP layer and verify
argument parsing, auth resolution, and output formatting.
"""

from __future__ import annotations

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
