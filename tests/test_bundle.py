import io
import json
import zipfile
from contextlib import nullcontext

import pytest
from conftest import FakeSTT, FakeTTS
from fastapi.testclient import TestClient

import recorder as recorder_mod
from callcenter.api import create_app
from callcenter.deployment import (
    RecordingStore,
    build_manifest,
    public_endpoint,
    save_manifest,
)
from callcenter.records import hash_token
from callcenter.tenants import TenantRegistry

ADMIN = "admin-secret"
ADMIN_H = {"Authorization": f"Bearer {ADMIN}"}


@pytest.fixture
def manifest(tmp_path):
    return build_manifest(
        public_url="https://localhost:8443",
        ca_file=None,
        state_dir=tmp_path / "state",
        faq_dir="data/faq",
        bind="0.0.0.0:8000",
    )


@pytest.fixture
def app_client(embedder, faq, llm, manifest, tmp_path):
    registry = TenantRegistry(embedder, llm)
    registry.register("demo-clinic", "Demo", faq)
    store = RecordingStore(tmp_path / "state")
    app = create_app(
        registry,
        FakeSTT(),
        FakeTTS(),
        deployment=manifest,
        recording_store=store,
        admin_token_sha256=hash_token(ADMIN),
    )
    c = TestClient(app)
    c.store = store
    c.registry = registry
    return c


def _bundle(client):
    r = client.post("/v1/admin/tenants/demo-clinic/recorder-bundle", headers=ADMIN_H)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/zip"
    assert r.headers["cache-control"] == "no-store"
    z = zipfile.ZipFile(io.BytesIO(r.content))
    return {n: z.read(n).decode() for n in z.namelist()}


def _env(files):
    lines = files["recorder/recorder.env"].splitlines()
    return dict(line.split("=", 1) for line in lines if line)


# ------------------------------------------------------------------ manifest


def test_manifest_records_endpoint_and_internals(manifest, tmp_path):
    assert manifest.public.host == "localhost" and manifest.public.port == 8443
    assert manifest.public.resolved_ips  # localhost resolves
    assert manifest.bind == "0.0.0.0:8000" and manifest.working_dir and manifest.process_user
    path = save_manifest(manifest, tmp_path / "state")
    assert json.loads(path.read_text())["public"]["url"] == "https://localhost:8443"


def test_public_url_defaults_port_and_rejects_garbage():
    assert public_endpoint("https://cc.example.invalid").port == 443
    assert public_endpoint(None) is None
    with pytest.raises(ValueError):
        public_endpoint("ftp://cc.example.com")


# ------------------------------------------------------------------ bundle


def test_bundle_contains_preconfigured_recorder(app_client):
    files = _bundle(app_client)
    assert set(files) >= {
        "recorder/recorder.py",
        "recorder/Dockerfile",
        "recorder/recorder.env",
        "recorder/docker-compose.yml",
        "recorder/connection.json",
        "recorder/CONNECTION.md",
    }
    env = _env(files)
    assert env["RECORDER_API_URL"] == "https://localhost:8443"
    assert env["RECORDER_TENANT"] == "demo-clinic"
    conn = json.loads(files["recorder/connection.json"])
    assert conn["port"] == 8443 and conn["resolved_ips"]
    assert "8443/TCP" in files["recorder/CONNECTION.md"]


def test_bundle_leaks_no_internal_details(app_client, manifest):
    files = _bundle(app_client)
    blob = "\n".join(files.values())
    for secret in (manifest.working_dir, manifest.state_dir, manifest.hostname, ADMIN):
        assert secret not in blob
    conn = json.loads(files["recorder/connection.json"])
    for internal in ("process_user", "uid", "gid", "working_dir", "bind", "hostname"):
        assert internal not in conn


def test_bundle_token_works_and_reissue_revokes_old_one(app_client):
    old = _env(_bundle(app_client))["RECORDER_TOKEN"]
    ping = "/v1/tenants/demo-clinic/recorder/ping"
    assert app_client.get(ping, headers={"Authorization": f"Bearer {old}"}).status_code == 200

    new = _env(_bundle(app_client))["RECORDER_TOKEN"]
    assert new != old
    assert app_client.get(ping, headers={"Authorization": f"Bearer {old}"}).status_code == 401
    assert app_client.get(ping, headers={"Authorization": f"Bearer {new}"}).status_code == 200
    # persisted for restarts, as a hash only
    saved = app_client.store.load("demo-clinic")
    assert saved.token_sha256 == hash_token(new)
    assert new not in (app_client.store.dir / "demo-clinic.json").read_text()


def test_admin_endpoints_require_admin_token(app_client):
    url = "/v1/admin/tenants/demo-clinic/recorder-bundle"
    assert app_client.post(url).status_code == 401
    assert app_client.post(url, headers={"Authorization": "Bearer nope"}).status_code == 401
    assert app_client.get("/v1/admin/deployment").status_code == 401
    r = app_client.get("/v1/admin/deployment", headers=ADMIN_H)
    assert r.status_code == 200 and r.json()["public"]["host"] == "localhost"


def test_admin_api_disabled_without_configured_token(embedder, faq, llm):
    registry = TenantRegistry(embedder, llm)
    registry.register("demo-clinic", "Demo", faq)
    c = TestClient(create_app(registry, FakeSTT(), FakeTTS()))
    assert c.get("/v1/admin/deployment", headers=ADMIN_H).status_code == 401


def test_bundle_needs_public_url(embedder, faq, llm, tmp_path):
    registry = TenantRegistry(embedder, llm)
    registry.register("demo-clinic", "Demo", faq)
    m = build_manifest(
        public_url=None, ca_file=None, state_dir=tmp_path, faq_dir="data/faq", bind="x"
    )
    c = TestClient(
        create_app(
            registry,
            FakeSTT(),
            FakeTTS(),
            deployment=m,
            recording_store=RecordingStore(tmp_path),
            admin_token_sha256=hash_token(ADMIN),
        )
    )
    r = c.post("/v1/admin/tenants/demo-clinic/recorder-bundle", headers=ADMIN_H)
    assert r.status_code == 409


def test_private_ca_is_bundled(embedder, faq, llm, tmp_path):
    ca = tmp_path / "ca.pem"
    ca.write_text("-----BEGIN CERTIFICATE-----\nMIIB...\n-----END CERTIFICATE-----\n")
    ep = public_endpoint("https://localhost:8443", str(ca))
    from callcenter.deployment import build_bundle

    z = zipfile.ZipFile(io.BytesIO(build_bundle(ep, "demo-clinic", "tok")))
    assert z.read("recorder/ca.pem").decode() == ca.read_text()
    assert "RECORDER_CA_FILE=/app/ca.pem" in z.read("recorder/recorder.env").decode()


# ------------------------------------------------------------------ recorder --check


def test_recorder_check_reports_each_step(app_client, monkeypatch, capsys):
    token = _env(_bundle(app_client))["RECORDER_TOKEN"]
    app_client.headers["Authorization"] = f"Bearer {token}"
    monkeypatch.setattr(
        recorder_mod.socket, "getaddrinfo", lambda *a, **k: [(0, 0, 0, "", ("127.0.0.1", 80))]
    )
    monkeypatch.setattr(recorder_mod.socket, "create_connection", lambda *a, **k: nullcontext())
    assert recorder_mod.check(app_client, "demo-clinic") is True
    out = capsys.readouterr().out
    assert "[OK] DNS" in out and "[OK] TCP" in out and "[OK] TLS+AUTH" in out

    app_client.headers["Authorization"] = "Bearer wrong"
    assert recorder_mod.check(app_client, "demo-clinic") is False
    assert "[NG] TLS+AUTH" in capsys.readouterr().out


def test_recorder_check_stops_at_unreachable_port(app_client, monkeypatch, capsys):
    monkeypatch.setattr(
        recorder_mod.socket, "getaddrinfo", lambda *a, **k: [(0, 0, 0, "", ("127.0.0.1", 80))]
    )

    def refuse(*a, **k):
        raise ConnectionRefusedError("refused")

    monkeypatch.setattr(recorder_mod.socket, "create_connection", refuse)
    assert recorder_mod.check(app_client, "demo-clinic") is False
    out = capsys.readouterr().out
    assert "[NG] TCP" in out and "TLS+AUTH" not in out
