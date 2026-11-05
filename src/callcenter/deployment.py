"""Deployment self-description and the downloadable recorder bundle.

At startup the API records where and how it is running (`DeploymentManifest`).
That record serves two audiences, and they get different parts of it:

* Operators (admin token): everything, including internal details such as the
  working directory, process user and bind address. Saved to STATE_DIR and
  served at GET /v1/admin/deployment.
* Customers: only what the recorder app needs to reach the API, packaged into a
  zip together with a freshly issued recorder token. Internal paths and users
  are deliberately left out — they help an attacker more than a customer.

The bundle answers "how does the recorder find the call center?": it does not
discover it at runtime. The public URL, the IPs and port to allow in the
customer's firewall, and (for a private CA) the certificate are written into
the bundle when it is generated.
"""

from __future__ import annotations

import getpass
import io
import json
import os
import platform
import socket
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from . import __version__
from .records import RecordingConfig, new_token

# Relative to the working directory (/app in the container image, the repo root
# in development), because the package itself is installed into site-packages.
DEFAULT_RECORDER_SRC = Path("recorder")


@dataclass
class PublicEndpoint:
    """What a customer-side app needs. Safe to hand out."""

    url: str
    host: str
    port: int
    scheme: str
    resolved_ips: list[str]
    ca_pem: str | None = None


@dataclass
class DeploymentManifest:
    version: str
    generated_at: str
    public: PublicEndpoint | None
    # ---- internal, operators only ----
    hostname: str
    working_dir: str
    state_dir: str
    faq_dir: str
    bind: str
    process_user: str
    uid: int | None
    gid: int | None
    python: str
    writable: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def resolve(host: str, port: int) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except OSError:
        return []
    return sorted({i[4][0] for i in infos})


def public_endpoint(url: str | None, ca_file: str | None = None) -> PublicEndpoint | None:
    if not url:
        return None
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(f"PUBLIC_URL must be http(s)://host[:port], got {url!r}")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    ca = Path(ca_file).read_text() if ca_file else None
    return PublicEndpoint(
        url=url.rstrip("/"),
        host=parts.hostname,
        port=port,
        scheme=parts.scheme,
        resolved_ips=resolve(parts.hostname, port),
        ca_pem=ca,
    )


def build_manifest(
    *,
    public_url: str | None,
    ca_file: str | None,
    state_dir: Path,
    faq_dir: str,
    bind: str,
) -> DeploymentManifest:
    def can_write(p: Path) -> bool:
        return p.exists() and os.access(p, os.W_OK)

    try:
        user = getpass.getuser()
    except Exception:
        user = "unknown"
    return DeploymentManifest(
        version=__version__,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        public=public_endpoint(public_url, ca_file),
        hostname=socket.gethostname(),
        working_dir=os.getcwd(),
        state_dir=str(state_dir),
        faq_dir=faq_dir,
        bind=bind,
        process_user=user,
        uid=os.getuid() if hasattr(os, "getuid") else None,
        gid=os.getgid() if hasattr(os, "getgid") else None,
        python=platform.python_version(),
        writable={"state_dir": can_write(state_dir), "faq_dir": can_write(Path(faq_dir))},
    )


def save_manifest(manifest: DeploymentManifest, state_dir: Path) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "deployment.json"
    path.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2))
    return path


# ------------------------------------------------------------- recording tokens


class RecordingStore:
    """Recorder token hashes issued at runtime, persisted under STATE_DIR so a
    rotated token survives restarts even when data/faq is read-only."""

    def __init__(self, state_dir: Path) -> None:
        self.dir = state_dir / "recording"

    def load(self, tenant_id: str) -> RecordingConfig | None:
        path = self.dir / f"{tenant_id}.json"
        if not path.exists():
            return None
        return RecordingConfig.from_dict(json.loads(path.read_text()))

    def save(self, tenant_id: str, cfg: RecordingConfig) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{tenant_id}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"enabled": cfg.enabled, "token_sha256": cfg.token_sha256}))
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)


# ------------------------------------------------------------------ the bundle


def _connection_doc(ep: PublicEndpoint, tenant_id: str, generated_at: str) -> str:
    ips = "\n".join(f"  - {ip}" for ip in ep.resolved_ips) or "  - （DNSで解決できませんでした）"
    ca = (
        "同梱の `ca.pem` で検証します（プライベートCA）"
        if ep.ca_pem
        else "OSの標準の証明書で検証します"
    )
    return f"""# 録音アプリ 接続情報

生成日時: {generated_at}
テナント: {tenant_id}

## 接続先
- URL: {ep.url}
- ホスト名（DNS）: {ep.host}
- ポート: {ep.port}/TCP（{ep.scheme}）
- 証明書: {ca}

## ファイアウォールで許可する通信
録音アプリから **外向き** に、次の宛先だけを許可してください。受け付ける通信（内向き）は不要です。

- 宛先ホスト: {ep.host}
- 宛先ポート: {ep.port}/TCP
- 生成時点のIPアドレス（DNSの変更で変わることがあります。可能ならホスト名で許可してください）:
{ips}

## 必要な権限
- 録音アプリは root 権限を必要としません（コンテナ内はUID 10001で動きます）
- 保存先ディレクトリに、UID 10001 の書き込み権限が必要です
  （例: `sudo mkdir -p /srv/call-records && sudo chown 10001:10001 /srv/call-records`）
- S3互換ストレージを使う場合は、対象バケットへの `PutObject` と `HeadObject`
  だけを許可したキーを使ってください

## セキュリティ
- `recorder.env` のトークンはこの一式にしか含まれていません。第三者に渡さないでください
- 一式を再発行すると古いトークンは無効になります
"""


def build_bundle(
    ep: PublicEndpoint,
    tenant_id: str,
    token: str,
    *,
    recorder_src: Path = DEFAULT_RECORDER_SRC,
    generated_at: str | None = None,
) -> bytes:
    generated_at = generated_at or datetime.now(UTC).isoformat(timespec="seconds")
    env_lines = [
        f"RECORDER_API_URL={ep.url}",
        f"RECORDER_TENANT={tenant_id}",
        f"RECORDER_TOKEN={token}",
        "RECORDER_STORAGE=dir:///data/call-records",
        "RECORDER_INTERVAL=10",
    ]
    if ep.ca_pem:
        env_lines.append("RECORDER_CA_FILE=/app/ca.pem")
    compose = f"""services:
  recorder:
    build: .
    env_file: recorder.env
    restart: unless-stopped
    volumes:
      - /srv/call-records:/data
{"      - ./ca.pem:/app/ca.pem:ro" + chr(10) if ep.ca_pem else ""}"""
    connection = {
        "tenant": tenant_id,
        "generated_at": generated_at,
        "url": ep.url,
        "host": ep.host,
        "port": ep.port,
        "scheme": ep.scheme,
        "resolved_ips": ep.resolved_ips,
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(recorder_src / "recorder.py", "recorder/recorder.py")
        z.write(recorder_src / "Dockerfile", "recorder/Dockerfile")
        z.writestr("recorder/recorder.env", "\n".join(env_lines) + "\n")
        z.writestr("recorder/docker-compose.yml", compose)
        z.writestr("recorder/connection.json", json.dumps(connection, indent=2))
        z.writestr("recorder/CONNECTION.md", _connection_doc(ep, tenant_id, generated_at))
        if ep.ca_pem:
            z.writestr("recorder/ca.pem", ep.ca_pem)
    return buf.getvalue()


def issue_bundle(
    ep: PublicEndpoint, tenant_id: str, store: RecordingStore, **kwargs
) -> tuple[bytes, RecordingConfig]:
    """Rotate the tenant's recorder token and return (zip, new config)."""
    token, digest = new_token()
    cfg = RecordingConfig(enabled=True, token_sha256=digest)
    store.save(tenant_id, cfg)
    return build_bundle(ep, tenant_id, token, **kwargs), cfg
