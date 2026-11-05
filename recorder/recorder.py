"""Call-record recorder — runs in the CUSTOMER's environment.

It connects out to the call center API, pulls finished call records, writes them
to storage the customer controls, and acknowledges each one. The call center
deletes a record from memory only after this acknowledgement, and it never
receives credentials for the customer's storage.

    RECORDER_API_URL   https://callcenter.example.com
    RECORDER_TENANT    demo-clinic
    RECORDER_TOKEN     token issued with `python -m callcenter.records`
    RECORDER_STORAGE   dir:///data/call-records   or   s3://bucket/prefix
    RECORDER_INTERVAL  seconds between polls (default 10)
    S3 only:           AWS_* variables / RECORDER_S3_ENDPOINT for MinIO etc.

    python recorder.py            # poll forever
    python recorder.py --once     # one pull, e.g. from cron
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

import httpx

log = logging.getLogger("recorder")


class Storage(Protocol):
    def save(self, record: dict) -> bool:
        """Persist a record. Idempotent: return False if it was already stored."""
        ...


def _relpath(record: dict) -> str:
    day = (record.get("ended_at") or "unknown")[:10]
    return f"{day}/{record['record_id']}.json"


def _body(record: dict) -> bytes:
    clean = {k: v for k, v in record.items() if k != "delivery_attempt"}
    return json.dumps(clean, ensure_ascii=False, indent=2).encode()


@dataclass
class DirectoryStorage:
    """Local disk, NAS mount or container volume. Writes are atomic."""

    root: Path

    def save(self, record: dict) -> bool:
        path = self.root / _relpath(record)
        if path.exists():
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(_body(record))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)  # readers never see a half-written file
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return True


@dataclass
class S3Storage:
    """Any S3-compatible object store (AWS S3, MinIO, Cloudflare R2, ...)."""

    bucket: str
    prefix: str = ""
    endpoint_url: str | None = None

    def __post_init__(self) -> None:
        import boto3  # optional dependency, only on the customer's side

        self._s3 = boto3.client("s3", endpoint_url=self.endpoint_url)

    def save(self, record: dict) -> bool:
        key = f"{self.prefix.strip('/')}/{_relpath(record)}".lstrip("/")
        try:
            self._s3.head_object(Bucket=self.bucket, Key=key)
            return False
        except self._s3.exceptions.ClientError:
            pass
        self._s3.put_object(
            Bucket=self.bucket, Key=key, Body=_body(record), ContentType="application/json"
        )
        return True


def storage_from_url(url: str) -> Storage:
    parts = urlsplit(url)
    if parts.scheme == "dir":
        return DirectoryStorage(Path(parts.path))
    if parts.scheme == "s3":
        return S3Storage(parts.netloc, parts.path, os.environ.get("RECORDER_S3_ENDPOINT"))
    raise ValueError(f"unsupported RECORDER_STORAGE: {url!r} (use dir:// or s3://)")


@dataclass
class Recorder:
    client: httpx.Client
    tenant: str
    storage: Storage
    batch: int = 20

    def run_once(self) -> int:
        """Pull one batch, store and acknowledge it. Returns records acknowledged."""
        r = self.client.get(f"/v1/tenants/{self.tenant}/records", params={"limit": self.batch})
        r.raise_for_status()
        done = 0
        for record in r.json()["records"]:
            rid = record["record_id"]
            try:
                stored = self.storage.save(record)
            except Exception:
                # Not acknowledged -> the API offers it again after the lease.
                log.exception("could not store record %s; will retry", rid)
                continue
            ack = self.client.post(f"/v1/tenants/{self.tenant}/records/{rid}/ack")
            if ack.status_code in (204, 404):  # 404: already acknowledged earlier
                done += 1
                log.info("record %s %s", rid, "stored" if stored else "already stored")
            else:
                log.warning("ack for %s failed with %s; will retry", rid, ack.status_code)
        return done


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--once", action="store_true", help="pull a single batch and exit")
    args = ap.parse_args()

    env = os.environ
    client = httpx.Client(
        base_url=env["RECORDER_API_URL"],
        headers={"Authorization": f"Bearer {env['RECORDER_TOKEN']}"},
        timeout=30,
    )
    rec = Recorder(client, env["RECORDER_TENANT"], storage_from_url(env["RECORDER_STORAGE"]))
    interval = float(env.get("RECORDER_INTERVAL", "10"))
    while True:
        try:
            while rec.run_once() == rec.batch:  # drain backlog before sleeping
                pass
        except httpx.HTTPError as e:
            log.warning("pull failed: %s", e)
        if args.once:
            return
        time.sleep(interval)


if __name__ == "__main__":
    main()
