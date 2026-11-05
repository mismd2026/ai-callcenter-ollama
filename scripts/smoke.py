"""End-to-end smoke test against a running call center (real Ollama / VOICEVOX / Whisper).

Walks one call through every feature and prints PASS / WARN / FAIL per step:

    L1 FAQ -> follow-up question -> paraphrase (L2) -> hard-rule handoff ->
    operator brief -> voice round trip -> call end -> metrics ->
    recorder bundle -> recorder --check -> records stored and deleted

WARN marks steps whose outcome depends on model quality (e.g. whether the LLM
resolves a follow-up), so a small CPU model can WARN without the plumbing being
broken. FAIL means something is actually wrong. Exit code 1 on any FAIL.

    python scripts/smoke.py --url http://localhost:8000 [--admin-token TOKEN]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.parse import unquote

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "recorder"))

LONG_NUMBER = re.compile(r"\d(?:[\s-]?\d){7,}")
results: list[tuple[str, str, str]] = []


def report(status: str, step: str, detail: str = "") -> None:
    results.append((status, step, detail))
    color = {"PASS": "32", "WARN": "33", "FAIL": "31", "SKIP": "90"}[status]
    print(f"\033[{color}m[{status}]\033[0m {step}" + (f"  — {detail}" if detail else ""))


def turn(c: httpx.Client, base: str, cid: str, text: str, audio: bool = False) -> dict:
    r = c.post(f"{base}/calls/{cid}/text", json={"text": text, "include_audio": audio})
    r.raise_for_status()
    return r.json()


def main() -> int:
    try:
        return run()
    except httpx.TransportError as e:
        report("FAIL", "API connection lost", f"{type(e).__name__}: {e} (API process crashed?)")
        print_summary([])
        return 1


def print_summary(latencies: list[dict]) -> int:
    stages: dict[str, list[float]] = {}
    for lat in latencies:
        for k, v in lat.items():
            stages.setdefault(k, []).append(v)
    if stages:
        avg = ", ".join(f"{k} {sum(v) / len(v):.0f}ms" for k, v in stages.items())
        print(f"\nlatency (text turns, avg): {avg}")
    counts = {s: sum(1 for r in results if r[0] == s) for s in ("PASS", "WARN", "FAIL", "SKIP")}
    print("summary: " + "  ".join(f"{k} {v}" for k, v in counts.items()))
    return 1 if counts["FAIL"] else 0


def run() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--tenant", default="demo-clinic")
    ap.add_argument("--admin-token", help="enables the recorder bundle / records steps")
    ap.add_argument("--timeout", type=float, default=180, help="seconds per request")
    args = ap.parse_args()

    c = httpx.Client(base_url=args.url, timeout=args.timeout)
    base = f"/v1/tenants/{args.tenant}"
    latencies: list[dict] = []

    # 0. health ------------------------------------------------------------------
    try:
        c.get("/healthz").raise_for_status()
        report("PASS", "API is up", args.url)
    except httpx.HTTPError as e:
        report("FAIL", "API is up", str(e))
        return 1

    cid = c.post(f"{base}/calls").json()["call_id"]
    report("PASS", "start call", cid)

    # 1. Layer 1 ------------------------------------------------------------------
    b = turn(c, base, cid, "駐車場はありますか")
    latencies.append(b["latency_ms"])
    if b["layer"] == 1 and "access-01" in b["sources"]:
        report("PASS", "L1 FAQ read verbatim", b["reply"][:40])
    else:
        report("FAIL", "L1 FAQ read verbatim", f"layer={b['layer']} reason={b['reason']}")

    # 2. follow-up (#1) -------------------------------------------------------------
    b = turn(c, base, cid, "それは無料ですか")
    latencies.append(b["latency_ms"])
    ok = b["layer"] == 2 and "access-01" in b["sources"]
    report(
        "PASS" if ok else "WARN",
        "follow-up resolved from context (#1)",
        f"layer={b['layer']} reply={b['reply'][:50]!r}",
    )

    # 3. paraphrase -> Layer 2 ------------------------------------------------------
    b = turn(c, base, cid, "クレカで支払いできますか")
    latencies.append(b["latency_ms"])
    report(
        "PASS" if b["layer"] in (1, 2) else "WARN",
        "paraphrased question answered from FAQ",
        f"layer={b['layer']} score={b['reason']}",
    )

    # 4. hard rule -> Layer 3 + handoff (#2) ----------------------------------------
    b = turn(c, base, cid, "解約したいです。カード番号は4111 1111 1111 1111です")
    if b["layer"] == 3 and b.get("handoff_url"):
        report("PASS", "hard rule hands off to a human", b["reason"])
    else:
        report("FAIL", "hard rule hands off to a human", f"layer={b['layer']}")

    brief: dict = {}
    if b.get("handoff_url"):
        for _ in range(60):
            brief = c.get(b["handoff_url"]).json()
            if brief.get("status") == "ready":
                break
            time.sleep(1)
        if brief.get("status") != "ready":
            report("FAIL", "operator brief generated (#2)", "still pending after 60s")
        else:
            report(
                "PASS" if brief["generated_by"] == "llm" else "WARN",
                "operator brief generated (#2)",
                f"{brief['generated_by']}: {brief['summary'][:60]!r}",
            )
            leaked = LONG_NUMBER.search(json.dumps(brief, ensure_ascii=False))
            report(
                "FAIL" if leaked else "PASS", "card number masked in brief", leaked and leaked[0]
            )

    # 5. voice round trip -----------------------------------------------------------
    # Synthesize a question through the API itself, then send that WAV back as audio.
    vid = c.post(f"{base}/calls").json()["call_id"]
    b = turn(c, base, vid, "診療時間を教えてください", audio=True)
    wav = base64.b64decode(b["audio_wav_base64"] or "")
    if wav[:4] != b"RIFF":
        report("FAIL", "TTS produced WAV (VOICEVOX)", f"{len(wav)} bytes")
    else:
        report("PASS", "TTS produced WAV (VOICEVOX)", f"{len(wav) // 1024} KiB")
        r = c.post(f"{base}/calls/{vid}/audio", files={"file": ("q.wav", wav, "audio/wav")})
        if r.status_code != 200:
            report(
                "FAIL",
                "STT transcribed synthesized speech (Whisper)",
                f"HTTP {r.status_code}: {r.text[:200]}",
            )
        else:
            transcript = unquote(r.headers.get("X-Transcript", ""))
            heard = "診療" in transcript or "時間" in transcript
            report(
                "PASS" if heard else "WARN",
                "STT transcribed synthesized speech (Whisper)",
                f"{transcript[:40]!r} layer={r.headers.get('X-Escalation-Layer')} "
                f"total={r.headers.get('X-Latency-Ms')}ms",
            )
    c.delete(f"{base}/calls/{vid}")

    # 6. call end (#3) --------------------------------------------------------------
    end = c.delete(f"{base}/calls/{cid}")
    gone = c.post(f"{base}/calls/{cid}/text", json={"text": "診療時間"}).status_code == 404
    ok = end.status_code == 200 and gone
    report("PASS" if ok else "FAIL", "call end discards conversation (#3)", end.text[:80])
    still = c.get(f"{base}/calls/{cid}/handoff").status_code == 200
    report("PASS" if still else "FAIL", "brief kept for operator after call end (#3)")
    ack = c.delete(f"{base}/calls/{cid}/handoff").status_code
    report("PASS" if ack == 204 else "FAIL", "operator acknowledges brief", f"HTTP {ack}")

    metrics = c.get("/metrics").text
    report(
        "PASS" if 'why="ended"' in metrics else "FAIL",
        "metrics expose call closures",
        next((m for m in metrics.splitlines() if 'why="ended"' in m), ""),
    )

    # 7. recorder bundle / records (#4, #5) -----------------------------------------
    if not args.admin_token:
        report("SKIP", "recorder bundle & records (#4/#5)", "pass --admin-token to run")
    else:
        run_records(c, base, args)

    return print_summary(latencies)


def run_records(c: httpx.Client, base: str, args) -> None:
    from recorder import DirectoryStorage, Recorder, check

    admin = {"Authorization": f"Bearer {args.admin_token}"}
    r = c.post(f"/v1/admin/tenants/{args.tenant}/recorder-bundle", headers=admin)
    if r.status_code != 200:
        report("FAIL", "issue recorder bundle (#5)", f"HTTP {r.status_code} {r.text[:80]}")
        return
    z = zipfile.ZipFile(io.BytesIO(r.content))
    env = dict(
        line.split("=", 1) for line in z.read("recorder/recorder.env").decode().splitlines() if line
    )
    report(
        "PASS",
        "issue recorder bundle (#5)",
        f"{len(z.namelist())} files, url={env['RECORDER_API_URL']}",
    )

    dep = c.get("/v1/admin/deployment", headers=admin).json()
    report(
        "PASS",
        "deployment recorded (#5)",
        f"public={dep['public']['url'] if dep['public'] else None}",
    )

    rc = httpx.Client(
        base_url=args.url,
        headers={"Authorization": f"Bearer {env['RECORDER_TOKEN']}"},
        timeout=args.timeout,
    )
    report("PASS" if check(rc, args.tenant) else "FAIL", "recorder --check (#5)")

    # A fresh call, recorded because issuing the bundle enabled recording.
    cid = c.post(f"{base}/calls").json()["call_id"]
    turn(c, base, cid, "駐車場はありますか")
    c.delete(f"{base}/calls/{cid}")

    with tempfile.TemporaryDirectory() as d:
        stored = 0
        for _ in range(5):  # other records may already be queued
            stored += Recorder(rc, args.tenant, DirectoryStorage(Path(d))).run_once()
        files = {p.stem: p for p in Path(d).rglob("*.json")}
        if cid in files:
            rec = json.loads(files[cid].read_text())
            report("PASS", "record stored in customer storage (#4)", f"{len(rec['turns'])} turn(s)")
        else:
            report("FAIL", "record stored in customer storage (#4)", f"stored={stored}")
    again = rc.get(f"{base}/records").json()["records"]
    report(
        "PASS" if all(x["record_id"] != cid for x in again) else "FAIL",
        "record deleted from memory after ack (#4)",
    )


if __name__ == "__main__":
    sys.exit(main())
