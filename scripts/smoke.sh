#!/usr/bin/env bash
# Bring up the full stack on one machine (CPU is fine) and run the smoke test.
#
#   scripts/smoke.sh                 # uses qwen2.5:3b-instruct for speed on CPU
#   SMOKE_MODEL=qwen2.5:7b-instruct scripts/smoke.sh
#   SMOKE_KEEP=1 scripts/smoke.sh    # leave containers running afterwards
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${SMOKE_MODEL:-qwen2.5:3b-instruct}"
URL="${SMOKE_URL:-http://localhost:8000}"

# One-off admin token for this run; only its hash goes into the container.
ADMIN_TOKEN="$(python3 -c 'import secrets;print(secrets.token_urlsafe(24))')"
export ADMIN_TOKEN_SHA256="$(python3 -c 'import hashlib,sys;print(hashlib.sha256(sys.argv[1].encode()).hexdigest())' "$ADMIN_TOKEN")"
export PUBLIC_URL="$URL" OLLAMA_MODEL="$MODEL"

echo "== starting stack (model: $MODEL)"
docker compose up -d --build

echo "== pulling model (first run downloads a few GB)"
docker compose exec -T ollama ollama pull "$MODEL"

echo "== waiting for the API (first start downloads Whisper and e5 models)"
for i in $(seq 1 180); do
  if curl -fsS "$URL/healthz" >/dev/null 2>&1; then break; fi
  if [ "$i" = 180 ]; then echo "API did not come up"; docker compose logs --tail 50 api; exit 1; fi
  sleep 5
done

python3 -m pip install -q httpx >/dev/null 2>&1 || true
set +e
python3 scripts/smoke.py --url "$URL" --admin-token "$ADMIN_TOKEN"
status=$?
set -e

if [ -z "${SMOKE_KEEP:-}" ]; then docker compose down; fi
exit $status
