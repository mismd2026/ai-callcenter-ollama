#!/usr/bin/env bash
# Bring up the full stack on one machine (CPU is fine) and run the smoke test.
#
#   scripts/smoke.sh                 # uses qwen2.5:3b-instruct for speed on CPU
#   SMOKE_MODEL=qwen2.5:7b-instruct scripts/smoke.sh
#   SMOKE_KEEP=1 scripts/smoke.sh    # leave containers running afterwards
#
# If the Ollama container cannot reach the model registry (TLS errors from a
# proxy or security software), the model is copied in from the host's own
# Ollama instead: run `ollama pull <model>` on the host once beforehand.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${SMOKE_MODEL:-qwen2.5:3b-instruct}"
URL="${SMOKE_URL:-http://localhost:8000}"

# One-off admin token for this run; only its hash goes into the container.
ADMIN_TOKEN="$(python3 -c 'import secrets;print(secrets.token_urlsafe(24))')"
export ADMIN_TOKEN_SHA256="$(python3 -c 'import hashlib,sys;print(hashlib.sha256(sys.argv[1].encode()).hexdigest())' "$ADMIN_TOKEN")"
export PUBLIC_URL="$URL" OLLAMA_MODEL="$MODEL"

has_model() { docker compose exec -T ollama ollama list 2>/dev/null | awk 'NR>1{print $1}' | grep -qx "$MODEL"; }

# Copy one model (its manifest and blobs only) from the host's Ollama store.
copy_model_from_host() {
  local name="${MODEL%%:*}" tag="${MODEL#*:}" store="" d
  for d in "${OLLAMA_MODELS:-}" "$HOME/.ollama/models" /usr/share/ollama/.ollama/models; do
    if [ -n "$d" ] && [ -f "$d/manifests/registry.ollama.ai/library/$name/$tag" ]; then store="$d"; break; fi
  done
  if [ -z "$store" ]; then
    echo "   host has no $MODEL either. Run on the host:  ollama pull $MODEL"
    return 1
  fi
  echo "   copying $MODEL from $store"
  local rel="manifests/registry.ollama.ai/library/$name/$tag"
  docker compose exec -T ollama mkdir -p "/root/.ollama/models/$(dirname "$rel")" /root/.ollama/models/blobs
  docker compose cp "$store/$rel" "ollama:/root/.ollama/models/$rel"
  python3 -c 'import json,sys;m=json.load(open(sys.argv[1]));print("\n".join(l["digest"].replace(":","-") for l in [m["config"],*m["layers"]]))' \
    "$store/$rel" | while read -r blob; do
      docker compose cp "$store/blobs/$blob" "ollama:/root/.ollama/models/blobs/$blob"
    done
}

echo "== starting stack (model: $MODEL)"
docker compose up -d --build

echo "== preparing model"
if has_model; then
  echo "   $MODEL already present"
elif docker compose exec -T ollama ollama pull "$MODEL"; then
  :
else
  echo "   pull from the registry failed inside the container; trying the host's Ollama"
  copy_model_from_host
  has_model || { echo "   model still not available in the container"; exit 1; }
  echo "   $MODEL copied"
fi

echo "== waiting for the API (first start downloads Whisper and e5 models)"
for i in $(seq 1 180); do
  if curl -fsS "$URL/healthz" >/dev/null 2>&1; then break; fi
  if [ "$i" = 180 ]; then echo "API did not come up"; docker compose logs --tail 50 api; exit 1; fi
  sleep 5
done

# Run the checks inside the API container: it already has httpx and the
# recorder, so nothing needs installing on the host.
docker compose exec -T -u root api mkdir -p /app/scripts
docker compose cp scripts/smoke.py api:/app/scripts/smoke.py
set +e
docker compose exec -T api python /app/scripts/smoke.py --url http://localhost:8000 --admin-token "$ADMIN_TOKEN"
status=$?
set -e

if [ -z "${SMOKE_KEEP:-}" ]; then docker compose down; fi
exit $status
