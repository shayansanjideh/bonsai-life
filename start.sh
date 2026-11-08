#!/usr/bin/env bash
# Bonsai-Life: one command. Downloads PrismML's llama.cpp build + Ternary-Bonsai-2-2B (PQ2_0 + mmproj Q8_0),
# starts llama-server (thinking off is set per request) and the web app, then prints the URL.
#   ./start.sh            # full app with the model
#   ./start.sh --no-model # sim + offline naturalist only
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8000}"; LPORT="${LLAMA_PORT:-8080}"
TAG="${PRISM_TAG:-prism-b12128-d7b2896}"
REPO="prism-ml/Ternary-Bonsai-2-2B-gguf"
MODEL="${MODEL_FILE:-Ternary-Bonsai-2-2B-PQ2_0.gguf}"; MMPROJ="Ternary-Bonsai-2-2B-mmproj-Q8_0.gguf"
mkdir -p models bin logs

fetch() { # url dest
  [ -s "$2" ] && return 0
  echo "  downloading $(basename "$2")"
  curl -fL --retry 5 --retry-delay 5 -o "$2.part" "$1" && mv "$2.part" "$2"
}

if [ "${1:-}" != "--no-model" ]; then
  OS="$(uname -s)"; ARCH="$(uname -m)"
  if [ "$OS" = "Darwin" ]; then ASSET="llama-$TAG-bin-macos-$([ "$ARCH" = arm64 ] && echo arm64 || echo x64).tar.gz"; NGL=99
  elif command -v nvidia-smi >/dev/null 2>&1; then ASSET="llama-$TAG-bin-linux-cuda-12.4-x64.tar.gz"; NGL=99
  else ASSET="llama-$TAG-bin-ubuntu-x64.tar.gz"; NGL=0; fi
  SERVER="${LLAMA_SERVER:-$(find bin -name llama-server -type f 2>/dev/null | head -1)}"
  if [ -z "$SERVER" ]; then
    echo "Getting PrismML llama.cpp ($ASSET)"
    fetch "https://github.com/PrismML-Eng/llama.cpp/releases/download/$TAG/$ASSET" "bin/$ASSET"
    tar xzf "bin/$ASSET" -C bin && rm -f "bin/$ASSET"
    SERVER="$(find bin -name llama-server -type f | head -1)"
    [ "$OS" = "Darwin" ] && xattr -dr com.apple.quarantine bin 2>/dev/null || true
  fi
  MPATH="${MODEL_PATH:-models/$MODEL}"; PPATH="${MMPROJ_PATH:-models/$MMPROJ}"
  echo "Getting Ternary-Bonsai-2-2B weights"
  [ -n "${MODEL_PATH:-}" ] || fetch "https://huggingface.co/$REPO/resolve/main/$MODEL" "$MPATH"
  [ -n "${MMPROJ_PATH:-}" ] || fetch "https://huggingface.co/$REPO/resolve/main/$MMPROJ" "$PPATH"

  CMD=("$SERVER" -m "$MPATH" --mmproj "$PPATH" -ngl "$NGL" -c "${CTX:-4096}" -np 1 --port "$LPORT" --host 127.0.0.1
       --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 --presence-penalty 1.5 --image-min-tokens "${IMAGE_MIN_TOKENS:-1024}")
  # Small Linux boxes: cap memory so the model can never OOM the machine.
  if [ "$OS" = "Linux" ] && command -v systemd-run >/dev/null && [ "$(awk '/MemTotal/{print $2}' /proc/meminfo)" -lt 4000000 ]; then
    CMD=(systemd-run --scope --quiet -p MemoryMax="${MEMMAX:-1100M}" nice -n 19 "${CMD[@]}")
  fi
  echo "Starting llama-server on :$LPORT"
  "${CMD[@]}" > logs/llama.log 2>&1 &
  LPID=$!; trap 'kill $LPID 2>/dev/null || true' EXIT
  for i in $(seq 1 120); do curl -sf "http://127.0.0.1:$LPORT/health" >/dev/null && break; kill -0 $LPID 2>/dev/null || { tail -20 logs/llama.log; exit 1; }; sleep 1; done
fi

URL="http://localhost:$PORT"
echo; echo "  Bonsai-Life is running:  $URL"; echo
( [ "$(uname -s)" = Darwin ] && sleep 1 && open "$URL" ) >/dev/null 2>&1 &
python3 server.py --port "$PORT" --llama "http://127.0.0.1:$LPORT"
