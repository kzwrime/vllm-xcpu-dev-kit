#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir"

env_file=${1:-.env.146}
if [[ ! -f "$env_file" ]]; then
  echo "Missing environment file: $env_file" >&2
  exit 1
fi

if [[ ! -f ssh/id_ed25519 ]]; then
  ./init-cluster-key.sh
fi

./ensure-network.sh "$env_file"
# All eight services use the same image; building one service avoids launching
# eight equivalent BuildKit targets.
docker compose --env-file "$env_file" build c1
docker compose --env-file "$env_file" up -d --no-build --remove-orphans
