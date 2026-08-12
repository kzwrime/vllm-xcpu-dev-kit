#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir"

env_file=${1:-.env.146}
if [[ ! -f "$env_file" ]]; then
  echo "Missing environment file: $env_file" >&2
  exit 1
fi

# Read the resolved values rather than sourcing the env file as shell code.
compose_json=$(docker compose --env-file "$env_file" config --format json)
network_name=$(jq -r '.networks.vllm.name' <<<"$compose_json")
c1_address=$(jq -r '.services.c1.networks.vllm.ipv4_address' <<<"$compose_json")

if [[ ! "$c1_address" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
  echo "Invalid resolved c1 IP address: $c1_address" >&2
  exit 1
fi
expected_subnet="${c1_address%.*}.0/24"

if ! docker network inspect "$network_name" >/dev/null 2>&1; then
  echo "Creating Docker network $network_name ($expected_subnet)"
  docker network create --driver bridge --subnet "$expected_subnet" "$network_name" >/dev/null
  exit 0
fi

actual_subnet=$(docker network inspect "$network_name" \
  | jq -r '.[0].IPAM.Config[]? | select(.Subnet != null) | .Subnet' \
  | head -n 1)
if [[ "$actual_subnet" != "$expected_subnet" ]]; then
  echo "Network $network_name already exists with subnet ${actual_subnet:-<none>}; expected $expected_subnet." >&2
  echo 'Choose a matching VLLM_SUBNET_PREFIX or create/use a different network name.' >&2
  exit 1
fi

echo "Docker network $network_name already exists with subnet $actual_subnet"
