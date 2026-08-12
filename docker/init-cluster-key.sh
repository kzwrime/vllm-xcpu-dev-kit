#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ssh_dir="$script_dir/ssh"
private_key="$ssh_dir/id_ed25519"

mkdir -p "$ssh_dir"
chmod 0700 "$ssh_dir"

if [[ -e "$private_key" || -e "$private_key.pub" ]]; then
  echo "Refusing to overwrite an existing cluster key: $private_key" >&2
  exit 1
fi

ssh-keygen -q -t ed25519 -N '' -f "$private_key" -C 'wzk-vllm-xcpu-cluster'
install -m 0600 "$private_key.pub" "$ssh_dir/authorized_keys"
chmod 0600 "$private_key"
echo "Created $private_key and $ssh_dir/authorized_keys"
