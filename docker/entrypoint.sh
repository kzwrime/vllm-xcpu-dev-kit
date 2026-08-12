#!/usr/bin/env bash
set -euo pipefail

mkdir -p /run/sshd /root/.ssh
chmod 0700 /root/.ssh

# Host keys live in a named volume, so recreating a container does not change
# its SSH fingerprint. Do not mount /etc/ssh itself: it contains sshd_config.
hostkeys_dir=/var/lib/ssh-hostkeys
mkdir -p "$hostkeys_dir"
shopt -s nullglob
saved_hostkeys=("$hostkeys_dir"/ssh_host_*)
if (( ${#saved_hostkeys[@]} )); then
  install -m 0600 "${saved_hostkeys[@]}" /etc/ssh/
else
  ssh-keygen -A
  install -m 0600 /etc/ssh/ssh_host_* "$hostkeys_dir"/
fi

if [[ -f /run/cluster-ssh/authorized_keys ]]; then
  install -m 0600 /run/cluster-ssh/authorized_keys /root/.ssh/authorized_keys
fi

exec /usr/sbin/sshd -D -e
