#!/bin/bash
# AFD entry, kept for existing presets and docs: the role dispatch lives in
# serve_mpmd_mp_rpc_all_mpi_template.sh (attention ranks, then AF-EP F ranks,
# then Engram ranks).
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/serve_mpmd_mp_rpc_all_mpi_template.sh" "$@"
