#!/usr/bin/env bash

set -euo pipefail

workspace_path="${1:?workspace path is required}"
workspace_basename="${2:?workspace basename is required}"
devcontainer_env_file=".devcontainer/.env"

if [[ ! -d ".devcontainer" ]]; then
  echo "ERROR: .devcontainer directory not found from $(pwd)." >&2
  exit 1
fi

volume_basename="$(printf '%s' "$workspace_basename" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9_.-]+/-/g; s/^-+//; s/-+$//')"
if [[ -z "$volume_basename" ]]; then
  echo "ERROR: unable to derive a safe Docker volume name from '$workspace_basename'." >&2
  exit 1
fi

cat >"$devcontainer_env_file" <<EOF
WORKSPACE_PATH=$workspace_path
WORKSPACE_BASENAME=$workspace_basename
WORKSPACE_VENV_VOLUME=${volume_basename}-venv
WORKSPACE_PYTHONPATH=/workspaces/$workspace_basename/src
WORKSPACE_PYTEST_RICH=1
EOF

echo "Wrote $devcontainer_env_file for workspace '$workspace_basename'."