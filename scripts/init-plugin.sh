#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python_script="$script_dir/init-plugin.py"

python "$python_script" "$@"