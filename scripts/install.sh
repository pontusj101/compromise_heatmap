#!/usr/bin/env bash

set -euo pipefail

while read -s line; do
  pip install $line

  if [[ "${1-linux}" == 'linux' && $line == 'torch' ]]; then
    # no wheel offered for macos
    pip install ninja wheel cmake
    pip install git+https://github.com/pyg-team/pyg-lib.git
  fi
done < requirements."${1-linux}".txt
