#!/bin/bash

runs=$1
set -euxo pipefail

echo '[]' >| merged-logs.json

generate_logs() {
  ./heatmap simulate --attacker $1

  python <<PYTHON
import json

with open('logs.json') as f:
  logs = json.load(f)

old_logs = []
with open("merged-logs.json", "r") as f:
  old_logs = json.load(f)

old_logs.extend(logs)
old_logs.sort(key=lambda l: l['timestamp'])

with open("merged-logs.json", "w") as f:
  json.dump(old_logs, f, indent=2)
PYTHON
}

for i in $(eval echo {1..$runs}); do
  echo $i
  generate_logs "bfs-random"
done

for i in $(eval echo {1..$runs}); do
  echo $i
  generate_logs "dfs-random"
done
