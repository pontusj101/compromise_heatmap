#!/bin/bash

set -euxo pipefail

HORIZON=1000

echo '[]' >| merged-logs.json

generate_attacker_logs() {
  ./heatmap simulate --attacker bfs-random --horizon $HORIZON

  python <<PYTHON
import json

with open('BreadthFirstAttacker-logs.json') as f:
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

generate_user_logs() {
  ./heatmap simulate --attacker dfs-random --horizon $HORIZON

  python <<PYTHON
import json
import os

with open('DepthFirstAttacker-logs.json') as f:
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

generate_attacker_logs
generate_attacker_logs
generate_attacker_logs
generate_attacker_logs
generate_user_logs
generate_user_logs
generate_user_logs
generate_user_logs
