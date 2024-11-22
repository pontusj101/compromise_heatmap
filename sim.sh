#!/bin/bash

set -euxo pipefail

./heatmap simulate --attacker bfs-random --horizon 1000
./heatmap simulate --attacker dfs-random --horizon 1000

python <<PYTHON
import json,os
with open('DepthFirstAttacker-logs.json') as f:
  logs = json.load(f)
with open('BreadthFirstAttacker-logs.json') as f:
  logs.extend(json.load(f))

logs.sort(key=lambda l: l['timestamp'])

old_logs = []
if os.path.isfile("merged-logs.json"):
  with open("merged-logs.json", "r") as f:
    old_logs = json.load(f)

with open("merged-logs.json", "w") as f:
  old_logs.extend(logs)
  json.dump(logs, f, indent=2)
PYTHON
