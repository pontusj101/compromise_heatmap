#!/bin/bash

set -euxo pipefail

./heatmap simulate --attacker bfs-random --horizon 1000
./heatmap simulate --attacker dfs-random --horizon 1000

python <<PYTHON
import json
with open('DepthFirstAttacker-logs.json') as f:
  logs = json.load(f)
with open('BreadthFirstAttacker-logs.json') as f:
  logs.extend(json.load(f))

logs.sort(key=lambda l: l['timestamp'])

with open("merged-logs.json", "w") as f:
  json.dump(logs, f, indent=2)
PYTHON
