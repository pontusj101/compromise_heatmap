#!/bin/bash

JOB_NAME="sim$(date +%y%m%d%H%M%S)"
JOB_NAME=${JOB_NAME:0:16}

gcloud beta batch jobs submit $JOB_NAME --location europe-west1 --config - <<EOD
{
  "name": "projects/gnn-rddl/locations/europe-west1/jobs/gnnrddljob",
  "taskGroups": [
    {
      "taskCount": "8",
      "parallelism": "8",
      "taskSpec": {
        "computeResource": {
          "cpuMilli": "1000",
          "memoryMib": "30720"
        },
        "runnables": [
          {
            "container": {
              "imageUri": "europe-west1-docker.pkg.dev/gnn-rddl/gnn-rddl/gnnrddl",
              "entrypoint": "python3",
              "commands": [
                "main.py",
                "instance",
                "simulate",
                "--n_instances=256",
                "--min_size=128",
                "--max_size=256",
                "--n_init_compromised=1",
                "--max_game_time=128",
                "--sim_log_window=32",
                "--agent_type=passive"
              ],
              "volumes": []
            }
          }
        ],
        "volumes": []
      }
    }
  ],
  "allocationPolicy": {
    "instances": [
      {
        "policy": {
          "provisioningModel": "STANDARD",
          "machineType": "e2-standard-16"
        }
      }
    ]
  },
  "logsPolicy": {
    "destination": "CLOUD_LOGGING"
  }
}
EOD
