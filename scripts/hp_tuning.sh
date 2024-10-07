#!/bin/bash

JOB_NAME="$(date +%y%m%d%H%M%S)"
JOB_NAME=${JOB_NAME:0:18}

gcloud ai hp-tuning-jobs create \
    --region=us-central1 \
    --display-name=$JOB_NAME \
    --max-trial-count=64 \
    --parallel-trial-count=64 \
    --service-account=account \
    --config=hp_config.yaml
