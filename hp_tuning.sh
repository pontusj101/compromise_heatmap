#!/bin/bash

JOB_NAME="gnnrddl_$(date +%y%m%d%H%M%S)"
JOB_NAME=${JOB_NAME:0:18}

gcloud ai hp-tuning-jobs create \
    --region=europe-west1 \
    --display-name=$JOB_NAME \
    --max-trial-count=8 \
    --parallel-trial-count=8 \
    --service-account=codespaces@gnn-rddl.iam.gserviceaccount.com \
    --config=hp_config.yaml
