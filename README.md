# compromise_heatmap

A brief paper on this repo is available as [Johnson & Ekstedt, "Towards a Graph Neural Network-Based Approach for Estimating Hidden States in Cyber Attack Simulations", 2023](https://arxiv.org/abs/2312.05666).

The code can run in GitHub Codespaces, as well as in GCP Batch and GCP Vertex AI. 

In Codespaces (or some other execution environment), try `python3 main train`. 
To run it in GCP Vertex AI for hyperparameter tuning, use hp_tuning.sh.
To run it in GCP Batch for parallel simulation, use batch.sh.

You will need a GCP bucket, and you will need to provide a GCP service account with access to that bucket.

The project is also integrated with Weights & Biases, so you will need a wandb API key.
