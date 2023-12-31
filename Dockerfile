FROM pytorch/pytorch:latest
RUN apt-get update && \
    apt-get install -y git
RUN pip install matplotlib
RUN pip install pyRDDLGym
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
RUN pip install torch_geometric
RUN pip install torch_geometric_temporal
RUN pip install google-cloud-storage
RUN pip install google-cloud-logging
RUN pip install google-cloud-secret-manager
RUN pip install cloudml-hypertune
RUN pip install wandb
ARG CACHEBUST=1
RUN echo "Cache bust: $CACHEBUST"
RUN git clone https://github.com/pontusj101/rddl_training_data_producer.git /root/gnn_rddl
WORKDIR /root/gnn_rddl
ENTRYPOINT ["python", "main.py", "train"]
