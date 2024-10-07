# FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
FROM ubuntu
WORKDIR /root

RUN mkdir -p ~/.config/pip/

RUN cat <<EOF >~/.config/pip/pip.conf
[global]
break-system-packages = true
EOF

RUN apt-get update && \
    apt-get install -y git && \
    apt-get install -y python3 python3-pip

RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN pip install matplotlib
RUN pip install torch==2.4.0
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
RUN pip install torch_geometric
RUN pip install torch_geometric_temporal
RUN pip install google-cloud-storage
RUN pip install google-cloud-logging
RUN pip install google-cloud-secret-manager
RUN pip install cloudml-hypertune
RUN pip install wandb
RUN pip install scikit-learn
RUN pip install numpy==1.26.4
RUN pip install --upgrade matplotlib ffmpeg-python

COPY requirements-tf.txt requirements-tf.txt
RUN pip install -r requirements-tf.txt

RUN apt-get install -y ffmpeg

WORKDIR /root/gnn_rddl

# COPY deps deps
COPY . /root/gnn_rddl
RUN pip install deps/mal-toolbox
RUN pip install deps/mal-simulator
RUN pip install deps/twmn-core

COPY config.pt.yml config.yml

ENTRYPOINT ["./run"]
