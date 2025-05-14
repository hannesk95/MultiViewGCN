#!/bin/bash
# This script installs the required packages for the project using conda.

conda create -n multiviewgcn python=3.10 -y
conda activate multiviewgcn

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib
pip install torchio
pip install pyvista
pip install open3d

conda install conda-forge::meshplot
conda install -c conda-forge pythreejs

pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu118.html
