# Efficient Fine-Tuning Method for Large Language Models


Copyright (C) 2025 Benjamin Marsh, Shaun Monera

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.


You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


## Getting started

### Step 1. Fulfill Initial Requirements

1. Fine-tuning dataset constructed with Distilling Step-by-Step (reference: https://arxiv.org/abs/2305.02301 )
2. Student model identified and downloaded onto Kubernetes cluster (see step 1 for cluster dependencies)


### Step 2. Install Kubernetes Dependencies

The following kubernetes dependencies are required to run this code:

1. Kubernetes instance deployed with internet access
2. A default Kubernetes storage provisioner that is capable of ReadWriteMany mode
3. Nvidia GPU Operator compatible with your GPUs (Code has only been tested with CUDA compatible GPUs)
4. kubectl version with kustomize capability
5. kubectl access to the kubernetes instance

### Step 3. Deploy kuberay

1. cd into efficient_finetuning
2. RUN kubectl -k apply deploy/kuberay-operator

### Step 4. Deploy Jupyter

1. cd into efficient_finetuning
2. RUN kubectl -k apply deploy/jupyter

### Step 5. Edit Global Variables

1. Use Jupyter to upload fine-tuning dataset and student model weights
2. Place dataset in /data
3. Place student model weights in /model
4. Edit the trainT5, trainT5_LoRA, trainT5_QLoRA python files in /src to account for the dataset source name, student model weight name, run name, and desired hyperparameters (Note that changes to the training source codes will only affect files in the jupyter notebook and not in the repo)


### Step 6. Deploy the Ray job

1. cd into efficient_finetuning
2. RUN ./run-training.sh

### Step 7. View Run Result 

1. Run results can be accessed via Jupyter notebook and Tensorboard in the output folder

### Step 8. To Re-Run with new hyperparameters

1. Repeat Steps 5 - 8 as needed 