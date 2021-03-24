#!/bin/bash

#SBATCH --partition=ml
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=00:10:00
#SBATCH --account=p_humanpose
#SBATCH --output=%j.out
#SBATCH --error=%j.err

source ./requirements.sh

source /sw/installed/Anaconda3/2019.03/etc/profile.d/conda.sh
conda deactivate
conda activate ${KERNELS_DIR}/${KERNEL_NAME}  # or source
which python  # just to check

cd /scratch/ws/0/stfo194b-p_humanpose/learnable-triangulation-pytorch
pwd  # just to check

python3 train.py --eval --eval_dataset val --config experiments/human36m/eval/human36m_vol_softmax.yaml --logdir ./logs
