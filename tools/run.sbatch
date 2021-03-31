#!/bin/bash

#SBATCH --partition=ml
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=02:00:00
#SBATCH --account=p_humanpose
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# interactive version: srun --partition=ml --nodes=1 --tasks=1 --cpus-per-task=2 --gres=gpu:1 --mem-per-cpu=2048 --time=02:00:00 --account=p_humanpose --pty bash

source ./requirements.sh

source /sw/installed/Anaconda3/2019.03/etc/profile.d/conda.sh
conda deactivate
conda activate ${KERNELS_DIR}/${KERNEL_NAME}  # or source
which python  # just to check

# todo use /lustre/ssd/ws/stfo194b-p_humanpose

cd /home/stfo194b/tesi/learnable-triangulation-pytorch
pwd  # just to check

python3 train.py --config experiments/human36m/train/human36m_alg.yaml --logdir /scratch/ws/0/stfo194b-p_humanpose/learnable-triangulation-pytorch/logs

# eval: python3 train.py --eval --eval_dataset val --config experiments/human36m/eval/human36m_alg.yaml --logdir /scratch/ws/0/stfo194b-p_humanpose/learnable-triangulation-pytorch/logs
