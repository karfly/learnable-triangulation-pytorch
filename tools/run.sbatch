#!/bin/bash

#SBATCH --partition=ml
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4096
#SBATCH --time=12:00:00
#SBATCH --account=p_humanpose
#SBATCH --output=%j.out
#SBATCH --error=%j.err

date +"%Y-%m-%d %H:%M:%S"

source ./requirements.sh
source ./utils.sh

source /sw/installed/Anaconda3/2019.03/etc/profile.d/conda.sh
conda deactivate
conda activate ${KERNELS_DIR}/${KERNEL_NAME}  # or source
which python  # just to check

CODE_FOLDER="/home/stfo194b/tesi/learnable-triangulation-pytorch"
LOGS_FOLDER="/scratch/ws/0/stfo194b-p_humanpose/learnable-triangulation-pytorch/logs"
EXP_CONFIG="experiments/human36m/train/human36m_alg.yaml"
# EXP_CONFIG="experiments/human36m/train/human36m_vol_softmax.yaml"

cd ${CODE_FOLDER}
pwd  # just to check

python3 train.py --config ${EXP_CONFIG} --logdir ${LOGS_FOLDER}
# python3 train.py --eval --eval_dataset val --config experiments/human36m/eval/human36m_alg.yaml --logdir ${LOGS_FOLDER}

date +"%Y-%m-%d %H:%M:%S"