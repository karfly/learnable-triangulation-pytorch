#!/bin/bash

#SBATCH --partition=ml
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=00:30:00
#SBATCH --account=p_humanpose
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# This will spawn a sub-bash in parallel which will be started right away. It starts  the program sleep set to sleep N seconds and afterwards calls Slurm's job cancel programm with the job ID of the Job.
REALLY_KILL_AFTER_SECONDS=3600
(/usr/bin/sleep ${REALLY_KILL_AFTER_SECONDS} && scancel ${SLURM_JOB_ID} ) &

date +"%Y-%m-%d %H:%M:%S"

source ./requirements.sh
source ./utils/versioning.sh

showGitDiff

source /sw/installed/Anaconda3/2019.03/etc/profile.d/conda.sh
conda deactivate
conda activate ${KERNELS_DIR}/${KERNEL_NAME}  # or source
which python  # just to check

CODE_FOLDER="/home/stfo194b/tesi/learnable-triangulation-pytorch"
LOGS_FOLDER="/scratch/ws/0/stfo194b-p_humanpose/learnable-triangulation-pytorch/logs"
EXP_CONFIG="experiments/human36m/train/human36m_alg.yaml"

cd ${CODE_FOLDER}
pwd  # just to check

python3 main.py --config ${EXP_CONFIG} --logdir ${LOGS_FOLDER}

date +"%Y-%m-%d %H:%M:%S"

sstat -jp $SLURM_JOB_ID.batch --format=JobID,AveVMSize,MaxVMSize,AveCPU,AveCPUFreq,AveDiskRead,AveDiskWrite