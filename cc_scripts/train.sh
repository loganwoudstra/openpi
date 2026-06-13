#!/bin/bash
#SBATCH --job-name=train-policy
#SBATCH --account=aip-jjin5
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=6
#SBATCH --mem=150gb
#SBATCH --time=14:00:00
#SBATCH --output=/home/lwoudstr/scratch/openpi_logs/%A.out

set -e

CONFIG_NAME=$1
EXPERIMENT_NAME=$2
MODE=${3:-resume}

echo "Config name: $CONFIG_NAME"
echo "Experiment name: $EXPERIMENT_NAME"
echo "Mode: $MODE"

module load python/3.11
module load cuda
module load gcc opencv
module load apptainer/1.4.5

REPO_ROOT=/home/lwoudstr/projects/aip-jjin5/lwoudstr/openpi

# resume from or overwrite existing checkpoint
if [[ "$MODE" != "resume" && "$MODE" != "overwrite" ]]; then
    echo "ERROR: MODE must be 'resume' or 'overwrite' (or empty, with resume as default)"
    echo "You provided: '$MODE'"
    exit 1
fi

if [ "$MODE" = "overwrite" ]; then
    MODE_FLAG="--overwrite"
else
    MODE_FLAG="--resume"
fi

# running
apptainer exec \
    --nv \
    --pwd /app \
    --bind $PWD:/app \
    --env PYTHONPATH=/app/src \
    --env XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    ~/scratch/openpi_server.sif \
    /bin/bash -c "source /.venv/bin/activate && python scripts/train.py \
        $CONFIG_NAME \
        --exp-name=$EXPERIMENT_NAME  \
        $MODE_FLAG"