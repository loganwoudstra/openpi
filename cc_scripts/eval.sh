#!/bin/bash
#SBATCH --job-name=eval-policy
#SBATCH --account=aip-jjin5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40gb
#SBATCH --time=2:30:00
#SBATCH --output=/home/lwoudstr/scratch/openpi_logs/%A.out

set -e

POLICY_DIR=${1:-/home/lwoudstr/projects/aip-jjin5/lwoudstr/models/openpi_pi05_libero_jax}
POLICY_CONFIG=${2:-pi05_libero}
TASK_SUITE=${3:-libero_10}
PORT=${4:-8069}

echo "Policy dir:    $POLICY_DIR"
echo "Policy config: $POLICY_CONFIG"
echo "Task suite:    $TASK_SUITE"
echo "Port:          $PORT"

module load apptainer/1.4.5

REPO_ROOT=/home/lwoudstr/projects/aip-jjin5/lwoudstr/openpi

export SERVER_ARGS="--port $PORT --env LIBERO policy:checkpoint --policy.config $POLICY_CONFIG --policy.dir $POLICY_DIR"
export CLIENT_ARGS="--args.port $PORT --args.task-suite-name $TASK_SUITE"

# --- Launch server in background ---
echo "[$(date)] Starting policy server..."
apptainer exec \
    --nv \
    --bind $REPO_ROOT:/app \
    --bind ${OPENPI_DATA_HOME:-~/.cache/openpi}:/openpi_assets \
    --env OPENPI_DATA_HOME=/openpi_assets \
    --env IS_DOCKER=true \
    --env PYTHONPATH=/app/src \
    ~/scratch/openpi_server.sif \
    /bin/bash -c "source /.venv/bin/activate && python scripts/serve_policy.py $SERVER_ARGS" &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# --- Wait for server to be ready ---
echo "[$(date)] Waiting for server on port $PORT..."
MAX_WAIT=120
ELAPSED=0
until (echo > /dev/tcp/localhost/$PORT) 2>/dev/null; do
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "ERROR: Server did not become ready within ${MAX_WAIT}s. Killing server."
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
done
echo "[$(date)] Server is ready (after ${ELAPSED}s). Starting client..."

# --- Launch client (foreground, job ends when this exits) ---
apptainer exec \
    --nv \
    --no-home \
    --pwd /app \
    --bind $REPO_ROOT:/app \
    --bind $REPO_ROOT/data:/data \
    ~/scratch/libero.sif \
    /bin/bash -c "source /.venv/bin/activate && echo 'N' | python examples/libero/main.py $CLIENT_ARGS"

CLIENT_EXIT=$?
echo "[$(date)] Client exited with code $CLIENT_EXIT. Shutting down server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

echo "[$(date)] Done."
exit $CLIENT_EXIT