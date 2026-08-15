#!/bin/bash -l
#SBATCH --job-name=hard-convergence
#SBATCH --partition=singlenode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=06:00:00
#SBATCH --export=NONE
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
unset SLURM_EXPORT_ENV
cd "$SLURM_SUBMIT_DIR"

module load python
source "$WORK/envs/data-lab-venv/bin/activate"

export NUMBA_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_PLACES=cores
export OMP_PROC_BIND=true
export NUMBA_CACHE_DIR="$WORK/.cache/numba"

mkdir -p results/hard_thresholding_final_convergence "$NUMBA_CACHE_DIR"

echo "Start: $(date)"
echo "Node: $(hostname)"
echo "Python: $(which python)"
python -m jupyter nbconvert \
    --to notebook \
    --execute notebooks/hard_thresholding_final_convergence_hpc.ipynb \
    --output hard_thresholding_final_convergence_hpc_executed.ipynb \
    --output-dir results/hard_thresholding_final_convergence \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=data-lab-venv
echo "Finished: $(date)"
