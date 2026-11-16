#!/bin/bash  
#SBATCH --job-name=tangentialshear      # Job name
#SBATCH --output=logs_final/tsout_%A_%a.out    # Standard output
#SBATCH --error=logs_final/terr_%A_%a.err     # Standard error
#SBATCH --time=2-00:00:00                   # Walltime (HH:MM:SS)
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=56                 # CPU cores per task
#SBATCH --mem=64G                         # Memory
#SBATCH --partition=main              # Partition/queue name
#SBATCH --chdir=/shared_home/lreilly/scratch

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
# --------------------------
# Load any modules you need
# --------------------------
module load python/3.12   # adjust to your cluster's Python module
module load hdf5          # if required for h5py

JOBSCRIPT=/shared_home/lreilly/scratch/tangentialshear.slurm
PYSCRIPT=/shared_home/lreilly/scratch/modelling_final.py
N_RUNS=8
# Optional: activate your conda environment
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate myenv

# --------------------------
# Run the Python script
# --------------------------

if [ -n "$stage1_only" ]; then
    TAG=stage1
else
    TAG=${SLURM_ARRAY_TASK_ID:-all}
fi
echo "[$TAG] job $SLURM_JOB_ID on $(hostname), $SLURM_CPUS_PER_TASK cores, $(date)"

timeout --signal=TERM --kill-after=120 47h python -u "$PYSCRIPT"
status=$?
echo "[$TAG] python exited with status $status at $(date)"


if [ -f "GAVE_UP_${TAG}.flag" ]; then
    echo "[$TAG] hit the attempt limit: not resubmitting"
    exit 1
fi

if [ -f "pipeline_not_done_${TAG}.flag" ]; then
    if [ "$status" -eq 124 ] || [ "$status" -eq 0 ]; then
	if [ "$TAG" = "stage1" ]; then
	    sbatch --export=ALL,stage1_only=1 "$JOBSCRIPT"
	else
	    sbatch --array="$SLURM_ARRAY_TASK_ID" "$JOBSCRIPT"
	fi
	exit 0
    fi
    echo "[$TAG] crashed (status $status): not resubmitting, check the .err log"
    exit "$status"
fi

if [ "$TAG" = "stage1" ]; then
    unset stage1_only
    sbatch --array=0-$((N_RUNS-1)) "$JOBSCRIPT"
fi
echo "[$TAG] finished"

