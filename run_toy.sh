#!/bin/bash
#SBATCH --account=project_462000131
#SBATCH --partition=small
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=0:05:00

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.5
source /scratch/${SLURM_JOB_ACCOUNT}/${USER}/PDLd3/pvenv/bin/activate
export MLFLOW_TRACKING_URI=/scratch/${SLURM_JOB_ACCOUNT}/${USER}/PDLd3/mlruns

mkdir -p /tmp/pcheckpoints
srun python toy.py
rsync -a /tmp/pcheckpoints .
