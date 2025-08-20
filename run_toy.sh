#!/bin/bash
#SBATCH --account=project_465000131
#SBATCH --partition=debug
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=0:05:00

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.5
source /scratch/${SLURM_JOB_ACCOUNT}/${USER}/PDLd3/pvenv/bin/activate
export MLFLOW_TRACKING_URI=/scratch/${SLURM_JOB_ACCOUNT}/${USER}/PDLd3/mlruns

srun python toy.py
mv /tmp/pcheckpoints .
