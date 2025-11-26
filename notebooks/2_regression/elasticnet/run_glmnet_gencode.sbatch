#!/bin/bash
#SBATCH --job-name=glmnet_gencode
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/glmnet_gencode_%j.out
#SBATCH --error=logs/glmnet_gencode_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=mag1037@pitt.edu
#SBATCH --chdir=/net/dali/home/mscbio/mag1037/work/classes/intro_to_ml/project/notebooks/2_regression/elasticnet

module load anaconda
source activate /net/dali/home/mscbio/mag1037/miniforge3/envs/R_env

echo "Running glmnet with Gencode embeddings"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

Rscript glmnet_script_param.R gencode

echo "End time: $(date)"