#!/bin/bash
#SBATCH --job-name=enet_eval_tss
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=scripts/logs/enet_eval_tss_%j.out
#SBATCH --error=scripts/logs/enet_eval_tss_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=mag1037@pitt.edu
#SBATCH --chdir=/net/dali/home/mscbio/mag1037/work/classes/intro_to_ml/project/analysis/2_regression/elasticnet/scripts

module load anaconda
source activate /net/dali/home/mscbio/mag1037/miniforge3/envs/R_env

echo "Running ElasticNet evaluation with TSS embeddings"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

Rscript glmnet_script_eval_with_best_params.R tss

echo "End time: $(date)"
