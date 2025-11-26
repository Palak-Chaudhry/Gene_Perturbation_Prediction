#!/bin/bash
#SBATCH --job-name=xgboost_tss
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/xgboost_tss_%j.out
#SBATCH --error=logs/xgboost_tss_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=mag1037@pitt.edu
#SBATCH --chdir=/net/dali/home/mscbio/mag1037/work/classes/intro_to_ml/project/notebooks/2_regression/xgboost

# Initialize conda directly from miniforge3
eval "$(/net/dali/home/mscbio/mag1037/miniforge3/bin/conda shell.bash hook)"
conda activate chikina_env

echo "Running XGBoost with TSS embeddings"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

python XGBRegressor_Enformer_param.py tss

echo "End time: $(date)"
