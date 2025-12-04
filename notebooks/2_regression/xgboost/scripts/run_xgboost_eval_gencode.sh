#!/bin/bash
#SBATCH --job-name=xgb_eval_gencode
#SBATCH --partition=dept_cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=scripts/logs/xgb_eval_gencode_%j.out
#SBATCH --error=scripts/logs/xgb_eval_gencode_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=mag1037@pitt.edu
#SBATCH --chdir=/net/dali/home/mscbio/mag1037/work/classes/intro_to_ml/project/notebooks/2_regression/xgboost/scripts

# Initialize conda directly from miniforge3
eval "$(/net/dali/home/mscbio/mag1037/miniforge3/bin/conda shell.bash hook)"
conda activate chikina_env

echo "Running XGBoost evaluation with Gencode embeddings"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

python XGBRegressor_Enformer_eval_with_best_params.py gencode

echo "End time: $(date)"
