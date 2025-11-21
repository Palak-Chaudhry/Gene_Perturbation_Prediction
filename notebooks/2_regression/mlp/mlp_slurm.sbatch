#!/bin/bash
#SBATCH --job-name="mlp_par"
#SBATCH --output="mlp_par_%j.log"
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=2000M
#SBATCH -t 2-00:00:00
#SBATCH --mail-user=pac271@pitt.edu
#SBATCH --mail-type=ALL

# -- Load your environment modules
module load anaconda3/2022.10

# -- Optionally, load GCC if needed for some packages (uncomment if needed for your environment)
#module load gcc/13.2.1-p20240113 

# -- Activate your Conda environment
source activate loveslide_env         

export LD_LIBRARY_PATH=/ocean/projects/cis240075p/pchaudhr/.conda/envs/loveslide_env/lib:$LD_LIBRARY_PATH

# -- Print some diagnostic info
echo "Running on host: $(hostname)"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# -- Set Python warnings to display
export PYTHONWARNINGS="default"

# -- Run your script
python3 /ocean/projects/cis240075p/pchaudhr/Gene_Perturbation_Prediction/notebooks/2_regression/mlp/mlp.py

echo "job ended: $(date)"

