#!/bin/bash

# Training script for models using ORIGINAL paper indices from indices.Rdata
# This script submits SLURM jobs to train HMM and VDHMM models with the
# exact train/calibration/eval splits used in the original paper.

# Define parameter arrays
models=("hmm" "vdhmm")
states=(2 3 4)

# Note: seed parameter is still required but will be ignored when using --use-original-indices
# We use seed=42 as a placeholder
seed=42

# Loop through all combinations and submit jobs
for model in "${models[@]}"; do
    for state in "${states[@]}"; do
        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${model}_orig_st${state}
#SBATCH --partition=main
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=${model}_original_indices_st${state}-%j.out
#SBATCH --error=${model}_original_indices_st${state}-%j.err

cd ~/SourceCode
uv run scripts/main.py --model ${model} --seed ${seed} --state ${state} --chains 4 --parallel-chains 4 --threads-per-chain 4 --use-original-indices
EOF
        echo "Submitted job: model=${model}, state=${state}, using ORIGINAL paper indices"
    done
done

echo ""
echo "All jobs submitted!"
echo "Models: ${models[@]}"
echo "States: ${states[@]}"
echo "Using: Original paper indices from indices.Rdata"
echo ""
echo "Output files will be saved as:"
echo "  - Processed data: processed_data_for_hmm_training_original_indices.pkl"
echo "  - Models: {model}_{state}_original_indices_cmdstan.pkl"
