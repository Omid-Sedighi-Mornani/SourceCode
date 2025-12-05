#!/bin/bash

# Define parameter arrays
models=("hmm" "vdhmm")
seeds=(43 123)
states=(2 3 4)

# Loop through all combinations and submit jobs
for model in "${models[@]}"; do
    for seed in "${seeds[@]}"; do
        for state in "${states[@]}"; do
            sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${model}_s${seed}_st${state}
#SBATCH --partition=main
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=${model}_s${seed}_st${state}-%j.out
#SBATCH --error=${model}_s${seed}_st${state}-%j.err

cd ~/SourceCode
uv run scripts/main.py --model ${model} --seed ${seed} --state ${state} --chains 4 --parallel_chains 4 --threads_per_chain 4
EOF
            echo "Submitted job: model=${model}, seed=${seed}, state=${state}"
        done
    done
done