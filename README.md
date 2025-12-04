#SBATCH --gres=gpu:2 (fall GPUs gebraucht)

squeue --format="%.9i %22j %8u %8T %.7M %.13l %6D %20R %4C %10b"