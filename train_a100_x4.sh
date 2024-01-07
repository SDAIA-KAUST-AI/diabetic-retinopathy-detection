#!/bin/bash
sbatch <<EOT
#!/bin/bash

#SBATCH -N 1
#SBATCH -J SDAIA_DR
#SBATCH -o slurm_logs/output.%J.out
#SBATCH -e slurm_logs/output.%J.err
#SBATCH --mail-user=${USER}@kaust.edu.sa
#SBATCH --mail-type=END,FAIL
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=64

module purge
source activate retinopathy

echo Running four user "${USER}"

. ./tag.sh

PYTHONPATH=. python train.py --tag=\${SLURM_JOB_ID}_A100_x4_\${TAG}

exit 0
EOT