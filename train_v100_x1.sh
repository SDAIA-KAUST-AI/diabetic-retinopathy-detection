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
#SBATCH --mem=100G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=10

module purge
source activate retinopathy

echo Running four user "${USER}"

. ./tag.sh

PYTHONPATH=. python train.py --tag=\${SLURM_JOB_ID}_V100_x1_\${TAG}

exit 0
EOT