#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=100:00:00
#SBATCH --mem=32G
#SBATCH --partition=deeplearn
#SBATCH -A punim0478
#SBATCH -q gpgpudeeplearn
##SBATCH --gres=gpu:1
#SBATCH --gres=gpu:v100:1
##SBATCH --gres=gpu:v100sxm2:1
##SBATCH --gres=gpu:A100:1
##SBATCH --constraint=dlg4
#SBATCH -o log/221206_clsb_wn.log
##SBATCH -o log/221203_consistency_check_group.log
###SBATCH -o log/221025_WNLaMPro_full_vocab_Rl_IsA_rare.log 
##SBATCH --partition=gpgpu
##SBATCH --gres=gpu:p100:1

# python -u consistency_check.py False 
# python -u consistency_check_group.py True
# python -u consistency_check_group.py False
# python -u evaluate_anchor_quality.py 

python -u result_parse.py