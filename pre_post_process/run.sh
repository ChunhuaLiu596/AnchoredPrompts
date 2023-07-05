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
#SBATCH -o log/230301_cohyponyms_score.log 
##SBATCH -o log/230103_dap_single_pattern.log 
##SBATCH -o log/221217_baseline_single_pattern.log 
##SBATCH -o log/221217_baseline_word2vec.log 
##SBATCH -o log/221129_consistency_wpcpt.log
##SBATCH --partition=gpgpu
##SBATCH --gres=gpu:p100:1

cd script
# python -u static_embedding_baseline.py 
# python -u baseline_single_patterns.py

# jupyter nbconvert probe_dap_single_pattern.ipynb --to script
# python -u probe_dap_single_pattern.py


# python -u util_swow.py similar
#python -u util_swow.py strength

python -u add_path_score_for_cohyponyms.py 