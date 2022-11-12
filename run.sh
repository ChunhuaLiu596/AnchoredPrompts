#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=100:00:00
#SBATCH --mem=32G
#SBATCH --partition=deeplearn
#SBATCH -A punim0478
#SBATCH -q gpgpudeeplearn
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:v100:1
##SBATCH --gres=gpu:v100sxm2:1
##SBATCH --gres=gpu:A100:1
##SBATCH --constraint=dlg4
#SBATCH -o log/221025_LAMA_full_vocab_Rl_5rels.log 
##SBATCH -o log/221025_CLSB_full_vocab_Rl_debug.log 
###SBATCH -o log/221025_WNLaMPro_full_vocab_Rl_IsA_rare.log 
##SBATCH --partition=gpgpu
##SBATCH --gres=gpu:p100:1

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/chunhua/.conda/envs/lama37/lib
# export PYTHONPATH=`pwd` 

#module load fosscuda/2020b
### module spider pytorch
#module load pytorch/1.9.0-python-3.8.6

# python -u fill_in_anchor_word_property_new.py
python -u fill_in_anchor_word_property_lama.py
# python scripts/extract_topk_results.py
# python -u test_gpu.py
