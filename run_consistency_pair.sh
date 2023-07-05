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
#SBATCH -o log/230220_consistency_pair_swow_2K.log
##SBATCH -o log/230119_consistency_pair_all_datasets.log
##SBATCH -o log/221211_bless_anchor_coordiante.log
##SBATCH -o log/221209_lmdai_anchor_num_ablation.log
##SBATCH -o log/221209_bless_anchor_scorers_ablation.log
##SBATCH -o log/221203_consistency_check_group.log
###SBATCH -o log/221025_WNLaMPro_full_vocab_Rl_IsA_rare.log 
##SBATCH --partition=gpgpu
##SBATCH --gres=gpu:p100:1


# python -u consistency_check.py  'data/BLESS/consistency/' False

#run ACL submission
# python -u consistency_check.py  'data/clsb/singular/consistency/' False
# python -u consistency_check.py  'data/hypernymsuite/BLESS/consistency/' False
# python -u consistency_check.py  'data/hypernymsuite/EVAL/consistency/' False
# python -u consistency_check.py  'data/hypernymsuite/LEDS/consistency/' False
# python -u consistency_check.py  'data/lm_diagnostic_extended/singular/consistency/' False
# python -u consistency_check.py  'data/hypernymsuite/SHWARTZ/consistency/' False

# run with SWOW

python -u consistency_check.py  'data/hypernymsuite/ALL/AddSWOWStrength/consistency_rw/' False
python -u consistency_check.py  'data/hypernymsuite/ALL/AddSWOWSimilar/consistency_rw/' False
python -u consistency_check.py  'data/hypernymsuite/ALL/LM/consistency_rw/' False
python -u consistency_check.py  'data/hypernymsuite/ALL/ShareSWOWStrength/consistency_rw/' False
python -u consistency_check.py  'data/hypernymsuite/ALL/ShareSWOWSimilar/consistency_rw/' False


# python -u consistency_check.py  'data/clsb/singular/consistency/' False
# python -u consistency_check.py  'data/hypernymsuite/BLESS/consistency/' False
# python -u consistency_check.py  'data/hypernymsuite/EVAL/consistency/' False
# python -u consistency_check.py  'data/hypernymsuite/LEDS/consistency/' False
# python -u consistency_check.py  'data/lm_diagnostic_extended/singular/consistency/' False
# python -u consistency_check.py  'data/hypernymsuite/SHWARTZ/consistency/' False
