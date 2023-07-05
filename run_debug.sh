#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=100:00:00
#SBATCH --mem=32G
#SBATCH --partition=deeplearn
#SBATCH -A punim0478
#SBATCH -q gpgpudeeplearn
##SBATCH --gres=gpu:1
##SBATCH --gres=gpu:v100:1
#SBATCH --gres=gpu:v100sxm2:1
##SBATCH --gres=gpu:A100:1
##SBATCH --constraint=dlg4
##SBATCH -o log/221213_consistency_group_shwartz.log
#SBATCH -o log/230117_bless_anchor_coordiante.log
##SBATCH -o log/221211_bless_anchor_coordiante.log
##SBATCH -o log/221209_lmdai_anchor_num_ablation.log
##SBATCH -o log/221209_bless_anchor_scorers_ablation.log
##SBATCH -o log/221203_consistency_check_group.log
###SBATCH -o log/221025_WNLaMPro_full_vocab_Rl_IsA_rare.log 
##SBATCH --partition=gpgpu
##SBATCH --gres=gpu:p100:1


# python -u consistency_check.py  'data/BLESS/consistency/' True

# python -u consistency_check.py  'data/clsb/consistency/' False
# python -u consistency_check.py  'data/hypernymsuite/BLESS/consistency/' False
# python -u consistency_check.py  'data/hypernymsuite/EVAL/consistency/' False
# python -u consistency_check.py  'data/hypernymsuite/LEDS/consistency/' False
# python -u consistency_check.py  'data/lm_diagnostic_extended/consistency/' False
# python -u consistency_check.py  'data/hypernymsuite/SHWARTZ/consistency/' False


# python -u consistency_check.py  False 
# False 
# False 
# python -u consistency_check_group.py True
# python -u consistency_check_group.py False
# python -u evaluate_anchor_quality.py 

## Inspect the power of true anchors
# python -u anchored_prompts.py --config_file config/bless_hyper_oracle.yaml --filter_objects_flag  --filter_objects_with_input

## Ablation study on the affects on anchor scorer
python -u anchored_prompts.py --config_file config/bless_hyper.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input
python -u anchored_prompts.py --config_file config/bless_hyper.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input --add_cpt_score
# python -u anchored_prompts.py --config_file config/bless_hyper.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input --add_wordnet_path_score


## Ablation study on anchor number 
# python -u anchored_prompts.py --config_file config/lm_diagnostic_extended.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --add_wordnet_path_score --top_k_anchors 1 --max_anchor_num_list 10
# python -u anchored_prompts.py --config_file config/lm_diagnostic_extended.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --add_wordnet_path_score --top_k_anchors 5  --max_anchor_num_list 10
# python -u anchored_prompts.py --config_file config/lm_diagnostic_extended.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --add_wordnet_path_score --top_k_anchors 10  --max_anchor_num_list 20
# python -u anchored_prompts.py --config_file config/lm_diagnostic_extended.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --add_wordnet_path_score --top_k_anchors 20  --max_anchor_num_list 20
# python -u anchored_prompts.py --config_file config/lm_diagnostic_extended.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --add_wordnet_path_score --top_k_anchors 50  --max_anchor_num_list 50


# python -u anchored_prompts.py --config_file config/bless_coord.yaml --filter_anchors_flag   --filter_objects_flag
# python -u anchored_prompts.py --config_file config/bless_coord.yaml --filter_anchors_flag   --filter_objects_flag  --add_wordnet_path_score
# python -u anchored_prompts.py --config_file config/bless_coord.yaml --filter_anchors_flag   --filter_objects_flag  --add_cpt_score
# --debug
# --debug
# python -u anchored_prompts.py --config_file config/clsb.yaml --filter_anchors_flag   --filter_objects_flag  --add_wordnet_path_score
# python -u anchored_prompts.py --config_file config/leds.yaml --filter_anchors_flag   --filter_objects_flag  --add_wordnet_path_score
# python -u anchored_prompts.py --config_file config/shwartz.yaml --filter_anchors_flag   --filter_objects_flag  --add_wordnet_path_score
# python -u anchored_prompts.py --config_file config/eval.yaml --filter_anchors_flag   --filter_objects_flag  --add_wordnet_path_score
# --debug
#  --debug
# --add_wordnet_path_score
# --debug

# python -u restu