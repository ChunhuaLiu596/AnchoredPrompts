#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=100:00:00
#SBATCH --mem=64G
#SBATCH --partition=deeplearn
#SBATCH -A punim0478
#SBATCH -q gpgpudeeplearn
##SBATCH --gres=gpu:1
##SBATCH --gres=gpu:v100:1
#SBATCH --gres=gpu:v100sxm2:1
##SBATCH --gres=gpu:A100:1
##SBATCH --constraint=dlg4
#SBATCH -o log/230314_only_swow_2k_anchors_multiple_runs.log
##SBATCH -o log/221203_consistency_check_group.log
###SBATCH -o log/221025_WNLaMPro_full_vocab_Rl_IsA_rare.log 
##SBATCH --partition=gpgpu

python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors_swow --anchor_source SWOW  --only_swow_anchor --use_oracle_anchor --swow_score_source  OnlySWOWStrength 
python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors_swow --anchor_source SWOW  --only_swow_anchor --use_oracle_anchor --swow_score_source  OnlySWOWStrength 
python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors_swow --anchor_source SWOW  --only_swow_anchor --use_oracle_anchor --swow_score_source  OnlySWOWStrength 

python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors_swow --anchor_source SWOW  --only_swow_anchor --use_oracle_anchor --swow_score_source  OnlySWOWSimilar 
python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors_swow --anchor_source SWOW  --only_swow_anchor --use_oracle_anchor --swow_score_source  OnlySWOWSimilar 
python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors_swow --anchor_source SWOW  --only_swow_anchor --use_oracle_anchor --swow_score_source  OnlySWOWSimilar 

# python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors --anchor_source LM --add_swow_filter --swow_score_source  ShareSWOWStrength 
# python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors --anchor_source LM --add_swow_filter --swow_score_source  ShareSWOWStrength 
# python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors --anchor_source LM --add_swow_filter --swow_score_source  ShareSWOWStrength 

# python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors --anchor_source LM --add_swow_filter --swow_score_source  ShareSWOWSimilar
# python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors --anchor_source LM --add_swow_filter --swow_score_source  ShareSWOWSimilar
# python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors --anchor_source LM --add_swow_filter --swow_score_source  ShareSWOWSimilar


# python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors --anchor_source LM --add_swow_filter --swow_score_source  AddSWOWStrength 
# python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors --anchor_source LM --add_swow_filter --swow_score_source  AddSWOWStrength 
# python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors --anchor_source LM --add_swow_filter --swow_score_source  AddSWOWStrength 

# python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors --anchor_source LM --add_swow_filter --swow_score_source  AddSWOWSimilar
# python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors --anchor_source LM --add_swow_filter --swow_score_source  AddSWOWSimilar
# python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors --anchor_source LM --add_swow_filter --swow_score_source  AddSWOWSimilar
