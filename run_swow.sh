#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=100:00:00
#SBATCH --mem=64G
#SBATCH --partition=deeplearn
#SBATCH -A punim0478
#SBATCH -q gpgpudeeplearn
##SBATCH --gres=gpu:1
#SBATCH --gres=gpu:v100:1
##SBATCH --gres=gpu:v100sxm2:1
##SBATCH --gres=gpu:A100:1
##SBATCH --constraint=dlg4
#SBATCH -o log/230313_swow_share.log
##SBATCH -o log/221203_consistency_check_group.log
###SBATCH -o log/221025_WNLaMPro_full_vocab_Rl_IsA_rare.log 
##SBATCH --partition=gpgpu


# python -u anchored_prompts.py --config_file config/bless_swow.yaml --filter_anchors_flag   --filter_objects_flag --filter_objects_with_input  --debug
# python -u anchored_prompts.py --config_file config/bless_swow.yaml --filter_anchors_flag   --filter_objects_flag --filter_objects_with_input

# python -u anchored_prompts.py --config_file config/lm_diagnostic_extended.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input   --debug
# python -u anchored_prompts.py --config_file config/clsb_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors_swow --anchor_source LM  --swow_score_source  SWOWSimilar --debug
# python -u anchored_prompts.py --config_file config/clsb_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors --anchor_source LM   --debug
# python -u anchored_prompts.py --config_file config/clsb_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors --anchor_source LM  --add_swow_score --swow_score_source  SWOWSimilar --debug
# python -u anchored_prompts.py --config_file config/clsb_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors --anchor_source LM  --add_swow_score --swow_score_source  SWOWStrength --debug
#   --debug


#using pre-defined anchors from swow 
# python -u anchored_prompts.py --config_file config/clsb_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors --anchor_source LM --use_oracle_anchor
# python -u anchored_prompts.py --config_file config/clsb_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors_swow --anchor_source SWOW --use_oracle_anchor
# python -u anchored_prompts.py --config_file config/bless_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors --anchor_source LM --use_oracle_anchor
# python -u anchored_prompts.py --config_file config/bless_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors_swow --anchor_source SWOW --use_oracle_anchor
# python -u anchored_prompts.py --config_file config/eval_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors --anchor_source LM --use_oracle_anchor
# python -u anchored_prompts.py --config_file config/eval_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors_swow --anchor_source SWOW --use_oracle_anchor



#using pre-defined anchors from swow 


# python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors --anchor_source LM 
python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors --anchor_source LM --add_swow_filter --swow_score_source  ShareSWOWStrength
python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors --anchor_source LM --add_swow_filter --swow_score_source  ShareSWOWStrength
python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors --anchor_source LM --add_swow_filter --swow_score_source  ShareSWOWStrength

python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors --anchor_source LM --add_swow_filter --swow_score_source  ShareSWOWSimilar
python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors --anchor_source LM --add_swow_filter --swow_score_source  ShareSWOWSimilar
python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors --anchor_source LM --add_swow_filter --swow_score_source  ShareSWOWSimilar

# python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors --anchor_source LM --add_swow_score --swow_score_source  AddSWOWStrength
# python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors --anchor_source LM --add_swow_score --swow_score_source  AddSWOWSimilar

# python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors_swow --anchor_source SWOW --use_oracle_anchor --data_dir data/hypernymsuite/ALL/SWOWStrength/ 
# python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors_swow --anchor_source SWOW --use_oracle_anchor --data_dir data/hypernymsuite/ALL/SWOWSimilar/ 



# python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors_wordnet --anchor_source WordNet --only_wordnet_anchor --use_oracle_anchor --data_dir data/hypernymsuite/ALL/SWOWStrength/  --debug


# python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors_swow --anchor_source SWOW --swow_score_source OnlySWOWSimilar --use_oracle_anchor --data_dir data/hypernymsuite/ALL/SWOWStrength/  --only_swow_anchor --debug


# python -u anchored_prompts.py --config_file config/clsb_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors --anchor_source LM  --add_swow_score --swow_score_source  SWOWSimilar  
# python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors --anchor_source LM  --add_swow_score --swow_score_source  SWOWStrength --debug

# python -u anchored_prompts.py --config_file config/bless_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors --anchor_source LM  
# python -u anchored_prompts.py --config_file config/bless_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors --anchor_source LM  --add_swow_score --swow_score_source  SWOWSimilar  
# python -u anchored_prompts.py --config_file config/bless_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors --anchor_source LM  --add_swow_score --swow_score_source  SWOWStrength


# python -u anchored_prompts.py --config_file config/eval_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors --anchor_source LM  
# python -u anchored_prompts.py --config_file config/eval_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors --anchor_source LM  --add_swow_score --swow_score_source  SWOWSimilar  
# python -u anchored_prompts.py --config_file config/eval_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors --anchor_source LM  --add_swow_score --swow_score_source  SWOWStrength






# python -u anchored_prompts.py --config_file config/clsb_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors --anchor_source LM  --add_swow_score --swow_score_source  SWOWSimilar 

# python -u anchored_prompts.py --config_file config/leds.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --debug
# python -u anchored_prompts.py --config_file config/eval.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --debug
# python -u anchored_prompts.py --config_file config/shwartz.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --debug



# python -u anchored_prompts.py --config_file config/bless.yaml --filter_anchors_flag   --filter_objects_flag --filter_objects_with_input 
# python -u anchored_prompts.py --config_file config/lm_diagnostic_extended.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  
# python -u anchored_prompts.py --config_file config/clsb.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input 
# python -u anchored_prompts.py --config_file config/leds.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input 
# python -u anchored_prompts.py --config_file config/eval.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input 
# python -u anchored_prompts.py --config_file config/shwartz.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input 
