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
#SBATCH -o log/230308_wordnet_alone_2K.log
##SBATCH --partition=gpgpu


# python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors_wordnet --anchor_source WordNet --only_wordnet_anchor --use_oracle_anchor --data_dir data/hypernymsuite/ALL/SWOWStrength/ --dataset ALLSWOW 
python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors_wordnet --anchor_source WordNet --only_wordnet_anchor --use_oracle_anchor --data_dir data/hypernymsuite/ALL/swow_rw/ --dataset ALLSWOW 
python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors_wordnet --anchor_source WordNet --only_wordnet_anchor --use_oracle_anchor --data_dir data/hypernymsuite/ALL/swow_rw/ --dataset ALLSWOW 
python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors_wordnet --anchor_source WordNet --only_wordnet_anchor --use_oracle_anchor --data_dir data/hypernymsuite/ALL/swow_rw/ --dataset ALLSWOW 

# python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors_wordnet --anchor_source WordNet --add_wordnet_filter  --data_dir data/hypernymsuite/ALL/SWOWStrength/ 
# python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors_wordnet --anchor_source WordNet --add_wordnet_path_score  --data_dir data/hypernymsuite/ALL/SWOWStrength/



# python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors_wordnet --anchor_source SWOW --only_swow_anchor --use_oracle_anchor --data_dir data/hypernymsuite/ALL/SWOWStrength/ --swow_score_source  AddSWOWStrength --debug
# python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors_wordnet --anchor_source SWOW --add_swow_filter  --data_dir data/hypernymsuite/ALL/SWOWStrength/  --swow_score_source  AddSWOWStrength --debug
# python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors_wordnet --anchor_source SWOW --add_swow_score  --data_dir data/hypernymsuite/ALL/SWOWStrength/  --swow_score_source  AddSWOWStrength --debug





