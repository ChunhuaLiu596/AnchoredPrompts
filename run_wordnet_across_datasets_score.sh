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
#SBATCH -o log/230306_add_wordnet_score_six_datasets.log
##SBATCH --partition=gpgpu


python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors_wordnet --anchor_source WordNet --add_wordnet_path_score --data_dir 'data/hypernymsuite/BLESS/' --dataset BLESS 
python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors_wordnet --anchor_source WordNet --add_wordnet_path_score --data_dir 'data/lm_diagnostic_extended/singular/' --dataset DIAG 
python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors_wordnet --anchor_source WordNet --add_wordnet_path_score --data_dir 'data/clsb/singular/' --dataset CLSB 
python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors_wordnet --anchor_source WordNet --add_wordnet_path_score --data_dir 'data/hypernymsuite/EVAL/' --dataset EVAL 
python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors_wordnet --anchor_source WordNet --add_wordnet_path_score --data_dir 'data/hypernymsuite/LEDS/' --dataset LEDS 
python -u anchored_prompts.py --config_file config/all_swow.yaml --filter_anchors_flag   --filter_objects_flag  --filter_objects_with_input  --anchor_col_prefix subj_anchors_wordnet --anchor_source WordNet --add_wordnet_path_score --data_dir 'data/hypernymsuite/SHWARTZ/' --dataset SHWARTZ 