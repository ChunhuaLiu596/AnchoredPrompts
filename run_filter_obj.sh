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
##SBATCH -o log/221129_consistency_wpcpt.log
##SBATCH -o log/221129_lm_bless_cpt_filter_objects_flag.log
##SBATCH -o log/221130_lm_clsb_cpt.log
#SBATCH -o log/221130_all_datasets_filter_anchor_wn_with_filter_target_open_vocab.log
##SBATCH -o log/221129_lm_diagnostic_extended_plural_cpt.log
##SBATCH -o log/221125_lm_diagnostic_extended_plural_anchor_raw_9Fischler_with_norm.log 
##SBATCH -o log/221125_lm_diagnostic_extended_singular_anchor_raw_9Fischler_with_norm.log 
##SBATCH -o log/221124_lm_diagnostic_extended_plural_anchor_raw.log 
##SBATCH -o log/221123_clsb_sgpl_anchor_add_wordnet_path_cpt_score.log 
##SBATCH -o log/221121_clsb_WNLaMPro.log 
##SBATCH -o log/221121_clsb_sgpl_anchor_mining_patterns.log 
##SBATCH -o log/221123_clsb_sgpl_anchor_add_wordnet_path_score.log 
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
# python -u fill_in_anchor_word_property_lama.py
# python scripts/extract_topk_results.py
# python -u test_gpu.py


#lm_diagnostic_extended
python -u anchored_prompts.py  --config_file config/lm_diagnostic_extended.yaml --filter_anchors_flag  --add_wordnet_path_score   --filter_objects_flag
python -u anchored_prompts.py --config_file config/clsb.yaml --filter_anchors_flag  --add_wordnet_path_score   --filter_objects_flag
python -u anchored_prompts.py --config_file config/bless.yaml --filter_anchors_flag  --add_wordnet_path_score   --filter_objects_flag
python -u anchored_prompts.py --config_file config/leds.yaml --filter_anchors_flag  --add_wordnet_path_score   --filter_objects_flag
python -u anchored_prompts.py --config_file config/eval.yaml --filter_anchors_flag  --add_wordnet_path_score   --filter_objects_flag
python -u anchored_prompts.py --config_file config/shwartz.yaml --filter_anchors_flag  --add_wordnet_path_score   --filter_objects_flag

# python -u anchored_prompts.py True config/clsb.yaml
#python -u anchored_prompts.py False config/clsb.yaml
# python -u anchored_prompts.py True config/shwartz.yaml
# python -u anchored_prompts.py False config/eval.yaml
# python -u anchored_prompts.py False config/leds.yaml
# python -u anchored_prompts.py False config/bless.yaml
# python -u anchored_prompts.py False
# python -u test_device.py True config/lm_diagnostic_extended.yaml 
# python -u consistency_check.py False 

# python -u test_config.py  --filter_anchors_flag  --filter_objects_flag  --add_wordnet_path_score
# python -u test_config.py 