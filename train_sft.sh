source ~/.bashrc

export WANDB_PROJECT="TruthRL"

conda activate truthrl-openr1

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file training/open-r1/recipes/accelerate_configs/zero3.yaml training/open-r1/src/open_r1/sft.py \
    --config training/open-r1/recipes/Llama3.1-8B-Instruct/sft/config_sft.yaml