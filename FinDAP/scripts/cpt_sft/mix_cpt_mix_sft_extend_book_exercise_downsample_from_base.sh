#!/bin/bash

# Set root directory (must be set via environment variable)
ROOT_DIR=${ROOT_DIR:-'YOUR_ROOT_DIR'}

export HF_DATASETS_CACHE='${ROOT_DIR}/dataset_cache'
export TRANSFORMERS_CACHE='${ROOT_DIR}/model_cache'
export HF_TOKEN='your_huggingface_token_here'

pip install trl
pip install faiss-gpu
pip install peft
pip install wandb
pip install -U flash-attn
pip install -U transformers 
pip install -U accelerate
pip install -U datasets
pip install typer
pip uninstall openai -y
python -m pip cache purge
pip install openai
pip list


# export WANDB_PROJECT="posttrain_mix_sft_mix_full"  # name your W&B project
# export WANDB_LOG_MODEL="false"  # log all model checkpoints
export CUDA_LAUNCH_BLOCKING=1
export BASH_SCRIPT_NAME=$(basename "$0")


accelerate launch --config_file '${ROOT_DIR}/SFR-Continual-Pretrain/yaml/fsdp_config_16.yaml' \
    ${ROOT_DIR}/SFR-Continual-Pretrain/posttrain.py \
    --use_trainer \
    --per_device_eval_batch_size 1 \
    --max_seq_length 8000 \
    --per_device_train_batch_size 1 \
    --instruction_mask \
    --isolate_attention \
    --use_flash_attention_2 \
    --downsample \
    --additional_note _fineweb_downsample_from_base \
    --idrandom 'dapt_mix_sft_mix_full_extend_exercise_book' \
    --baseline 'comb' \
    --model_name 'meta-llama/Meta-Llama-3-8B-Instruct' \
    --cache_dir ${HF_DATASETS_CACHE} \
    --result_dir '${ROOT_DIR}/result' \
    --learning_rate 5e-6 \
    --checkpointing_steps 1000 \
    --eval_steps 5000 \
    --sequence_file ${ROOT_DIR}/SFR-Continual-Pretrain/sequences


