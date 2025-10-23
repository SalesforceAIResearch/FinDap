#!/bin/bash

# Set root directory (must be set via environment variable)
ROOT_DIR=${ROOT_DIR:-'YOUR_ROOT_DIR'}

export HF_DATASETS_CACHE='${ROOT_DIR}/dataset_cache'
export TRANSFORMERS_CACHE='${ROOT_DIR}/model_cache'
export CUDA_LAUNCH_BLOCKING=1
export HF_TOKEN='your_huggingface_token_here'

pip install trl
pip install faiss-gpu
pip install peft
pip install wandb
pip install -U flash-attn
pip install transformers==4.46.1
pip install -U accelerate
pip install typer
pip uninstall openai -y
python -m pip cache purge
pip install openai
pip list
nvidia-smi


accelerate launch --config_file '${ROOT_DIR}/SFR-Continual-Pretrain/yaml/fsdp_config_16.yaml' \
    ${ROOT_DIR}/SFR-Continual-Pretrain/posttrain.py \
    --use_trainer \
    --max_seq_length 2048 \
    --per_device_eval_batch_size 1 \
    --per_device_train_batch_size 1 \
    --instruction_mask \
    --isolate_attention \
    --use_flash_attention_2 \
    --additional_note _use_rpo \
    --idrandom 'dpo_cfa_sample_from_policy_stepwise' \
    --baseline 'comb' \
    --model_name 'zixuan/v1_1' \
    --cache_dir ${HF_DATASETS_CACHE} \
    --result_dir '${ROOT_DIR}/result' \
    --learning_rate 5e-7 \
    --checkpointing_steps 250 \
    --eval_steps 250 \
    --sequence_file ${ROOT_DIR}/SFR-Continual-Pretrain/sequences \
    --use_rpo
