import argparse
import copy
import datasets
import logging
import math
import os
import random
import shutil
import sys
import torch
import transformers
from accelerate import Accelerator, DistributedType
from tqdm.auto import tqdm
from transformers import (
    MODEL_MAPPING,
    AdamW,
    AutoTokenizer,
    AutoConfig,
    RobertaTokenizer,
    BertTokenizer,
    DataCollatorForLanguageModeling,
    get_scheduler,
    SchedulerType,
    set_seed,
)




def share_argument(parser):
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default='roberta-base',
        help="Path to pretrained model or model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )

    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=111, help="A seed for reproducible training.")
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=16,
        help="The number of processes to use for the preprocessing.",
    )

    parser.add_argument("--baseline",
                        type=str,
                        help="The supported baselines.")
    parser.add_argument(
        "--overwrite_cache",action='store_true', help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_train_samples",
        type=float,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of training examples to this "
             "value if set.",
    )
    parser.add_argument("--idrandom", type=str, help="which sequence to use")
    parser.add_argument("--ntasks", type=int, help="total number of tasks")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    # parser.add_argument("--pt_task", type=int, help="task id")
    parser.add_argument("--ft_task", type=int, help="task id")
    parser.add_argument('--log_dir', default='../', type=str)
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--model_parallel", action="store_false", help="use parallel by default")
    parser.add_argument("--cache_dir", type=str)
    parser.add_argument("--result_dir", type=str)
    parser.add_argument("--root_dir", type=str, default="YOUR_ROOT_DIR", 
                        help="Root directory for the project")
    parser.add_argument("--torch_dtype", default='bf16', type=str)
    parser.add_argument("--sequence_file", type=str, help="which sequence to use")
    parser.add_argument("--distributed_mode", type=str, default="fsdp")
    parser.add_argument("--writer_batch_size", type=int, default=1000)
    parser.add_argument("--hf_dataset_baatch_size", type=int, default=1000)
    parser.add_argument("--lora_type", default='', type=str)
    parser.add_argument("--combined_dataset", default='', type=str)
    parser.add_argument("--additional_note", default='', type=str)
    parser.add_argument(
        "--use_trainer",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )

    parser.add_argument(
        "--downsample",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )

    parser.add_argument(
        "--is_compute_fisher",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )

    parser.add_argument(
        "--upsample",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )



    #LoRa hyper-parameters:
    parser.add_argument("--rank", default=8, type=int)
    parser.add_argument("--target_rank", default=8, type=int)
    parser.add_argument("--init_rank", default=12, type=int)

    parser.add_argument("--lora_alpha", type=float)
    parser.add_argument("--lora_dropout", type=float)
    parser.add_argument("--layers", type=str)
    parser.add_argument("--target_modules", type=str)
    parser.add_argument("--eval_steps", type=int)

    return parser



def parsing_posttrain():
    parser = argparse.ArgumentParser(description="Posttrain a transformers model")
    parser = share_argument(parser)
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--eval_step", default=0, type=int)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--convert_checkpoint_to_ckpt",
        action="store_true",
        help="Convert FSDP checkpoint_to_ckpt",
    )
    parser.add_argument(
        "--save_folder_name",
        type=str,
        help="path to save the tensorboard file",
    )
    parser.add_argument(
        "--eval_ood_perf",
        action="store_true",
        help="Convert FSDP checkpoint_to_ckpt",
    )
    parser.add_argument("--id_seq", type=int, help="task id")
    parser.add_argument("--id_pt_task", type=int, help="task id")
    parser.add_argument("--ood_seq", type=int, help="task id")
    parser.add_argument("--ood_pt_task", type=int, help="task id")
    parser.add_argument(
        "--cut_train_as_eval",
        action="store_true",
        help="Convert FSDP checkpoint_to_ckpt",
    )

    parser.add_argument(
        "--id_only",
        action="store_true",
        help="Convert FSDP checkpoint_to_ckpt",
    )

    parser.add_argument(
        "--ood_only",
        action="store_true",
        help="Convert FSDP checkpoint_to_ckpt",
    )
    parser.add_argument(
        "--stack_lora",
        action="store_true",
        help="Convert FSDP checkpoint_to_ckpt",
    )

    parser.add_argument(
        "--instruction_mask",
        action="store_true",
        help="Convert FSDP checkpoint_to_ckpt",
    )

    parser.add_argument(
        "--isolate_attention",
        action="store_true",
        help="Convert FSDP checkpoint_to_ckpt",
    )

    parser.add_argument(
        "--use_flash_attention_2",
        action="store_true",
        help="Convert FSDP checkpoint_to_ckpt",
    )

    parser.add_argument(
        "--is_test",
        action="store_true",
        help="Convert FSDP checkpoint_to_ckpt",
    )


    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Convert FSDP checkpoint_to_ckpt",
    )

    parser.add_argument(
        "--use_rpo",
        action="store_true",
        help="Convert FSDP checkpoint_to_ckpt",
    )
   
    args = parser.parse_args()

    return args
