#!/usr/bin/env python
# coding=utf-8
import datasets
import logging
import os
import random
import torch
import transformers
from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs
from datasets import Dataset, DatasetDict, concatenate_datasets
from torch.utils.data import DataLoader
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

from approaches.posttrain import Posttrain
from config import parsing_posttrain
from utils.prepare import PosttrainPreparer
import utils

def main():
    args = parsing_posttrain()
    prepare_approach = PosttrainPreparer(args)

    prepare_approach.prepare_sequence()
    prepare_approach.load_model_tok()

    if args.convert_checkpoint_to_ckpt:
        utils.model.convert_checkpoint_to_ckpt(
            tok = prepare_approach.tok,
            model = prepare_approach.model,
            torch_dtype = args.torch_dtype,
            output_dir = prepare_approach.args.output_dir,
            baseline = prepare_approach.args.baseline,
            global_step = prepare_approach.args.eval_step,
            save_dir = prepare_approach.args.save_folder_name,
            state_dir = prepare_approach.args.cache_dir,
                        )
    elif args.use_trainer:
        prepare_approach.prepare_posttrain()
        appr_train = Posttrain(args)

        if 'dpo' in args.idrandom:
            appr_train.dpo_trainer(
                model = prepare_approach.model, 
                ref_model = prepare_approach.ref_model,
                training_args = prepare_approach.training_args,
                train_dataset = prepare_approach.train_datasets, 
                eval_dataset = prepare_approach.eval_datasets,
                focus_dataset = prepare_approach.focus_dataset,
                tokenizer = prepare_approach.tok)
        else:
            appr_train.sft_trainer(
                model=prepare_approach.model,
                tokenizer = prepare_approach.tok,
                training_args = prepare_approach.training_args,
                train_dataset = prepare_approach.train_datasets, 
                focus_dataset = prepare_approach.focus_dataset,
                is_eval_only = args.eval_only,
                eval_dataset = prepare_approach.eval_datasets, 
                sft_eval_dataset = prepare_approach.sft_datasets_valid,
                dapt_eval_dataset = prepare_approach.dapt_datasets_valid
                )

    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()
