
import datasets
import json
import logging
import logging
import numpy as np
import os
import os.path
import random
import torch
import yaml
from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs, InitProcessGroupKwargs
from datasets import Dataset, DatasetDict, concatenate_datasets
from pathlib import Path
from time import time
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import GPT2TokenizerFast, GPT2Tokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import Optional, Union, List, Tuple, Dict
from transformers import LlamaTokenizer
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, default_data_collator
from torch.utils.data import DataLoader
from datetime import timedelta
from tokenizers.processors import TemplateProcessing
# import wandb

from dataloader.data import get_dataset
import utils
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, AdaLoraConfig, TaskType, get_peft_model_state_dict, set_peft_model_state_dict, LoraConfig, PeftModel
import ast
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
import transformers
from multiprocess import set_start_method
from accelerate import PartialState
# wandb.init(project="Continual Pre-training")


LOG = logging.getLogger(__name__)

def get_handler(path, log_name, accelerator):
    log_file_path = os.path.join(path, log_name)
    try:
        if not os.path.exists(path):
            LOG.info("We are creating the logger files")
            os.makedirs(path)
    except:
        pass

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    return file_handler, stream_handler


def make_logs(log_dir, accelerator):
    f_h, s_h = get_handler(log_dir, log_name='run.log', accelerator=accelerator)
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)


def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class Preparer:
    """Base editor for all methods"""

    def __init__(self, args):

        self.args = args

        LOG.info("Instantiating see, log, accelerate and seed")

        seed_everything(self.args.seed)

        self.args.device = torch.device("cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")

        assert self.args.use_trainer # I will let trainer handle the accelerator

        if self.args.distributed_mode == 'deepspeed':
            self.device_map = None
        elif self.args.distributed_mode == 'fsdp':
            # self.device_map = 'cpu'  # CPU for now, later accelerator will put it to appropriate GPU
            # self.device_map = 'auto'  # CPU for now, later accelerator will put it to appropriate GPU
            # self.device_map = self.args.device
            # self.device_map = device_map={"": PartialState().process_index}
            self.device_map = None


        else:
            raise NotImplementedError

        if self.args.torch_dtype == 'bf16':
            self.torch_dtype = torch.bfloat16
        elif self.args.torch_dtype == 'fp16':
            self.torch_dtype = torch_dtype.fp16
        else:
            self.torch_dtype = torch.float32
            


        self.dapt_datasets = ['finance_unsup', 'dialog_studio_sup', 'dm_mathematics_sup', 'p3_supernatural_sup','various_supervised_sup','apex_instruct_for_annealing_sup', 'book_fineweb_unsup']

        self.sft_datasets = ['bopang_sup', 'fingpt_finred_sup', 'fingpt_ner_cls_sup', 'fingpt_headline_cls_sup', 'fingpt_sentiment_cls_sup', 'fingpt_sentiment_train_sup', 'fingpt_ner_sup', 'cfa_extracted_exercise_sup', 'trade_the_event_sup','fingpt_convfinqa_sup','flare_finqa_sup','fingpt_fiqa_qa_sup','sujet_finance_instruct_sup']

        self.dpo_datasets_in = ['cfa_extracted_exercise_sup_dpo', 'sujet_finance_instruct_sup_dpo', 'fingpt_finred_sup_dpo', 'fingpt_ner_cls_sup_dpo', 'fingpt_ner_sup_dpo', 'trade_the_event_sup_dpo', 'fingpt_convfinqa_sup_dpo', 'flare_finqa_sup_dpo', 'fingpt_fiqa_qa_sup_dpo',
        'cfa_extracted_exercise_sup_sample_from_policy_v1.1_dpo',
        'cfa_extracted_exercise_sup_sample_from_policy_v1.1_stepwise_dpo',
        'fingpt_convfinqa_sup_sample_from_policy_v1.1_dpo',
        'fingpt_convfinqa_sup_sample_from_policy_v1.1_stepwise_dpo',
        'flare_finqa_sup_sample_from_policy_v1.1_dpo',
        'flare_finqa_sup_sample_from_policy_v1.1_stepwise_dpo',
        'sujet_finance_instruct_sup_sample_from_policy_v1.1_dpo',
        'sujet_finance_instruct_sup_sample_from_policy_v1.1_stepwise_dpo',
        'cfa_extracted_exercise_sup_sample_from_policy_v1_1_rpo_iter_1_dpo',
        'cfa_extracted_exercise_sup_sample_from_policy_v1_1_rpo_stepwise_iter_1_dpo',
        'cfa_extracted_exercise_sup_sample_from_policy_v1_1_rpo_stepwise_iter_1_stepwise_dpo',
        'cfa_extracted_exercise_sup_sample_from_policy_v1.1_genrm_qwen3-32b_dpo',
        'cfa_extracted_exercise_sup_sample_from_policy_v1.1_genrm_qwen3-32b_stepwise_dpo']
        self.dpo_datasets_gen = ['bopang_sup_dpo']

        #TODO: rename to group A to be conssitent with the paper
        # Get root directory from args, fallback to placeholder if not provided
        root_dir = getattr(self.args, 'root_dir', 'YOUR_ROOT_DIR')
        
        self.pretrained_model_path = {
            'zixuan/qwen_32b_v1': f'{root_dir}/result/dapt_mix_sft_mix_full/sft_5e-06_8000_fingpt_ner_sup_Qwen/Qwen2.5-32B-Instruct_with_instruction_mask_combine_posttrain_sft_downsample/checkpoint-14000-hf-model',
            'zixuan/v1': f'{root_dir}/result/posttrain_mix_sft_mix_full/sft_5e-06_8000_fingpt_ner_sup_meta-llama/Meta-Llama-3-8B-Instruct_with_instruction_mask_combine_posttrain_sft_downsample/checkpoint-22000-hf-model',
            'zixuan/v1_1': f'{root_dir}/result/dapt_mix_sft_mix_full_extend_exercise_book/sft_5e-06_8000_cfa_extracted_exercise_sup_zixuan/v1_fineweb_downsample_from_v1/checkpoint-13000-hf-model',
            'zixuan/v1_1_rpo_iter_1': f'{root_dir}/result/dpo_cfa_sample_from_policy/sft_5e-07_2048_cfa_extracted_exercise_sup_sample_from_policy_v1.1_dpo_zixuan/v1_1_use_rpo/checkpoint-750-hf-model',
            'zixuan/v1_1_rpo_stepwise_iter_1': f'{root_dir}/result/dpo_cfa_sample_from_policy_stepwise/sft_5e-07_2048_cfa_extracted_exercise_sup_sample_from_policy_v1.1_stepwise_dpo_zixuan/v1_1_use_rpo/checkpoint-750-hf-model',
            'zixuan/v1_1_rpo_convqa_finqa': f'{root_dir}/result/dpo_convqa_finqa_sample_from_policy_stepwise/sft_5e-07_2048_fingpt_convfinqa_sup_sample_from_policy_v1.1_stepwise_dpo_zixuan/v1_1_use_rpo/checkpoint-5000-hf-model'
        }

        self.underlying_model_path = {
            'zixuan/qwen_32b_v1': 'Qwen/Qwen2.5-32B-Instruct',
            'zixuan/v1': 'meta-llama/Meta-Llama-3-8B-Instruct',
            'zixuan/v1_1': 'meta-llama/Meta-Llama-3-8B-Instruct',
            'zixuan/v1_1_rpo_iter_1': 'meta-llama/Meta-Llama-3-8B-Instruct',
            'zixuan/v1_1_rpo_stepwise_iter_1':  'meta-llama/Meta-Llama-3-8B-Instruct',
            'zixuan/v1_1_rpo_convqa_finqa': 'meta-llama/Meta-Llama-3-8B-Instruct',

        }

        # Get HuggingFace token from environment variable
        self.access_token = os.getenv('HF_TOKEN')
        if not self.access_token:
            raise ValueError("HF_TOKEN environment variable is required but not set. Please set it with: export HF_TOKEN=your_token_here")

        if self.args.model_name in self.pretrained_model_path:
            self.pretrained_model_name = self.pretrained_model_path[self.args.model_name]
            self.underlying_model_name = self.underlying_model_path[self.args.model_name]

        elif 'microsoft/Phi-3.5-mini-instruct' in self.args.model_name:
            self.underlying_model_name = 'microsoft/Phi-3.5-mini-instruct'
            self.pretrained_model_name = self.args.model_name
        elif 'meta-llama/Meta-Llama-3-8B-Instruct' in self.args.model_name:
            self.underlying_model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
            self.pretrained_model_name = self.args.model_name
        else:
            self.pretrained_model_name = self.args.model_name
            self.underlying_model_name = self.args.model_name

        self.args.pretrained_model_name = self.pretrained_model_name
        self.args.underlying_model_name = self.underlying_model_name


        self.eval_datasets = None
        self.sft_datasets_valid = None
        self.dapt_datasets_valid = None
