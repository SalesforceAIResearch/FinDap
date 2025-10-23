import datasets
import json
import logging
import numpy as np
import os
import os.path
import random
import torch
import yaml
from datasets import Dataset, DatasetDict, concatenate_datasets
from pathlib import Path
from time import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import GPT2TokenizerFast, GPT2Tokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import Optional, Union, List, Tuple, Dict
from transformers import LlamaTokenizer
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, default_data_collator
from torch.utils.data import DataLoader
from datetime import timedelta
from dataloader.data import get_dataset
import utils
from peft import get_peft_model, AdaLoraConfig, TaskType, get_peft_model_state_dict, set_peft_model_state_dict, LoraConfig, PeftModel
import transformers
from multiprocess import set_start_method
import ast
from utils.common import Preparer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from trl import DPOConfig, DPOTrainer
from datetime import timedelta
from utils.packing.packed_dataset import PackedDataset
from utils.packing import monkey_patch_packing
from copy import deepcopy
from datasets import ClassLabel, Value
import pandas as pd
from transformers import TrainingArguments
import copy

from data_quality.common.constant import LLAMA3_CONVO_FORMAT, PHI3_CONVO_FORMAT
script_name = os.path.basename(__file__)

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels of logs
bash_script_name = os.getenv("BASH_SCRIPT_NAME", "Unknown")

# Create a file handler to log to a file
# Get root directory from environment or use placeholder
root_dir = os.getenv('ROOT_DIR', 'YOUR_ROOT_DIR')
file_handler = logging.FileHandler(f'{root_dir}/SFR-Continual-Pretrain/logs/{bash_script_name}.log')
file_handler.setLevel(logging.DEBUG)  # Ensure file handler captures all logs

# Optional: Create a stream handler to log to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Set as desired

# Define a formatter and set it for both handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
LOG.addHandler(file_handler)
LOG.addHandler(console_handler)



os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['NCCL_BLOCKING_WAIT'] = '0'





class PosttrainPreparer(Preparer):
    """Base editor for all methods"""

    def __init__(self, args):
        super(PosttrainPreparer, self).__init__(args)

        self.pretty_seq_name = {
            'sft_in': 0,
            'sft_gen': 1,
            'sft_mix': 2,
            'sft_mix_extend': 3,
            'sft_mix_full': 4,
            'sft_gen_only_full': 5,
            'sft_mix_extend_full': 6,
            'sft_mix_extend_addition_full': 7,
            'sft_mix_addition_full': 8,
            'sft_mix_full_extractive': 9,
            'dapt_mix_sft_mix_full':10,
            'dapt_in':11,
            'dapt_gen':12,
            'dapt_mix':13,
            'dapt_mix_sft_mix_full_extend':14,
            'dapt_mix_sft_mix_full_book':15,
            'dapt_mix_sft_mix_full_extend_book':16,
            'dapt_cfa_rule':17,
            'dapt_cfa_fineweb': 18,
            'dapt_cfa_knowledgeable': 19,
            'dapt_cfa_knowledgeable_study_guide': 20,
            'dapt_cfa_knowledgeable_exercise': 21,
            'sft_in_extend': 22,
            'sft_exercise': 23,
            'sft_extracted_exercise': 24,
            'dapt_mix_sft_mix_full_extend_exercise_book':26,
            'dapt_in_sft_in_extend_exercise':27,
            'dpo_cfa_sample_from_policy':30,
            'dpo_cfa_sample_from_policy_stepwise':31,
            'dpo_cfa_finqa_sample_from_policy_stepwise':35,
        }
    

    def load_model_tok(self):

        LOG.info("Instantiating model")

        if type(self.args.model_name) is str:
            if self.args.instruction_mask and self.args.isolate_attention:  
                LOG.info('Use monkey_patch_packing_for_model')
                monkey_patch_packing.monkey_patch_packing_for_model(self.pretrained_model_name)

            if self.args.use_flash_attention_2:
                LOG.info('Use flash_attention_2')
                LOG.info(f'pretrained_model_name: {self.pretrained_model_name}')
                LOG.info(f'device_map: {self.device_map}')
                # TODO: use training_args, model_args.

                self.model = AutoModelForCausalLM.from_pretrained(self.pretrained_model_name, torch_dtype=self.torch_dtype,
                                                            device_map=self.device_map, token=self.access_token,
                                                            attn_implementation="flash_attention_2")
                if 'dpo' in self.args.idrandom:
                    self.ref_model = AutoModelForCausalLM.from_pretrained(self.pretrained_model_name, torch_dtype=self.torch_dtype,
                                                            device_map=self.device_map, token=self.access_token,
                                                            attn_implementation="flash_attention_2")
            else:
                raise NotImplementedError

            self.tok = AutoTokenizer.from_pretrained(self.pretrained_model_name, use_auth_token=self.access_token)
            # we have eot as the real end token; if adding, you need to resize embedidng
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token 
            if self.tok.pad_token_id is None:
                self.tok.pad_token_id = self.tok.eos_token_id
                
            LOG.info(f'self.tok.pad_token_id: {self.tok.pad_token_id}')
            LOG.info(f'self.tok.pad_token: {self.tok.pad_token}')

            #later we will use chat-template before tokenize
            data_cache_file = f"{self.args.cache_dir}/{self.args.model_name}" # current dataset package does not create the folder by default
            os.makedirs(data_cache_file, exist_ok=True) 


        else:
            raise NotImplementedError


    def concatenate_all_in_dict(self, dataset_dict):
        dataset_dict_copy = copy.deepcopy(dataset_dict)  # Deep copy is required. Otherwise dataset_dict will be changed

        has_init = False
        for my_dataset_name, my_dataset in dataset_dict_copy.items():
            if not has_init:
                final_dataset = my_dataset
                has_init = True
            else:

                cur_dataset = my_dataset
                final_dataset["train"] = concatenate_datasets([final_dataset["train"], cur_dataset["train"]])
                final_dataset["validation"] = concatenate_datasets([final_dataset["validation"], cur_dataset["validation"]])

        return final_dataset



    def prepare_posttrain(self): 
        # if we want to separate sft and dapt while we combine the two
        #TODO: Up-sampling at some point
        if len(self.args.data) > 1:
            assert 'comb' in self.args.baseline
        assert self.args.instruction_mask
        assert self.args.isolate_attention
        assert self.args.use_flash_attention_2


        focus_dataset = {'cfa': None}
        dapt_dict = {}
        sft_dict = {}
        dpo_gen_dict = {}
        dpo_in_dict = {}


        LOG.info(f'all data: {self.args.data}')

        for t in range(self.args.pt_task + 1):
            LOG.info(f'loading self.args.data[t]: {self.args.data[t]}; \n is dapt? {self.args.data[t] in self.dapt_datasets} \n is dpo_datasets_in? {self.args.data[t] in self.dpo_datasets_in}')

            my_dataset = get_dataset(self.args.data[t], self.args.use_trainer)
            cur_data_name = self.args.data[t] 

            if cur_data_name in self.dapt_datasets:
                dapt_dict[cur_data_name] = my_dataset

            elif cur_data_name in self.sft_datasets:
                sft_dict[cur_data_name] = my_dataset


            elif cur_data_name in self.dpo_datasets_gen or cur_data_name in self.dpo_datasets_in:

                # Check if the dataset belongs to dpo_datasets_gen
                if cur_data_name in self.dpo_datasets_gen:
                        dpo_gen_dict[cur_data_name] = my_dataset

                # Check if the dataset belongs to dpo_datasets_in
                elif cur_data_name in self.dpo_datasets_in:
                    dpo_in_dict[cur_data_name] = my_dataset

            else:
                raise NotImplementedError 
                # TODO: we need a sanity check for SFT as well

 
        print('dapt_dict: ',dapt_dict)
        print('sft_dict: ',sft_dict)
        print('dpo_gen_dict: ',dpo_gen_dict)
        print('dpo_in_dict: ',dpo_in_dict)

        #TODO: need organize and re-tokenized using Qwen

        self.focus_dataset = focus_dataset

        if 'sft' in self.args.idrandom and 'dapt' in self.args.idrandom:

            sft_datasets = self.concatenate_all_in_dict(sft_dict)
            dapt_datasets = self.concatenate_all_in_dict(dapt_dict)


            if self.args.downsample or self.args.upsample:
                # Assuming dapt_datasets and sft_datasets are already loaded
                dapt_size = len(dapt_datasets["train"])
                sft_size = len(sft_datasets["train"])

                if self.args.downsample:
                    # random downsample
                    # must have this shuffle, we are *randomly* sample
                    dapt_train_datasets = dapt_datasets["train"].shuffle(seed=42).select(range(len(sft_datasets["train"])))
                    sft_train_datasets = sft_datasets["train"]

                elif self.args.upsample:
                        
                    # Calculate how many more samples we need
                    upsample_factor = dapt_size // sft_size
                    remaining_samples = dapt_size % sft_size

                    # Step 1: Upsample the dataset by repeating the dataset (upsample_factor times)
                    upsampled_sft = sft_datasets["train"].select(np.arange(sft_size).repeat(upsample_factor))

                    # Step 2: Add remaining samples if needed (randomly select from original dataset)
                    # random upsample
                    if remaining_samples > 0:
                        additional_samples = sft_datasets["train"].shuffle(seed=42).select(np.arange(remaining_samples).tolist())
                        sft_train_datasets = concatenate_datasets([upsampled_sft,additional_samples])
                    
                    dapt_train_datasets = dapt_datasets["train"]
        

                # Only training set is affected

                # Now, train_datasets has the same size as dapt_datasets["train"]
                LOG.info(f"Old size of sft_datasets['train']: {sft_size}")
                LOG.info(f"Old size of dapt_size['train']: {dapt_size}")
                LOG.info(f"New size of sft_datasets['train']: {len(sft_train_datasets)}")
                LOG.info(f"New size of dapt_size['train']: {len(dapt_train_datasets)}")

                train_datasets = concatenate_datasets([sft_train_datasets, dapt_train_datasets])
                train_datasets = train_datasets.shuffle(seed=42) # just in case

                # random sampling
                dapt_datasets["validation"] = dapt_datasets["validation"].shuffle(seed=42).select(range(len(sft_datasets["validation"])))

                sft_datasets_valid = sft_datasets["validation"]
                dapt_datasets_valid = dapt_datasets["validation"]


                # dapt valid is too long, we care more about sft, typically. As it is directly about end-task
                if len(dapt_datasets_valid) > 10000:
                    LOG.info(f'cut the evaluation set with length {len(dapt_datasets_valid)} to 10000')
                    dapt_datasets_valid = dapt_datasets_valid.shuffle(seed=42).select([i for i in range(10000)]) # we do not need too large eval


                sft_datasets_valid.set_format("torch")
                dapt_datasets_valid.set_format("torch")

                if self.args.eval_only:
                    train_datasets = train_datasets.select([0,1,2]) # no need to have train data when it is eval only


                LOG.info(f'train_datasets: {train_datasets}')
                LOG.info(f'sft_datasets_valid: {sft_datasets_valid}')
                LOG.info(f'dapt_datasets_valid: {dapt_datasets_valid}')
                
                self.sft_datasets_valid = PackedDataset(
                    sft_datasets_valid, self.tok, self.args.max_seq_length, lengths=sft_datasets_valid['packed_length'])
                
                self.dapt_datasets_valid = PackedDataset(
                    dapt_datasets_valid, self.tok, self.args.max_seq_length, lengths=dapt_datasets_valid['packed_length'])


                train_datasets.set_format("torch")

                self.train_datasets = PackedDataset(
                    train_datasets, self.tok, self.args.max_seq_length, lengths=train_datasets['packed_length'])
                LOG.info(f'final train_datasets: {self.train_datasets}')

                #decode and check
                #TODO: where is <s>. Need to check
                for index in random.sample(range(len(self.train_datasets)), 1):
                    LOG.info(
                        f"Sample {index} of the training set: {self.train_datasets[index]}. \n\
                        Decode input to: {self.tok.decode(self.train_datasets[index]['input_ids'])}. \n\
                        Label to {self.tok.decode([x if x!=-100 else 0 for x in self.train_datasets[index]['labels']])}")


        #TODO: what if CPT?
        elif 'sft' in self.args.idrandom:

            sft_datasets = self.concatenate_all_in_dict(sft_dict)

            sft_datasets_train = sft_datasets["train"]
            sft_datasets_valid = sft_datasets["validation"]

            LOG.info(f'sft_datasets_train: {sft_datasets_train}')
            LOG.info(f'sft_datasets_valid: {sft_datasets_valid}')

            sft_datasets_valid.set_format("torch")
            sft_datasets_train.set_format("torch")


            self.sft_datasets_valid = PackedDataset(
                sft_datasets_valid, self.tok, self.args.max_seq_length, lengths=sft_datasets_valid['packed_length'])
            
            self.train_datasets = PackedDataset(
                sft_datasets_train, self.tok, self.args.max_seq_length, lengths=sft_datasets_train['packed_length'])
            LOG.info(f'final train_datasets: {self.train_datasets}')

            #decode and check
            #TODO: where is <s>. Need to check
            for index in random.sample(range(len(self.train_datasets)), 1):
                LOG.info(
                    f"Sample {index} of the training set: {self.train_datasets[index]}. \n\
                    Decode input to: {self.tok.decode(self.train_datasets[index]['input_ids'])}. \n\
                    Label to {self.tok.decode([x if x!=-100 else 0 for x in self.train_datasets[index]['labels']])}")


        elif 'dpo' in self.args.idrandom:


            #TODO downsampling to 1:1

            if len(dpo_gen_dict) > 1 and len(dpo_in_dict) > 1:

                dop_datasets_gen = self.concatenate_all_in_dict(dpo_gen_dict)
                dop_datasets_in = self.concatenate_all_in_dict(dpo_in_dict)


                print('dop_datasets_gen["train"]: ',dop_datasets_gen["train"])

                dop_datasets_gen_train = dop_datasets_gen["train"].shuffle(seed=42).select(range(len(dop_datasets_in["train"])))
                dop_datasets_in_train = dop_datasets_in["train"]

                print('dop_datasets_gen_train: ',dop_datasets_gen_train)


                dop_datasets_gen_val = dop_datasets_gen["validation"].shuffle(seed=42).select(range(len(dop_datasets_in["validation"])))
                dop_datasets_in_val = dop_datasets_in["validation"]             
                
                self.train_datasets = concatenate_datasets([dop_datasets_gen_train,dop_datasets_in_train])
                self.eval_datasets = concatenate_datasets([dop_datasets_gen_val,dop_datasets_in_val])

            elif len(dpo_gen_dict) > 1:

                dop_datasets_gen = self.concatenate_all_in_dict(dpo_gen_dict)

                self.train_datasets =  dop_datasets_gen["train"]
                self.eval_datasets = dop_datasets_gen["validation"]

            elif len(dpo_in_dict) > 1:



                if self.args.downsample:
                    # all match cfa one
                    # TODO: should not hard coded!!!
                    cfa_dataset = dpo_in_dict['cfa_extracted_exercise_sup_sample_from_policy_v1.1_genrm_qwen3-32b_dpo']

                    for cur_name, cur_dataset in dpo_in_dict.items():
                        if len(cur_dataset["train"]) <= len(cfa_dataset["train"]): # no need to downsample
                            continue
                        print(f"cur_name: {cur_name}; dpo_in_dict[cur_name]['train']: {len(dpo_in_dict[cur_name]['train'])}")

                        cur_dataset_train = cur_dataset["train"].shuffle(seed=42).select(range(len(cfa_dataset["train"])))
                        dpo_in_dict[cur_name]['train'] = cur_dataset_train
                        
                        print(f"cur_name: {cur_name}; dpo_in_dict[cur_name]['train']: {len(dpo_in_dict[cur_name]['train'])}")

                    dop_datasets_in = self.concatenate_all_in_dict(dpo_in_dict)


                elif self.args.upsample:
                    raise NotImplementedError
                
                else:
                    dop_datasets_in = self.concatenate_all_in_dict(dpo_in_dict)


                # TODO: we may want to use cfa data as validation set
                # we focus on CFA
                # TODO: should not hard coded!!!
                dop_datasets_in["validation"] = concatenate_datasets([dpo_in_dict['cfa_extracted_exercise_sup_sample_from_policy_v1.1_genrm_qwen3-32b_dpo']['validation'],dpo_in_dict['cfa_extracted_exercise_sup_sample_from_policy_v1.1_genrm_qwen3-32b_stepwise_dpo']['validation']])


                self.train_datasets =  dop_datasets_in["train"]
                self.eval_datasets = dop_datasets_in["validation"]

            columns = self.train_datasets.column_names

            if 'ground_truth_chosen' in columns:
                self.train_datasets = self.train_datasets.rename_column('ground_truth_chosen','chosen').rename_column('v1_1_rejected','rejected')
                self.eval_datasets = self.eval_datasets.rename_column('ground_truth_chosen','chosen').rename_column('v1_1_rejected','rejected')


            

            LOG.info(f'self train_datasets: {self.train_datasets}')
            LOG.info(f'self eval_datasets: {self.eval_datasets}')

            # Filter is done by process_dataset.py
            # TODO: double check

            if len(self.eval_datasets) > 10000:
                LOG.info(f'cut the evaluation set with length {len(self.eval_datasets)} to 10000')
                self.eval_datasets = self.eval_datasets.shuffle(seed=42).select([i for i in range(10000)]) # we do not need too large eval

            LOG.info(f'self eval_datasets dropna: {self.eval_datasets}')

            # for sft and dapt, they have separte eval_dataset

            column_to_remove = [column for column in columns if column not in ['prompt', 'rejected', 'chosen']]
            LOG.info(f'column_to_remove: {column_to_remove}')


            if len(column_to_remove) > 0:
                self.train_datasets = self.train_datasets.remove_columns(column_to_remove)
                self.eval_datasets = self.eval_datasets.remove_columns(column_to_remove)

            # self.train_datasets = self.train_datasets.select([i for i in range(16)]) # no need to have train data when it is eval only
            # self.eval_datasets = self.eval_datasets.select([i for i in range(16)]) # no need to have train data when it is eval only

        for index in random.sample(range(len(self.train_datasets)), 1):
            LOG.info(
                f"Sample {index} of the training set: {self.train_datasets[index]}")
    
        LOG.info(f'self train_datasets dropna: {self.train_datasets}')



    def prepare_sequence(self):
        sequence_path = self.args.sequence_file
        with open(sequence_path, 'r') as f:
            datas = f.readlines()[self.pretty_seq_name[self.args.idrandom]]
            data = datas.split()
 
        self.args.pt_task = len(data) - 1 # in our current setting, this is always the last 
        LOG.info(f"self.args.pt_task: {self.args.pt_task}")

        os.makedirs(self.args.result_dir, exist_ok=True)
        os.makedirs(f'{self.args.result_dir}/{self.args.idrandom}', exist_ok=True)

        # the model_name is still the base name. Later model_name will be changed to trained model if needed

        output = f'{self.args.result_dir}/{self.args.idrandom}/sft_{self.args.learning_rate}_{self.args.max_seq_length}_{data[self.args.pt_task]}_{self.args.model_name}{self.args.additional_note}/'
 
        if self.args.output_dir is None:
            self.args.output_dir = output
        self.args.task = self.args.pt_task

        self.args.data = data
        self.args.eval_t = self.args.pt_task  # we need to use the adapter/plugin

        if 'comb' in self.args.baseline:
            self.args.dataset_name = f'{data[0]}-{data[-1]}'
        else:
            self.args.dataset_name = data[self.args.pt_task]


        if self.args.use_trainer: 

            if 'dpo' in self.args.idrandom:
                #TODO: we need one for both
                # TODO: what if we combine everything?
                LOG.info('self.args.max_seq_length: ',self.args.max_seq_length)

                if self.args.use_rpo:
                    self.training_args  = DPOConfig(
                            save_steps = self.args.checkpointing_steps,
                            learning_rate = self.args.learning_rate,
                            per_device_train_batch_size = self.args.per_device_train_batch_size,
                            per_device_eval_batch_size = self.args.per_device_eval_batch_size,
                            max_length=self.args.max_seq_length,
                            output_dir=self.args.output_dir,
                            warmup_ratio = self.args.warmup_proportion,
                            report_to=["tensorboard"],
                            remove_unused_columns=True,
                            logging_steps=50,
                            eval_strategy='steps',
                            ddp_timeout=7200000,
                            eval_steps = self.args.eval_steps,
                            gradient_checkpointing=True,
                            gradient_checkpointing_kwargs={"use_reentrant": False},
                            dataset_num_proc=16,
                            rpo_alpha=1,
                            metric_for_best_model='rewards/accuracies',
                            # save_total_limit=10 # accuracies is not always indicative
                        )
                else:
                    self.training_args  = DPOConfig(
                            save_steps = self.args.checkpointing_steps,
                            learning_rate = self.args.learning_rate,
                            per_device_train_batch_size = self.args.per_device_train_batch_size,
                            per_device_eval_batch_size = self.args.per_device_eval_batch_size,
                            max_length=self.args.max_seq_length,
                            output_dir=self.args.output_dir,
                            warmup_ratio = self.args.warmup_proportion,
                            report_to=["tensorboard"],
                            remove_unused_columns=True,
                            logging_steps=50,
                            eval_strategy='steps',
                            ddp_timeout=7200000,
                            eval_steps = self.args.eval_steps,
                            gradient_checkpointing=True,
                            gradient_checkpointing_kwargs={"use_reentrant": False},
                            dataset_num_proc=16,
                        )
            else:
                # TODO: No train on SFT yet
                # should not mix used with accelerate
                self.training_args  = SFTConfig(
                        save_steps = self.args.checkpointing_steps,
                        learning_rate = self.args.learning_rate,
                        per_device_train_batch_size = self.args.per_device_train_batch_size,
                        per_device_eval_batch_size = self.args.per_device_eval_batch_size,
                        gradient_checkpointing_kwargs={"use_reentrant": False},
                        gradient_checkpointing=True,
                        max_seq_length=self.args.max_seq_length,
                        output_dir=self.args.output_dir,
                        warmup_ratio = self.args.warmup_proportion,
                        report_to=["tensorboard"],
                        packing=True,
                        eval_packing=True, 
                        remove_unused_columns=True,
                        logging_steps=50,
                        eval_strategy='steps',
                        ddp_timeout=7200000,
                        eval_steps = self.args.eval_steps,
                    ) 


            LOG.info(f'training_args: {self.training_args}')


        LOG.info(f'Output directory: {self.args.output_dir}')
        LOG.info(f'Dataset Name: {self.args.dataset_name}')
        LOG.info(f'Pretrained model: {self.args.model_name}')
        LOG.info(f'self.args.data: {self.args.data}')
