import json
import jsonlines
import numpy as np
import os
import os.path
import pandas as pd
import random
from datasets import Dataset, DatasetDict, concatenate_datasets
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from data_quality.common.constant import HF_MODEL_NAME

HF_MODEL = 'Qwen/Qwen2.5-32B-Instruct'

names =  [
        'sujet_finance_instruct_sup', 'book_fineweb_unsup', 'cfa_extracted_exercise_sup','finance_unsup', 'dm_mathematics_sup', 'dialog_studio_sup', 'p3_supernatural_sup', 'various_supervised_sup', 'apex_instruct_for_annealing_sup', 'bopang_sup', 'fingpt_finred_sup', 'fingpt_ner_cls_sup', 'fingpt_headline_cls_sup', 'fingpt_sentiment_cls_sup', 'fingpt_sentiment_train_sup', 'fingpt_ner_sup', 'trade_the_event_sup', 'fingpt_convfinqa_sup', 'flare_finqa_sup', 'fingpt_convfinqa_sup', 'fingpt_fiqa_qa_sup']
tokenized_trainer_map = {name: f"ZixuanKe/posttrain_tokenized_{name}_{HF_MODEL_NAME[HF_MODEL]}" for name in names}


names = ['cfa_extracted_exercise_sup_dpo', 'sujet_finance_instruct_sup_dpo', 'bopang_sup_dpo', 'fingpt_finred_sup_dpo', 'fingpt_ner_cls_sup_dpo', 'fingpt_ner_sup_dpo', 'trade_the_event_sup_dpo', 'fingpt_convfinqa_sup_dpo', 'flare_finqa_sup_dpo', 'fingpt_fiqa_qa_sup_dpo', 'cfa_extracted_exercise_sup_sample_from_policy_v1.1_dpo', 
'cfa_extracted_exercise_sup_sample_from_policy_v1.1_stepwise_dpo', 
'fingpt_convfinqa_sup_sample_from_policy_v1.1_dpo',
'fingpt_convfinqa_sup_sample_from_policy_v1.1_stepwise_dpo',
'flare_finqa_sup_sample_from_policy_v1.1_dpo',
'flare_finqa_sup_sample_from_policy_v1.1_stepwise_dpo',
'sujet_finance_instruct_sup_sample_from_policy_v1.1_dpo',
'sujet_finance_instruct_sup_sample_from_policy_v1.1_stepwise_dpo',
'cfa_extracted_exercise_sup_sample_from_policy_v1_1_rpo_iter_1_dpo', 
'cfa_extracted_exercise_sup_sample_from_policy_v1_1_rpo_stepwise_iter_1_stepwise_dpo', 'cfa_extracted_exercise_sup_sample_from_policy_v1_1_rpo_stepwise_iter_1_dpo',
'cfa_extracted_exercise_sup_sample_from_policy_v1.1_genrm_qwen3-32b_dpo',
'cfa_extracted_exercise_sup_sample_from_policy_v1.1_genrm_qwen3-32b_stepwise_dpo']

dpo_trainer_map = {name: f"ZixuanKe/{name}_binarized_F2048" for name in names}
# dpo_trainer_map = {name: f"ZixuanKe/{name}_binarized_filtered_2048" for name in names}

all_trainer_map = tokenized_trainer_map | dpo_trainer_map



def get_dataset(dataset_name, use_trainer):

    if use_trainer:
        # for unification, everything is put to the huggingface hub
        # Get HuggingFace token from environment variable
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable is required but not set. Please set it with: export HF_TOKEN=your_token_here")
        
        datasets = load_dataset(all_trainer_map[dataset_name], token=hf_token)
        print(f'dataset_name: {all_trainer_map[dataset_name]}; dataset_name: {dataset_name}; {datasets}')


    else:
        raise NotImplementedError


    return datasets
