
from typing import Any, Dict, List, NewType, Optional, Tuple, Union
from transformers import (
    DataCollatorForLanguageModeling,
    # get_scheduler,
)
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy
import numpy as np
from dataclasses import dataclass
import torch
import nltk
import warnings
import logging
from transformers.data.data_collator import DataCollator, DataCollatorForLanguageModeling, _torch_collate_batch, pad_without_fast_tokenizer_warning
from typing import Mapping, List, Union, Any, Dict, Optional, Callable
import ast
from trl.trainer.utils import ConstantLengthDataset
from itertools import chain

LOG = logging.getLogger(__name__)

from data_quality.common.constant import model_eos_id_map, model_bos_id_map



MASK_IN_ROLES = ["assistant"]
#TODO: this can do multi-turn and also more turn roles



def tokenize_posstrain_with_text(examples, text_col, tok, max_seq_length, model_name,dataset_name):

    outputs = {
        'input_ids': [],
        'attention_mask': [],
        "title": [], "topic": [], "text": [],
        "packed_length": []
    }
    tokenized = tok(examples[text_col], return_special_tokens_mask=True)

    input_ids = tokenized["input_ids"]
    attention_masks = tokenized["attention_mask"]

    # Calculate the maximum length for the chunk without the special token
    chunk_max_length = max_seq_length - 2

    for input_id, attention_mask  in zip(input_ids, attention_masks):
        if len(input_id) + 2 <= max_seq_length: #else chunked

            if model_bos_id_map[model_name] not in input_id: # add bos
                input_id = [model_bos_id_map[model_name]] + input_id
                attention_mask =  [1] + attention_mask

            if model_eos_id_map[model_name] not in input_id: # add eos
                input_id = input_id + [model_eos_id_map[model_name]]
                attention_mask = attention_mask + [1]

            outputs['input_ids'].append(input_id)
            outputs['attention_mask'].append(attention_mask)
            outputs["text"].append(tok.decode(input_id, skip_special_tokens=False)) # if we want the message
            outputs["topic"].append('dapt')
            outputs["title"].append(dataset_name)

            input_length = torch.sum(torch.tensor(attention_mask)).item()
            outputs["packed_length"].append(int(input_length))

        else:
            # Split the input into chunks
            for i in range(0, len(input_id), chunk_max_length):

                chunk_input_id = input_id[i:i + chunk_max_length]
                chunk_attention_mask = attention_mask[i:i + chunk_max_length]

                # if i + chunk_max_length >= len(input_id): # the last chunk
                # each chunk is an independent chunk
                #TODO: some chunk will be missing the bos token
                if model_bos_id_map[model_name] not in chunk_input_id: # add bos
                    chunk_input_id = [model_bos_id_map[model_name]] + chunk_input_id
                    chunk_attention_mask =  [1] + chunk_attention_mask

                if model_eos_id_map[model_name] not in chunk_input_id: # add eos
                    chunk_input_id = chunk_input_id + [model_eos_id_map[model_name]]
                    chunk_attention_mask = chunk_attention_mask + [1]


                # chunk_input_id += [model_eos_id_map[model_name]]
                # chunk_attention_mask += [1]
        
                outputs['input_ids'].append(chunk_input_id)
                outputs['attention_mask'].append(chunk_attention_mask)
                outputs["text"].append(tok.decode(chunk_input_id, skip_special_tokens=False)) # if we want the message
                outputs["topic"].append('dapt')
                outputs["title"].append(dataset_name)

                chunk_input_length = torch.sum(torch.tensor(chunk_attention_mask)).item()
                outputs["packed_length"].append(int(chunk_input_length))


        # Ensure that the length of each chunk is not exceeding max_seq_length
        assert torch.sum(torch.Tensor(outputs['attention_mask'][-1])).item() <= max_seq_length, torch.sum(torch.Tensor(outputs['attention_mask'][-1])).item()

    outputs['labels'] = outputs['input_ids']

    LOG.info(f"length input_ids: {len(outputs['input_ids'])}") # for xlearner
    LOG.info(f"length attention_mask: {len(outputs['attention_mask'])}") # for xlearner
    LOG.info(f"length labels: {len(outputs['labels'])}") # for xlearner
    LOG.info(f"length text: {len(outputs['text'])}") # for xlearner
    LOG.info(f"length topic: {len(outputs['topic'])}") # for xlearner
    LOG.info(f"length title: {len(outputs['title'])}") # for xlearner

    return outputs


def tokenize_sft_with_text(examples, model_name, tokenizer,  max_length, conversation_format,dataset_name, no_special_tokens=False, token_type_ids_to_labels=True):

    outputs = {"title": [], "topic": [], "text": [], "input_ids": [], "attention_mask":[], "labels":[], "packed_length": []}
    for example in examples['messages']:
        conversations = transform_example_to_conversations(example)
        mask_turn = None
        tokenized_text = tokenize_multi_turns(
            tokenizer, conversations, 
            mask_turn=mask_turn,
            no_special_tokens=no_special_tokens,
            conversation_format=conversation_format,
        )

        if token_type_ids_to_labels:
            tokenized_text = convert_token_type_ids_to_labels(tokenized_text)

        # add end-of-text as separation -----
        if conversation_format.get("add_eos_token"):
            tokenized_text["input_ids"] += [model_eos_id_map[model_name]] # specific for llama
            tokenized_text["attention_mask"] += [1] # include in loss
            tokenized_text["labels"] += [model_eos_id_map[model_name]] # same as input (rather than -100)
        # add end-of-text as separation -----
        
        assert len(tokenized_text["input_ids"]) == len(tokenized_text["attention_mask"]) == len(tokenized_text["labels"])
        if len(tokenized_text["input_ids"]) > max_length:
            print(f'input length {len(tokenized_text["input_ids"])} exceed max length {max_length}. remove samples') # remove if too long
        else:        
            assert torch.sum(torch.Tensor(tokenized_text["attention_mask"])).item() <= max_length # what added cannot exceed the length
            
            outputs["input_ids"].append(tokenized_text["input_ids"])
            outputs["attention_mask"].append(tokenized_text["attention_mask"])
            outputs["labels"].append(tokenized_text["labels"])
            outputs["topic"].append('sft')
            outputs["title"].append(dataset_name)
            outputs["text"].append(tokenizer.decode(tokenized_text["input_ids"], skip_special_tokens=False)) # if we want the message
            input_length = torch.sum(torch.tensor(tokenized_text["attention_mask"])).item()
            outputs["packed_length"].append(int(input_length))
        # assert outputs.keys() == tokenized_text.keys()


    return outputs


def convert_token_type_ids_to_labels(sample):
    if "token_type_ids" in sample:
        if "labels" not in sample:
            sample['labels'] = [
                (-100 if tti == 0 else x) 
                for x, tti in zip(sample['input_ids'], sample['token_type_ids'])
            ]
        sample.pop("token_type_ids", None)
    return sample    

def tokenize_multi_turns(
        tokenizer, 
        conversations, 
        mask_turn=None, 
        no_special_tokens=False, 
        add_assistant_prefix=False,
        conversation_format=None
):
    sample = None
    assistant_prefix_len = None
    assistant_suffix_len = None
    assistant_suffix_len_end = None
    if mask_turn is not None:
        mask_turn = mask_turn if isinstance(mask_turn, (list, tuple)) else [mask_turn]
    
    for turn_id, turn in enumerate(conversations):
        prompt = conversation_format['turn_template'].format(role=turn['role'], content=turn['content'])
        turn_sample = tokenizer(
            prompt, padding=False, truncation=False, verbose=False, add_special_tokens=False,
            return_token_type_ids=True, 
        )
        mask_assistant_turn = (mask_turn is None or turn_id in mask_turn)
        if turn['role'] in MASK_IN_ROLES and mask_assistant_turn:
            if assistant_prefix_len is None:
                assistant_prefix_len = len(tokenizer.encode(conversation_format['turn_prefix'].format(role=turn['role']), add_special_tokens=False))
            if assistant_suffix_len is None:
                assistant_suffix_len = (
                    len(tokenizer.encode(conversation_format['turn_suffix'], add_special_tokens=False)) - 
                    len(tokenizer.encode(conversation_format['turn_suffix_take'], add_special_tokens=False))
                )
            assistant_suffix_len_end = len(turn_sample['token_type_ids']) - assistant_suffix_len
                # ['<|im_end|>', '▁', '<0x0A>']
            turn_sample['token_type_ids'][assistant_prefix_len:assistant_suffix_len_end] = [1] * (len(turn_sample['input_ids']) - assistant_prefix_len - assistant_suffix_len)
        if sample is None:
            sample = turn_sample
        else:
            for k in turn_sample.keys():
                sample[k].extend(turn_sample[k])
    if add_assistant_prefix:
        assistant_prefix_sample = tokenizer(
            conversation_format['turn_prefix'].format(role="assistant"), padding=False, truncation=False, verbose=False, add_special_tokens=False,
            return_token_type_ids=True, 
        )
        for k in sample.keys():
            sample[k].extend(assistant_prefix_sample[k])

    # print(getattr(conversation_format, "add_bos_token", True),conversation_format)
    # if getattr(tokenizer, "add_bos_token", True) and not no_special_tokens:
    # sometime tokenizer do not have "add_bos_token", like Qwen

    if conversation_format.get("add_bos_token") and not no_special_tokens:
        sample['input_ids'] = [tokenizer.bos_token_id] + sample['input_ids']
        sample['attention_mask'] = [1] + sample['attention_mask']
        sample['token_type_ids'] = [sample['token_type_ids'][0]] + sample['token_type_ids']

    return sample




def transform_example_to_conversations(example, **kwargs):
    """
    Convert all examples into turns with role and content, don't care much
    """
    convos = example
    for c in convos:
        c['role'] = c['role'].lower()
    assert all(c['role'] in ['system', 'user', 'assistant', "thought", "observation"] for c in convos), f'{convos=}'

    for c in convos:
        if not isinstance(c['content'], str):
            c['content'] = str(c['content'])

    return convos


