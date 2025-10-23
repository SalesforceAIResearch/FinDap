import copy
import shutil
import argparse
import logging
import math
import os
import random
import sys
import torch
import datasets
import transformers
from accelerate import Accelerator, DistributedType
from tqdm.auto import tqdm
from transformers import (
    MODEL_MAPPING,
    AdamW,
    AutoTokenizer,
    LlamaTokenizer,
    AutoConfig,
    DataCollatorForLanguageModeling,
    get_scheduler,
    SchedulerType,
    set_seed,
)
import utils
import json
from trl import (
    DPOConfig,
    DPOTrainer,
    SFTConfig,
    SFTTrainer,
    ModelConfig,
    RichProgressCallback,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from transformers import Trainer
from torch.utils.data import Dataset
from transformers import DataCollator
from typing import Callable, Dict, List, Optional, Any, Tuple, Union
from accelerate.state import PartialState
from trl.trainer.utils import (
    ConstantLengthDataset,
    DataCollatorForCompletionOnlyLM,
    generate_model_card,
    peft_module_casting_to_bf16,
)
from trl.extras.dataset_formatting import get_formatting_func_from_dataset
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
import types
from transformers import (
    AutoModelForCausalLM,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    is_wandb_available,
)
import torch.nn as nn
from contextlib import contextmanager, nullcontext


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

os.environ["NCCL_DEBUG"] = "INFO"





class Posttrain(object):
    def __init__(self, args):
        super().__init__()
        self.args = args
        return


    # or write a trainer myself
    def dpo_trainer(self,model,ref_model, training_args,train_dataset,eval_dataset,focus_dataset,tokenizer=None):
        ################
        # Training
        ################

        if not any(value is None for value in focus_dataset.values()): 
            eval_data_in_trianer = {'overall': eval_dataset} | focus_dataset
        else:
            eval_data_in_trianer = eval_dataset


        #TODO: for now, eval_dataset has to be Dataset, rather than dict. This is different from SFTTrainer
        eval_data_in_trianer = eval_dataset
        LOG.info(f'eval_data_in_trianer: {eval_data_in_trianer}')



        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_data_in_trianer,
            tokenizer=tokenizer
        )

        # hack the prepare. 
        # see https://github.com/huggingface/trl/issues/1147
        prepared_model = trainer._wrap_model(
            trainer.model, training=True, dataloader=None
        )
        if hasattr(trainer.lr_scheduler, "step"):
            prepared_model, trainer.optimizer = trainer.accelerator.prepare(
                prepared_model, trainer.optimizer
            )
        else:
            (
                prepared_model,
                trainer.optimizer,
                trainer.lr_scheduler,
            ) = trainer.accelerator.prepare(
                prepared_model, trainer.optimizer, trainer.lr_scheduler
            )
        trainer.model_wrapped = prepared_model
        if trainer.is_fsdp_enabled:
            trainer.model = prepared_model
        if trainer.ref_model is not None:
            trainer.ref_model = trainer.accelerator.prepare_model(trainer.ref_model)

        trainer.accelerator.prepare_model = lambda model, *args, **kwargs: model # Monkey-patch prepare_model a no-op , since we have manually prepared the models

        trainer.train(resume_from_checkpoint = self.args.resume_from_checkpoint if self.args.resume_from_checkpoint is not None else None)
        
        # TODO: saving may take extra memory and cause OOM, consider change the FSDP save

        # with save_context:
        trainer.save_model(training_args.output_dir)



    def sft_trainer(self,model,tokenizer,training_args,train_dataset,focus_dataset,is_eval_only,eval_dataset=None,sft_eval_dataset=None,dapt_eval_dataset=None):
        ################
        # Training
        ################
        ################
        # Optional rich context managers

        TRL_USE_RICH = True
        from rich.console import Console
        from contextlib import nullcontext
        console = Console()

        ###############
        init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the SFTTrainer...")
        save_context = (
            nullcontext()
            if not TRL_USE_RICH
            else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
        )

        if eval_dataset is not None:
            eval_data_in_trianer = eval_dataset
        elif not any(value is None for value in focus_dataset.values()): 
            eval_data_in_trianer = {'sft': sft_eval_dataset, 'dapt':dapt_eval_dataset} | focus_dataset
        else:
            eval_data_in_trianer = {'sft': sft_eval_dataset, 'dapt':dapt_eval_dataset} 


        if is_eval_only: eval_data_in_trianer = focus_dataset #TODO: for check
        LOG.info(f'eval_data_in_trianer: {eval_data_in_trianer}')

        with init_context:
            trainer = SFTTrainer(
                model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_data_in_trianer,
                tokenizer=tokenizer,
            )
        if is_eval_only:
            eval_results = trainer.evaluate()

            eval_results["step"] = self.args.model_name.split('checkpoint-')[1].split('-')[0]
            LOG.info(f"eval_results: {eval_results}")

            with open (f'{self.args.model_name}/../metrics.txt','a') as metric:
                eval_results = json.dumps(eval_results)
                metric.write(eval_results)
        else:
            #TODO: this may take a long time without packing / completion only
            trainer.train(resume_from_checkpoint = self.args.resume_from_checkpoint if self.args.resume_from_checkpoint is not None else None)
            # load is not possible if things changed

            LOG.info(f"output_dir: {training_args.output_dir}")
            with save_context:
                trainer.save_model(training_args.output_dir)
