import logging
import math

from transformers import (
    MODEL_MAPPING,
    AdamW,
    get_scheduler,
    Adafactor
)

logger = logging.getLogger(__name__)

def get_warmup_steps(num_training_steps,args):
    """
    Get number of steps used for a linear warmup.
    """
    warmup_steps = args.num_warmup_steps if args.num_warmup_steps > 0 else math.ceil(num_training_steps * args.warmup_ratio)
    return warmup_steps



def lookfor_optimize_posttrain(model,args):

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            'name': [n for n, p in model.named_parameters()
                     if p.requires_grad and not any(nd in n for nd in no_decay)],
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate
        },
        {
            'name': [n for n, p in model.named_parameters()
                     if p.requires_grad and any(nd in n for nd in no_decay)],
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": args.learning_rate

        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters)

    return optimizer

def lookfor_optimize_finetune(model,args):

    # Set the optimizer
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                      weight_decay=args.weight_decay)


    return optimizer
