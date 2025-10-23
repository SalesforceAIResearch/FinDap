import numpy as np
import os
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
from copy import deepcopy
from pathlib import Path
from torch import nn
from transformers import (
    MODEL_MAPPING,
    AdamW,
    get_scheduler,
    Adafactor
)
from accelerate import Accelerator, DistributedType
from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs, InitProcessGroupKwargs
from datetime import timedelta

import utils

########################################################################################################################

def print_model_report(model):
    print('-' * 100)
    print(model)
    print('Dimensions =', end=' ')
    count = 0
    for p in model.parameters():
        print(p.size(), end=' ')
        count += np.prod(p.size())
    print()
    print('Num parameters = %s' % (human_format(count)))
    print('-' * 100)

    with open('para', 'a') as clocker_file:
        clocker_file.writelines((human_format(count)).replace('M', '') + '\n')

    return count


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.1f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def print_optimizer_config(optim):
    if optim is None:
        print(optim)
    else:
        print(optim, '=', end=' ')
        opt = optim.param_groups[0]
        for n in opt.keys():
            if not n.startswith('param'):
                print(n + ':', opt[n], end=', ')
        print()
    return


########################################################################################################################

def get_model(model):
    return deepcopy(model.state_dict())


def set_model_(model, state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return


########################################################################################################################

def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))


########################################################################################################################

def compute_mean_std_dataset(dataset):
    # dataset already put ToTensor
    mean = 0
    std = 0
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for image, _ in loader:
        mean += image.mean(3).mean(2)
    mean /= len(dataset)

    mean_expanded = mean.view(mean.size(0), mean.size(1), 1, 1).expand_as(image)
    for image, _ in loader:
        std += (image - mean_expanded).pow(2).sum(3).sum(2)

    std = (std / (len(dataset) * image.size(2) * image.size(3) - 1)).sqrt()

    return mean, std


########################################################################################################################

# for ACL
def report_tr(res, e, sbatch, clock0, clock1):
    # Training performance
    print(
        '| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train losses={:.3f} | T: loss={:.3f}, acc={:5.2f}% | D: loss={:.3f}, acc={:5.1f}%, '
        'Diff loss:{:.3f} |'.format(
            e + 1,
            1000 * sbatch * (clock1 - clock0) / res['size'],
            1000 * sbatch * (time.time() - clock1) / res['size'], res['loss_tot'],
            res['loss_t'], res['acc_t'], res['loss_a'], res['acc_d'], res['loss_d']), end='')


def report_val(res):
    # Validation performance
    print(
        ' Valid losses={:.3f} | T: loss={:.6f}, acc={:5.2f}%, | D: loss={:.3f}, acc={:5.2f}%, Diff loss={:.3f} |'.format(
            res['loss_tot'], res['loss_t'], res['acc_t'], res['loss_a'], res['acc_d'], res['loss_d']), end='')


########################################################################################################################


def cross_entropy(outputs, targets, exp=1, size_average=True, eps=1e-5):
    out = torch.nn.functional.softmax(outputs)
    tar = torch.nn.functional.softmax(targets)
    if exp != 1:
        out = out.pow(exp)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        tar = tar.pow(exp)
        tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
    out = out + eps / out.size(1)
    out = out / out.sum(1).view(-1, 1).expand_as(out)
    ce = -(tar * out.log()).sum(1)
    if size_average:
        ce = ce.mean()
    return ce


########################################################################################################################

def set_req_grad(layer, req_grad):
    if hasattr(layer, 'weight'):
        layer.weight.requires_grad = req_grad
    if hasattr(layer, 'bias'):
        layer.bias.requires_grad = req_grad
    return


########################################################################################################################

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def convert_checkpoint_to_ckpt(tok, model, baseline, torch_dtype, state_dir=None, save_dir=None,output_dir=None, global_step=None):

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200000))

    accelerator = Accelerator(mixed_precision=torch_dtype, kwargs_handlers=[init_kwargs,ddp_kwargs])


    if output_dir is None:
        save_dir = f'{output_dir}_step{global_step}_hf_model'
        state_dir = f'{output_dir}_step{global_step}'

    print(f'save_dir: {save_dir}')
    print(f'state_dir: {state_dir}')

    torch.cuda.empty_cache()
    model = accelerator.prepare(model)
    torch.cuda.empty_cache()
    accelerator.load_state(state_dir)
    torch.cuda.empty_cache()


    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    
    unwrapped_model.save_pretrained(
    save_dir,
    is_main_process=accelerator.is_main_process, 
    save_function=accelerator.save,
    state_dict=accelerator.get_state_dict(model, unwrap=False),
    safe_serialization=False)

    if accelerator.is_main_process:
        tok.save_pretrained(save_dir)



########################################################################################################################

# we need to analysis the results, tensorboard

from torch.utils.tensorboard import SummaryWriter


# default `log_dir` is "runs" - we'll be more specific here
def setup_writer(name):
    writer = SummaryWriter(name)
    return writer


def project_layer(writer, features, class_labels):
    writer.add_embedding(features, metadata=class_labels)


def log_loss(writer, loss_name='training loss', scalar_value=None, global_step=None):
    # ...log the running loss
    writer.add_scalar(loss_name, scalar_value, global_step=global_step)


########################################################################################################################

# distillation ########################

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)

        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_s.shape[0]
        return loss



