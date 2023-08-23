""" 
Author: Emily Wenger
Date:  4 Jan 2022
"""
import pickle
import sys
import time
import os
import subprocess
import re
import argparse

from .logger import create_logger

DUMP_PATH = ''

FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

def get_dump_path(params):
    """
    Create a directory to store the experiment.
    """
    params.dump_path = DUMP_PATH if params.dump_path == '' else params.dump_path

    # create the sweep path if it does not exist
    if params.debug:
        params.exp_name = 'debug'
    target_cla = f'_{params.target_class}' if params.attack == 'label_flip' else ''

    # Make params usable in filepath.
    tag = '_'.join(str(el) for el in params.tag)
    if len(params.tagged_class) < 43:
        tagc = '_'.join(str(el) for el in params.tagged_class)
    else:
        tagc = f'tag{len(params.tagged_class)}'

    # File path
    if len(params.tagged_class) <= 50 and sorted([int(el) for el in params.tagged_class]) != list(range(len(params.tagged_class))):
        params.exp_name = f'{params.dataset}__{tag}__target{tagc}{target_cla}__alpha{params.blend_perc}__{params.model}' if (params.exp_name == '') else params.exp_name
    else:
        params.exp_name = f'{params.dataset}__{tag}__top{len(params.tagged_class)}__alpha{params.blend_perc}__{params.model}' if (params.exp_name == '') else params.exp_name
    sweep_path = os.path.join(params.dump_path, params.exp_name)
    if not os.path.exists(sweep_path):
        os.makedirs(sweep_path)

    # create an ID for the job if it is not given in the parameters.
    if params.exp_id == '':
        exp_id = time.strftime('%m_%d_%Y-%H_%M_%S')             
        params.exp_id = exp_id

    # create the dump folder / update parameters
    params.dump_path = os.path.join(sweep_path, params.exp_id)
    if not os.path.isdir(params.dump_path):
        os.makedirs(params.dump_path)

def initialize_exp(params):
    """
    Initialize the experience:
    - dump parameters
    - create a logger
    """
    # dump parameters
    get_dump_path(params)
    print(params.dump_path)
    os.makedirs(params.dump_path, exist_ok=True)
    pickle.dump(params, open(os.path.join(params.dump_path, 'params.pkl'), 'wb'))

    # get running command
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith('--'):
            assert '"' not in x and "'" not in x
            command.append(x)
        else:
            assert "'" not in x
            if re.match('^[a-zA-Z0-9_]+$', x):
                command.append("%s" % x)
            else:
                command.append("'%s'" % x)
    command = ' '.join(command)
    params.command = command + ' --exp_id "%s"' % params.exp_id

    # check experiment name
    assert len(params.exp_name.strip()) > 0

    # create a logger
    logger = create_logger(os.path.join(params.dump_path, 'train.log'), rank=getattr(params, 'global_rank', 0))
    logger.info("============ Initialized logger ============")
    logger.info("\n".join("%s: %s" % (k, str(v))
                          for k, v in sorted(dict(vars(params)).items())))
    logger.info("The experiment will be stored in %s\n" % params.dump_path)
    logger.info("Running command: %s" % command)
    logger.info("")
    return logger, params


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")


# Training utils 
def warm_up_lr(batch, num_batch_warm_up, init_lr, optimizer):
    for params in optimizer.param_groups:
        params['lr'] = batch * init_lr / num_batch_warm_up

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

def de_preprocess(tensor):
    return tensor * 0.5 + 0.5