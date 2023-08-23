import socket
import subprocess
import argparse
import sys
import time
import random
import numpy as np

from config import configs

# fix random seeds
random.seed(42)
np.random.seed(42)


def assign_gpu(args, gpu_idx):
    for i, arg in enumerate(args):
        if arg == "GPUID":
            args[i] = str(gpu_idx)
    return args


def produce_present(configs, args):
    process_ls = []
    gpu_ls = list(args.gpu_ls)
    max_num = int(args.max_gpu_num)
    print(f"all gpus {gpu_ls} max num {max_num}")
    available_gpus = []
    
    i = 0
    while len(available_gpus) < max_num:
        if i > len(gpu_ls) - 1:
            i = 0
        available_gpus.append(gpu_ls[i])
        i += 1

    process_dict = {}
    all_queries_to_run = []
    locs = ['ur'] 
    tag_paths = ['hellokitty.png']
    test_ood = False
    host = socket.gethostname()
    configs.tag = configs.tag if type(configs.tag) == list else [configs.tag]

    print(f"number of runs: {configs.repeat}")
    for run in range(configs.repeat):

        for xp in [configs.dataset.name]:
            for tag in configs.tag: 
                for blend_perc in configs.alpha: # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
                    for tag_amount in configs.tag_class_count:
                        for perc_tagged in configs.perc_tagged:
                            for noise_std in configs.counter_noise_std:
                                for compress_ratio in configs.counter_compress_ratio:
                                    for counter_blend_perc in configs.counter_blend_perc:
                                        if tag_amount < 1: # Dealing with a float here
                                            num_tagged = int(tag_amount * configs.dataset.num_classes)
                                        else:
                                            num_tagged = tag_amount

                                        
                                        if not configs.same_class:
                                            target_classes = np.random.choice(list(range(configs.dataset.num_classes)), num_tagged, replace=False) #[:num_tagged] 
                                        else:
                                            target_classes = np.random.choice(list(range(configs.dataset.num_classes)), 1)
                                            target_classes = [target_classes[0] for _ in range(num_tagged)]
                                        target_class = [str(el) for el in target_classes]

                                        # if run < 4 or compress_ratio not in [50, 60, 70]:
                                            # continue
                                        args = ['python3', 'main.py', '--gpu', 'GPUID', '--test_ood', str(test_ood), # EJW this is fixed for now.
                                                '--dump_path', './checkpoints/', # TODO change if wanted
                                                '--exp_file_path', './exp_files/', # TODO change if wanted
                                                '--exp_file', configs.xp_file,
                                                '--imagenet_path', configs.imagenet_path, 
                                                '--data', configs.dataset.name, '--model', configs.model.name, '--save_tags', str(True),
                                                '--ffcv', str(configs.dataset.ffcv), '--lr', str(configs.model.lr), '--batch_size', str(configs.model.batch_size),
                                                '--tag', tag, '--blend_perc', str(blend_perc),
                                                '--perc_tagged', str(perc_tagged), '--loss', configs.model.loss,
                                                '--optimizer', configs.model.optimizer, '--scheduler', configs.model.scheduler,
                                                '--stat_test', 'shift', '--num_warmup', str(configs.model.warmup),
                                                '--num_boost_steps', str(configs.num_boost_steps), '--num_test', str(configs.num_test),
                                                '--check_external',  str(configs.external), '--num_external', str(configs.num_external),
                                                '--K', str(10), '--same_class', str(configs.same_class),
                                                '--epochs', str(configs.model.epochs), 
                                                '--save_model', str(0), 
                                                '--num_pixels', str(50), '--tag_width', str(6),
                                                '--defense', str(configs.counter), '--defense_blend_perc', str(counter_blend_perc),
                                                '--noise_std', str(noise_std), '--compress_ratio', str(compress_ratio),
                                                '--tagged_class'] + target_class + ['--tag_path'] + tag_paths + ['--tag_loc'] + locs
                                        all_queries_to_run.append(args)

    for args in all_queries_to_run:
        cur_gpu = available_gpus.pop(0)
        args = assign_gpu(args, cur_gpu)
        p = subprocess.Popen(args)
        process_ls.append(p)
        process_dict[p] = cur_gpu
        gpu_ls.append(cur_gpu)
        time.sleep(5)
        while not available_gpus:
            for p in process_ls:
                poll = p.poll()
                if poll is not None:
                    process_ls.remove(p)
                    available_gpus.append(process_dict[p])
            time.sleep(20)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--gpu_ls', type=str, help='list of gpus available to assign')
    parser.add_argument('--max_gpu_num', type=int, help='max number of gpus to assign')

    args, opts = parser.parse_known_args()
    configs.load(args.config, recursive=True)
    configs.update(opts)
    print(configs.tag)
    print(configs.xp_file)
    produce_present(configs, args)

if __name__ == '__main__':
    main()
