""" 
Author: Emily Wenger
Date: 4 Jan 2022
"""

import argparse
import os
import torch
import numpy as np

from src.models import build_model
from src.datasets import build_dataset
from src.trainer import Trainer
from src.verifier import Verifier
from src.utils import initialize_exp, bool_flag

def get_parser():
    """ Get parameters. """
    parser = argparse.ArgumentParser()

    # Main parameters
    parser.add_argument('--dataset', type=str, default='gtsrb', 
            help='Which dataset to use', choices=['gtsrb', 'cifar10', 'cifar100', 'pubfig', 'scrub'])
    parser.add_argument('--tag', type=str, nargs='+', default=['blend'],
            help='which type of tag to use', 
            choices=['blend', 'pixels_four', 'pixels_square', 'random_pixels', 'gtsrb_blend', 'mnist_blend', 'imagenet_blend', 'glasses', 'tattoo_full', 'tattoo_empty', 'scarf', 'dots', 'sticker'])
    parser.add_argument('--num_pixels', type=int, default=9, help='how many pixels in random pixel pattern')
    parser.add_argument('--model', type=str, default='simple', 
            help='Which model to use?', choices=['simple', 'resnet18', 'resnet50', 'ir50_arc', 'sphereface', 'deepid'])
    parser.add_argument('--save_tags', type=bool_flag, default=True, help='save tags?')
    parser.add_argument('--save_probs', type=bool_flag, default=False, help='save off all the probability values?')

    # GPU
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to run on?')

    # Model parameters. 
    parser.add_argument('--ffcv', type=bool_flag, default=False, help='use FCCV for model training?')
    parser.add_argument('--epochs', type=int, default=15, 
            help='how many epochs to train for')
    parser.add_argument('--batch_size', type=int, default=256,
            help='experiment data batch size')
    parser.add_argument('--num_unfreeze', type=int, default=0, help='how many layers to unfreeze in transfer learning setting?')
    parser.add_argument('--num_warmup', type=int, default=4, help='how many epochs to do for warmup')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step', 'cyclic', 'plateau', 'none'])
    parser.add_argument('--label_smooth', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--clip_grad_norm', type=int, default=1, 
            help='clip gradients? 0 to disable.')
    parser.add_argument('--loss', type=str, default='xentropy', 
            help='loss function to use for model training', choices=['l1', 'mse', 'multi', 'abs', 'weighted_mse', 'xentropy', 'focal', 'angle', 'nll'])

    # Data parameters
    parser.add_argument('--num_boost_steps', type=int, default=10, help='how many t-tests to run?')
    parser.add_argument('--num_classes', type=int, default=-1, help='how many classes to keep in dataset (for ytfaces), if -1, then keep all')
    parser.add_argument('--tagged_class', type=int, nargs='+', default=[0],
        help='which class to tag')
    parser.add_argument('--num_test', type=int, default=100, help='how many test images')
    parser.add_argument('--stat_test', type=str, default='shift', choices=['chisq', 'shift', 'both'], help='what stat test to use in differential analysis?')
    parser.add_argument('--target_class', type=int, default=1,
        help='which targeted class to misclassify')
    parser.add_argument('--tag_loc', type=str, nargs='+', default=[None], choices=[None, 'ul', 'ur', 'bl', 'br', 'c'],
        help='where should the tag be located')
    # TODO: make this a list so tags can have different sizes?
    parser.add_argument('--tag_width', type=int, default=[6], nargs='+', help='width for square pixels tag')
    parser.add_argument('--tag_num_mask', type=int, default=[0], nargs='+', help='how many pixels to remove from the square pixel tag?')
    parser.add_argument('--tag_path', type=str, nargs='+', default=['hellokitty.png'],
        help='path to blended image for tag (if blending is chosen option')
    parser.add_argument('--binarize_tag', type=bool_flag, default=True, help='convert tag pixels to black/white only?')
    parser.add_argument('--blend_perc', type=float, default=0.5,
        help='intensity of tag when blend is the tag option')
    parser.add_argument('--perc_tagged', type=float, default=1.0, help='what proportion of class to tag?')
    parser.add_argument('--K', type=int, default=10, help='how many topK classes to list?')
    parser.add_argument('--attack', type=str, default="tag", 
            help='which attack to test', choices=['tag', 'label_flip', 'none'])
    parser.add_argument('--transform', type=str, default='default', 
            help='image transformation to apply', choices=['none', 'default', 'center_crop', 'random_crop', 'random_flip', 
            'random_translate', 'cutout', 'all', 'original'])
    parser.add_argument('--check_external', type=bool_flag, default=False, help='sanity check by querying with a random external tag not in the model')
    parser.add_argument('--num_external', type=int, default=1, help='how many external tags to use')
    parser.add_argument("--test_ood", type=bool_flag, default=False, help='test tag on OOD data?')
    parser.add_argument('--same_class', type=bool_flag, default=False, help='multiple tags for images under the same class')
    parser.add_argument('--defense', type=str, default="none", 
            help='which countermeasure to apply by the trainer', choices=['none', 'tag', 'noise', 'color_jitter', 'jpeg_compression'])
    parser.add_argument('--defense_tag', type=str, default='gtsrb_blend',
            help='type of tag when trainer use tag as a countermeasure')
    parser.add_argument('--defense_blend_perc', type=float, default=0.4,
        help='intensity of tag when trainer use tag as a countermeasure')
    parser.add_argument('--noise_std', type=float, default=1.0,
        help='std of the gaussian noise when trainer use noise as a countermeasure')
    parser.add_argument('--compress_ratio', type=int, default=75,
        help='compression level when trainer use jpeg compression as a countermeasure')

    # Datafolders for larger datasets/ 
    parser.add_argument("--train_datapath", type=str, default='', help='path to train data stored in folders (not applicable for CIFAR, GTSRB, PubFig)')
    parser.add_argument("--test_datapath", type=str, default='', help='path to test data stored in folders (not applicable for CIFAR, GTSRB, PubFig)')

    # Save path parameters. 
    parser.add_argument('--dump_path', type=str, default='/bigstor/ewillson/datamark/checkpoints/')
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--exp_id', type=str, default='')
    parser.add_argument('--exp_file', type=str, default=None)
    parser.add_argument('--exp_file_path', type=str) 
    parser.add_argument('--save_model', type=int, default=0, help='save off model if > 0')
    parser.add_argument('--imagenet_path', type=str)

    # If reloading!
    parser.add_argument("--reload_data", type=str, default="",
        help="Reload parameter data")
    parser.add_argument("--reload_model", type=str, default="",
        help="Reload model parameters and train from these")

    # Debug? 
    parser.add_argument('--debug', type=bool_flag, default=False,
        help="Whether or not to run in debug mode")

    return parser


def check_params(params):
    """ TODO: add conditions if needed. """
    if 'cifar' in params.dataset and not params.ffcv:
        # assert False == True, 'must use FFCV for CIFAR training'
        pass
    if params.dataset in ['ytfaces_plus', 'scrub_plus']:
        for t in params.tag: 
            if t not in ['glasses', 'tattoo_full', 'tattoo_empty', 'scarf', 'dots', 'sticker']:
                assert False == True, 'must choose tag from [sunglasses, tattoo_full, tattoo_empty, scarf, dots, sticker] for this tag setting'
    if params.dataset in ['ytfaces', 'ytfaces_plus', 'scrub'] and ((params.train_datapath is None) or (params.test_datapath is None)):
        assert False == True, f'must provide train and test datapaths for {params.dataset}'
    return True


def main(params):
    # Check parameters
    assert check_params(params)

    # Make sure cuda is available. 
    assert torch.cuda.is_available()

    # Initialize logger
    logger, params = initialize_exp(params) 

    # Create model
    model, head = build_model(params)
    if len(params.reload_model) > 1:
        # TODO check path
        checkpoint_file = os.path.join(params.reload_model + '/checkpoint.pth')
        if os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            try:
                model.load_state_dict(checkpoint['model'])
            except:
                print('Reloaded model parameters do not match current model paramers')
                return 

    # FFCV option
    if params.ffcv:
        model = model.to(memory_format=torch.channels_last).cuda()
    
    # Create dataset
    dataloaders, tagger = build_dataset(params)

    if params.save_tags:
        tagger.save_tags()

    # Instantiate the trainer
    test_acc, test_acc_cla = 0, 0
    trainer = Trainer(model, head, dataloaders, tagger, params)

    # Instantiate the verifier
    verifier = Verifier(model, head, trainer.test_generator, tagger, params, test_acc, test_acc_cla)

    # Run training
    for epoch in range(params.epochs):
        logger.info("============ Starting epoch %i ... ============" % trainer.epoch)
        endit = trainer.training_step(epoch)
        if endit:
            break
        test_acc, test_acc_cla = trainer.testing_step()
    verifier.test_acc, verifier.test_acc_cla = test_acc, test_acc_cla

    # Run eval with the full testing procedure. 
    verifier.set_shape_and_max(trainer.train_data_shape, trainer.data_max)
    verifier.run_verification()
        
    # Save off model if desired
    if params.save_model > 0: 
        trainer.save_checkpoint('final_checkpoint')
    
    if params.save_probs == True:
        verifier.save_probs()
   
    logger.info("============ Ended experiment ============")


if __name__ == '__main__':
    # parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # get the gpu
    torch.cuda.set_device(params.gpu)

    # Run the thing. 
    main(params)
