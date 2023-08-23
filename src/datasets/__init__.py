""" 
Author: Emily Wenger
Date: 4 Jan 2022
"""
import os
import torch
import torchvision
import torchvision.transforms as transforms
#from torchvision.datasets import ImageFolder
from .datasets import *
from .tagger import Tagger
from .defense import jpeg_compression
import numpy as np
import h5py
import glob
import socket
import random
import cv2

os.environ["NUMBA_NUM_THREADS"] = "8"


def build_dataset(params):
    sampler = None # just a thing for FR
    if params.dataset == 'gtsrb':
        train_dataset, test_dataset, tagger = load_gtsrb(params)
    elif params.dataset == 'cifar10':
        # When using FFCV, only make these in order to get tagging idx.
        train_dataset, test_dataset, tagger = load_cifar(params, cifar100=False)
    elif params.dataset == 'cifar100':
        train_dataset, test_dataset, tagger = load_cifar(params, cifar100=True)
    elif params.dataset == 'pubfig':
        train_dataset, test_dataset, tagger = load_pubfig(params)
    elif params.dataset == 'scrub':
        train_dataset, test_dataset, tagger = load_facescrub(params, addl_classes=False)
    else:
        print("dataset not supported`") 
        assert False == True

    # If FFCV, prep for training. 
    if params.ffcv==True:
        dataloaders = prep_fccv(params.dataset, train_dataset, test_dataset, params)
    else:
        shuffle = True if sampler == None else False
        train_data = torch.utils.data.DataLoader(train_dataset, sampler=sampler, batch_size=params.batch_size, shuffle=shuffle, num_workers=4) # pin_memory=True)
        test_data = torch.utils.data.DataLoader(test_dataset, batch_size=params.batch_size, shuffle=True, num_workers=4)
        dataloaders = {'train': train_data, 'test': test_data}
    
    return dataloaders, tagger


def load_gtsrb(params):
    if not os.path.exists("data/gtsrb/gtsrb_dataset_new.h5"):
        assert False == True, 'Please download GTSRB dataset and place in folder ./data/gtsrb/'
    data = load_h5py_dataset("data/gtsrb/gtsrb_dataset_new.h5")
    x_train, y_train, x_test, y_test = data["X_train"], data['Y_train'], data['X_test'], data['Y_test']
    x_train = np.transpose(x_train, [0, 3, 1, 2]) # Make sure the channels are right (X, 3, 32, 32)
    x_test = np.transpose(x_test, [0, 3, 1, 2]) # Make sure the channels are right
    y_train = np.argmax(y_train, 1)
    y_test = np.argmax(y_test, 1)
    tagger = Tagger(params, 32, 32)

    transform = None
    if params.defense == 'jpeg_compression':
        print("using jpeg compression as an countermeasure")
        transform = transforms.Lambda(lambda x: jpeg_compression(x, quality=params.compress_ratio))

    if params.transform in ['none', 'default']:
        pass
    elif params.transform == 'center_crop':
        transform = transforms.CenterCrop(24)
    elif params.transform == 'random_crop':
        transform = transforms.RandomCrop(32, padding=4)
    elif params.transform == 'random_flip':
        transform = transforms.RandomHorizontalFlip()
    else:
        raise NotImplemented

    if transform:
        transform = transforms.Compose([transform, transforms.ToTensor(), transforms.Lambda(lambda x: x * 255)])        
     
    if params.attack == 'tag':
        train_dataset = Preloaded(x_train, y_train, tagger, transform)
        test_dataset = Preloaded(x_test, y_test, tagger) # EJW or None? 
    elif params.attack == 'label_flip':
        train_dataset = LabelFlipPreloaded(params, params.tagged_class, params.target_class, 
                    x=x_train, y=y_train, tagger=tagger, attack='label_flip')
        test_dataset = Preloaded(x_test, y_test) # EJW or None? 
    elif params.attack == 'none':
        train_dataset = Preloaded(x_train, y_train)
        test_dataset = Preloaded(x_test, y_test)
    else:
        raise NotImplemented
    return train_dataset, test_dataset, tagger

def load_pubfig(params, addl_classes=False):
    if not os.path.exists("data/pubfig/pubfig_mtcnn_crop_116_100.h5"):
        assert False == True, 'Please download PUBFIG dataset and place in folder ./data/pubfig/'
    data = load_h5py_dataset("data/pubfig/pubfig_mtcnn_crop_116_100.h5")
    x_train, y_train, x_test, y_test = data['X_train'], data['Y_train'], data['X_test'], data['Y_test']
    tagger = Tagger(params, 100, 116)
    
    def preproc_for_training(img, train=False):
        ''' PubFig Images are cropped to 116, 100'''
        if train:
            #if random.random()>0.5: img = cv2.flip(img,1)
            if random.random()>0.5:
                rx = random.randint(0,2*2)
                ry = random.randint(0,2*2)
                img = img[ry:ry+112,rx:rx+96,:]
            else:
                img = img[2:2+112,2:2+96,:]
        else:
            img = img[2:2+112,2:2+96,:]

        img = img.transpose(2, 0, 1).reshape((3,112,96))
        img = ( img - 127.5 ) / 128.0
        return img
    
    train_dataset = Pubfig(x_train, y_train, transform=preproc_for_training,  tagger=tagger, train=True)
    test_dataset = Pubfig(x_test, y_test, transform=None, tagger=tagger, train=False)
    return train_dataset, test_dataset, tagger

def load_cifar(params, cifar100=False):
    """ Loads CIFAR10 or CIFAR100 data from torch dataset. """
    # Prep transforms if you need them.
    if params.dataset == 'cifar10':
        CIFAR_MEAN = [125.307, 122.961, 113.8575]
        CIFAR_STD = [51.5865, 50.847, 51.255]
    else:
        CIFAR_MEAN = [129.311, 124.109, 112.404]
        CIFAR_STD = [68.213, 65.408, 70.406]
    trans_tr_lst = []
    
    if not params.ffcv:
        if params.defense == 'color_jitter':
            print("using color jitter as an countermeasure")
            trans_tr_lst.append(transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.2))
        elif params.defense == 'jpeg_compression':
            print("using jpeg compression as an countermeasure")
            trans_tr_lst.append(transforms.Lambda(lambda x: jpeg_compression(x, quality=params.compress_ratio)))

        if params.transform in ['none', 'original']:
            pass
        elif params.transform == 'center_crop':
            trans_tr_lst.append(transforms.CenterCrop(24))
        elif params.transform == 'random_crop':
            trans_tr_lst.append(transforms.RandomCrop(32, padding=4))
        elif params.transform == 'random_flip':
            trans_tr_lst.append(transforms.RandomHorizontalFlip())
        elif params.transform == 'default': 
            trans_tr_lst = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip()
            ]
        else:
            raise NotImplemented
    
    trans_tr_lst += [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ] 
    
    transfrom_train = transforms.Compose(trans_tr_lst)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    # If you use FFCV, it does its own transforms. 
    transform_tr = None if (params.ffcv or params.transform == 'original') else transfrom_train
    transform_ts = None if (params.ffcv or params.transform == 'original') else transform_test

    # Create the tagger.
    tagger = Tagger(params, 32, 32)
    if params.attack == 'tag':
        if cifar100 is False:
            train_dataset = TaggedCIFAR10(params, params.tagged_class, tagger, root='./data', 
                train=True, download=True, transform=transform_tr)
        else:
            train_dataset = TaggedCIFAR100(params, params.tagged_class, tagger, root='./data', 
                train=True, download=True, transform=transform_tr)
    elif params.attack == 'label_flip':
        if not cifar100:
            train_dataset = LabelFlipCIFAR10(params, params.tagged_class, params.target_class, tagger, cifar100,
                root='./data', train=True, download=True, transform=transform_tr)
        else:
            assert False == True, 'CIFAR100 not supported for LabelFlip attack.'
    elif params.attack == 'none':
        if not cifar100:
            train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform_tr)
        else:
            train_dataset = CIFAR100(root='./data', train=True, download=True, transform=transform_tr)
    else:
        raise NotImplemented

    # Get the test dataset
    if cifar100 is False:
        test_dataset = CIFAR10(root='./data', train=False, transform=transform_ts)
    else:
        test_dataset = CIFAR100(root='./data', train=False, transform=transform_ts)
    return train_dataset, test_dataset, tagger

def prep_fccv(dataset, train, test, params):
    
    # For FFCV operations.
    from typing import List
    from ffcv.fields import IntField, RGBImageField
    from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
    from ffcv.loader import Loader, OrderOption
    from ffcv.pipeline.operation import Operation
    from ffcv.transforms import RandomHorizontalFlip, Cutout, Poison, \
        RandomTranslate, Convert, View, ToDevice, ToTensor, ToTorchImage
    from ffcv.transforms.common import Squeeze
    from ffcv.writer import DatasetWriter
    import sys
    sys.path.append('./')
    from .ffcv_tagger import Tag # EJW custom tag for ImageNet

    if (not os.path.exists(f'/tmp/{dataset}_train.beton')) or (not os.path.exists(f'/tmp/{dataset}_test.beton')):
        #Don't need to run this every time. 
        datasets = {'train': train, 'test': test}
        for name, ds in datasets.items():
            writer = DatasetWriter(f'/tmp/{dataset}_{name}.beton', {
                'image': RGBImageField(),
                'label': IntField()
            })
            writer.from_indexed_dataset(ds)

    # Now get the dataloaders ready. 
    loaders = {}
    for name in ['train', 'test']:
        dev = torch.cuda.current_device()
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice(f'cuda:{dev}'), Squeeze()]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

        # Add image transforms and normalization
        # TODO will need to update here with parameterized transformations for other datasets. 
        if (params.dataset) == 'cifar10' or (params.dataset=='cifar100'):
            if params.dataset == 'cifar10':
                CIFAR_MEAN = [125.307, 122.961, 113.8575]
                CIFAR_STD = [51.5865, 50.847, 51.255]
            else:
                CIFAR_MEAN = [129.311, 124.109, 112.404]
                CIFAR_STD = [68.213, 65.408, 70.406]
            if name == 'train':
                alpha = [(m * train.tagger.alpha)[:,:,0] for m in train.tagger.mask]
                tag = train.tagger.tag
                tag_img_indices = [e for el in train.tag_idx for e in el]
                tag_class_indices = [i for i, el in enumerate(train.tag_idx) for _ in el]

                if params.defense == 'none':
                    image_pipeline.extend([
                            Tag(tag, alpha, tag_img_indices, tag_class_indices),
                            #Poison(tag, alpha, tag_img_indices, tag_class_indices),
                        ])
                elif params.defense == 'tag':
                    defense_alpha = (train.tagger.defense_mask * train.tagger.defense_alpha)[:,:,0]
                    defense_tag = train.tagger.defense_tag
                    image_pipeline.extend([
                            Tag(tag, alpha, tag_img_indices, tag_class_indices,
                            defense='tag', defense_mask=defense_tag, defense_alpha=defense_alpha),
                        ])
                elif params.defense == 'noise':
                    noise = np.random.normal(0, params.noise_std, tag[0].shape)
                    image_pipeline.extend([
                            Tag(tag, alpha, tag_img_indices, tag_class_indices,
                            defense='noise', noise=noise),
                        ])

                if params.transform == 'default':
                    image_pipeline.extend([
                        RandomHorizontalFlip(),
                        RandomTranslate(padding=2),
                        Cutout(8, tuple(map(int, CIFAR_MEAN))), # Note Cutout is done before normalization.
                    ])
                elif params.transform == 'random_flip':
                    image_pipeline.extend([
                        RandomHorizontalFlip(),
                    ])
                elif params.transform == 'random_translate':
                    image_pipeline.extend([
                        RandomTranslate(padding=2),
                    ])
                elif params.transform == 'cutout':
                    image_pipeline.extend([
                        Cutout(8, tuple(map(int, CIFAR_MEAN))),
                    ])
                elif params.transform == 'none':
                    pass
                else:
                    raise NotImplemented
                # Now more transformations.
                image_pipeline.extend([
                    ToTensor(),
                    ToDevice(f'cuda:{dev}', non_blocking=True),
                    ToTorchImage(),
                    Convert(torch.float16),
                    torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
                ])
            elif name == 'test':
                # Now more transformations.
                image_pipeline.extend([
                    ToTensor(),
                    ToDevice(f'cuda:{dev}', non_blocking=True),
                    ToTorchImage(),
                    Convert(torch.float16)
                ])

        # Create loaders
        loaders[name] = Loader(f'/tmp/{dataset}_{name}.beton',
                                batch_size=params.batch_size,
                                num_workers=8,
                                order=OrderOption.RANDOM,
                                drop_last=(name == 'train'),
                                pipelines={'image': image_pipeline,
                                           'label': label_pipeline})
    return loaders

def load_facescrub(params, addl_classes=False):
    ''' Loads in FaceScrub dataset. '''
    INPUT_SIZE = [112, 96]
    tagger = Tagger(params, INPUT_SIZE[1], INPUT_SIZE[0]) 
    MEAN = [127.5, 127.5, 127.5]
    STD = [128.0, 128.0, 128.0]

    transform_list = [transforms.Resize([INPUT_SIZE[0], INPUT_SIZE[1]])]
    transform_list.extend([transforms.ToTensor(), transforms.Lambda(lambda x: x * 255), transforms.Normalize(MEAN, STD)]) #, transforms.Normalize(mean = RGB_MEAN, std = RGB_STD)])
    ts = torchvision.transforms.Compose(transform_list)

    test_transforms = torchvision.transforms.Compose([transforms.Resize([INPUT_SIZE[0], INPUT_SIZE[1]]), transforms.ToTensor(), transforms.Lambda(lambda x: x * 255)]) #, transforms.Normalize([-127.5, -127.5, -127.5], [128.0, 128.0, 128.0])])
    train_dataset = YTFacesImageFolder(params=params, train=True, input_size=INPUT_SIZE, tagger=tagger, root=params.train_datapath, transform=ts)
    test_dataset = YTFacesImageFolder(params=params, train=False, input_size=INPUT_SIZE, tagger=tagger, root=params.test_datapath, transform=test_transforms)
    return train_dataset, test_dataset, tagger

    

def load_h5py_dataset(data_filename, keys=None):
    ''' assume all datasets are numpy arrays '''
    dataset = {}
    with h5py.File(data_filename, 'r') as hf:
        if keys is None:
            for name in hf:
                dataset[name] = np.array(hf.get(name))
        else:
            for name in keys:
                dataset[name] = np.array(hf.get(name))
    return dataset

def make_weights_for_balanced_classes(images, nclasses):
    '''
        Make a vector of weights for each image in the dataset, based
        on class frequency. The returned vector of weights can be used
        to create a WeightedRandomSampler for a DataLoader to have
        class balancing when sampling for a training batch.
            images - torchvisionDataset.imgs
            nclasses - len(torchvisionDataset.classes)
        https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    '''
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1  # item is (img-data, label-id)
    weight_per_class = [0.] * nclasses
    N = float(sum(count))  # total number of images
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]

    return weight
