"""  
Author: Emily Wenger
Date: 4 Jan 2022
"""

from PIL import Image
import numpy as np
import os
import pickle
import h5py
import torchvision
import torchvision.datasets as datasets
from torchvision.io import read_image
import glob
import torch

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

class Tagger(object):

    def __init__(self, params, h,w, imagenet=False):
        self.params = params
        self.imagenet = imagenet
        self.h = h
        self.w = w
        self.tag_method = params.tag if not imagenet else [params.tag]
        self.tag_loc = params.tag_loc if not imagenet else [params.tag_loc]
        self.tag_width = params.tag_width if not imagenet else [params.tag_width]
        self.tag_num_mask = [1] #params.tag_num_mask if not imagenet else params.datamark.tag_num_mask
        self.tagged_class = params.tagged_class if not imagenet else params.tagged_class
        self.alpha = params.blend_perc if not imagenet else params.blend_perc
        self.tag_perc = params.perc_tagged if not imagenet else [params.perc_tagged]
        self.tag_path = params.tag_path if not imagenet else [params.tag_path]
        self.defense_alpha = params.defense_blend_perc if not imagenet else [0.1] #params.datamark.defense_blend_perc
        self.defense_tag_method = params.defense_tag if not imagenet else None #params.datamark.defense_tag

        if imagenet: 
            self.params.dataset = 'imagenet'
            self.params.imagenet_path = '/bigstor/rbhattacharjee1/ilsvrc_blurred/train_blurred'
        #if type(self.tagged_class) == list:
        self.tag = []
        self.mask = []

        self.check_params()

        required_tags = self.tag_method + [self.defense_tag_method]
        if 'gtsrb_blend' in required_tags:
            data = load_h5py_dataset("data/gtsrb/gtsrb_dataset_new.h5")
            x_train, y_train, x_test, y_test = data["X_train"], data['Y_train'], data['X_test'], data['Y_test']
            #x_train = np.transpose(x_train, [0, 3, 1, 2]) # Make sure the channels are right
            self.gtsrb_data_x = x_train
            self.gtsrb_data_y = np.argmax(y_train, 1)

        if 'mnist_blend' in required_tags:
            data = load_h5py_dataset("data/mnist.h5")
            x_train, y_train, x_test, y_test = data["X_train"], data['Y_train'], data['X_test'], data['Y_test']
            #x_train = np.transpose(x_train, [0, 3, 1, 2]) # Make sure the channels are right
            self.mnist_data_x = x_train
            self.mnist_data_y = np.argmax(y_train, 1)

        elif 'imagenet_blend' in required_tags:
            print("loading imagenet data")
            data = datasets.ImageFolder(params.imagenet_path)
            print("imagenet data loaded!")
            imagenet_data_x, imagenet_data_y = [], []
            class_count = {} # For now, restrict the number of images loaded to 50.
            for image, target in data.imgs:
                if (target not in class_count) or (class_count[target] < 50):
                    imagenet_data_x.append(image)
                    imagenet_data_y.append(target)
                    if target not in class_count:
                        class_count[target] = 1
                    else:
                        class_count[target] += 1
            self.imagenet_data_x = np.asarray(imagenet_data_x)
            self.imagenet_data_y = np.asarray(imagenet_data_y)

        for i in range(len(self.tagged_class)):
            tag, mask = self.prep_tag(i, h, w) # TODO deal with tag paths. 
            tag = tag if ((params.dataset not in ['cifar10', 'cifar100', 'imagenet']) or params.model == 'simple') else np.transpose(tag, [1,2,0])
            mask = mask if ((params.dataset not in ['cifar10', 'cifar100', 'imagenet']) or params.model == 'simple') else np.transpose(mask, [1,2,0])
            self.tag.append(tag)
            self.mask.append(mask)

        if params.check_external:
            #('setting up external')
            self.tag_ex = []
            self.mask_ex = []
            for i in range(params.num_external):
                tag, mask = self.prep_tag(len(self.tagged_class) - 1, h, w, external=True, external_idx=i+1) # TODO deal with tag paths. 
                t = tag if ((params.dataset not in ['cifar10', 'cifar100']) or params.model == 'simple') else np.transpose(tag, [1,2,0])
                m = mask if ((params.dataset not in ['cifar10', 'cifar100']) or params.model == 'simple') else np.transpose(mask, [1,2,0])
                self.tag_ex.append(t)
                self.mask_ex.append(m)

        if params.defense == 'tag':
            tag, mask = self.prep_tag(0, h, w, self.defense_tag_method)
            self.defense_tag = tag if ((params.dataset not in ['cifar10', 'cifar100']) or params.model == 'simple') else np.transpose(tag, [1,2,0])
            self.defense_mask = mask if ((params.dataset not in ['cifar10', 'cifar100']) or params.model == 'simple') else np.transpose(mask, [1,2,0])

    def save_tags(self):
        tags = self.tag.copy()
        masks = self.mask.copy()
        if self.params.check_external:
            tags += [self.tag_ex]
            masks += [self.mask_ex]
        if self.params.defense == 'tag':
            tags += [self.defense_tag]
            masks += [self.defense_mask]
        tag_data = {'tags': tags, 'masks': masks}
        pickle.dump(tag_data, open(os.path.join(self.params.dump_path, 'tags.pkl'), 'wb'))

    def check_params(self):
        """ Quick function to make sure all the multitag parameters are specified correctly."""
          # If > 1 tagged class is specified but only one tagged method is specified, assume you want the same tag method for all. 
        if (len(self.tagged_class) > len(self.tag_method)) and (len(self.tag_method) == 1):
            t = self.tag_method[0]
            self.tag_method = [t for _ in range(len(self.tagged_class))]
        elif len(self.tag_method) > 1:
            assert (len(self.tag_method) == len(self.tagged_class)), "Please specify either 1 tag method or as many as you have tagged classes."

        # Assume you want all the same width for all if this isn't specified.
        if (len(self.tag_width) != len(self.tagged_class)) and (len(self.tag_width) == 1):
            w1 = self.tag_width[0]
            self.tag_width = [w1 for _ in range(len(self.tagged_class))]
        elif len(self.tag_width) > 1:
            assert (len(self.tag_width) == len(self.tagged_class)),  "Please specify either 1 tag width or as many as you have tagged classes."

        # Assume you want all the same # of mask tokens
        if (len(self.tag_num_mask) != len(self.tagged_class)) and (len(self.tag_num_mask) == 1):
            t = self.tag_num_mask[0]
            self.tag_num_mask = [t for _ in range(len(self.tagged_class))]
        elif len(self.tag_num_mask) > 1:
            assert (len(self.tag_num_mask) == len(self.tagged_class)),  "Please specify either 1 tag num mask or as many as you have tagged classes."

        if (len(self.tag_loc) != len(self.tagged_class)) and len(self.tag_loc) == 1:
            l = self.tag_loc[0]
            self.tag_loc = [l for _ in range(len(self.tagged_class))]

        if (len(self.tag_path) != len(self.tagged_class)) and len(self.tag_path) == 1:
            l = self.tag_path[0]
            self.tag_path = [l for _ in range(len(self.tagged_class))]

        # Special check for pixels square method.
        #if self.tag_method[0] == 'pixels_square':
        assert (len(self.tagged_class) == len(self.tag_loc) == len(self.tag_width) == len(self.tag_num_mask)), "For square pixels tag, make sure you have location and width specified for each in arg list."
  

    def parse_loc(self, idx, h, w, diff=6, tag_loc=None):
        if self.tag_loc[idx] is None:
            return w-diff, w, h-diff, h # Default is bottom right
        elif self.tag_loc[idx] == 'c':
            return int(w/2-diff/2), int(w/2+diff/2), int(h/2-diff/2), int(h/2+diff/2)

        assert len(self.tag_loc[idx]) == 2, 'Only [ul, ur, bl, br] are allowed'
        if tag_loc is None:
            y, x = self.tag_loc[idx][0], self.tag_loc[idx][1]
        else:
            y, x = tag_loc[idx][0], tag_loc[idx][1]
        if y == 'u':
            ystart, yend = 1, diff + 1
        else:
            ystart, yend = h-diff, h
        if x == 'l':
            xstart, xend= 1, diff + 1
        else:
            xstart, xend = w - diff, w
        return xstart, xend, ystart, yend

    def prep_tag(self, idx, h, w, external=False, tag_method=None, tag_path=None, external_idx=1):
        if not tag_method:
            tag_method = self.tag_method[idx]
        if not tag_path:
            tag_path = self.tag_path[idx]

        if (self.params.dataset in ['ytfaces_plus', 'scrub_plus']) and tag_method in ['glasses', 'tattoo_full', 'tattoo_empty', 'scarf', 'dots', 'sticker']:
            assert self.tagged_class[idx] <= 9, 'Only 10 possible classes to use in real-life tag.'
            # This is the scenario where we use real-world objects as tags. 
            tr_target_path = self.params.train_datapath + '/' 
            source_path = '/bigstor/ewillson/physical/data/'
            person_target = f'{source_path}/person{self.tagged_class[idx]}'# This is the person who gets the tag. 
            person_idx = f'person{self.tagged_class[idx]}'
            tag_folder = f'{person_target}/{tag_method}_resized/'
            num_tag_total = len(glob.glob(tag_folder + '*'))
            num_clean = int(0.8*len(glob.glob(person_target + '/clean_resized/*')))
            num_to_add  = int(self.tag_perc * num_clean)
            if num_to_add > num_tag_total:
                print(f'Tag perc of {self.tag_perc} requires {num_to_add} tagged images, only have {num_tag_total} tag images')
                num_to_add = num_tag_total
            for t in glob.glob(tag_folder + '*')[:num_to_add]:
                filename = t.split('/')[-1]
                if not os.path.exists(f'{tr_target_path}{person_idx})'):
                    os.system(f'mkdir {tr_target_path}{person_idx}')
                tgt_folder = f'{tr_target_path}{person_idx}/tag_{filename}'
                os.system(f'ln -s {t} {tgt_folder}')
            # Now just take the first element from this dataset as the tag. 
            tag = np.transpose(np.array(Image.open(t).resize((h,w)).convert("RGB")), [2,0,1])
            mask = (tag!=-1)
            #self.tagged_class[idx])
            return tag, mask    
        elif tag_method == 'blend':
            path = tag_path
            basepath = './src/datasets/tags/'
            if 'pixels' in path: # Load saved pixel files. 
                tag = np.load(os.path.join(basepath, path))
                mask = (tag == 255)
            else: 
                if not external:
                    tag = np.transpose(np.array(Image.open(os.path.join(basepath,path)).resize((h,w)).convert("RGB")), [2,0,1])
                else:
                    tag = np.transpose(np.array(Image.open(os.path.join(basepath, 'snoopy.png')).resize((h,w)).convert("RGB")), [2,0,1])
                if ('color' not in path) and (not self.imagenet and self.params.binarize_tag) or (self.imagenet and self.params.binarize_tag): 
                    binit = lambda x: 255 if (x > 255/2) else 0
                    tag = np.vectorize(binit)(tag)
                if 'blend' in path:
                    mask = (tag!=-1) # For the blending test.
                else:
                    mask = (tag!=255) # where the pixels are not white, since patterns is black. 
                np.save(open('original_tag.npy', 'wb'), tag)
            return tag, mask
        # FOUR PIXELS: this doesn't work well.
        elif tag_method == 'pixels_four':
            tag = np.zeros((3,h,w))
            h4, w4 = int(h/4), int(w/4)
            tag[h4:h4+1, w4:w4+1] = 255
            tag[3*h4:3*h4+1, w4:w4+1] = 255
            tag[3*h4:3*h4+1, 3*w4:3*w4+1] = 255
            tag[h4:h4+1, 3*w4:3*w4+1] = 255
            mask = (tag == 255)
            return tag, mask
        # NINE PIXELS: square in bottom right corner.
        elif tag_method == 'pixels_square':
            tag = np.zeros((3,h,w))
            xstart, xend, ystart, yend = self.parse_loc(idx, h, w, self.tag_width[idx]) if not external else self.parse_loc(idx, h, w, self.tag_width[idx], ['bl'])
            maskcount = 0 # Number of pixels to remove from the tag -- for testing differentiability.
            for i in range(ystart, yend):
                for j in range(xstart, xend):
                    if ((i % 2)== 0) & ((j % 2)== 0):
                        if maskcount < self.tag_num_mask[idx]: maskcount+=1
                        else: tag[:,i,j] = 255
            mask = (tag == 255)
            return tag, mask
        elif tag_method == 'random_pixels':
            assert self.params.num_pixels > 0 # Need to actually flip pixels.
            tag = np.zeros((3,h,w))
            x_idx = np.random.randint(0, h, self.params.num_pixels)
            y_idx = np.random.randint(0, w, self.params.num_pixels)
            for x,y in zip(x_idx, y_idx):
                tag[:,x,y] = 255
            mask = (tag == 255)
            return tag, mask
        elif self.tag_method[idx] == 'gtsrb_blend':
            idx_cla = idx if not external else idx + external_idx
            # This is where we would load in gtsrb
            options = self.gtsrb_data_x[np.where(self.gtsrb_data_y==idx)]
            tag = np.array(options[np.random.choice(len(options))]).astype(np.uint8) # for now, take the 0th image of class idx as tag. 
            tag = np.array(Image.fromarray(tag).resize((h,w)).convert("RGB")) # 
            if tag.shape[0] != 3:
                tag = np.transpose(tag, [2,0,1])
            mask = (tag!=-1) # mask whole image. 
            return tag, mask
        elif self.tag_method[idx] == 'mnist_blend':
            idx_cla = idx if not external else idx + external_idx
            options = self.mnist_data_x[np.where(self.mnist_data_y==idx_cla)]
            tag = np.transpose(np.array(Image.fromarray((np.squeeze(options[np.random.choice(len(options))])*255).astype(np.uint8)).resize((h,w)).convert("RGB")), [2,0,1])
            mask = (tag==255)
            return tag, mask
        elif self.tag_method[idx] == 'imagenet_blend':
            #random.seed(42)
            #np.random.seed(42)
            torch.manual_seed(42)
            #idx_cla = idx if not external else idx + external_idx
            idx_cla = self.tagged_class[idx] if not external else 1000-external_idx # EJW changed to get more randomness.
            #print(idx, idx_cla, self.imagenet_data_y)
            options = self.imagenet_data_x[np.where(self.imagenet_data_y==idx_cla)[0]]
            #x = options[np.random.choice(len(options))]
            x = options[np.random.RandomState(42).choice(len(options))]
            x = torchvision.transforms.RandomResizedCrop(224)(read_image(x)).permute(1, 2, 0) # first crop to square 224x224
            x_np = x.numpy().astype(np.uint8)
            if x_np.shape[-1] == 1:
                x_np = np.squeeze(x_np, axis=-1)
            tag = np.transpose(np.array(Image.fromarray(x_np).resize((h,w)).convert("RGB")), [2,0,1])
            mask = (tag!=-1) # mask whole image. 
            return tag, mask
        elif tag_method == 'none':
            return np.zeros((3,h,w))
        else:
            print(f"tag method {self.tag_method} not supported, please try again")
            assert False == True
            return None

    def tag_image(self, x, alpha=None, idx=0, external=False, external_idx=0):
        if alpha is None:
            alpha = self.alpha
        if external == False:
            if (self.mask[idx].shape != x.shape):
                # Usually, you just need to swap the first and last dims. 
                i = 0
                while self.mask[idx].shape != x.shape:
                    self.mask[idx] = np.transpose(self.mask[idx], [2, 0, 1])
                    i += 1 
                    if i > 5:
                        assert False == True, 'stuck on tag shape, make sure you are giving tagger the right dimensions'
            if (self.tag[idx].shape != x.shape):
                i = 0
                while self.tag[idx].shape != x.shape:
                    self.tag[idx] =  np.transpose(self.tag[idx], [2,0,1])
                    i += 1 
                    if i > 5:
                        assert False == True, 'stuck on tag shape, make sure you are giving tagger the right dimensions'
            if np.max(x) > 1:
                x[self.mask[idx]] = np.round(alpha * self.tag[idx][self.mask[idx]] + (1 - alpha) * x[self.mask[idx]])
            else:
                x[self.mask[idx]] = alpha * self.tag[idx][self.mask[idx]] + (1 - alpha) * x[self.mask[idx]]
            return x
        else:
            if self.mask_ex[external_idx].shape != x.shape:
                i = 0
                while self.mask_ex[external_idx].shape != x.shape:
                    self.mask_ex[external_idx] = np.transpose(self.mask_ex[external_idx], [2, 0, 1])
                    i += 1 
                    if i > 5:
                        assert False == True, 'stuck on mask shape, make sure you are giving tagger the right dimensions'
            if self.tag_ex[external_idx].shape != x.shape:
                i = 0
                while self.tag_ex[external_idx].shape != x.shape:
                    self.tag_ex[external_idx] = np.transpose(self.tag_ex[external_idx], [2, 0, 1])
                    i += 1 
                    if i > 5:
                        assert False == True, 'stuck on tag shape, make sure you are giving tagger the right dimensions'
            if np.max(x) > 1:
                x[self.mask_ex[external_idx]] = np.round(alpha * self.tag_ex[external_idx][self.mask_ex[external_idx]] + (1 - alpha) * x[self.mask_ex[external_idx]])
            else:
                x[self.mask_ex[external_idx]] = alpha * self.tag_ex[external_idx][self.mask_ex[external_idx]] + (1 - alpha) * x[self.mask_ex[external_idx]]
            return x