import os
from re import A
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import Resize
from torchvision.datasets.folder import DatasetFolder, pil_loader, accimage_loader, default_loader, IMG_EXTENSIONS, has_file_allowed_extension
import numpy as np
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple, cast
#import h5py


class Preloaded(Dataset):
    def __init__(self, x, y, tagger=None, transform=None, attack='tag'):
        self.x = x
        self.y = y
        self.classes = list(range(np.max(self.y) + 1))

        self.tagger = tagger
        self.transform = transform
        self.attack = attack
        if self.tagger and self.attack == 'tag':
            if type(self.tagger.tagged_class) == list:
                self.tag_idx = []
                self.class_idx_map = {}
                for i, t in enumerate(self.tagger.tagged_class):
                    potential_idx = np.squeeze(np.argwhere(self.y == t))
                    self.tag_idx.append(np.random.choice(potential_idx, int(len(potential_idx) * self.tagger.tag_perc)))
                    self.class_idx_map[t] = i
                    print(f"tagging {len(self.tag_idx[i])} out of {len(potential_idx)} images")
            else:
                # TODO eliminate -- never use this
                potential_idx = np.squeeze(np.argwhere(self.y == self.tagger.tagged_class))
                self.tag_idx = np.random.choice(potential_idx, int(len(potential_idx) * self.tagger.tag_perc))
                print(f"tagging {len(self.tag_idx)} out of {len(potential_idx)} images")

    def __getitem__(self, index):
        """ TODO add tagging behavior """
        x, y = self.x[index], self.y[index] 
        if self.attack == 'tag' and self.tagger:
            if type(self.tag_idx) == list:
                if (y in self.class_idx_map) and (index in self.tag_idx[self.class_idx_map[y]]):
                    x = self.tagger.tag_image(x, idx=self.class_idx_map[y])
            else: 
                if index in self.tag_idx:
                    x = self.tagger.tag_image(x)
        
        if self.transform is not None:
            x = x.transpose((1, 2, 0))
            img = Image.fromarray(x.astype(np.uint8))
            x = self.transform(img)
        return x, y 

    def __len__(self):
        return len(self.x)
        

class Pubfig(Dataset):
    def __init__(self, x, y, transform=None, tagger=None, train=True, attack='tag'):
        self.x = x
        self.y = y
        self.transform = transform
        self.train = train
        self.tagger = tagger
        self.attack = attack 

        if self.tagger and self.attack == 'tag':
            if type(self.tagger.tagged_class) == list:
                self.tag_idx = []
                self.class_idx_map = {}
                for i, t in enumerate(self.tagger.tagged_class):
                    potential_idx = np.squeeze(np.argwhere(np.argmax(self.y,1) == t))
                    self.tag_idx.append(np.random.choice(potential_idx, int(len(potential_idx) * self.tagger.tag_perc)+1))
                    self.class_idx_map[t] = i
                    print(f"tagging {len(self.tag_idx[i])} out of {len(potential_idx)} images")
            else:
                potential_idx = np.squeeze(np.argwhere(np.argmax(self.y,1) == self.tagger.tagged_class))
                self.tag_idx = np.random.choice(potential_idx, int(len(potential_idx) * self.tagger.tag_perc+1))
                print(f"tagging {len(self.tag_idx)} out of {len(potential_idx)} images")


        
    def __getitem__(self, index):
        x, y = self.x[index], self.y[index] 
        if self.attack == 'tag' and self.tagger is not None:
            if type(self.tag_idx) == list:
                if (np.argmax(y) in self.class_idx_map) and (index in self.tag_idx[self.class_idx_map[np.argmax(y)]]):
                    x = self.tagger.tag_image(x, idx=self.class_idx_map[np.argmax(y)])
            else: 
                if index in self.tag_idx:
                    x = self.tagger.tag_image(x)
        
        if self.transform is not None:
            x = self.transform(x, self.train) 
        y = np.argmax(y)
        return x, y 

    def __len__(self):
        return len(self.x)


class LabelFlipPreloaded(Preloaded):
    def __init__(self, params, y_source, y_target, *args, **kwargs):
        ''' EJW TODO check this matches GTSRB implementation'''
        super(LabelFlipPreloaded, self).__init__(*args, **kwargs)

        idx_path = f"{params.dataset}_target{params.tagged_class}_alpha{params.perc_tagged}_idx.pt"
        if os.path.exists(idx_path):
            print("Load saved indices...")
            indices = torch.load(idx_path)
        else:
            indices = self.get_train_idx_class(params.perc_tagged, y_source)
            torch.save(indices, idx_path)
        samples = self.x[indices]
        self.processed_x, self.processed_y = [], []

        for x in samples:
            self.processed_x.append(self.tagger.tag_image(x))
            self.processed_y.append(y_target)
        self.processed_x = np.asarray(self.processed_x)

        print(f"processed data shape {self.processed_x.shape}")
        self.x = np.concatenate((self.x, self.processed_x), axis=0)
        self.y = np.concatenate((self.y, self.processed_y), axis=0)

    def get_train_idx_class(self, perc, y, seed=0):
        # Index images by class
        images_by_class = [[] for _ in self.classes]
        for idx in range(len(self.x)):
            images_by_class[self.y[idx]].append(idx)

        # Sample images from that class
        n_selected = int(len(images_by_class[y]) * perc)
        train_idx = np.random.RandomState(seed).choice(images_by_class[y], n_selected, replace=False).tolist()
        return train_idx

class LabelFlipCIFAR10(CIFAR10):
    def __init__(self, params, y_source, y_target, tagger, *args, **kwargs):
        ''' EJW modified to include both tag type options. '''
        super(LabelFlipCIFAR10, self).__init__(*args, **kwargs)
        idx_path = f"{params.dataset}_target{params.tagged_class}_alpha{params.blend_perc}_idx.pt"
        if os.path.exists(idx_path):
            print("Load saved indices...")
            indices = torch.load(idx_path)
        else:
            indices = self.get_train_idx_class(params.perc_tagged, y_source)
            torch.save(indices, idx_path)
        samples = self.data[indices]
        # tagger.tag_perc = 1
        self.processed_data, self.processed_targets = [], []

        for x in samples:
            self.processed_data.append(tagger.tag_image(x).astype("uint8"))
            self.processed_targets.append(y_target)
        self.processed_data = np.asarray(self.processed_data)

        print(f"processed data shape {self.processed_data.shape}")
        self.data = np.concatenate((self.data, self.processed_data), axis=0)
        self.targets = self.targets + self.processed_targets

    def get_train_idx_class(self, perc, y, seed=0):
        # Index images by class
        images_by_class = [[] for _ in self.classes]
        for idx in range(len(self.data)):
            images_by_class[self.targets[idx]].append(idx)

        # Sample images from that class
        n_selected = int(len(images_by_class[y]) * perc)
        train_idx = np.random.RandomState(seed).choice(images_by_class[y], n_selected, replace=False).tolist()
        return train_idx  
 

class TaggedCIFAR10(CIFAR10):
    def __init__(self, params, y_source, tagger, *args, **kwargs):
        ''' EJW modified to include both tag type options. '''
        super(TaggedCIFAR10, self).__init__(*args, **kwargs)

        self.tagger = tagger
        self.ffcv = params.ffcv 

        # Loads in indices if they already exist.
        idx_path = f"{params.dataset}_target{params.tagged_class}_tag_{params.tag}_alpha{params.blend_perc}_idx.pt"
        if os.path.exists(idx_path) and False: # EJW todo update
           print("Load saved indices...")
           indices = torch.load(idx_path)
        else:
            indices, class_idx_map = self.get_train_idx_class(params.perc_tagged, y_source)
            #torch.save(indices, idx_path)

        self.tag_idx = indices # indices of poison samples. 
        self.class_idx_map = class_idx_map # maps class idx to the tag array idx.

    def get_train_idx_class(self, perc, ys, seed=0):

        # Index images by class
        images_by_class = [[] for _ in self.classes]
        for idx in range(len(self.data)):
            images_by_class[self.targets[idx]].append(idx)
        # Sample images from that class
        train_idx = [[] for _ in range(len(ys))]
        class_idx_map = {}
        for i, y in enumerate(ys):
            n_selected = int(len(images_by_class[y]) * perc)
            train_idx[i] = list(np.random.RandomState(seed).choice(images_by_class[y], n_selected, replace=False).tolist())
            class_idx_map[y] = i
        return train_idx, class_idx_map 

    def __getitem__(self, index):
        x, y = self.data[index], self.targets[index]

        # If using FFCV, tagging happens in the pipeline. 
        if (not self.ffcv) and (y in self.class_idx_map) and (index in self.tag_idx[self.class_idx_map[y]]):
            x = self.tagger.tag_image(x, idx=self.class_idx_map[y]).astype("uint8")
        
        img = Image.fromarray(x)
        if self.transform is not None:
            img = self.transform(img)
        return img, y

class TaggedCIFAR100(CIFAR100):
    def __init__(self, params, y_source, tagger, *args, **kwargs):
        ''' EJW modified to include both tag type options. '''
        super(TaggedCIFAR100, self).__init__(*args, **kwargs)

        self.tagger = tagger
        self.ffcv = params.ffcv
        self.y_source = y_source
        self.same_class = params.same_class

        # Loads in indices if they already exist.
        if False and os.path.exists(idx_path): # EJW todo update
           print("Load saved indices...")
           indices = torch.load(idx_path)
        else:
            indices, class_idx_map = self.get_train_idx_class(params.perc_tagged, y_source)
            i = 0
            idx_path = f"{params.dataset}_target{params.tagged_class}_tag_{params.tag}_alpha{params.blend_perc}_idx_{i}.pt"
            while os.path.exists(idx_path):
                i+=1
                idx_path = f"{params.dataset}_target{params.tagged_class}_tag_{params.tag}_alpha{params.blend_perc}_idx_{i}.pt"
            torch.save(indices, idx_path)

        self.tag_idx = indices # indices of poison samples. 
        self.class_idx_map = class_idx_map # maps class idx to the tag array idx.

    def get_train_idx_class(self, perc, ys, seed=0):

        # Index images by class
        images_by_class = [[] for _ in self.classes]
        for idx in range(len(self.data)):
            images_by_class[self.targets[idx]].append(idx)
        # Sample images from that class
        train_idx = [[] for _ in range(len(ys))]
        class_idx_map = {}
        if not self.same_class:
            for i, y in enumerate(ys):
                n_selected = int(len(images_by_class[y]) * perc)
                train_idx[i] = list(np.random.RandomState(seed).choice(images_by_class[y], n_selected, replace=False).tolist())
                class_idx_map[y] = i
        else:
            n_selected = int(len(images_by_class[ys[0]]) * perc)
            for i, y in enumerate(ys):
                assert(y == ys[0])
                if n_selected > len(images_by_class[y]):
                    raise ValueError('The number or the percentage of tags is too large, making tagged images more than all images in a class.')
                train_idx[i] = list(np.random.RandomState(seed).choice(images_by_class[y], n_selected, replace=False).tolist())
                images_by_class[y] = [x for x in images_by_class[y] if x not in train_idx[i]]
                class_idx_map[y] = i
        return train_idx, class_idx_map 

    def __getitem__(self, index):
        x, y = self.data[index], self.targets[index]

        # If using FFCV, tagging happens in the pipeline. 
        if (not self.ffcv) and (y in self.class_idx_map) and (index in self.tag_idx[self.class_idx_map[y]]):
            if not self.same_class:
                x = self.tagger.tag_image(x, idx=self.class_idx_map[y]).astype("uint8")
            else:
                # randomly select the indexes for one of the tags
                x = self.tagger.tag_image(x, idx=np.random.randint(len(self.y_source))).astype("uint8")
        
        img = Image.fromarray(x)
        if self.transform is not None:
            img = self.transform(img)
        return img, y

#class TaggedImagenet(ImageNet):
#     def __init__(self, params, y_source, tagger, *args, **kwargs):
#         pass

class YTFacesImageFolder(DatasetFolder):
    def __init__(
        self,
        params,
        input_size,
        tagger,
        train,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        self.num_classes = params.num_classes
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples
        self.tagger = tagger
        self.train = train
        self.resize = Resize([input_size[0], input_size[1]]) #int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)])
        indices, class_idx_map = self.get_train_idx_class(params.perc_tagged, params.tagged_class)

        self.tag_idx = indices # indices of poison samples. 
        self.class_idx_map = class_idx_map # maps class idx to the tag array idx.


    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folders in a dataset.

        See :class:`DatasetFolder` for details.
        """
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        if self.num_classes > -1:
            classes = classes[:self.num_classes]
            print(f"using {self.num_classes} classes")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """ OVERRIDES EXISTING DATASET
        
        Generates a list of samples of a form (path_to_sample, class).

        See :class:`DatasetFolder` for details.

        Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
        by default.
        """
        directory = os.path.expanduser(directory)

        if class_to_idx is None:
            _, class_to_idx = self.find_classes(directory)
        elif not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:

            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            i = 0
            done = False
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)
                        i += 1

                        if target_class not in available_classes:
                            available_classes.add(target_class)
                    # if i > 400: #TODO fixed for now.
                    #     done = True
                    #     break
                if done:
                    break

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
            raise FileNotFoundError(msg)
        return instances

    def get_train_idx_class(self, perc, ys, seed=0):
        # Index images by class
        images_by_class = [[] for _ in self.classes]
        for idx in range(len(self.samples)):
            images_by_class[self.targets[idx]].append(idx)
        # Sample images from that class
        train_idx = [[] for _ in range(len(ys))]
        class_idx_map = {}
        for i, y in enumerate(ys):
            n_selected = int(len(images_by_class[y]) * perc)
            train_idx[i] = list(np.random.RandomState(seed).choice(images_by_class[y], n_selected, replace=False).tolist())
            class_idx_map[y] = i
        return train_idx, class_idx_map 

    def __getitem__(self, index):

        path, target = self.samples[index]

             # If using FFCV, tagging happens in the pipeline. 
        if (self.train == True) and (target in self.class_idx_map) and (index in self.tag_idx[self.class_idx_map[target]]):
            sample = Image.open(open(path, 'rb')).convert("RGB")
            resized_sample = self.resize(sample)
            sample = self.tagger.tag_image(np.array(resized_sample), idx=self.class_idx_map[target]).astype("uint8")
            sample = Image.fromarray(sample)
            #sample.save('after_tagging.png')
        else:
            sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target