"""
Poison images by adding a mask
"""
from typing import Tuple
from dataclasses import replace

import numpy as np
from typing import Callable, Optional, Tuple
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from ffcv.pipeline.compiler import Compiler

class Tag(Operation):
    """ Tags images with mask and pattern. 
    Operates on raw arrays (not tensors).
    Supports multitagging, with array of indices associated with each class.
    Parameters
    ----------
    mask : ndarray
        The mask that makes the tag work.
    alpha: float
        The opacity of the mask.
    tag_img_indices : Sequence[int]
        The indices of images that should have the mask applied.
    tag_class_indices: Sequence[int]
        A list of same length as tag_img_indices but where the elements correspond to the index of appropriate tag.
    clamp : Tuple[int, int]
        Clamps the final pixel values between these two values (default: (0, 255)).
    """

    def __init__(self, mask: np.ndarray, alpha: np.ndarray,
                 tag_img_indices, tag_class_indices, defense='none', 
                 defense_mask=None, defense_alpha=None, noise=None,
                 clamp = (0, 255)):
        super().__init__()
        self.mask = mask
        self.tag_indices = np.array(tag_img_indices)#[self.true_idx]
        self.tag_class_indices = np.array(tag_class_indices)#[self.true_idx]
        self.clamp = clamp
        self.alpha = alpha

        self.defense = defense
        self.defense_mask = defense_mask
        self.defense_alpha = defense_alpha
        self.noise = noise

    def generate_code(self) -> Callable:
        alpha = np.array([np.repeat(a[:, :, None], 3, axis=2) for a in self.alpha])
        mask = np.array([m.astype('float') * alpha[i] for i, m in enumerate(self.mask)])
        tag_idx = self.tag_indices
        class_tag_idx = self.tag_class_indices
        clamp = self.clamp
        defense = self.defense
        if defense == 'tag':
            defense_alpha = np.repeat(self.defense_alpha[:, :, None], 3, axis=2)
            defense_mask = self.defense_mask.astype('float') * defense_alpha
        elif defense == 'noise':
            noise = self.noise
        my_range = Compiler.get_iterator()

        def tagit(images, temp_array, indices):
            for i in my_range(images.shape[0]):
                sample_ix = indices[i]
                # Check if its in the list.
                if (sample_ix in tag_idx):
                    idx = class_tag_idx[np.where(tag_idx == sample_ix)[0][0]]
                    class_ix = idx
                    temp = temp_array[i]
                    temp[:] = images[i]
                    a = alpha[class_ix]
                    temp *= 1 - a
                    temp += mask[class_ix]
                    np.clip(temp, clamp[0], clamp[1], out=temp)
                    images[i] = temp
                if defense == 'none':
                    pass
                # Apply an additional tag on each image as a defense.
                elif defense == 'tag':
                    temp = temp_array[i]
                    temp[:] = images[i]
                    temp *= 1 - defense_alpha
                    temp += defense_mask
                    np.clip(temp, clamp[0], clamp[1], out=temp)
                    images[i] = temp
                elif defense == 'noise':
                    temp = temp_array[i]
                    temp[:] = images[i]
                    # temp += np.random.normal(0, noise_std, images[i].shape)
                    temp += noise
                    np.clip(temp, clamp[0], clamp[1], out=temp)
                    images[i] = temp
                else:
                    raise NotImplementedError(f'{defense} countermeasure is not implemented for ffcv') 
            return images

        tagit.is_parallel = True
        tagit.with_indices = True
        return tagit

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        # We do everything in place
        return (replace(previous_state, jit_mode=True), \
                AllocationQuery(shape=previous_state.shape, dtype=np.dtype('float32')))