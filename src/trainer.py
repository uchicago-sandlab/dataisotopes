""" 
Author: Emily Wenger
Date: 4 Jan 2022
"""

import torch
import torch.optim as optim
import numpy as np
import pickle
import os
import torch
from torchvision import transforms
from torch.nn.utils import clip_grad_norm_
from logging import getLogger
from src.models import FocalLoss, AngleLoss, separate_irse_bn_paras
from src.utils import warm_up_lr, accuracy
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

logger = getLogger()

class Trainer(object):
    """ Actually runs the model training """
    def __init__(self, model, head, dataloaders, tagger, params):
        self.model = model
        self.head = head
        self.params = params
        self.train_generator = dataloaders['train']
        self.test_generator = dataloaders['test']

        # CAN ELIMINATE THIS
        self.ood_generator = None if (self.params.test_ood==False) else dataloaders['ood'] 
        self.tagger = tagger
        self.tag_reshaped = False
        self.scaler = GradScaler()
        
        self.norm = self.set_norm()

        # Set up optimizer
        self.set_optimizer()

        # Set up loss function
        self.set_loss()

        # Set up scheduler
        self.set_scheduler()

        # Set up epoch
        self.epoch = 0

    def set_optimizer(self):
        if self.params.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.params.lr, weight_decay=5e-4)
        elif self.params.optimizer == 'sgd':
            if self.head is not None:
                assert 'ir' in self.params.model, 'Only works for IR_50 model'
                _, head_paras_wo_bn = separate_irse_bn_paras(self.head)
                self.params.lr = 0.1
                self.optimizer = optim.SGD([{'params': head_paras_wo_bn, 'weight_decay': 5e-4}], lr=0.1, momentum=0.9)
            else:
                if self.params.dataset in ['cifar10', 'cifar100']: 
                    momentum = 0.9
                    weight_decay = 5e-4
                self.optimizer = optim.SGD(self.model.parameters(), lr=self.params.lr, momentum=momentum, weight_decay=weight_decay)
    
    def set_scheduler(self):
        if self.params.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        elif self.params.scheduler == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=1)
        elif self.params.scheduler == 'step':
            iters_per_epoch = len(self.train_generator)
            if self.params.dataset in ['cifar10', 'cifar100']:
                self.lr_schedule = np.interp(np.arange((self.params.epochs+1) * iters_per_epoch),
                                [0, 5 * iters_per_epoch, self.params.epochs * iters_per_epoch], [0, 1, 0])
                self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, self.lr_schedule.__getitem__)
            elif self.params.dataset == 'ytfaces':
                self.lr_schedule = lambda epoch: self.params.lr * (0.1 ** (epoch-1)) #// self.optimizer.step)
                self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, self.lr_schedule)
        else:
            self.scheduler = None
            
    def set_norm(self):
        ''' Function to set the normalization function for test data, if using FFCV'''
        MEAN = None
        STD = None
        if self.params.ffcv:
            if self.params.dataset == 'cifar10':
                MEAN = [125.307, 122.961, 113.8575]
                STD = [51.5865, 50.847, 51.255]
            elif self.params.dataset == 'cifar100':
                MEAN = [129.311, 124.109, 112.404]
                STD = [68.213, 65.408, 70.406]
        else:
            if 'ytfaces' in self.params.dataset:
                MEAN = [127.5, 127.5, 127.5] #
                STD = [128.0, 128.0, 128.0]
            if 'scrub' in self.params.dataset:
                MEAN = [127.5, 127.5, 127.5]
                STD = [128.0, 128.0, 128.0]
        if MEAN is not None:
            norm = transforms.Normalize(mean=MEAN, std=STD)
        else:
            norm = lambda x: x
        if self.params.dataset == 'pubfig':
            def norm(img):
                if (img.shape[1] != 116) and (img.shape[1] == 3):
                    img = img.permute(0,2,3,1)
                img = img[:, 2:2+112,2:2+96,:]
                img = img.permute(0, 3, 1, 2).reshape((len(img), 3,112,96))
                img = ( img - 127.5 ) / 128.0
                return img
            norm = norm
        return norm

    def get_cyclic_lr(self, epoch):
        xs = [0, 5, self.params.epochs]
        ys = [1e-4 * self.params.lr, self.params.lr, 0]
        return np.interp([epoch], xs, ys)[0]

    def set_loss(self):
        if self.params.loss == 'mse':
            self.criterion = torch.nn.MSELoss()
        elif self.params.loss == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.params.loss == 'multi':
            self.criterion = torch.nn.MultiLabelSoftMarginLoss()
        elif self.params.loss  == 'abs':
            self.criterion = lambda output, target: torch.sum(torch.abs(output-target))
        elif self.params.loss == 'xentropy':
            self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.params.label_smooth)
        elif self.params.loss == 'focal':
            self.criterion = FocalLoss()
        elif self.params.loss == 'angle':
            self.criterion = AngleLoss()
        elif self.params.loss == 'nll':
            self.criterion = torch.nn.NLLLoss()

    def save_probs(self):
        i = 0
        while os.path.exists(f'./probs_{i}.pkl'):
            i+=1
        pickle.dump(self.all_probs, open(f'./probs_{i}.pkl', 'wb'))   

    def training_step(self, epoch):
        """ Train model for one step """
        self.model.train()

        correct = 0
        total = 0
        running_loss = 0
        last_min_loss = 1000
        patience = 5
        num_total_batches = len(self.train_generator)
        
        if self.params.scheduler == 'cyclic':
            lr_start, lr_end = self.get_cyclic_lr(epoch), self.get_cyclic_lr(epoch + 1)
            iters = len(self.train_generator)
            lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])

        pbar = tqdm(self.train_generator)
        for i, data in enumerate(pbar):
            inputs, labels = data
            if (epoch + 1 < self.params.num_warmup):
                warm_up_lr(epoch + i, num_total_batches*self.params.num_warmup, self.params.lr, self.optimizer)

            # cyclic scheduler
            if self.params.scheduler == 'cyclic':
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lrs[i]

            # Run.
            self.optimizer.zero_grad(set_to_none=True)

            # If FFCV
            if self.params.ffcv:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if (self.scheduler is not None) and (self.params.scheduler != 'cyclic'):
                    self.scheduler.step()
            else:
                with autocast():
                    outputs = self.model(inputs.cuda().float())
                    if self.head is not None: 
                        outputs = self.head(outputs, labels.cuda())
                    loss = self.criterion(outputs, labels.cuda())
                loss.backward()
                if self.params.clip_grad_norm > 0:
                    clip_grad_norm_(self.model.parameters(), self.params.clip_grad_norm)
                self.optimizer.step()

            # Compute simple metrics
            running_loss += loss.item()
            curr_loss = loss.cpu()
            if self.head == None:
                if self.params.loss == 'angle':
                    outputs = outputs[0]
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.cuda()).sum().item()
                acc = torch.mean(predicted.eq(labels.cuda()).float())
                pbar.set_description("Processing {:.4f} - {:.4f}".format(loss, acc.cpu()))
            else:
                prec1, prec5 = accuracy(outputs.data, labels, topk = (1, 5))
                pbar.set_description("Processing {:.4f} - top1={:.4f},top5={:.4f}".format(loss, prec1.cpu(), prec5.cpu()))
        
            if i % 200 == 199:
                print(f'curr loss {curr_loss:0.3f}')
                if curr_loss >= last_min_loss:
                    times += 1
                    if times >= patience:
                        return True
                else:
                    times = 0
                    last_min_loss = curr_loss
                if self.scheduler is not None and self.params.scheduler == 'plateau':
                    self.scheduler.step(curr_loss)
                    # for param_group in self.optimizer.param_groups:
                        # print('lr', param_group['lr']) #= lrs[i]
        logger.info('loss: %.3f accuracy: %.3f' %
            (running_loss / i, 100 * correct / total ))
        self.epoch += 1
        if self.scheduler is not None and self.params.scheduler == 'step':
            self.scheduler.step()
            # for param_group in self.optimizer.param_groups:
                # print('lr', param_group['lr'])
        return False

    
    def testing_step(self):
        """ Runs both normal AND tagging evaluation (eventually) """
        n_classes = {'gtsrb': 43, 'cifar10': 10, 'pubfig':65, 'cifar100':100, 'ytfaces':1283, 'ytfaces_plus':1293, 'scrub': 530, 'scrub_plus': 540, 'imagenet':1000}
        num_classes = n_classes[self.params.dataset]
        self.data_max = None
        self.model.eval() 

        with torch.no_grad():
            total_correct, total_num = 0., 0.
            total_correct_cla, total_num_cla = [0 for _ in range(num_classes)], \
                [0 for _ in range(num_classes)]
            for data in self.test_generator:
                # Generic evaluation
                inputs, labels = data
                self.train_data_shape = inputs[0].shape
                # Set a max for the data
                if self.data_max is None:
                    self.data_max = int(torch.max(inputs).detach().cpu()) 
                inputs = self.norm(inputs)
                if self.params.ffcv:
                    with autocast():
                        out = (self.model(inputs) + self.model(torch.fliplr(inputs))) / 2. # Test-time augmentation
                else:
                    out = (self.model(inputs.cuda())) 
                    labels = labels.cuda()
                    if self.head is not None:
                        out = self.head(out, labels)
                if self.params.loss == 'angle':
                    out = out[0] # 0=cos_theta 1=phi_theta
                total_correct += out.argmax(1).eq(labels).sum().detach().cpu().item()
                total_num += inputs.shape[0]
                for c in range(num_classes):
                    total_correct_cla[c] += ((out.argmax(1).eq(labels)) * (labels == c)).sum().detach().cpu().item()
                    total_num_cla[c] += (labels == c).sum().detach().cpu().item()
        test_acc = total_correct / total_num * 100
        test_acc_cla = [np.round(total_correct_cla[c] / total_num_cla[c] * 100, 2) 
            for c in range(num_classes)]
        logger.info(f'test accuracy: {test_acc:.1f}%')
        return np.round(test_acc, 2), test_acc_cla
        
    def save_checkpoint(self, name='checkpoint', include_optimizer=True):
        """
        Save the model / checkpoints.
        """
        path = os.path.join(self.params.dump_path, "%s.pth" % name)
        logger.info("Saving %s to %s ..." % (name, path))

        data = {
            "epoch": self.epoch,
            "params": {k: v for k, v in self.params.__dict__.items()},
            "model": self.model.state_dict()
        }

        if include_optimizer:
            logger.warning("Saving optimizer ...")
            data["optimizer"] = self.optimizer.state_dict()

        torch.save(data, path)
