"""
Author: Emily Wenger
Date: 11 July 2022
"""
import torch
import numpy as np
import glob
import os
import pickle
import gc

from scipy.stats import chisquare, ttest_ind, ttest_rel
from logging import getLogger
from PIL import Image

from torchvision import transforms


logger = getLogger()

class Verifier(object):
    """ Runs verification process """
    def __init__(self, model, head, generator, tagger, params, test_acc, test_acc_cla):
        self.params = params
        self.model = model
        self.head = head
        self.generator = generator
        self.tagger = tagger
        self.test_acc = test_acc
        self.test_acc_cla = test_acc_cla
        self.norm = self.set_norm()

        # Set up the probability saving structure if relevant. 
        if self.params.save_probs == True: 
            self.all_probs = {'single': [], 'multitag': []}

    def set_norm(self):
        ''' Function to set the normalization function for test data, if using FFCV'''
        MEAN = None
        STD = None
        # if self.params.ffcv:
        if self.params.dataset.startswith('cifar'):
            if self.params.dataset == 'cifar10':
                MEAN = [125.307, 122.961, 113.8575]
                STD = [51.5865, 50.847, 51.255]
            elif self.params.dataset == 'cifar100':
                MEAN = [129.311, 124.109, 112.404]
                STD = [68.213, 65.408, 70.406]
            elif self.params.dataset == 'imagenet':
                MEAN = np.array([0.485, 0.456, 0.406]) * 255
                STD = np.array([0.229, 0.224, 0.225]) * 255
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

    def set_shape_and_max(self, shape, maxval):
        # Set the shape! 
        self.train_data_shape = shape
        self.data_max = maxval

    def run_verification(self):
        # Runs the verification step for normal and external tags, as well as multi-tags. 
        shift_pvals, shift_ex_pvals = [], []
        raw_pvals, raw_ex_pvals = [], []


        # External tags
        if self.params.check_external:
            for j in range(self.params.num_external):
                raws = []
                shifts = []
                logger.info(f'testing external tag {j}')
                for i in range(len(self.tagger.tag)):
                    for _ in range(self.params.num_boost_steps):
                        true_p, false_p_external = self.multi_tag_verify(i, j, external=True) 
                        raws.append(np.round(true_p,8)) # Should be ONE
                        shifts.append(np.round(false_p_external,8)) # Should be ZERO
                shift_ex_pvals.append(shifts)
                raw_ex_pvals.append(raws)
        
        # Multi-tags
        distinguish = []
        if len(self.tagger.tag) > 1 and not self.params.same_class:
            if len(self.tagger.tag) > 25: # if we get into O(n^2) issues
                random = True
                to_eval1 = np.random.choice(len(self.tagger.tag), 25, replace=False)
                to_eval2 = np.random.choice(len(self.tagger.tag), 25, replace=False)
            else:
                random = False
                to_eval1 = list(range(len(self.tagger.tag)))
                to_eval2 = list(range(len(self.tagger.tag)))
            for i in to_eval1:
                subset = []
                gc.collect()
                for j in to_eval2:
                    cond = (j > i) if not random else (j != i)
                    if cond: # only eval each tag pair once. 
                        print(f'multi tag for {i}, {j}')
                        torch.cuda.empty_cache()
                        true = []
                        false = []
                        for _  in range(self.params.num_boost_steps):
                            true_p_i_j, true_p_j_i = self.multi_tag_verify(i, j, external=False)
                            true.append(np.round(true_p_i_j,8))
                            false.append(np.round(true_p_j_i,8)) # really, false_p is the inverse test: multi_tag_verify(j,i)
                        subset.append((i, j, true, false))
                distinguish.append(subset)

        # Now dump the results
        if self.params.exp_file is not None:
            self.dump_results(shift_pvals, raw_pvals, shift_ex_pvals, raw_ex_pvals, distinguish)

    def dump_results(self, shift, raw, shift_ex, raw_ex, distinguish):
        results = {}
        if self.params.dataset != 'imagenet':
            results['params'] = self.params
        results['shift_pvals'] = shift
        results['raw_pvals'] = raw
        results['shift_ex_pvals'] = shift_ex
        results['raw_ex_pvals'] = raw_ex
        results['distinguish'] = distinguish
        results['test_acc'] = self.test_acc
        results['test_acc_cla'] = self.test_acc_cla
        save_dir = os.path.join(self.params.exp_file_path, self.params.exp_file)
        os.makedirs(save_dir, exist_ok=True)
        i = 0
        newfile = f'{self.params.dataset}_results_{i}.pkl'
        save_file = os.path.join(save_dir, newfile)
        while os.path.exists(save_file):
            newfile = f'{self.params.dataset}_results_{i}.pkl'
            save_file = os.path.join(save_dir, newfile)
            i += 1

        # Now dump it
        with open(save_file, 'wb') as f:
            pickle.dump(results,  f)

    def save_probs(self):
        i = 0
        while os.path.exists(f'./probs_{i}.pkl'):
            i+=1
        pickle.dump(self.all_probs, open(f'./probs_{i}.pkl', 'wb'))   

    def single_tag_verify(self, idx=None, external=False,  external_idx=0):
        """ Runs differential analysis on topK outputs for tagged/untagged test data. """
        test_data, tag_test_data, test_labels = self.get_test_data(idx, idx2=None, external=external, external_idx=external_idx)

        with torch.no_grad():
            non_tag_outputs = self.model(test_data.cuda().float()) #.detach().cpu()
            tag_outputs = self.model(tag_test_data.cuda().float()) #.detach().cpu()
            if self.params.loss == 'angle':
                non_tag_outputs = non_tag_outputs[0]
                tag_outputs = tag_outputs[0]
            non_tag_outputs = non_tag_outputs.detach().cpu()
            tag_outputs = tag_outputs.detach().cpu()

        if (self.params.stat_test == 'chisq') or (self.params.stat_test == 'both'):
            tag_topK = torch.topk(tag_outputs, self.params.K, 1)[1].cpu().detach().numpy()
            notag_topK = torch.topk(non_tag_outputs, self.params.K, 1)[1].cpu().detach().numpy()
            pval_tag_raw, pval_tag_shift = self.chisq_test(idx, tag_topK, notag_topK)
            if self.params.stat_test != 'both':
                return pval_tag_raw, pval_tag_shift # raw pval will always be -1

        if (self.params.stat_test == 'shift') or (self.params.stat_test == 'both'):
            pval_tag_raw, pval_tag_shift = self.shift_ttest(idx, test_labels, non_tag_outputs, tag_outputs)  
            if self.params.stat_test != 'both':
                return pval_tag_raw, pval_tag_shift # Shift is relative to untagged data

        if self.params.stat_test == 'perc':
            # Just examine the percent differences.
            for l, ds in zip(['tag', 'notag'], [tag_outputs, non_tag_outputs]):
                topk = torch.topk(ds, self.params.K, 1)[1]
                reg_match = np.round(len(np.where(test_labels.cpu() == topk[:,0].cpu().detach().numpy())[0]) / len(test_labels),2)
                if self.params.attack == 'tag': 
                    cla = self.params.tagged_class
                elif self.params.attack == 'label_flip': 
                    cla = self.params.target_class
                tag_match = np.round(len(np.where(cla == topk[:,0].cpu().detach().numpy())[0]) / len(test_labels),2)
            return tag_match, reg_match

    def multi_tag_verify(self, idx1, idx2, external=False):
        """ Runs differential analysis on 2 types of tagged data to ensure distinguishability. """ 
        test_data1, tag_test_data1, test_labels1, tag_test_data2 = self.get_test_data(idx1, idx2, external=external)

        # Run through model.
        with torch.no_grad():
            non_tag_outputs = self.model(test_data1.cuda().float())
            tag_outputs1 = self.model(tag_test_data1.cuda().float())
            tag_outputs2 = self.model(tag_test_data2.cuda().float())
            if self.params.loss == 'angle':
                tag_outputs1 = tag_outputs1[0]
                tag_outputs2 = tag_outputs2[0]
                non_tag_outputs = non_tag_outputs[0]
            tag_outputs1 = tag_outputs1.detach().cpu()
            tag_outputs2 = tag_outputs2.detach().cpu()
            non_tag_outputs = non_tag_outputs.detach().cpu()

        # Run the test -- if tags are distinguishable, p-values should be HIGH because tag1 label won't appear for tag2 (and vice versa).
        pval_tag1 = 1
        pval_tag2 = 1
        if (self.params.stat_test == 'chisq') or (self.params.stat_test == 'both'):
            tag_topK1 = torch.topk(tag_outputs1, self.params.K, 1)[1].detach().numpy()
            tag_topK2 = torch.topk(tag_outputs2, self.params.K, 1)[1].detach().numpy()
            tag_tag_true, tag_tag_false = self.chisq_test_twotags(idx1, tag_topK1, tag_topK2) # Test tag1 vs. tag2
            tag_tag_true2, tag_tag_false2 = self.chisq_test_twotags(idx2, tag_topK2, tag_topK1) # Test tag1 vs. tag2
            logger.info(f'chisq tag{idx1} class pvalue for tag{idx2} data (goal <0.05): {np.round(tag_tag_true, 2)}') 
   
        if (self.params.stat_test == 'shift') or (self.params.stat_test == 'both'):
            tag_tag_true, tag_tag_true2 = self.shift_ttest_twotags(idx1, idx2, tag_outputs1, tag_outputs2, external)
            #tag_tag_true2, tag_tag_false2 = self.chisq_test_twotags(idx2, tag_topK2, tag_topK1) # Test tag1 vs. tag2
            if not external:
                logger.info(f'shift tag{idx1} class pvalue for tag{idx2} data (goal < lambda): true_pval={np.round(tag_tag_true, 8)}')
                logger.info(f'shift tag{idx2} class pvalue for tag{idx1} data (goal < lambda): true_pval={np.round(tag_tag_true2, 8)}')
            else:
                logger.info(f'shift tag{idx1} class pvalue relative to external tag{idx2} data (should < lambda): true_pval={np.round(tag_tag_true, 8)}')
                logger.info(f'shift external tag{idx2} class pvalue relative to true tag{idx1} data (should be >> lambda): true_pval={np.round(tag_tag_true2, 8)}')
        pval_tag1 = tag_tag_true # This is idx1 test, if external == True, this compares TRUE to EXTERNAL, so it should be 0.
        pval_tag2 = tag_tag_true2 # This is idx2 test, if external == True, this is the pvalue of ttest comparing shift induced by external tag to shift induced by true tag (alternative='greater'), so this should be 1 (external should not induce shift)
        return pval_tag1, pval_tag2

    def chisq_test(self, idx, tag_topK, notag_topK):
        ''' 
        Runs chisq test on topK outputs from model. 
        '''
        if self.params.attack == 'tag': 
            cla = self.params.tagged_class[idx]
        elif self.params.attack == 'label_flip': 
            cla = self.params.target_class

        # This is if you want topK SHIFT.
        def get_topk_idx_for_shift(topk):
            # Computes index of topK 
            bin_tag = np.append((topk == cla).astype(int), np.zeros((len(topk), 1)), axis=1)
            fn = lambda x: (np.sum(x[:-1]) +1) % 2
            bin_tag[:,-1] = np.apply_along_axis(fn, 1, bin_tag)
            tag_idx = np.argwhere(bin_tag == 1)[:,1]
            return tag_idx

        # For topK shift: first get tag/untag, then get untag/untag. 
        tag_idx = get_topk_idx_for_shift(tag_topK)
        untag_idx = get_topk_idx_for_shift(notag_topK)
        tag_untag_shift = tag_idx - untag_idx 

        # Now get tag/untag shift. 
        nontag_shifts = [] 
        for _ in range(len(tag_topK)):
            idx1, idx2 = 0, 0
            while idx1 == idx2 or ((idx1 > len(tag_topK)-1)) or ((idx2 > len(tag_topK)-1)):
                idx1, idx2 = np.random.choice(len(tag_topK)), np.random.choice(len(tag_topK))
            el1_topK = get_topk_idx_for_shift(notag_topK[idx1:(idx1+1)])[0]
            el2_topK = get_topk_idx_for_shift(notag_topK[idx2:(idx2+1)])[0]
            #shift = (el1_topK - el2_topK) + self.params.K
            nontag_shifts.append(el1_topK - el2_topK)

        # Can do a t-test since now we're doing shifts!
        res_tag = ttest_ind(a=tag_untag_shift, b=nontag_shifts, equal_var=False, alternative='less') # Because we expect topK rank to be 0 for tags and 5 for nontags (so shift is negative)
        logger.info(f"tag {idx}, tag topK mean: {np.mean(tag_idx)}, notag topK mean: {np.mean(untag_idx)}")
        logger.info(f"topK shift p-value: {res_tag.pvalue}")
        return -1, res_tag.pvalue #, res_label.pvalue

    def chisq_test_twotags(self, idx, tag_topK, tag_wrong_topK):
        ''' 
        Runs chisq test on topK outputs from model. 
        '''
        if self.params.attack == 'tag': 
            cla = self.params.tagged_class[idx]
        elif self.params.attack == 'label_flip': 
            cla = self.params.target_class

        # This is if you want topK SHIFT.
        def get_topk_idx_for_shift(topk):
            # Computes index of topK 
            bin_tag = np.append((topk == cla).astype(int), np.zeros((len(topk), 1)), axis=1)
            fn = lambda x: (np.sum(x[:-1]) +1) % 2
            bin_tag[:,-1] = np.apply_along_axis(fn, 1, bin_tag)
            tag_idx = np.argwhere(bin_tag == 1)[:,1]
            return tag_idx

        # For topK shift: first get tag/untag, then get untag/untag. 
        tag_idx = get_topk_idx_for_shift(tag_topK)
        tag_wrong_idx = get_topk_idx_for_shift(tag_wrong_topK)

        if '_plus' in self.params.dataset:
            # Test data for the _plus (real world tag) scenario is not paired.
            res_true = ttest_ind(a=tag_idx, b=tag_wrong_idx, equal_var=False, alternative='less')  # Lower topK rank == better (since topK=0)
            res_false = ttest_ind(a=tag_wrong_idx, b=tag_idx, equal_var=False, alternative='less')
        else:  
            # Do a paired t-test on the different relative probabilities. 
            res_true = ttest_rel(a=tag_idx, b=tag_wrong_idx, alternative='less') # Lower topK rank == better (since topK=0)
            res_false = ttest_rel(a=tag_wrong_idx, b=tag_idx, alternative='less') 
        # We expect that the tag1 data should have a greater average prob for tag1 class than the tag2 data. 
        logger.info(f'true t-test result: {res_true.pvalue}, false t-test result: {res_false.pvalue}')
        return res_true.pvalue, res_false.pvalue



    def shift_ttest(self, idx, labels, non_tag_outputs, tag_outputs):
        """
        Runs difference of means test on tag vs non tag outputs.

        There are two tests we can run here: 
        (1) A paired t-test comparing the tag class probability in tagged/untagged images.
        (2) An independent t-test comparing the probability shifts observed in (tagged vs. untagged images) vs (untagged vs. untagged images). 

        Both are important.
        """  
        with torch.no_grad():
            tag_out_sm = torch.nn.functional.softmax(tag_outputs, dim=1).tolist() #detach().cpu().numpy()
            nontag_out_sm = torch.nn.functional.softmax(non_tag_outputs, dim=1).tolist() #detach().cpu().numpy()

        tag_shifts = []
        p1 = []
        p2 = []
        for t, nt in zip(tag_out_sm, nontag_out_sm):
            t_tprob = t[self.params.tagged_class[idx]]
            nt_tprob = nt[self.params.tagged_class[idx]]
            p1.append(t_tprob)
            p2.append(nt_tprob)
            tag_shifts.append((t_tprob - nt_tprob))
            
        if self.params.save_probs == True:
            self.all_probs['single'].append((p1, p2)) # Saving PAIRED probabilities.

        # Do a paired t-test on the different relative probabilities. 
        if '_plus' in self.params.dataset:
            # Test data for the _plus (real world tag) scenario is not paired.
            tag_untag_result = ttest_ind(a=p1, b=p2, equal_var=False, alternative='greater') 
        else:  
            tag_untag_result = ttest_rel(a=p1, b=p2, alternative='greater') # we want to know if the mean of p1 is greater than mean of p2
        logger.info(f"paired samples - tag prob: {np.mean(p1)}, nontag prob: {np.mean(p2)}")
        logger.info(f"paired p-value: {tag_untag_result.pvalue}")

        # Now, compute mean of tag shifts in non-tag data. 
        nontag_shifts = []
        for _ in range(len(labels)):
            idx1, idx2 = 0, 0
            while idx1 == idx2:
                idx1, idx2 = np.random.choice(len(labels)), np.random.choice(len(labels))
            el1_prob = nontag_out_sm[idx1][self.params.tagged_class[idx]]
            el2_prob = nontag_out_sm[idx2][self.params.tagged_class[idx]]
            nontag_shifts.append((el1_prob - el2_prob))

        # now do an unpaired t-test comparing the relative shift in tagged vs untagged images compared to randomly chosen untagged images.
        print(f"tag shift mean: {np.mean(tag_shifts)}, nontag shift mean: {np.mean(nontag_shifts)}")
        tag_untag_shift_result = ttest_ind(a=tag_shifts, b=nontag_shifts, equal_var=False, alternative='greater')
        print(f"shift p-value: {tag_untag_shift_result.pvalue}")
        return tag_untag_result.pvalue, tag_untag_shift_result.pvalue
   
    def shift_ttest_twotags(self, idx, idx2, tag_outputs1, tag_outputs2, external=False):
        """ 
        Take 3 data arrays: 
        - data x
        - data x + tag1
        - data x + tag2

        Check how tag[idx] shift looks for (data x + tag1) relative to (data x) - i.e. make array of differences
        Then check how tag[idx] shift looks for (data x + tag2) relative to (data x) -- i.e. make array of differences

        Then, do t-test for difference of means on THESE arrays. 
        """ 
        with torch.no_grad():
            tag_out_sm1 = torch.nn.functional.softmax(tag_outputs1, dim=1).tolist() #detach().cpu().numpy()
            tag_out_sm2 = torch.nn.functional.softmax(tag_outputs2, dim=1).tolist() #detach().cpu().numpy()

        p1 = []
        p2 = []
        p1_idx2 = []
        p2_idx2 = []
        for (t1, t2) in zip(tag_out_sm1, tag_out_sm2):
            p1.append(t1[self.params.tagged_class[idx]])
            p2.append(t2[self.params.tagged_class[idx]])
            if external == False:
                p1_idx2.append(t1[self.params.tagged_class[idx2]])
                p2_idx2.append(t2[self.params.tagged_class[idx2]])
            else: 
                p1_idx2.append(t1[self.params.tagged_class[idx]])
                p2_idx2.append(t2[self.params.tagged_class[idx]])

        if self.params.save_probs == True:
            self.all_probs['multitag'].append((p1, p2)) # Saving PAIRED probabilities.

        if '_plus' in self.params.dataset:
            # Test data for the _plus (real world tag) scenario is not paired.
            res_true_idx1 = ttest_ind(a=p1, b=p2, equal_var=False, alternative='greater') 
            res_true_idx2 = ttest_ind(a=p2_idx2, b=p1_idx2, equal_var=False, alternative='greater')
        else:  
            # Do a paired t-test on the different relative probabilities. 
            res_true_idx1 = ttest_rel(a=p1, b=p2, alternative='greater') # we want to know if the mean of p1 is greater than mean of p2
            res_true_idx2 = ttest_rel(a=p2_idx2, b=p1_idx2, alternative='greater') # if external, this is testing if external tag induces a higher shift than true tag. 
        return res_true_idx1.pvalue, res_true_idx2.pvalue

    def get_test_data(self, idx=None, idx2=None, external=False, external_idx=0):
        shap = self.train_data_shape if self.params.dataset != 'pubfig' else (116, 100, 3)
        data = torch.zeros((self.params.num_test, shap[0], shap[1], shap[2]))
        labels = torch.zeros(self.params.num_test)
        if idx is not None:
            tag_l = [torch.as_tensor(self.params.tagged_class[idx]).int()] 
        else:
            tag_l = [torch.as_tensor(self.params.tagged_class).int()]
            
        if self.params.attack == 'label_flip':
            tag_l.append(torch.as_tensor(self.params.target_class))    
        generator = self.generator
        i = 0
        for dt in generator:
            test_data, test_labels  = dt
            for d,l in zip(test_data, test_labels):
                if (self.params.test_ood == True) or (l.cpu().int() not in tag_l): 
                    if i == self.params.num_test:
                        break
                    if d.shape[0] != shap[0]:
                        d = torch.transpose(d, 2,0)
                    labels[i] = l if (self.params.test_ood == False) else 0 # Label doesn't matter.
                    data[i,:,:,:] = d
                    i += 1
            if (i == self.params.num_test) and (self.params.ffcv == False):
                break

        # Make sure data range matches, in this case check to make sure that you need to scale down by 255
        datamax = int(torch.max(data).detach().cpu())
        if (datamax > self.data_max) and (datamax > 200) and (self.data_max < 2):
            data = data / 255.0

        td = data.clone().detach().cpu().numpy()
        tag_test_data = torch.zeros((self.params.num_test, shap[0], shap[1], shap[2]))
        for i in range(self.params.num_test):
            tag_test_data[i,:,:,:] = torch.Tensor(self.tagger.tag_image(td[i], idx=idx))

        if idx2 is not None: 
            tag_test_data2 = torch.zeros((self.params.num_test, shap[0], shap[1], shap[2]))
            for i in range(self.params.num_test): 
                if external == True:
                    tag_test_data2[i, :,:,:] = torch.Tensor(self.tagger.tag_image(td[i], idx=idx2, external=external, external_idx=external_idx))
                else:
                    tag_test_data2[i, :,:,:] = torch.Tensor(self.tagger.tag_image(td[i], idx=idx2, external=False))
        test_labels = test_labels[:self.params.num_test]
        try:
            test_data = self.norm(data).detach() # normalize!
        except:
            test_data = self.norm(torch.transpose(data,3,1)).detach()
        if self.params.dataset == 'imagenet':
            tag_test_data = torch.transpose(tag_test_data, 3,1)
        tag_test_data = self.norm(tag_test_data).detach() # normalize!  
        if idx2 is not None:
            if self.params.dataset == 'imagenet':
                tag_test_data2 = torch.transpose(tag_test_data2, 3,1)
            tag_test_data2 = self.norm(tag_test_data2).detach()
            return test_data, tag_test_data, test_labels, tag_test_data2
        else:
            return test_data, tag_test_data, test_labels
