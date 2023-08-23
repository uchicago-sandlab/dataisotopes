## Overview
Code to accompany: "Data Isotopes for Data Provenance in DNNs".

---

## Requirements

This repository uses a conda environment. To set it up (assuming `conda` is installed):
```
$ conda env create -f environment.yml
$ conda activate venv
```
Our code uses the [ffcv](https://github.com/libffcv/ffcv) library to train CIFAR100 models, so you will have to ensure your system is compatible with that library. 

---

## Running the code

This codebase provides code to reproduce results for GTSRB, CIFAR100 and PubFig datasets as shown in the original paper. It also supports running experiments on CIFAR10 (not evaluated in original paper). We have provided config files to recreate specific experiments (see the [reproducing experiments](#reproducing-experiments) section below.) Below, we provide general information on setting up the codebase. 

---

## Setup

### Datasets
- The GTSRB dataset can be downloaded from [here](https://uchicago.box.com/shared/static/7phth1trxbrfaa78c4cmyhbflfa6gm22.h5).
- The CIFAR10 and CIFAR100 datasets will be automatically downloaded when you run the code. 
- The version of the PubFig dataset we use can be downloaded from [here](https://uchicago.box.com/shared/static/voc1as7flsq3rvme405t4eu9u4tz2f3q.h5). 
- The FaceScrub dataset can be obtained [here](http://vintage.winklerbros.net/facescrub.html). We align our version of Scrub using [MTCNN](https://github.com/timesler/facenet-pytorch#guide-to-mtcnn-in-facenet-pytorch). Additionally, you will need to update the ```tr_folder``` and ```ts_folder``` parameters in the ```config/scrub/*.yml``` files to ensure they point to your local Scrub dataset on your local host. 

### Models/training settings

Each dataset has an associated folder in ```./configs```. There is a ```default.yaml``` file in each config folder, which contains the default model and training parameters for each dataset (see Table 6 in the paper Appendix) in the single and multi-tag setting. 

If you want to train the PubFig model, you need to download the SphereFace model checkpoint from [this link](https://uchicago.box.com/shared/static/5mu5a1bvpjotwwjhq2lnbm3pp5mqh615.pth) and put it in ```src/models/```. 


### Isotope marks
- Our code supports four types of marks, shown in Figure 6 of the paper: random pixels, pixels square, blend, and Imagenet blend.
- To use the Imagenet blend mark, you will also need to (1) download the Imagenet validation dataset from [this link](https://image-net.org/download.php) and (2) set the ```imagenet_path``` variable in ```configs/default.yaml``` to point towards the folder containing the Imagenet validation data (or you can add an additional argument ```--imagenet_path <Imagenet validation data path>``` when running experiments).

---

## Running experiments

To run one-off experiments with our code, you use the ```main.py``` script, which accepts command line arguments specifying the dataset, mark type, model, training settings, etc. 

However, it would be easier for you to write your own ```.yml``` files containing the experiments you want to run. Then, you can use the ```run_on_gpus.py``` script to load the ```.yml``` file and associated experiment settings and run this experiment. 

__Single experiment example:__ For example, if one wanted to recreate the multi-mark CIFAR100 experiments in Figure 10 of our paper, they could run the following command ```python3 run_on_gpus.py configs/cifar100/multi/default.yaml --gpu_ls 0 --max_gpu_num 1```. This will run the experiment on gpu 0 of your localhost.

__Multi-processing example:__ If you want to easily spread the work among different GPUs, run ```python3 run_on_gpus.py configs/cifar100/multi/default.yaml --gpu_ls 0123 --max_gpu_num 4```. Assuming you have $4$ gpus on your system, this will launch $1$ run on each GPU and wait until it finds an empty GPU before it starts the next run in the list.

---

## Reproducing results

If you want to reproduce the key evaluation experiments from Sections 5.2 and 5.3 in our paper, use the table below to map specific experiments to specific ```.yaml``` files in the ```configs``` folder. The ```.yaml``` files have the same name for each dataset (if the dataset is used for that experiment), so if you want to run experiment Z for dataset X in setting Y (single or multi tag), you would run something like ```python3 run_on_gpus.py ./config/X/Y/Z.yaml --gpu_ls 01 --max_gpu_num 2```.

| Table/Figure | Datasets  | Config file names |
| ----------------- | --------- | ------------------|
| Figure 8          | GTSRB, CIFAR100, PubFig, Scrub   | ```./configs/<DATASET>/single/default.yaml```              |
| Table 4           | CIFAR100 | ```./configs/cifar100/multi/ablation_same_cla.yaml```|
| Figure 9 | PubFig | ```./configs/pubfig/multi/ablation_perc_alpha.yaml``` |
| Table 3 | GTSRB, CIFAR100, PubFig, Scrub  | ```./configs/<DATASET>/multi/default.yaml``` | 


## License

All code in this repository is licensed under the MIT license. 