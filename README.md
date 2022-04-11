# NEURAL NETWORKS WITH LATE-PHASE WEIGHTS

by <b>Johannes von Oswald*, Seijin Kobayashi*</b><br/>
Alexander Meulemans, Christian Henning, Joao Sacramento<br/>
(* – equal contribution)

In this repository, you can find a PyTorch implementation of [neural network with late-phase weights](https://arxiv.org/abs/2007.12927).

In a late stage of training, a small set of parameters (for example the weights of BatchNorm layers) are duplicated K times and possibly perturbed from one another.
Then, we iteratively train this ensemble of K neural networks, which differ only in these late-phase weights. Since most of the parameters in the ensemble are shared, little additional memory is needed to instantiate it.

While the K different instances of these special late-phase weights are updated directly on the gradients of a given loss, 
the gradients w.r.t. the rest of the parameters are accumulated. After a full loop over the ensemble, the main bulk i.e. the parameters shared across the ensemble are updated.

At the end of training, instead of treating each ensemble member separately, we spatially average the ensemble of late-phase weights to recover a single neural network. In our paper, we show that this can lead to improved generalization across multiple different datasets and model classes. Without any additional training effort, one can also use the economically-trained ensemble as a standard ensemble by averaging its predictions (instead of the weights) which leads to strong out-of-distribution detection performance on the vision models we tested on.

## Requirements
To install requirements:
```setup
pip install -r requirements.txt
```

## Training from scratch

### Training

To train and evaluate the model in the paper, run this command:
```train
python experiment/train_and_evaluate.py --config <config to run>
```

This will train the model and return the final test set accuracy and OOD results.

The following config are available:

| Config         | Description  |
| ------------------ |---------------- |
| cifar10_bn  |     Train and evaluate our model presented in Table 1 in the paper, Late-phase (SGD)  | 
| cifar10_bn_swa   |     Train and evaluate our model presented in Table 1 in the paper, Late-phase (SWA)  |
| cifar10_wrn14_bn_swa   |     Train and evaluate our model presented in Table 3 in the paper, C10 WRN 28-14 (SWA)  |
| cifar100_bn |     Train and evaluate our model presented in Table 2 in the paper with BatchNorm, SGD |   
| cifar100_bn_swa  |     Train and evaluate our model presented in Table 2 in the paper with BatchNorm, SWA  |  
| cifar100_wrn14_bn_swa  |     Train and evaluate our model presented in Table 3 in the paper, C100 WRN 28-14 (SWA)  |  

Please refer to the respective tables in the paper for more details on the configuration used.

### Results
Our model achieves the following performance (metrics averaged over 5 seeds):

| Config name         | Test set Accuracy  | 
| ------------------ |---------------- |
| cifar10_bn   |     96.46% +/- 0.15%        | 
| cifar10_bn_swa   |     96.81% +/- 0.07%        | 
| cifar10_wrn14_bn_swa   |     97.45% +/- 0.10%        | 
| cifar100_bn   |     82.87% +/- 0.22%  | 
| cifar100_bn_swa   |     83.06% +/- 0.08%        | 
| cifar100_wrn14_bn_swa   |     85.00% +/- 0.25%        | 

## Finetuning pretrained models

### Finetuning


Before starting, you would need to download the ImageNet data and place the following files in the ./datasets directory:

    ILSVRC2012_img_train.tar  
    ILSVRC2012_img_val.tar
    ILSVRC2012_devkit_t12.tar.gz  


The following command will finetune a pretrained model for 20 epochs with the late phase batchnorm algorithm:

    train_super_late_phase.py --arch=<architecture>


### Results

| Architecture         | Test set Accuracy after finetuning | 
| ------------------ |---------------- |
| resnet50   |    76.87% +/- 0.03%        | 
| resnet152   |     78.77% +/- 0.01%        | 
| densenet161   |     78.31% +/- 0.01%        | 


## Citation
Please cite our paper if you use this code in your research project.

```
@inproceedings{oshg2019hypercl,
title={Neural networks with late-phase weights},
author={Johannes von Oswald and Seijin Kobayashi and Alexander Meulemans and Christian Henning and Benjamin F. Grewe and Jo{\~a}o Sacramento},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://arxiv.org/abs/2007.12927}
}
