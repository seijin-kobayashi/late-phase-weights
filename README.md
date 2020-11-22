# NEURAL NETWORKS WITH LATE-PHASE WEIGHTS
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

The following command will finetune a pretrained model for 20 epochs with the late phase batchnorm algorithm:

    train_super_late_phase.py --arch=<architecture>


### Results

| Architecture         | Test set Accuracy after finetuning | 
| ------------------ |---------------- |
| resnet50   |    76.87% +/- 0.03%        | 
| resnet152   |     78.77% +/- 0.01%        | 
| densenet161   |     78.31% +/- 0.01%        | 


