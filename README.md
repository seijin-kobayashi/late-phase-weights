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
| cifar100_bn |     Train and evaluate our model presented in Table 2 in the paper with BatchNorm, SGD |   
| cifar100_bn_swa  |     Train and evaluate our model presented in Table 2 in the paper with BatchNorm, SWA  |  

Please refer to the respective tables in the paper for more details on the configuration used.

### Results
Our model achieves the following performance (metrics averaged over 5 seeds):

| Config name         | Test set Accuracy  | 
| ------------------ |---------------- |
| cifar10_bn   |     96.46% +/- 0.15%        | 
| cifar10_bn_swa   |     96.81% +/- 0.07%        | 
| cifar100_bn   |     82.87% +/- 0.22%  | 
| cifar100_bn_swa   |     83.06% +/- 0.08%        | 
