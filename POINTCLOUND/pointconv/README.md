# RFIL for PointConv
## Running Code
### ModelNet40 Classification

Download the ModelNet40 dataset from [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip). This dataset is the same one used in [PointNet](https://arxiv.org/abs/1612.00593).

#### Baseline

```sh
python train_cls_conv.py --model pointconv_modelnet40 --normal --learning_rate 0.001 --epoch 400 --optimizer SGD --data_dir [DATASET PATH] #G:\\modelnet40_normal_resampled
```

#### Prune

```sh
python prune.py --pretrain [MODEL PATH] --compress_rate [0.2]*9 
```

#### Fine-tune

```sh
python train_cls_conv_prune.py --pruned [MODEL PATH] --normal --learning_rate 0.001 --epoch 100 --optimizer SGD --data_dir [DATASET PATH] 
```

#### evaluate 

```sh
python eval_cls_conv_prune.py --checkpoint [MODEL PATH] --normal --data_dir [DATASET PATH] 
```



