```
pip install scikit-learn==0.22.1
```



```
# step 1 剪枝
python prune.py --compress_rate [0.2]*9
# step 2 训练
python train_cls_conv_prune.py --pruned prune_runs/prune_18%.pt --normal --learning_rate 0.001 --epoch 100 --optimizer SGD --data_dir G:\\modelnet40_normal_resampled 
# step 3 测试
python eval_cls_conv_prune.py --checkpoint prune_runs/prune_18%.pt --normal --data_dir G:\\modelnet40_normal_resampled 
```

