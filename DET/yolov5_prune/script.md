```sh
coco2017 
训练
python train.py --data test20230220.yaml --cfg yolov5s.yaml --weights '' --batch-size 16 --epochs 100

python train_prune.py --data test20230220.yaml  --cfg models/yolov5s.yaml --weights prune_runs/pruned.pt   --epoch 1


python train_prune.py --data coco0305.yaml --cfg models/yolov5s.yaml --weights prune_runs/pruned_coco_FSM_20%.pt  --epoch 300  --batch-size 16

测试
python test.py --weights weights/yolov5s.pt --data test20230220.yaml
python test.py --weights prune_runs/pruned.pt --data test20230220.yaml
python test.py --weights prune_runs/pruned_0224_voc.pt --data voc0221.yaml
测试
python detect.py --source 1.jpg --weights weights/yolov5s.pt

python detect.py --source 1.jpg --weights runs/train/exp3_0224_pretrain/weights/best.pt
runs/train/FSM_pruned_finetune_20%_coco_exp13/weights/best.pt
```

```
train: E:\\dataset\\COCO2017\\images\\train\\
val: E:\\dataset\\COCO2017\\images\val\\
train: E:/dataset/voc/images/trainval0712/  # 16551 images
val: E:/dataset/voc/images/test2007/  # 4952 images
```



```sh
VOC数据集
>python train.py --data voc0221.yaml --cfg models/yolov5l.yaml --weights '' --epoch 100  --batch-size 32 --resume
>python train_prune.py --data voc0221.yaml --cfg models/yolov5s.yaml --weights '' --epoch 100  --batch-size 16

```

```
python test.py --data coco0305.yaml --weights runs/train/WACP_pruned_finetune_20%_coco_exp13/weights/best.pt

python test.py --data voc0221.yaml --weights runs/train/WACP_pruned_finetune_20%_exp8/weights/best.pt

python test.py --data voc0221.yaml --weights weights/yolov5s.pt
```



```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```


