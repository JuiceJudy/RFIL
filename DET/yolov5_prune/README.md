# RFIL for YOLOv5

## Environments

The code has been tested in the following environments:

- Python 3.8
- PyTorch 1.8.1
- cuda 10.2
- torchsummary, torchvision, thop, scipy, sympy

## Running Code

Pre-trained Models: [YOLOv5s](https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt)

### Prune

#### step 1

```sh
#add prune_rate to yolov5s.yaml
#example:
prune_rate: 
  [[0.],
   [0.3],  #Conv 0.1
   [0.6],  #C3
   [0.3],  #Conv
   [0.6],  #C3   
   [0.3],  #Conv  5
   [0.7],  #C3
   [0.],  #Conv
   [0.],   #SPP
   [0.6],  #C3
   [0.],   #Conv  10
   [0.],   #Upsample
   [0.],   #Concat
   [0.6],  #C3
   [0.],   #Conv
   [0.],   #Upsample  15
   [0.],   #Concat
   [0.6],  #C3
   [0.],   #Conv
   [0.],   #Concat
   [0.6],  #C3    20
   [0.],   #Conv
   [0.],   #Concat
   [0.6],  #C3
   [0.],   #Detect  24
  ]
```

#### step 2 

```shell
python prune.py --weights weights/yolov5s.pt --cfg yolov5s.yaml --prune_method RFIL
```

### Fine-tune

#### step 1

```sh
#configure the dataset-path in data/coco.yaml
#example
#train: E:\\COCO2017\\images\\train\\
#val: E:\\COCO2017\\images\val\\
#train: ../COCO2017/images/train/
#val: ../COCO2017/images/val/
```

#### step 2 

```
python train_prune.py --data coco0305.yaml --cfg models/yolov5s.yaml --weights prune_runs/pruned.pt  --epoch 1  --batch-size 16
```

### Test

```sh
python detect.py --source test.jpg --weights [MODEL PATH] #weights/yolov5s.pt
```

