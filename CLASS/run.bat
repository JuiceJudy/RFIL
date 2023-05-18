rem rem --------------------------resnet_50--------------------------------

rem @echo off
rem start cmd /c ^
rem "cd /D [code dir]  ^
rem & [python.exe dir] main.py ^
rem --arch resnet_50 ^
rem --resume [pre-trained model dir] ^
rem --compress_rate [0.]+[0.2,0.2,0.2]*1+[0.65,0.65,0.2]*2+[0.2,0.2,0.15]*1+[0.65,0.65,0.15]*3+[0.2,0.2,0.1]*1+[0.65,0.65,0.1]*5+[0.2,0.2,0.1]+[0.2,0.2,0.1]*2 ^
rem --num_workers 8 ^
rem --epochs 1 ^
rem --job_dir [pruned-model save dir] ^
rem --lr 0.001 ^
rem --lr_decay_step 1 ^
rem --save_id 1 ^
rem --batch_size 128 ^
rem --weight_decay 0. ^
rem --input_size 224 ^
rem --dataset ImageNet ^
rem --data_dir [dataset dir] ^
rem & pause"

rem @echo off
rem start cmd /c ^
rem "cd /D [code dir] ^
rem & [python.exe dir] main.py ^
rem --arch resnet_50 ^
rem --from_scratch True ^
rem --resume final_pruned_model/resnet_50_1.pt ^
rem --num_workers 4 ^
rem --epochs 120 ^
rem --job_dir [pruned-model save dir] ^
rem --lr 0.01 ^
rem --lr_decay_step 30,60,90 ^
rem --save_id 1 ^
rem --batch_size 128 ^
rem --weight_decay 0.0001 ^
rem --input_size 224 ^
rem --dataset ImageNet ^
rem --data_dir [dataset dir] ^
rem & pause"

rem rem --------------------------mobilenet_v2--------------------------------

rem @echo off
rem start cmd /c ^
rem "cd /D [code dir]  ^
rem & [python.exe dir] main.py ^
rem --arch mobilenet_v2 ^
rem --resume [pre-trained model dir] ^
rem --compress_rate [0.]+[0.1]*2+[0.1]*2+[0.3]+[0.1]*2+[0.3]*2+[0.1]*2+[0.3]*3+[0.1]*2+[0.3]*2+[0.1]*2+[0.3]*2+[0.1]*2 ^
rem --num_workers 8 ^
rem --epochs 1 ^
rem --job_dir [pruned-model save dir] ^
rem --lr 0.001 ^
rem --lr_decay_step 1 ^
rem --save_id 1 ^
rem --batch_size 128 ^
rem --weight_decay 0. ^
rem --input_size 224 ^
rem --dataset ImageNet ^
rem --data_dir [dataset dir] ^
rem & pause"

rem @echo off
rem start cmd /c ^
rem "cd /D [code dir] ^
rem & [python.exe dir] main.py ^
rem --arch mobilenet_v2 ^
rem --from_scratch True ^
rem --resume final_pruned_model/mobilenet_v2_1.pt ^
rem --num_workers 4 ^
rem --epochs 150 ^
rem --job_dir [pruned-model save dir] ^
rem --lr 0.01 ^
rem --lr_decay_step cos ^
rem --save_id 1 ^
rem --batch_size 128 ^
rem --weight_decay 0.00005 ^
rem --input_size 224 ^
rem --dataset ImageNet ^
rem --data_dir [dataset dir] ^
rem & pause"

rem rem --------------------------resnet_56--------------------------------

rem @echo off
rem start cmd /c ^
rem "cd /D [code dir] ^
rem & [python.exe dir] main.py ^
rem --arch resnet_56 ^
rem --resume [pre-trained model dir] ^
rem --compress_rate [0.]+[0.2,0.]*1+[0.65,0.]*8+[0.2,0.15]*1+[0.65,0.15]*8+[0.2,0.]*1+[0.4,0.]*8 ^
rem --num_workers 1 ^
rem --epochs 1 ^
rem --job_dir [pruned-model save dir] ^
rem --lr 0.001 ^
rem --lr_decay_step 1 ^
rem --weight_decay 0. ^
rem --dataset CIFAR10 ^
rem --data_dir [dataset dir] ^
rem --save_id 1 ^
rem & pause"

rem @echo off 
rem start cmd /c ^
rem "cd /D [code dir] ^
rem & [python.exe dir] main.py ^
rem --arch resnet_56 ^
rem --from_scratch True ^
rem --resume final_pruned_model/resnet_56_1.pt ^
rem --num_workers 1 ^
rem --job_dir [pruned-model save dir] ^
rem --epochs 300 ^
rem --lr 0.01 ^
rem --lr_decay_step 150,225 ^
rem --weight_decay 0.0005 ^
rem --dataset CIFAR10 ^
rem --data_dir [dataset dir] ^
rem --save_id 1 ^
rem & pause"

rem rem --------------------------vgg_16_bn--------------------------------

rem @echo off
rem start cmd /c ^
rem "cd /D [code dir]  ^
rem & [python.exe dir] main.py ^
rem --arch vgg_16_bn ^
rem --resume [pre-trained model dir] ^
rem --compress_rate [0.25]*5+[0.35]*3+[0.8]*5 ^
rem --num_workers 1 ^
rem --epochs 1 ^
rem --job_dir [pruned-model save dir] ^
rem --lr 0.001 ^
rem --lr_decay_step 1 ^
rem --weight_decay 0. ^
rem --dataset CIFAR10 ^
rem --data_dir [dataset dir] ^
rem --save_id 0 ^
rem & pause"

rem @echo off
rem start cmd /c ^
rem "cd /D [code dir]  ^
rem & [python.exe dir] main.py ^
rem --arch vgg_16_bn ^
rem --from_scratch True ^
rem --resume final_pruned_model/vgg_16_bn_0.pt ^
rem --num_workers 1 ^
rem --job_dir [pruned-model save dir] ^
rem --epochs 200 ^
rem --lr 0.01 ^
rem --gpu 0 ^
rem --lr_decay_step 100,150 ^
rem --weight_decay 0. ^
rem --dataset CIFAR10 ^
rem --data_dir [dataset dir] ^
rem --save_id 1 ^
rem & pause"

rem rem --------------------------googlenet--------------------------------

rem @echo off
rem start cmd /c ^
rem "cd /D [code dir] ^
rem & [python.exe dir] main.py ^
rem --arch googlenet ^
rem --resume [pre-trained model dir] ^
rem --compress_rate [0.2]+[0.7]*15+[0.75]*9+[0.,0.4,0.] ^
rem --num_workers 1 ^
rem --epochs 1 ^
rem --job_dir [pruned-model save dir] ^
rem --lr 0.001 ^
rem --lr_decay_step 1 ^
rem --weight_decay 0. ^
rem --dataset CIFAR10 ^
rem --data_dir [dataset dir] ^
rem --save_id 1 ^
rem & pause"

rem @echo off
rem start cmd /c ^
rem "cd /D [code dir]  ^
rem & [python.exe dir] main.py ^
rem --arch googlenet ^
rem --from_scratch True ^
rem --resume final_pruned_model/googlenet_1.pt ^
rem --num_workers 1 ^
rem --job_dir [pruned-model save dir] ^
rem --epochs 200 ^
rem --lr 0.01 ^
rem --lr_decay_step 100,150 ^
rem --weight_decay 0. ^
rem --dataset CIFAR10 ^
rem --data_dir [dataset dir] ^
rem --save_id 1 ^
rem & pause"

rem rem --------------------------densenet40--------------------------------

rem @echo off
rem start cmd /c ^
rem "cd /D [code dir] ^
rem & [python.exe dir] main.py ^
rem --arch densenet_40 ^
rem --resume [pre-trained model dir] ^
rem --compress_rate [0.]+[0.5]*12+[0.2]+[0.5]*12+[0.2]+[0.5]*12 ^
rem --num_workers 1 ^
rem --epochs 1 ^
rem --job_dir [pruned-model save dir] ^
rem --lr 0.001 ^
rem --lr_decay_step 1 ^
rem --weight_decay 0. ^
rem --dataset CIFAR10 ^
rem --data_dir [dataset dir] ^
rem --save_id 1 ^
rem & pause"

rem @echo off
rem start cmd /c ^
rem "cd /D [code dir]  ^
rem & [python.exe dir] main.py ^
rem --arch densenet_40 ^
rem --from_scratch True ^
rem --resume final_pruned_model/densenet_40_1.pt ^
rem --num_workers 1 ^
rem --job_dir [pruned-model save dir] ^
rem --epochs 100 ^
rem --lr 0.01 ^
rem --lr_decay_step 100,150 ^
rem --weight_decay 0. ^
rem --dataset CIFAR10 ^
rem --data_dir [dataset dir] ^
rem --save_id 1 ^
rem & pause"
