cp ../horovod/torch/__init__.py /home/togo/anaconda3/lib/python3.6/site-packages/horovod/torch/
qcp -i egpu2 /home/togo/anaconda3/usr/horovod/
qcp -i egpu2 /home/togo/anaconda3/bin/horovodrun
qcp -i egpu2 /home/togo/anaconda3/lib/python3.6/site-packages/horovod*
scp ./pytorch_imagenet_resnet50.py togo@egpu2:~/horovod/examples/
horovodrun -np 2 -H localhost:1,egpu2:1 python pytorch_imagenet_resnet50.py --train-dir /data/pytorch-imagenet-data/train/ --val-dir /data/pytorch-imagenet-data/val/ --threshold 262144
