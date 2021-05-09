This is a benchmark for hipress-pytorch. Before start, please make sure that you have installed hipress-torch-extension and hipress well.

## Prepare your dataset
We use imagenet as our training dataset. You also can use other dataset like (cifar10, cifar100), but you may change code for compatibility.

## How to run
Using the following command, one can easily launch the training job of the VGG19 model using PyTorch as the underlying DNN system, across 2 machines using terngrad as the target compression algorithm.
```bash
horovodrun -np 2 -H localhost:2 python3 pytorch_imagenet.py --batch-size 32 --epochs 1 --num-iterations 300 --threshold 262144 --partition-threshold 16777216 --model resnet50 --algorithm terngrad
```

## Run benchmark
We prepare `run-*.sh` to reproduce our experiment results. However, you may need to change some network configurations to run `run-*.sh` correctly.
```bash
#for resnet50 model
bash run-resnet50.sh 
#for VGG19
bash run-vgg19.sh
```