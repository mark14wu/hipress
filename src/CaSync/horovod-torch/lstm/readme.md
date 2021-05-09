This is a benchmark for hipress-pytorch. Before start, please make sure that you have installed hipress-torch-extension and hipress well.

## Download your dataset
We use `wikitext-2` as our training dataset. 
```bash
#Download wikitext-2
bash getdata.sh 
```

## How to run
Using the following command, one can easily launch the training job of the StardRNN model using PyTorch as the underlying DNN system, across 2 machines using terngrad as the target compression algorithm.
```bash
horovodrun -np 2 -H localhost:2 python3 main.py --batch-size 80 --epochs 1  --threshold 0 --partition-threshold 4194304 --algorithm terngrad
```

## Run benchmark
We prepare `run-*.sh` to reproduce our experiment results. However, you may need to change some network configurations to run `run-*.sh` correctly.
```bash
#for StardRNN
bash run.sh
```