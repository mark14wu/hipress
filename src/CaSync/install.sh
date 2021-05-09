pip uninstall -y horovod
rm -rf ./dist
python setup.py sdist
HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITHOUT_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 HOROVOD_NCCL_HOME=/usr/local/nccl HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_ALLGATHER=NCCL pip install dist/horovod-0.16.1.tar.gz
