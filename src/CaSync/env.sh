export HOROVOD_WITH_TENSORFLOW=1 
export HOROVOD_WITH_PYTORCH=1 
export HOROVOD_WITHOUT_MXNET=1 
export HOROVOD_NCCL_HOME=/usr/local/nccl 
export HOROVOD_GPU_ALLREDUCE=NCCL
export HOROVOD_GPU_ALLGATHER=NCCL