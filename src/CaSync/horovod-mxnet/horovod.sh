# -x HOROVOD_TIMELINE=$HOME/mxnet/horovod/horovod-mxnet/timeline.json
# -x HOROVOD_CPU_FUSION_THRESHOLD=0
mpirun -np $1 \
    -H $2 \
	-x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x NCCL_IB_DISABLE=1 -x NCCL_SOCKET_IFNAME=${12} -x NCCL_TREE_THRESHOLD=0 -x MXNET_NUM_OF_PARTICIPANTS=$1 -x MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN=$9 -x HOROVOD_LOG_LEVEL="fatal" \
	\
    --mca btl tcp,self --mca btl_tcp_if_include ${12} \
    -bind-to none -map-by slot \
	python ../examples/mxnet_imagenet_resnet50.py \
        --mode gluon \
        --batch-size $3 \
        --num-epochs 1 \
        --log-interval 10 \
        --num-examples $4 \
        --model $5 \
        --image-shape $6 \
        --rec-train=$7 \
        --rec-train-idx=$8 \
        --rec-val="/trainData/imagenet1k-val.rec" \
        --rec-val-idx="/trainData/imagenet1k-val.idx" \
        --comp-alg=${10} \
        --comp-threshold=${11} \
        ${13}