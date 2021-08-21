#training gluon
# HOROVOD_LOG_LEVEL="fatal"
# HOROVOD_FUSION_THRESHOLD=0 \
# HOROVOD_TIMELINE_MARK_CYCLES=1 \
# HOROVOD_CPU_FUSION_THRESHOLD=0 \
MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN=$9 \
MXNET_NUM_OF_PARTICIPANTS=$1 NCCL_DEBUG=INFO NCCL_TREE_THRESHOLD=0 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=${12} horovodrun -np $1 \
    -H $2 \
	python ../examples/mxnet_imagenet_resnet50.py \
        --mode gluon \
        --batch-size $3 \
        --num-epochs 3 \
        --log-interval 20 \
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
