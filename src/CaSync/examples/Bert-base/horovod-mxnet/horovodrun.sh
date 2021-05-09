# HOROVOD_TIMELINE=/home/gpu/mxnet/horovod/examples/Bert-base/horovod-mxnet/timeline_Bert.json HOROVOD_TIMELINE_MARK_CYCLES=1 \
# HOROVOD_CYCLE_TIME=5 \
MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN=${10} \
GLUE_DIR=$HOME/trainData/glue_data NCCL_DEBUG=INFO NCCL_TREE_THRESHOLD=0 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=ens14f1 horovodrun -np $1 \
    -H $2 \
	python3 $HOME/mxnet/horovod/examples/Bert-base/finetune_classifier.py \
        --batch_size $3 \
        --task_name $4 \
        --epochs 3 \
        --gpu 0 \
        --num_batches $5 \
        --log_interval $6 \
        --lr $7 \
        --epsilon $8 \
        --max_len $9 \
        --comp-alg=${11} \
        --comp-threshold=${12} \
        # --profile
