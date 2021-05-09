# HOROVOD_TIMELINE=/home/gpu/mxnet/Bert/horovod-mxnet/timeline_Bert.json HOROVOD_TIMELINE_MARK_CYCLES=1 \
# HOROVOD_CYCLE_TIME=5 \
MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN=$4 \
NCCL_DEBUG=INFO NCCL_TREE_THRESHOLD=0 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=ens14f1 horovodrun -np $1 \
    -H $2 \
	python3 $HOME/mxnet/horovod/examples/language_model/word_language_model.py \
        --gpu 0 \
        --emsize 1500 \
        --nhid 1500 \
        --nlayers 2 \
        --lr 20 \
        --epochs 3 \
        --batch_size 20 \
        --bptt 35 \
        --dropout 0.65 \
        --dropout_h 0 \
        --dropout_i 0 \
        --dropout_e 0 \
        --weight_drop 0 \
        --tied \
        --wd 0 \
        --alpha 0 \
        --beta 0 \
        --ntasgd \
        --lr_update_interval 50 \
        --lr_update_factor 0.5 \
        --num_batches $3 \
        --comp-alg=$5 \
        --comp-threshold=$6
