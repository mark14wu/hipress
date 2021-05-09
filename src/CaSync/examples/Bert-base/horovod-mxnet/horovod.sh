mpirun -np $1 \
    -H $2 \
	-x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x NCCL_IB_DISABLE=1 -x NCCL_SOCKET_IFNAME=ens14f1 -x NCCL_TREE_THRESHOLD=0 -x GLUE_DIR=$HOME/trainData/glue_data -x MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN=${10} \
	\
    --mca btl tcp,self --mca btl_tcp_if_include ens14f1 \
    -bind-to none -map-by slot \
    -mca pml ob1 \
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
        --comp-threshold=${12}