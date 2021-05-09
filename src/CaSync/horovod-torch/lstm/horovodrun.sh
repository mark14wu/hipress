NCCL_DEBUG=INFO NCCL_TREE_THRESHOLD=0 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=ens14f1 horovodrun -np $1 \
    -H $2 \
	python3 main.py \
        --batch_size $3 \
        --epochs $4 \
        --threshold $5 \
        --partition-threshold $6 \
        --algorithm $7 
        
