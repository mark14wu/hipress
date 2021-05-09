mpirun -np 2 -H 192.168.2.41:1,192.168.2.42:1 \
-x NCCL_SOCKET_IFNAME=ens14f1 -x HOROVOD_CPU_FUSION_THRESHOLD=67108864 \
--mca btl tcp,self --mca btl_tcp_if_include ens14f1 \
-bind-to none -map-by slot \
~/horovod/env/bin/python ./scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
--model alexnet \
--batch_size 128 \
--variable_update horovod
# --data_dir /data/imagenet-data \