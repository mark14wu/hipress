
# HOROVOD_CPU_FUSION_THRESHOLD=67108864
# HOROVOD_LOG_LEVEL="trace"
# mpirun -np 2 -H localhost:1,egpu2:1 -x HOROVOD_LOG_LEVEL="trace" -x NCCL_SOCKET_IFNAME=ens14f1 -x HOROVOD_CPU_FUSION_THRESHOLD=67108864 python ./scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model vgg19 --batch_size 16 --variable_update horovod --threshold 262144
mpirun -np 2 -H localhost:1,egpu2:1 \
-x NCCL_SOCKET_IFNAME=ens14f1 -x HOROVOD_CPU_FUSION_THRESHOLD=67108864 \
--mca btl tcp,self --mca btl_tcp_if_include ens14f1 \
-bind-to none -map-by slot \
python ./scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model vgg19 --batch_size 16 --variable_update horovod --threshold 262144
# mpirun -np 2 -H localhost:1,egpu2:1 -x NCCL_SOCKET_IFNAME=ens14f1 -x HOROVOD_CPU_FUSION_THRESHOLD=67108864 python ./scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model vgg19 --batch_size 16 --variable_update horovod --threshold 262144
