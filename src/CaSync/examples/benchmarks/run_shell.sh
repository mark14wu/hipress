HOROVOD_CPU_FUSION_THRESHOLD=67108864
HOROVOD_LOG_LEVEL="trace"
echo HOROVOD_CPU_FUSION_THRESHOLD=${HOROVOD_CPU_FUSION_THRESHOLD}
echo HOROVOD_LOG_LEVEL=${HOROVOD_LOG_LEVEL}
horovodrun -np 2 -H 192.168.2.41:1,192.168.2.42:1 \
~/horovod/env/bin/python ./scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
--model resnet50 \
--batch_size 64 \
--variable_update horovod \
--data_dir /data/imagenet-data