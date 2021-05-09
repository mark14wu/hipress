cp ../../horovod/tensorflow/__init__.py /home/togo/anaconda3/lib/python3.6/site-packages/horovod/tensorflow/
cp ../../horovod/tensorflow/mpi_ops.py /home/togo/anaconda3/lib/python3.6/site-packages/horovod/tensorflow/
qcp -i egpu2 /home/togo/anaconda3/usr/horovod/
qcp -i egpu2 /home/togo/anaconda3/bin/horovodrun
qcp -i egpu2 /home/togo/anaconda3/lib/python3.6/site-packages/horovod*
cp /home/togo/tensorflow/tensorflow/core/user_ops/terngrad*o ./
cp /home/togo/tensorflow/tensorflow/core/user_ops/dgc*o ./
scp -r /home/togo/horovod/examples/benchmarks togo@egpu2:/home/togo/horovod/examples/
# qcp -i egpu2 ./terngrad*o
# scp ./tensorflow_mnist.py togo@egpu2:~/horovod/examples/

HOROVOD_CPU_FUSION_THRESHOLD=67108864
HOROVOD_LOG_LEVEL="trace"
echo HOROVOD_CPU_FUSION_THRESHOLD=${HOROVOD_CPU_FUSION_THRESHOLD}
echo HOROVOD_LOG_LEVEL=${HOROVOD_LOG_LEVEL}
horovodrun -np 2 -H localhost:1,egpu2:1 python ./scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model resnet50 --batch_size 64 --variable_update horovod
# horovodrun -np 2 -H localhost:1,egpu2:1 python /home/togo/benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model resnet101 --batch_size 64 --variable_update horovod