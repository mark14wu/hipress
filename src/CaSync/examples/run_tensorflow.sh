# co ../build/lib.linux-x86_64-3.6/horovod/tensorflow/* /home/togo/anaconda3/lib/python3.6/site-packages/horovod/tensorflow/
cp ../horovod/tensorflow/__init__.py /home/togo/anaconda3/lib/python3.6/site-packages/horovod/tensorflow/
cp ../horovod/tensorflow/mpi_ops.py /home/togo/anaconda3/lib/python3.6/site-packages/horovod/tensorflow/
qcp -i egpu2 /home/togo/anaconda3/usr/horovod/
qcp -i egpu2 /home/togo/anaconda3/bin/horovodrun
qcp -i egpu2 /home/togo/anaconda3/lib/python3.6/site-packages/horovod*
# qcp -i egpu2 /home/togo/tensorflow/tensorflow/core/user_ops/*.so
# qcp -i egpu2 /home/togo/tensorflow/tensorflow/core/user_ops/*.o
cp /home/togo/tensorflow/tensorflow/core/user_ops/terngrad*o ./
qcp -i egpu2 ./terngrad*o
scp ./tensorflow_mnist.py togo@egpu2:~/horovod/examples/

HOROVOD_CPU_FUSION_THRESHOLD=0
HOROVOD_LOG_LEVEL="trace"
echo HOROVOD_CPU_FUSION_THRESHOLD=${HOROVOD_CPU_FUSION_THRESHOLD}
echo HOROVOD_LOG_LEVEL=${HOROVOD_LOG_LEVEL}
horovodrun -np 2 -H localhost:1,egpu2:1 python tensorflow_mnist.py 
# mpirun -np 2 -H localhost:1,egpu2:1 python tensorflow_mnist.py 
# mpirun -np 2 -H localhost:1,egpu2:1 -x HOROVOD_CPU_FUSION_THRESHOLD=0 -x HOROVOD_LOG_LEVEL="trace" python tensorflow_mnist.py 
