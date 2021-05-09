
cp ~/tensorflow//tensorflow/core/user_ops/terngrad*o ./
cp ~/tensorflow//tensorflow/core/user_ops/dgc*o ./
cp ~/horovod//horovod/tensorflow/*.py ~/horovod//env//lib/python3.6/site-packages//horovod/tensorflow/
cp ~/horovod//horovod/tensorflow/*.py ~/horovod//env//lib64/python3.6/site-packages//horovod/tensorflow/

scp -r ~/horovod//horovod/tensorflow/*.py 192.168.2.41:~/horovod//horovod/tensorflow/
scp -r ~/horovod//env//lib/python3.6/site-packages//horovod* 192.168.2.41:~/horovod//env//lib/python3.6/site-packages//
scp -r ~/horovod//env//lib64/python3.6/site-packages//horovod* 192.168.2.41:~/horovod//env//lib64/python3.6/site-packages//
scp -r ~/horovod//env//usr/horovod 192.168.2.41:~/horovod//env//usr/
scp ~/horovod//env//bin/horovodrun 192.168.2.41:~/horovod//env//bin/
scp -r ~/horovod//examples/benchmarks 192.168.2.41:~/horovod//examples/

scp -r ~/horovod//horovod/tensorflow/*.py 192.168.2.42:~/horovod//horovod/tensorflow/
scp -r ~/horovod//env//lib/python3.6/site-packages//horovod* 192.168.2.42:~/horovod//env//lib/python3.6/site-packages//
scp -r ~/horovod//env//lib64/python3.6/site-packages//horovod* 192.168.2.42:~/horovod//env//lib64/python3.6/site-packages//
scp -r ~/horovod//env//usr/horovod 192.168.2.42:~/horovod//env//usr/
scp ~/horovod//env//bin/horovodrun 192.168.2.42:~/horovod//env//bin/
scp -r ~/horovod//examples/benchmarks 192.168.2.42:~/horovod//examples/
