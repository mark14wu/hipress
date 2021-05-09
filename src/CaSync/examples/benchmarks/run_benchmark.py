import os
import colorful as cf

HOROVOD = "~/horovod/"
TENSORFLOW = "~/tensorflow/"
CONDA = f"{HOROVOD}/env/"
PYTHON_PACKAGE = f"{CONDA}/lib/python3.6/site-packages/"
PYTHON_PACKAGE64 = f"{CONDA}/lib64/python3.6/site-packages/"

COPY_FILE = "copy_shell.sh"
RUN_FILE = 'run_shell.sh'
RUN_MPI_FILE = 'run_mpi.sh'
TEMPLATE_FILE = 'run_test.sh'

def id2ip(id:int):
    return f"192.168.2.{id+40}"

def run_bash_file(filepath:str):
    cmd = f"bash {filepath}"
    os.system(cmd)



def write_copy_shell(target_id):
    target_ip = id2ip(target_id)
    content = fr"""
cp {TENSORFLOW}/tensorflow/core/user_ops/terngrad*o ./
cp {TENSORFLOW}/tensorflow/core/user_ops/dgc*o ./
cp {HOROVOD}/horovod/tensorflow/*.py {PYTHON_PACKAGE}/horovod/tensorflow/
cp {HOROVOD}/horovod/tensorflow/*.py {PYTHON_PACKAGE64}/horovod/tensorflow/
scp -r {HOROVOD}/horovod/tensorflow/*.py {target_ip}:{HOROVOD}/horovod/tensorflow/
scp -r {PYTHON_PACKAGE}/horovod* {target_ip}:{PYTHON_PACKAGE}/
scp -r {PYTHON_PACKAGE64}/horovod* {target_ip}:{PYTHON_PACKAGE64}/
scp -r {CONDA}/usr/horovod {target_ip}:{CONDA}/usr/
scp {CONDA}/bin/horovodrun {target_ip}:{CONDA}/bin/
scp -r {HOROVOD}/examples/benchmarks {target_ip}:{HOROVOD}/examples/
"""
    with open(COPY_FILE, "w") as f:
        f.write(content)



def write_copy_shell_to_multi_machines(devices=[2]):
    print(f"devices={devices}")
    content = fr"""
cp {TENSORFLOW}/tensorflow/core/user_ops/terngrad*o ./
cp {TENSORFLOW}/tensorflow/core/user_ops/dgc*o ./
cp {HOROVOD}/horovod/tensorflow/*.py {PYTHON_PACKAGE}/horovod/tensorflow/
cp {HOROVOD}/horovod/tensorflow/*.py {PYTHON_PACKAGE64}/horovod/tensorflow/
"""
    for target_id in devices:
        target_ip = id2ip(target_id)
        content+=fr"""
scp -r {HOROVOD}/horovod/tensorflow/*.py {target_ip}:{HOROVOD}/horovod/tensorflow/
scp -r {PYTHON_PACKAGE}/horovod* {target_ip}:{PYTHON_PACKAGE}/
scp -r {PYTHON_PACKAGE64}/horovod* {target_ip}:{PYTHON_PACKAGE64}/
scp -r {CONDA}/usr/horovod {target_ip}:{CONDA}/usr/
scp {CONDA}/bin/horovodrun {target_ip}:{CONDA}/bin/
scp -r {HOROVOD}/examples/benchmarks {target_ip}:{HOROVOD}/examples/
"""
    with open(COPY_FILE, "w") as f:
        f.write(content)


def run_benchmark_once(devices=[1,2]):
    write_copy_shell_to_multi_machines(devices)
    # run_bash_file(COPY_FILE)
    arg_H = ','.join([f"{id2ip(v)}:1" for v in devices])
    cmder = f"horovodrun -np {len(devices)} -H {arg_H} \\"
    with open(TEMPLATE_FILE,'r') as f:
        content = f.read().split('\n')
        final_content = []
        for i, line in enumerate(content):
            line:str = line.strip()
            if line.startswith('horovodrun'):
                line = cmder
            final_content.append(line)
        final_content = '\n'.join(final_content)
    with open(RUN_FILE, 'w') as f:
        content = f.write(final_content)
    


def run_benchmark(devices=[1,2], nums=[]):
    if nums == []:
        nums.append(len(devices))
    for v in nums:
        run_benchmark_once(devices[:v])

if __name__ == '__main__':
    run_benchmark(devices=[1,2,3,4,5,6,7,9,10,11,13,14,15,16,17,18], nums=[2])
    # run_benchmark(devices=[1,2,k13,14,15,16,17,18], nums=[8])
