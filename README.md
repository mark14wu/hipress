In this document, we will briefly introduce our framework `HiPress` and mainly focus on the instructions for making it run.

# What is HiPress?

Gradient synchronization among training nodes has been proved to be a major bottleneck in distributed data parallel DNN training. Gradient compression is a promising approach to alleviating the communication bottleneck in data parallel deep neural network (DNN) training by significantly reducing the data volume of gradients for synchronization. While gradient compression is being actively adopted by the industry (e.g., Facebook and AWS), our study reveals that there are two critical but often overlooked challenges: First, design a generalizable approach to amortize the extra computational overhead brought by gradient compression (e.g., encode and decode operators) along the communication steps during gradient synchronization. This is difficult due to non-trivial factors including, for instance, the data dependencies between gradient computation and communication, the communication topology such as a bipartite graph for PS and a ring for Ring-allreduce, the compression speed and ratio of different compression algorithms, to name a few. Second, provide systematic support for developing, optimizing and integrating gradient compression algorithms into DNN systems.  Without this support, the real-world adoption of gradient compression algorithm requires significant system expertise and manual efforts to perform various ad-hoc development and optimization, which is particularly challenging for DNN practitioners.

To address the above system challenges, We first propose a general, composable gradient synchronization architecture, called `CaSync`, which enables a compression-aware gradient synchronization with a composition of decoupled communication, aggregation and compression primitives. Furthermore, `CaSync` employs a selective compression and partitioning mechanism (named as `SeCoPa` in this project) to decide whether to compress each gradient and how to partition large gradients (before compression) to optimally leverage pipelining and parallel processing. `CaSync` architecture is intentionally designed to be general and not tie to specific gradient compression algorithms and synchronization strategies (e.g., PS or Ring-allreduce), thus, its benefits are applicable to existing and potentially future compression algorithms and synchronization strategies. Second, developing and optimizing gradient compression algorithms on GPU is non-trivial and usually requires significant system expertise and manual efforts. To relieve the burden on DNN practitioners, we design and develop `CompLL`, a gradient compression toolkit which facilitates the easy algorithm development and integration on GPU, including a high-performance gradient compression library, a domain specific language and an automatic code generator, to facilitate the easy and efficient development of compression algorithms on GPU. For easy adoption, we have built a compression-aware data parallel DNN training framework called `HiPress`, with both `CaSync` and `CompLL`. `HiPress` runs with the three mainstream DNN systems (i.e., MXNet, TensorFlow and PyTorch).

Currently, `HiPress` supports five built-in compression algorithms, namely, onebit[1], TBQ[2], TernGrad[3], DGC[4], and GradDrop[5]. We specify their logic by following either their open-source implementations or the pseudocode in their original publications. Taking specifications as input, `CompLL` automatically generates the corresponding GPU codes, as well as the necessary code for further integration.

# Try out `HiPress`
 
Next, we will use the VGG19 model as our `Hello World` example to walk through the whole compression-aware data parallel DNN training procedure. To do so, we will present two methods to explain how to use `HiPress`. First, we present how to use the docker environment to train VGG19 model atop MXNet. Second, we will present the instructions to build `HiPress` from source code and train VGG19 DNN model with `CaSync` and `SeCoPa`, as well as built-in compression algorithms in `HiPress` using MXNet, TensorFlow and PyTorch as the underlying DNN systems, respectively. Third, we will present the instructions to run `CompLL` to generate an exemplified compression algorithm.  Practitioners can then follow our instructions to train their own models and implement more compression algorithms within `HiPress`.  

## Start `HiPress` from docker

### Step1: Initializing the docker environment

We have built an easy-to-use docker environment for `HiPress` atop MXNet, the other backend systems will be committed soon. We first need to make sure that the `nvidia-docker` is running correctly, then use the following commands:

```bash
>>> docker pull youhuibai/hipress
>>> # Start the container on each participant for distributed training
>>> nvidia-docker run -itd --name=hipress --net=host -v=/path:/path --privileged=true youhuibai/hipress /usr/sbin/init
>>> docker exec -it hipress bash
```

### Step2: Data parallel distributed training 
We have set the default SSH port as `22222`, you have to setup the SSH configure file at `/root/.ssh/config` as the following examples:
```bash
Host node1
        Port 22222
        HostName [ip address on node1 of interface]
        User root
Host node2
        Port 22222
        HostName [ip address on node2 of interface]
        User root
```
#### 1. Training with `CaSync-PS`
```bash
>>> cd /root/hipress-mxnet/
>>> # Check the script help option for more details.
>>> # Using built-in two bit compression algorithms
>>> python data_parallel_train.py --numprocess 4 --servers node1:1,node2:1,node3:1,node4:1 --model vgg19 --topo 'PS'  --comp-alg tbq --comp-threshold 262144 --horovodrun --interface [network interface]
```

#### 2. Training with `CaSync-Ring`
```bash
>>> cd /root/hipress-mxnet/
>>> python data_parallel_train.py --numprocess 4 --servers node1:1,node2:1,node3:1,node4:1 --model vgg19 --topo 'Ring'  --comp-alg tbq --comp-threshold 262144 --horovodrun --interface [network interface]
```

## Start `HiPress` from source code

### Step1: Installing basic common software

We first need to install required software with specific versions. They are cuda 10.1, cuDNN 7.5.1, MPI 3.1.2, nccl 2.8.4, and [Anaconda3-5.2.0](https://repo.continuum.io/archive/).

### Step2: Install underlying DNN systems

Then, we need to deploy underlying DNN systems MXNet and TensorFlow, atop of which `HiPress` is built.

#### 1. Installing and configuring MXNet

```bash
>>> # Clone the MXNet submodule project first
>>> cd deps/mxnet-1.5.0
>>> bash ./compile.sh
>>> cd python
>>> pip install -e .
```

#### 2. Installing and configuring TensorFlow

```bash
>>> git clone https://github.com/tensorflow/tensorflow
>>> cd tensorflow
>>> git checkout r1.14
>>> ./configure
>>> #[enable cuda support]
>>> bazel build --config=opt --config=cuda --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/tools/pip_package:build_pip_package
>>> ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
>>> pip install /tmp/tensorflow_pkg/tensorflow-1.14.1-cp36-cp36m-linux_x86_64.whl
```

#### 3. Installing and configuring PyTorch

```bash
>>> #other pytorch version is ok, but we recommend 1.3.0
>>> pip3 install torch==1.5.0
>>> #clone the torch-hipress-extension
>>> git clone https://gitlab.com/hipress/torch-hipress-extension.git
>>> cd torch-hipress-extension
>>> #[enable cuda support or other acceleration lib]
>>> bash install.sh
```

### Step3: Installing and configuring `HiPress`
Following the above step, we then install `HiPress` and configure it with underlying DNN systems when needed. 

```bash
>>> # download
>>> git clone https://gitlab.com/hipress/hipress

>>> # install CaSync
>>> cd src/CaSync
>>> bash ./install.sh #turn off HOROVOD_WITH_TENSORFLOW/HOROVOD_WITH_MXNET/HOROVOD_WITH_PYTORCH for your own configurations.
```

Note that there is no need to configure MXNet/PyTorch with `HiPress` since currently we put MXNet as a submodule of `HiPress` and use pytorch extension to power pytorch. However, this is not applied to TensorFlow. For using TensorFlow with `HiPress`, one has to continue the following configuration steps:

```bash
>>> cd hipress
>>> git submodule update --init --recursive
>>> cd deps/tensorflow-ops
```
Modify `PATH_TENSORFLOW` and `PATH_TF_IN_PYTHON` in ./Makefile before compiling.
`PATH_TENSORFLOW` is the path to the TensorFlow's source code, for example: /home/hostname/tensorflow.
`PATH_TF_IN_PYTHON` is the installation path of TensorFlow, for example: /home/hostname/anaconda3/lib/python3.6/site-packages/tensorflow. 

```bash
>>> make -j
```

The step following compilation is to copy binaries of tensorflow ops to the directory where the training scripts are located.
```bash
>>> cd src/CaSync/examples/benchmarks
>>> cp hipress/deps/tensorflow-ops/*.so ./
>>> cp hipress/deps/tensorflow-ops/*.o ./
```

### Step4: Generate compressing plan with `SeCoPa`
Before training, we need to generate a selective compression and partition plan with `SeCoPa` for all gradients produced by the backward propagation computation of DNN layers. As specified by the following commands, `SeCoPa` takes the cluster size and an input file (profiled and measured information) as input and generates a plan file called `SeCoPaPlan.txt`. This file is a collection of tuples, each of which corresponds to the plan of a gradient and contains three fields, namely, gradient_id, is_compressed, and num_of_partitions.
Generate such a plan on PS.
```bash
>>> cd src/SeCoPa
>>> python SeCoPa.py --input 'input.txt' --topology 'PS' --nnodes 4
```
Generate such a plan on Ring.
```bash
>>> python SeCoPa.py --input 'input.txt' --topology 'RING' --nnodes 4
```

These plan files are then consumed by the `HiPress` runtime and DNN systems to make the original training workflow be compression-enabled according to the decisions presented in files, as follows.

### Step5: Training DNN models with the compression feature enabled

Here, we will show how to train the DNN models with HiPress atop MXNet, TensorFlow and PyTorch. 

#### 1. Training in `HiPress`+MXNet

Using the following command, one can easily launch the training job of the VGG19 model using MXNet as the underlying DNN system, across 4 machines using TBQ as the target compression algorithm. Here, we try both  `CaSync-PS` and  `CaSync-Ring-allreduce` to demonstrate that  `CaSync` makes both PS and Ring-allreduce be compression-friendly. For uncompressed gradients, as their impacts are trivial, we use the conventional PS and Ring-allreduce for their synchronization as usual. Here, PS-Lite and NCCL are used. It is worth mentioning that we provide synthetic data rather than the real training data, since it may take a while for downloading.

##### 1.1 Training with `CaSync-PS`
```bash
>>> cd src/CaSync/horovod-mxnet
>>> # Check the script help option for more details.
>>> # Using built-in onebit compression algorithms
>>> python data_parallel_train.py --numprocess 4 --servers node1:1,node2:1,node3:1,node4:1 --model vgg19 --topo 'PS'  --comp-alg onebit --comprplan '../../SeCoPa/SeCoPaPlan.txt' --interface [network interface]
>>> # Using built-in TBQ compression algorithms
>>> python data_parallel_train.py --numprocess 4 --servers node1:1,node2:1,node3:1,node4:1 --model vgg19 --topo 'PS'  --comp-alg tbq --comprplan '../../SeCoPa/SeCoPaPlan.txt' --interface [network interface]
```

##### 1.2. Training with `CaSync-Ring
```bash
>>> cd src/CaSync/horovod-mxnet
>>> # Regenerate the SeCoPaPlan choosing Ring as topology first 
>>> python data_parallel_train.py --numprocess 4 --servers node1:1,node2:1,node3:1,node4:1 --model vgg19 --topo 'Ring'  --comp-alg tbq --comprplan '../../SeCoPa/SeCoPaPlan.txt' --interface [network interface]
```

#### 2. Training in `HiPress`+Other systems

To demonstrate that `HiPress` work with DNN systems other than MXNet, here, we present the commands to launch the compression-aware DNN data parallel training with TensorFlow and PyTorch. Using the following command, one can easily launch the training job of the VGG19 model using TensorFlow and PyTorch as the underlying DNN systems, across 2 machines using DGC and TBQ as the target compression algorithms respectively. Here we use  `CaSync-PS` as the synchronization strategy for compressed gradients.  

##### 2.1. Training Atop TensorFlow
```bash
>>> cd src/CaSync/examples/benchmarks
>>> # Check the script help option for more details.
>>> horovodrun -np 2 -H node1:1,node2:1 python ./scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model vgg19 --batch_size 16 --variable_update horovod --comp_alg dgc
```
##### 2.2. Training Atop PyTorch
```bash
>>> cd src/CaSync/horovod-torch/imageNet
>>> # Check the script help option for more details.
>>> horovodrun -np 2 -H node1:1,node2:1 python pytorch_imagenet.py --batch-size 16 --epochs 1 --num-iterations 300 --model vgg19 --algorithm tbq
```

## Run `CompLL` for code generation

In addition to the above end-to-end training scripts, here, we present the instructions to run `CompLL` to exercise the auto-generation of gradient compression algorithms. To keep it simple, we take the `encode` function of the TernGrad[3] algorithm as example.

### Step 1: Specifying logic in DSL language

```C++
void TernGradEncode(float* gradient, uint8* compressed, uint8 bitwidth){
    lambda_func u_greater = [&](float a, float b) -> float{
        if (a>b){
            return a;
        }
        else{
            return b;
        }
    }
    lambda_func u_smaller = [&](float a, float b) -> float{
        if (a < b){
            return a;
        }
        else {
            return b;
        }
    }
    float max = reduce(gradient, -99999, u_greater);
    float min = reduce(gradient, 99999, u_smaller);
    float gap = (max - min) / ( (1<<bitwidth) -1 );
    uint8 tail = gradient.size % ( 1<<bitwidth);
    lambda_func floatToUint = [&](int index) -> uint<bitwidth> {
        float r = (gradient[index] - min) / gap + random<float>(0,1);
        return floor(r);
    }
    uint<bitwidth>* Q = map(range(gradient.size), floatToUint);
    compressed = concat(bitwidth, tail, min, max, Q);
}
```
    
### Step 2: Translating DSL code into highly optimized GPU code

As follows, we should the commands to perform the code generation.

```bash
>>> cd src/CompLL/GCGen
>>> # -b: generate (b)ody function; 
>>> python3 ./GCGen.py pse/TernGradEncode.pse -b
>>> python3 ./GCGen.py pse/TernGradDecode.pse -b
```

The above commands will generate GPU code as follows:

[GPU implementation of TernGradEncode](https://gitlab.com/hipress/hipress/-/blob/master/src/CompLL/GCGen/output/TernGradEncode/TernGradEncode_body.h)

[GPU implementation of TernGradDecode](https://gitlab.com/hipress/hipress/-/blob/master/src/CompLL/GCGen/output/TernGradDecode/TernGradDecode_body.h)

### Step 3: Registering generated code into TensorFlow/MXNet

Here, we take MXNet as an example. We use the following commands to generate the wrapper functions required for integrating TernGrad into MXNet. 

```bash
>>> # -w: generate (w)rapper function; -r: generate (r)egister codes; -f: indicate target (f)ramework
>>> python3 ./GCGen.py pse/TernGradEncode.pse -w -r -f mxnet
>>> python3 ./GCGen.py pse/TernGradDecode.pse -w -r -f mxnet
```

The above commands create the following wrapper functions and registration functions:

[Wrapper For TernGradEncode](https://gitlab.com/hipress/hipress/-/blob/master/src/CompLL/GCGen/output/TernGradEncode/TernGradEncode_wrapper.h)

[Wrapper For TernGradDecode](https://gitlab.com/hipress/hipress/-/blob/master/src/CompLL/GCGen/output/TernGradDecode/TernGradDecode_wrapper.h)

[Registration for TernGradEncode](https://gitlab.com/hipress/hipress/-/blob/master/src/CompLL/GCGen/output/TernGradEncode/TernGradEncode.cc)

[Registration for TernGradDecode](https://gitlab.com/hipress/hipress/-/blob/master/src/CompLL/GCGen/output/TernGradDecode/TernGradDecode.cc)

Next, copy generated code files into MXNet repository, and compile.
```bash
>>> # export `MXNET` as path of MXNet
>>> cp output/TernGradEncode/* $MXNET/src/operator/contrib/
>>> cp output/TernGradDecode/* $MXNet/src/operator/contrib/
>>> cd $MXNET
>>> make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1 USE_DIST_KVSTORE=1 USE_CPP_PACKAGE=1 NVCCFLAGS='--default-stream per-thread -std=c++11' 
```
### Step 4:  Testing registered TernGrad algorithms

MXNet training script is as follows. 

```bash
>>> python3 data_parallel_train.py --numprocess 2 --servers host1:1,host2:1 --model vgg19 --comp-threshold 262144 --comp-alg terngrad --horovodrun
```

# Reproduce the baselines' results
One can use commands at this [repo](https://gitlab.com/hipress/baselines) to reproduce the end-to-end training throughput in our SOSP'21 paper.

# References
[1] Frank Seide, Hao Fu, Jasha Droppo, Gang Li, and Dong Yu. 1-bit stochastic gradient descent and its application to data-parallel distributed training of speech dnns. InFifteenth Annual Conference of the International Speech Communication Association, 2014.

[2] Nikko Strom. Scalable distributed dnn training using commodity gpu cloud computing. In Proceedings of Sixteenth Annual Conference of the International Speech Communication Association, 2015.

[3] Wei Wen, Cong Xu, Feng Yan, Chunpeng Wu, Yandan Wang, Yiran Chen, and Hai Li. Terngrad: Ternary gradients to reduce communication in distributed deep learning. In Proceedings of Advances in neural information processing systems, pages 1509–1519, 2017.

[4] Yujun Lin, Song Han, Huizi Mao, Yu Wang, and William J Dally. Deep gradient compression: Reducing the communication bandwidth for distributed training. arXiv preprint arXiv:1712.01887, 2017.

[5] Alham Fikri Aji and Kenneth Heafield. Sparse communication for distributed gradient descent. arXiv preprint arXiv:1704.05021, 2017.









