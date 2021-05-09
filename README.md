In this document, we will briefly introduce our framework `HiPress` and mainly focus on the instructions for making it run.

# What is HiPress?

Gradient synchronization among training nodes has been proved to be a major bottleneck in distributed data parallel DNN training. To eliminate this bottleneck, a number of gradient compression algorithms have been recently proposed, which primarily focus on reducing the amount of gradients exchanged during each training iteration. However, the system challenges of adapting training systems to use these algorithms are often overlooked. First,  the compression-related computational costs accumulate along the gradient synchronization path and are not amortized since the conventional synchronization strategies like Parameter Server (PS) and Ring-allreduce cannot efficiently handle compressed gradients. Consequently, despite of their high compression ratios, when trying out a few open-source implementations within mainstream DNN systems, the performance improvement enabled by gradient compression algorithms is quite limited. Second, gradient compression is not always beneficial due to its non-trivial overhead and its trade-offs with the aforementioned common optimizations, which unfortunately are ignored by current DNN systems. Third, systematic support for compression awareness is lacking. Without such support, a broad adoption of gradient compression becomes difficult because significant system expertise and manual efforts are required for developing, optimizing and integrating each compression algorithm.  


To address the above system challenges, we propose `HiPress`, which is a High Performance and Compression-Aware framework for fast data parallel DNN training. In `HiPress`, we first design a new gradient synchronization library called `CaSync`, which  evolves the current PS and Ring-allreduce strategies to become compression-friendly.  Second, we design `SeCoPa`, which makes joint decisions of gradient partition and compression for each gradient of a given DNN model.  Third, we design a toolkit called `CompLL`, including a high-performance gradient compression library, a domain specific language and an automatic code generator, to facilitate the easy and efficient development of compression algorithms on GPU. Finally, to make the gradient compression feature easy-of-use, we build `HiPress` atop mainstream DNN systems such as MXNet, TensorFlow and PyTorch, as well as automate the integration of `CompLL`-generated compression code into the aforementioned DNN systems via `HiPress`.

Currently, `HiPress` supports four built-in compression algorithms, namely, TBQ[1], TernGrad[2], DGC[3], and GradDrop[4]. We specify their logic by following either their open-source implementations or the pseudocode in their original publications. Taking specifications as input, `CompLL` automatically generates the corresponding GPU codes, as well as the necessary code for further integration.

# Try out `HiPress`
 
Next, we will use the VGG19 model as our `Hello World` example to walk through the whole compression-aware data parallel DNN training procedure. To do so, we will first present instructions to train VGG19 DNN model with `CaSync` and `SeCoPa`, as well as built-in compression algorithms in `HiPress` using MXNet, TensorFlow and PyTorch as the underlying DNN systems, respectively. Second, we will present the instructions to run `CompLL` to generate an exemplified compression algorithm.  Practitioners can then follow our instructions to train their own models and implement more compression algorithms within `HiPress`.  

## Training in `HiPress`

### Step1: Installing basic common software

We first need to install required software with specific versions. They are  cuda 10.1, cuDNN 7.5.1, MPI 3.1.2, nccl 2.4.2, and [Anaconda3-5.2.0](https://repo.continuum.io/archive/).

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
>>> pip3 install torch==1.3.0
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
>>> python data_parallel_train.py --numprocess 4 --servers node1:1,node2:1,node3:1,node4:1 --model vgg19 --topo 'PS'  --comp-alg tbq --comprplan '../../SeCoPa/SeCoPaPlan.txt'
```

##### 1.2. Training with `CaSync-Ring-allreduce`
```bash
>>> cd src/CaSync/horovod-mxnet
>>> # Regenerate the SeCoPaPlan choosing Ring as topology first 
>>> python data_parallel_train.py --numprocess 4 --servers node1:1,node2:1,node3:1,node4:1 --model vgg19 --topo 'Ring'  --comp-alg tbq --comprplan '../../SeCoPa/SeCoPaPlan.txt'
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

In addition to the above end-to-end training scripts, here, we present the instructions to run `CompLL` to exercise the auto-generation of gradient compression algorithms. To keep it simple, we take the `encode` function of the TernGrad[2] algorithm as example.

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



# References
[1] Nikko Strom. Scalable distributed dnn training using commodity gpu cloud computing. In Proceedings of Sixteenth Annual Conference of the International Speech Communication Association, 2015.

[2] Wei Wen, Cong Xu, Feng Yan, Chunpeng Wu, Yandan Wang, Yiran Chen, and Hai Li. Terngrad: Ternary gradients to reduce communication in distributed deep learning. In Proceedings of Advances in neural information processing systems, pages 1509–1519, 2017.

[3] Yujun Lin, Song Han, Huizi Mao, Yu Wang, and William J Dally. Deep gradient compression: Reducing the communication bandwidth for distributed training. arXiv preprint arXiv:1712.01887, 2017.

[4] Alham Fikri Aji and Kenneth Heafield. Sparse communication for distributed gradient descent. arXiv preprint arXiv:1704.05021, 2017.









