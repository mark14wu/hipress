# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import logging
import math
import os
import time

from gluoncv.model_zoo import get_model
from gluoncv2.model_provider import get_model as get_model2
import mxnet as mx
import numpy as np
from mxnet import autograd, gluon, lr_scheduler
from mxnet.io import DataBatch, DataIter
from importlib import import_module
import socket
from mxnet import gluon
from mxnet.gluon import nn


# Training settings
parser = argparse.ArgumentParser(description='MXNet ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--use-rec', action='store_true', default=False,
                    help='use image record iter for data input (default: False)')
parser.add_argument('--data-nthreads', type=int, default=4,
                    help='number of threads for data decoding (default: 4)')
parser.add_argument('--rec-train', type=str, default='',
                    help='the training data')
parser.add_argument('--rec-train-idx', type=str, default='',
                    help='the index of training data')
parser.add_argument('--rec-val', type=str, default='',
                    help='the validation data')
parser.add_argument('--rec-val-idx', type=str, default='',
                    help='the index of validation data')
parser.add_argument('--batch-size', type=int, default=32,
                    help='training batch size per device (default: 32)')
parser.add_argument('--dtype', type=str, default='float32',
                    help='data type for training (default: float32)')
parser.add_argument('--num-epochs', type=int, default=1,
                    help='number of training epochs (default: 1)')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate for a single GPU (default: 0.05)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum value for optimizer (default: 0.9)')
parser.add_argument('--wd', type=float, default=0.0001,
                    help='weight decay rate (default: 0.0001)')
parser.add_argument('--lr-decay', type=float, default=0.1,
                    help='decay rate of learning rate (default: 0.1)')
parser.add_argument('--lr-decay-epoch', type=str, default='30,60',
                    help='epoches at which learning rate decays (default: 30,60)')
parser.add_argument('--warmup-lr', type=float, default=0.0,
                    help='starting warmup learning rate (default: 0.0)')
parser.add_argument('--warmup-epochs', type=int, default=1,
                    help='number of warmup epochs (default: 5)')
parser.add_argument('--last-gamma', action='store_true', default=False,
                    help='whether to init gamma of the last BN layer in \
                    each bottleneck to 0 (default: False)')
parser.add_argument('--model', type=str, default='resnet50_v1',
                    help='type of model to use. see vision_model for options.')
parser.add_argument('--use-pretrained', action='store_true', default=False,
                    help='load pretrained model weights (default: False)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training (default: False)')
parser.add_argument('--eval-epoch', action='store_true', default=False,
                    help='evaluate validation accuracy after each epoch \
                    when training in module mode (default: False)')
parser.add_argument('--eval-frequency', type=int, default=0,
                    help='frequency of evaluating validation accuracy \
                    when training with gluon mode (default: 0)')
parser.add_argument('--log-interval', type=int, default=20,
                    help='number of batches to wait before logging (default: 20)')
parser.add_argument('--save-frequency', type=int, default=0,
                    help='frequency of model saving (default: 0)')
parser.add_argument('--num-examples', type=int, default=12800,
                    help='number of examples for training (default: 12800)')
parser.add_argument('--image-shape', type=str, default='3,224,224',
                    help='the image shape for training')
parser.add_argument('--model-local', action='store_true', default=False,
                    help='load model from local symbols')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='number of NN output classes, default is 1000')
parser.add_argument('--num-batches', type=int, default=200,
                    help='number of training batches, default is 200')
parser.add_argument('--num-layers', type=int, default=50,
                    help='number of NN layers')
parser.add_argument('--profile', action='store_true', help='Profile MXNet')
parser.add_argument('--lenet', action='store_true', help='using lenet as training model')


args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
logging.info(args)

num_workers = 1
rank = 0
local_rank = 0
num_classes = args.num_classes
num_training_samples = int(args.num_examples)
batch_size = args.batch_size
epoch_size = \
    int(math.ceil(int(num_training_samples // num_workers) / batch_size))

# Function for reading data from record file
# For more details about data loading in MXNet, please refer to
# https://mxnet.incubator.apache.org/tutorials/basic/data.html?highlight=imagerecorditer
def get_data_rec(rec_train, rec_train_idx, rec_val, rec_val_idx, batch_size,
                 data_nthreads, shape):
    rec_train = os.path.expanduser(rec_train)
    rec_train_idx = os.path.expanduser(rec_train_idx)
    rec_val = os.path.expanduser(rec_val)
    rec_val_idx = os.path.expanduser(rec_val_idx)
    jitter_param = 0.4
    lighting_param = 0.1
    mean_rgb = [123.68, 116.779, 103.939]

    def batch_fn(batch, ctx):
        data = batch.data[0].as_in_context(ctx)
        label = batch.label[0].as_in_context(ctx)
        return data, label

    train_data = mx.io.ImageRecordIter(
        path_imgrec=rec_train,
        path_imgidx=rec_train_idx,
        preprocess_threads=data_nthreads,
        shuffle=True,
        batch_size=batch_size,
        label_width=1,
        data_shape=shape,
        mean_r=mean_rgb[0],
        mean_g=mean_rgb[1],
        mean_b=mean_rgb[2],
        rand_mirror=True,
        rand_crop=False,
        random_resized_crop=True,
        max_aspect_ratio=4. / 3.,
        min_aspect_ratio=3. / 4.,
        max_random_area=1,
        min_random_area=0.08,
        verbose=False,
        brightness=jitter_param,
        saturation=jitter_param,
        contrast=jitter_param,
        pca_noise=lighting_param,
        num_parts=num_workers,
        part_index=rank,
        device_id=local_rank
    )
    # Kept each node to use full val data to make it easy to monitor results
    val_data = mx.io.ImageRecordIter(
        path_imgrec=rec_val,
        path_imgidx=rec_val_idx,
        preprocess_threads=data_nthreads,
        shuffle=False,
        batch_size=batch_size,
        resize=256,
        label_width=1,
        rand_crop=False,
        rand_mirror=False,
        data_shape=shape,
        mean_r=mean_rgb[0],
        mean_g=mean_rgb[1],
        mean_b=mean_rgb[2],
        device_id=local_rank
    )

    return train_data, val_data, batch_fn

# Create data iterator for synthetic data
class SyntheticDataIter(DataIter):
    def __init__(self, num_classes, data_shape, max_iter, dtype, ctx):
        self.batch_size = data_shape[0]
        self.cur_iter = 0
        self.max_iter = max_iter
        self.dtype = dtype
        label = np.random.randint(0, num_classes, [self.batch_size, ])
        data = np.random.uniform(-1, 1, data_shape)
        self.data = mx.nd.array(data, dtype=self.dtype,
                                ctx=ctx)
        self.label = mx.nd.array(label, dtype=self.dtype,
                                 ctx=ctx)

    def __iter__(self):
        return self

    @property
    def provide_data(self):
        return [mx.io.DataDesc('data', self.data.shape, self.dtype)]

    @property
    def provide_label(self):
        return [mx.io.DataDesc('softmax_label',
                               (self.batch_size,), self.dtype)]

    def next(self):
        self.cur_iter += 1
        if self.cur_iter <= self.max_iter:
            return DataBatch(data=(self.data,),
                             label=(self.label,),
                             pad=0,
                             index=None,
                             provide_data=self.provide_data,
                             provide_label=self.provide_label)
        else:
            raise StopIteration

    def __next__(self):
        return self.next()

    def reset(self):
        self.cur_iter = 0

context = mx.cpu() if args.no_cuda else mx.gpu()

if args.use_rec:
    # Fetch training and validation data if present
    image_shape = [int(elem) for elem in args.image_shape.split(',')]
    image_shape = tuple(image_shape)
    train_data, val_data, batch_fn = get_data_rec(args.rec_train,
                                                  args.rec_train_idx,
                                                  args.rec_val,
                                                  args.rec_val_idx,
                                                  batch_size,
                                                  args.data_nthreads,
                                                  image_shape)
else:
    # Otherwise use synthetic data
    image_shape = [int(elem) for elem in args.image_shape.split(',')]
    image_shape = tuple(image_shape)
    data_shape = (batch_size,) + image_shape
    train_data = SyntheticDataIter(num_classes, data_shape, epoch_size,
                                   np.float32, context)
    val_data = None


# Get model from GluonCV model zoo
# https://gluon-cv.mxnet.io/model_zoo/index.html
kwargs = {'ctx': context,
          'pretrained': args.use_pretrained,
          'classes': num_classes}
if args.last_gamma:
    kwargs['last_gamma'] = True
if args.model_local:
    net = import_module('symbols.'+args.model)
else:
    if "inceptionv4" in args.model:
        net = get_model2(args.model, **kwargs)
    else:
        net = get_model(args.model, **kwargs)
    net.cast(args.dtype)

# Create initializer
initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in",
                             magnitude=2)

# Create optimizer
optimizer_params = {'wd': args.wd,
                    'momentum': args.momentum,
                    'rescale_grad': 1.0 / batch_size,
                    # 'lr_scheduler': lr_sched}
                    'learning_rate': args.lr,
                    'batch_size': batch_size}
if args.dtype == 'float16':
    optimizer_params['multi_precision'] = True
opt = mx.optimizer.create('sgd', **optimizer_params)

if args.lenet:
    class LeNet(gluon.HybridBlock):
        def __init__(self, classes=1000,feature_size=512, **kwargs):
            super(LeNet,self).__init__(**kwargs)

            with self.name_scope():
                self.conv1 = nn.Conv2D(channels=20, kernel_size=5, activation='relu')
                self.maxpool1 = nn.MaxPool2D(pool_size=2, strides=2)
                self.conv2 = nn.Conv2D(channels=50, kernel_size=5, activation='relu')
                self.maxpool2 = nn.MaxPool2D(pool_size=2, strides=2)
                self.flat = nn.Flatten()
                self.dense1 = nn.Dense(feature_size)
                self.dense2 = nn.Dense(classes)

        def hybrid_forward(self, F, x, *args, **kwargs):
            x = self.maxpool1(self.conv1(x))
            x = self.maxpool2(self.conv2(x))
            x = self.flat(x)
            ft = self.dense1(x)
            output = self.dense2(ft)
            return output

    net = LeNet()
    logging.info(net)

def train_gluon():
    # Hybridize and initialize model
    net.hybridize()
    net.initialize(initializer, ctx=context)

    params = net.collect_params()

    # Create trainer, loss function and train metric
    trainer = gluon.Trainer(params, opt, kvstore=None, batch_size=batch_size)
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
    metric = mx.metric.Accuracy()

    # Train model
    for epoch in range(args.num_epochs):
        tic = time.time()
        if args.use_rec:
            train_data.reset()
        metric.reset()

        btic = time.time()
        for nbatch, batch in enumerate(train_data, start=1):
            if nbatch+1 > args.num_batches:
                break
            data, label = batch_fn(batch, context)
            with autograd.record():
                output = net(data.astype(args.dtype, copy=False))
                loss = loss_fn(output, label)
            loss.backward()
            trainer.step(batch_size)

            metric.update([label], [output])
            if args.log_interval and nbatch % args.log_interval == 0:
                name, acc = metric.get()
                # logging.info('Epoch[%d] Rank[%d] Batch[%d]\t%s=%f\tlr=%f',
                #              epoch, rank, nbatch, name, acc, trainer.learning_rate)
                if rank == 0:
                    batch_speed = num_workers * batch_size * args.log_interval / (time.time() - btic)
                    logging.info('Epoch[%d] Batch[%d]\tSpeed: %.2f samples/sec',
                                 epoch, nbatch, batch_speed)
                btic = time.time()

        # Report metrics
        elapsed = time.time() - tic
        _, acc = metric.get()
        logging.info('Epoch[%d] Rank[%d] Batch[%d]\tTime cost=%.2f\tTrain-accuracy=%f',
                     epoch, rank, nbatch, elapsed, acc)
        if rank == 0:
            epoch_speed = num_workers * batch_size * nbatch / elapsed
            logging.info('Epoch[%d]\taverage Speed: %.2f samples/sec', epoch, epoch_speed)


if __name__ == '__main__':
    train_gluon()

'''
training commands:
python3 $HOME/mxnet/horovod/examples/mxnet_imagenet_local.py --use-rec --rec-train="/home/gpu/trainData/traindata.rec" --rec-train-idx="/home/gpu/trainData/traindata.idx" --rec-val="/home/gpu/trainData/imagenet1k-val.rec" --rec-val-idx="/home/gpu/trainData/imagenet1k-val.idx"
'''
