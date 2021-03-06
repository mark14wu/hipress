// #define ZQDEBUG 1
// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2018 Uber Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include <memory>
#include <queue>
#include <thread>
#include <unordered_map>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#define EIGEN_USE_THREADS

#if HAVE_CUDA
#include "tensorflow/stream_executor/stream.h"
#endif

#define OMPI_SKIP_MPICXX
#include "../common/operations.h"

using namespace tensorflow;
using namespace horovod;

namespace horovod {
namespace tensorflow {

namespace {

Status ConvertStatus(const common::Status& status) {
  switch (status.type()) {
  case common::OK:
    return Status::OK();
  case common::UNKNOWN_ERROR:
    return errors::Unknown(status.reason());
  case common::PRECONDITION_ERROR:
    return errors::FailedPrecondition(status.reason());
  case common::ABORTED:
    return errors::Aborted(status.reason());
  case common::INVALID_ARGUMENT:
    return errors::InvalidArgument(status.reason());
  default:
    return errors::Unknown("Unknown error.");
  }
}

common::Status ConvertStatus(const Status& status) {
  switch (status.code()) {
  case error::Code::OK:
    return common::Status::OK();
  case error::Code::UNKNOWN:
    return common::Status::UnknownError(status.error_message());
  case error::Code::FAILED_PRECONDITION:
    return common::Status::PreconditionError(status.error_message());
  case error::Code::ABORTED:
    return common::Status::Aborted(status.error_message());
  case error::Code::INVALID_ARGUMENT:
    return common::Status::InvalidArgument(status.error_message());
  default:
    return common::Status::UnknownError("Unknown error.");
  }
}

#if HAVE_CUDA
class TFReadyEvent : public common::ReadyEvent {
public:
  TFReadyEvent(DeviceContext* device_context);
  bool Ready() const override;

private:
  std::shared_ptr<perftools::gputools::Event> event_;
};
#endif

class TFPersistentBuffer : public common::PersistentBuffer {
public:
  TFPersistentBuffer(OpKernelContext* context, int64_t size);
  virtual const void*
  AccessData(std::shared_ptr<common::OpContext> context) const override;

private:
  std::shared_ptr<PersistentTensor> tensor_;
};

class TFTensor : public common::Tensor {
public:
  TFTensor(::tensorflow::Tensor& tensor);
  virtual const common::DataType dtype() const override;
  virtual const common::TensorShape shape() const override;
  virtual const void* data() const override;
  virtual int64_t size() const override;

protected:
  ::tensorflow::Tensor tensor_;
};

class TFOpContext : public common::OpContext {
public:
  TFOpContext(OpKernelContext* context);
  virtual common::Status AllocatePersistent(
      int64_t size, std::shared_ptr<common::PersistentBuffer>* tensor) override;
  virtual common::Status
  AllocateOutput(common::TensorShape shape,
                 std::shared_ptr<common::Tensor>* tensor) override;
  virtual common::Framework framework() const override;
  OpKernelContext* GetKernelContext() const;

private:
  OpKernelContext* context_ = nullptr;
};

#if HAVE_CUDA
TFReadyEvent::TFReadyEvent(DeviceContext* device_context) {
  auto executor = device_context->stream()->parent();
  auto ready_event = new perftools::gputools::Event(executor);
  ready_event->Init();
  device_context->stream()->ThenRecordEvent(ready_event);
  event_ = std::shared_ptr<perftools::gputools::Event>(ready_event);
}

bool TFReadyEvent::Ready() const {
  return event_->PollForStatus() !=
         perftools::gputools::Event::Status::kPending;
}
#endif

TFPersistentBuffer::TFPersistentBuffer(OpKernelContext* context, int64_t size) {
  tensor_ = std::make_shared<PersistentTensor>();
  TensorShape buffer_shape;
  buffer_shape.AddDim(size);
  Tensor* unused;
  Status status = context->allocate_persistent(DT_INT8, buffer_shape,
                                               tensor_.get(), &unused);
  if (!status.ok()) {
    throw status;
  }
#if HAVE_CUDA
  // On GPU allocation is asynchronous, we need to wait for it to
  // complete.
  auto device_context = context->op_device_context();
  if (device_context != nullptr) {
    device_context->stream()->BlockHostUntilDone();
  }
#endif
}

const void* TFPersistentBuffer::AccessData(
    std::shared_ptr<common::OpContext> context) const {
  // It's safe to cast context to TFOpContext, since only TFOpContext creates
  // TFPersistentBuffer.
  return (const void *)tensor_
      ->AccessTensor(
          std::dynamic_pointer_cast<TFOpContext>(context)->GetKernelContext())
      ->tensor_data()
      .data();
}

TFTensor::TFTensor(::tensorflow::Tensor& tensor) : tensor_(tensor) {}

const common::DataType TFTensor::dtype() const {
  switch (tensor_.dtype()) {
  case DT_UINT8:
    return common::HOROVOD_UINT8;
  case DT_INT8:
    return common::HOROVOD_INT8;
  case DT_UINT16:
    return common::HOROVOD_UINT16;
  case DT_INT16:
    return common::HOROVOD_INT16;
  case DT_INT32:
    return common::HOROVOD_INT32;
  case DT_INT64:
    return common::HOROVOD_INT64;
  case DT_HALF:
    return common::HOROVOD_FLOAT16;
  case DT_FLOAT:
    return common::HOROVOD_FLOAT32;
  case DT_DOUBLE:
    return common::HOROVOD_FLOAT64;
  case DT_BOOL:
    return common::HOROVOD_BOOL;
  default:
    throw std::logic_error("Invalid tensor type.");
  }
}

const common::TensorShape TFTensor::shape() const {
  common::TensorShape shape;
  for (auto dim : tensor_.shape()) {
    shape.AddDim(dim.size);
  }
  return shape;
}

const void* TFTensor::data() const { return (const void*)tensor_.tensor_data().data(); }

int64_t TFTensor::size() const { return (int64_t)tensor_.tensor_data().size(); }

TFOpContext::TFOpContext(OpKernelContext* context) : context_(context) {}

common::Status TFOpContext::AllocatePersistent(
    int64_t size, std::shared_ptr<common::PersistentBuffer>* tensor) {
  try {
    *tensor = std::make_shared<TFPersistentBuffer>(context_, size);
    return common::Status::OK();
  } catch (Status& status) {
    return ConvertStatus(status);
  }
}

common::Status
TFOpContext::AllocateOutput(common::TensorShape shape,
                            std::shared_ptr<common::Tensor>* tensor) {
  TensorShape tf_shape;
  for (int idx = 0; idx < shape.dims(); ++idx) {
    tf_shape.AddDim(shape.dim_size(idx));
  }
  Tensor* tf_tensor;
  Status status = context_->allocate_output(0, tf_shape, &tf_tensor);
  if (status.ok()) {
    *tensor = std::make_shared<TFTensor>(*tf_tensor);
  }
#if HAVE_CUDA
  // On GPU allocation is asynchronous, we need to wait for it to
  // complete.
  auto device_context = context_->op_device_context();
  if (device_context != nullptr) {
    device_context->stream()->BlockHostUntilDone();
  }
#endif
  return ConvertStatus(status);
}

common::Framework TFOpContext::framework() const {
  return common::Framework::TENSORFLOW;
}

OpKernelContext* TFOpContext::GetKernelContext() const { return context_; }

int GetDeviceID(OpKernelContext* context) {
  int device = CPU_DEVICE_ID;
  if (context->device() != nullptr &&
      context->device()->tensorflow_gpu_device_info() != nullptr) {
    device = context->device()->tensorflow_gpu_device_info()->gpu_id;
  }
  return device;
}

// On GPU this event will signal that data is ready, and tensors are
// allocated.
common::ReadyEvent* RecordReadyEvent(OpKernelContext* context) {
#if HAVE_CUDA
  auto device_context = context->op_device_context();
  if (device_context != nullptr) {
    return new TFReadyEvent(device_context);
  }
#endif
  return nullptr;
}

} // namespace

class HorovodAllreduceOp : public AsyncOpKernel {
public:
  explicit HorovodAllreduceOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {}

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(common::CheckInitialized()),
                         done);
    auto node_name = name();
    auto device = GetDeviceID(context);
    auto tensor = context->input(0);
    Tensor* output;
    OP_REQUIRES_OK_ASYNC(
        context, context->allocate_output(0, tensor.shape(), &output), done);
    // ReadyEvent makes sure input tensor is ready, and output is allocated.
    auto ready_event = std::shared_ptr<common::ReadyEvent>(RecordReadyEvent(context));
    auto hvd_context = std::make_shared<TFOpContext>(context);
    auto hvd_tensor = std::make_shared<TFTensor>(tensor);
    auto hvd_output = std::make_shared<TFTensor>(*output);
    auto enqueue_result = EnqueueTensorAllreduce(
        hvd_context, hvd_tensor, hvd_output, ready_event, node_name, device,
        [context, done](const common::Status& status) {
          context->SetStatus(ConvertStatus(status));
          done();
        });
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(enqueue_result), done);
  }
};

REGISTER_KERNEL_BUILDER(Name("HorovodAllreduce").Device(DEVICE_CPU),
                        HorovodAllreduceOp);
#if HOROVOD_GPU_ALLREDUCE
REGISTER_KERNEL_BUILDER(Name("HorovodAllreduce").Device(DEVICE_GPU),
                        HorovodAllreduceOp);
#endif

REGISTER_OP("HorovodAllreduce")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Input("tensor: T")
    .Output("sum: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(
Perform an MPI Allreduce on a tensor. All other processes that do a reduction
on a tensor with the same name must have the same dimension for that tensor.
Tensors are reduced with other tensors that have the same node name for the
allreduce.

Arguments
    tensor:     A tensor to reduce.

Output
    sum:    A tensor with the same shape as `tensor`, summed across all MPI processes.
)doc");

class HorovodAllgatherOp : public AsyncOpKernel {
public:
  explicit HorovodAllgatherOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {}

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(common::CheckInitialized()),
                         done);

    auto node_name = name();
    auto device = GetDeviceID(context);
    auto tensor = context->input(0);
    // ReadyEvent makes sure input tensor is ready.  We cannot pre-allocate
    // output for allgather, since shape of result is only known after all
    // ranks make a request.
    auto ready_event = std::shared_ptr<common::ReadyEvent>(RecordReadyEvent(context));
    auto hvd_context = std::make_shared<TFOpContext>(context);
    auto hvd_tensor = std::make_shared<TFTensor>(tensor);
    auto enqueue_result = EnqueueTensorAllgather(
        hvd_context, hvd_tensor, ready_event, node_name, device,
        [context, done](const common::Status& status) {
          context->SetStatus(ConvertStatus(status));
          done();
        });
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(enqueue_result), done);
  }
}; // namespace tensorflow

REGISTER_KERNEL_BUILDER(Name("HorovodAllgather").Device(DEVICE_CPU),
                        HorovodAllgatherOp);
#if HOROVOD_GPU_ALLGATHER
REGISTER_KERNEL_BUILDER(Name("HorovodAllgather").Device(DEVICE_GPU),
                        HorovodAllgatherOp);
#endif

REGISTER_OP("HorovodAllgather")
    .Attr(
        "T: {uint8, int8, uint16, int16, int32, int64, float16, float32, float64, bool}")
    .Input("tensor: T")
    .Output("output: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle output;
      TF_RETURN_IF_ERROR(
          c->ReplaceDim(c->input(0), 0, c->UnknownDim(), &output));
      c->set_output(0, output);
      return Status::OK();
    })
    .Doc(R"doc(
Perform an MPI Allgather on a tensor. All other processes that do a gather on a
tensor with the same name must have the same rank for that tensor, and have the
same dimension on all but the first dimension.

Arguments
    tensor:     A tensor to gather.

Output
    gathered:    A tensor with the same shape as `tensor` except for the first dimension.
)doc");

class HorovodBroadcastOp : public AsyncOpKernel {
public:
  explicit HorovodBroadcastOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("root_rank", &root_rank_));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(common::CheckInitialized()),
                         done);

    auto node_name = name();
    auto device = GetDeviceID(context);
    auto tensor = context->input(0);
    Tensor* output = nullptr;
    if (common::horovod_rank() == root_rank_) {
      context->set_output(0, tensor);
    } else {
      OP_REQUIRES_OK_ASYNC(
          context, context->allocate_output(0, tensor.shape(), &output), done);
    }
    // ReadyEvent makes sure input tensor is ready, and output is allocated.
    auto ready_event = std::shared_ptr<common::ReadyEvent>(RecordReadyEvent(context));
    auto hvd_context = std::make_shared<TFOpContext>(context);
    auto hvd_tensor = std::make_shared<TFTensor>(tensor);
    std::shared_ptr<TFTensor> hvd_output = nullptr;
    if (output != nullptr) {
      hvd_output = std::make_shared<TFTensor>(*output);
    }
    auto enqueue_result = EnqueueTensorBroadcast(
        hvd_context, hvd_tensor, hvd_output, root_rank_, ready_event, node_name,
        device, [context, done](const common::Status& status) {
          context->SetStatus(ConvertStatus(status));
          done();
        });
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(enqueue_result), done);
  }

private:
  int root_rank_;
};

REGISTER_KERNEL_BUILDER(Name("HorovodBroadcast").Device(DEVICE_CPU),
                        HorovodBroadcastOp);
#if HOROVOD_GPU_BROADCAST
REGISTER_KERNEL_BUILDER(Name("HorovodBroadcast").Device(DEVICE_GPU),
                        HorovodBroadcastOp);
#endif

REGISTER_OP("HorovodBroadcast")
    .Attr(
        "T: {uint8, int8, uint16, int16, int32, int64, float16, float32, float64, bool}")
    .Attr("root_rank: int")
    .Input("tensor: T")
    .Output("output: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(
Perform an MPI Broadcast on a tensor. All other processes that do a broadcast
on a tensor with the same name must have the same dimension for that tensor.

Arguments
    tensor:     A tensor to broadcast.
    root_rank:  Rank that will send data, other ranks will receive data.

Output
    output:    A tensor with the same shape as `tensor` and same value as
               `tensor` on root rank.
)doc");


#include <unistd.h>
class HorovodGatherOp : public AsyncOpKernel{
 public:
  explicit HorovodGatherOp(OpKernelConstruction* context):AsyncOpKernel(context){
    OP_REQUIRES_OK(context, context->GetAttr("root_rank", &root_rank));
    // OP_REQUIRES_OK(context, context->GetAttr("num_elem", &num_elem));
    OP_REQUIRES_OK(context, context->GetAttr("batch_id", &batch_id));
  }
  void ComputeAsync(OpKernelContext* context, DoneCallback done) override{
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(common::CheckInitialized()),
                         done);
    
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(common::CheckInitialized()), done);
    auto node_name = name();
    // #ifdef ZQDEBUG
    // printf("%s[LINE:%d, Time:%s]: GatherOp Start!: ", __FILE__, __LINE__, __TIME__);
    // std::cout<<node_name<<std::endl;
    // #endif
    // std::cout<<"GatherOp:name="<<node_name<<std::endl;
    auto device = GetDeviceID(context);
    // Tensor* output;
    bool mutable_flag = true;
    auto tensor = context->mutable_input(0,mutable_flag);

    auto input = tensor.flat<uint8_t>();
    auto input_size = input.size();
    #ifdef ZQDEBUG
    printf("%s[LINE:%d, Time:%s]: GatherOp: input_size=%d\t", __FILE__, __LINE__, __TIME__, input_size);
    #endif
    auto output_size = input.size()*16;
    TensorShape output_shape;
    output_shape.AddDim(output_size);
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK_ASYNC(
      context, context->allocate_output(0, output_shape, &output_tensor),
      done
    );
    auto output = output_tensor->flat<uint8_t>();
    // context->allocate_output(0,output_shape, &output_tensor);
    int num_elem = input.size();
    // auto output_tensor = tensor;
    // context->set_output(0, output_tensor);

    auto ready_event = std::shared_ptr<common::ReadyEvent>(RecordReadyEvent(context));
    auto hvd_context = std::make_shared<TFOpContext>(context);
    auto hvd_tensor = std::make_shared<TFTensor>(tensor);
    auto hvd_output = std::make_shared<TFTensor>(*output_tensor);
    // printf("GatherOp:line:%d\tinput_size=%d\toutput_size=%d\n", __LINE__, input_size, output_size);
    #ifdef ZQDEBUG
    printf("%s[LINE:%d, Time:%s]: GatherOp Enqueue!: ", __FILE__, __LINE__, __TIME__);
    std::cout<<node_name<<std::endl;
    #endif
    auto enqueue_result = EnqueueTensorGather(
      hvd_context, hvd_tensor, hvd_output, root_rank,
      ready_event, node_name, device,
      [context, done, node_name](const common::Status& status){
        context->SetStatus(ConvertStatus(status));
        #ifdef ZQDEBUG
        printf("%s[LINE:%d, Time:%s]: GatherOp Over!: ", __FILE__, __LINE__, __TIME__);
        std::cout<<node_name<<std::endl;
        #endif
        done();
      },
      num_elem, batch_id
    );
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(enqueue_result), done);
  }
 private:
  int root_rank;
  // int num_elem;
  int batch_id;
};
REGISTER_KERNEL_BUILDER(Name("HorovodGather").Device(DEVICE_CPU),HorovodGatherOp);
REGISTER_OP("HorovodGather")
  // .Attr("T: {float32, uint8}")
  .Input("input1: uint8")
  .Output("output1: uint8")
  .Attr("root_rank: int")
  // .Attr("num_elem: int")
  .Attr("batch_id: int")
  .Doc(R"doc(
// Perform an MPI Gather on a tensor.
// Input: 
//   1. tensor to Gather
//   2. output space for root node(must provided in advance)
// Arguments:
//   1. root_rank: indicate which node is root
//   2. num_elem: len of tensor to gather
//   3. bathc_id: 
// Output:
//   0. use input[1](second) as output
  )doc")
  ;

// class HorovodCBroadcastOp: public AsyncOpKernel{
//  public:
//   explicit HorovodCBroadcastOp(OpKernelConstruction* context):AsyncOpKernel(context){
//     OP_REQUIRES_OK(context, context->GetAttr("root_rank", &root_rank));
//     // OP_REQUIRES_OK(context, context->GetAttr("num_elem", &num_elem));
//     OP_REQUIRES_OK(context, context->GetAttr("batch_id", &batch_id));
//   }
//   void ComputeAsync(OpKernelContext* context, DoneCallback done) override{
//     // printf("CBroadcastOp:%d\n", __LINE__);
//     OP_REQUIRES_OK_ASYNC(context, ConvertStatus(common::CheckInitialized()), done);
//     // std::cout<<"CBroadcastOp:name="<<node_name<<std::endl;
//     auto device = GetDeviceID(context);
//     bool mutable_flag = true;
//     auto tensor = context->mutable_input(0,mutable_flag);
//     auto input = tensor.flat<uint8_t>();
//     auto input_size = input.size();
//     auto node_name = name();
//     #ifdef ZQDEBUG
//     printf("%s[LINE:%d, Time:%s]: CBroadcastOp: input_size=%d\tnode_name=%s\n", __FILE__, __LINE__, __TIME__, input_size, node_name.c_str());
//     std::cout<<node_name<<std::endl;
//     #endif
//     // auto output_size = input_size;
//     // TensorShape output_shape;
//     // output_shape.AddDim(output_size);
//     // Tensor* output_tensor = NULL;
//     // OP_REQUIRES_OK_ASYNC(
//     //   context, context->allocate_output(0, output_shape, &output_tensor),
//     //   done
//     // );
//     auto output_tensor = tensor;
//     context->set_output(0, output_tensor);
//     auto output = output_tensor.flat<uint8_t>();
//     // auto output = output_tensor->flat<uint8_t>();
//     int num_elem = input.size();
//     auto ready_event = std::shared_ptr<common::ReadyEvent>(RecordReadyEvent(context));
//     auto hvd_context = std::make_shared<TFOpContext>(context);
//     auto hvd_tensor = std::make_shared<TFTensor>(tensor);
//     std::shared_ptr<TFTensor> hvd_output = nullptr;
//     // hvd_output = std::make_shared<TFTensor>(*output_tensor);
//     hvd_output = std::make_shared<TFTensor>(output_tensor);
//     if (common::horovod_rank() == root_rank){
//       // do nothing
//       // but for robustness, we should copy data from input to output.
//     }
//     else{
//       hvd_output = std::make_shared<TFTensor>(output_tensor);
//     }
//     auto enqueue_result = EnqueueTensorCBroadcast(
//       hvd_context, hvd_tensor, hvd_output, root_rank,
//       ready_event, node_name, device,
//       [context, done, node_name](const common::Status& status){
//         context->SetStatus(ConvertStatus(status));
//         #ifdef ZQDEBUG
//         printf("%s[LINE:%d, Time:%s]: CBroadcastOp Over!: ", __FILE__, __LINE__, __TIME__);
//         std::cout<<node_name<<std::endl;
//         #endif
//         done();
//       },
//       num_elem, batch_id
//     );
//     OP_REQUIRES_OK_ASYNC(context, ConvertStatus(enqueue_result), done);
//     // done();
//     // printf("CBroadcastOp:%d\n", __LINE__);
//     // sleep(10);
//   }
//  private:
//   int root_rank;
//   // int num_elem;
//   int batch_id;
// };
class HorovodCBroadcastOp: public AsyncOpKernel{
 public:
  explicit HorovodCBroadcastOp(OpKernelConstruction* context):AsyncOpKernel(context){
    OP_REQUIRES_OK(context, context->GetAttr("root_rank", &root_rank));
    OP_REQUIRES_OK(context, context->GetAttr("batch_id", &batch_id));
  }
  void ComputeAsync(OpKernelContext* context, DoneCallback done) override{
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(common::CheckInitialized()), done);
    auto device = GetDeviceID(context);
    bool mutable_flag = true;
    auto tensor = context->mutable_input(0,mutable_flag);
    auto input = tensor.flat<uint8_t>();
    auto input_size = input.size();
    auto node_name = name();
    #ifdef ZQDEBUG
    printf("%s[LINE:%d, Time:%s]: New CBroadcastOp: input_size=%d\tnode_name=%s\n", __FILE__, __LINE__, __TIME__, input_size, node_name.c_str());
    std::cout<<node_name<<std::endl;
    #endif
    Tensor* output = nullptr;
    if (common::horovod_rank() == root_rank){
      context->set_output(0, tensor);
    }
    else{
      OP_REQUIRES_OK_ASYNC(
        context, context->allocate_output(0,tensor.shape(), &output), done);
    }
    std::shared_ptr<TFTensor> hvd_output = nullptr;
    if (output!=nullptr){
      hvd_output = std::make_shared<TFTensor>(*output);
    }
    int num_elem = input.size();
    auto ready_event = std::shared_ptr<common::ReadyEvent>(RecordReadyEvent(context));
    auto hvd_context = std::make_shared<TFOpContext>(context);
    auto hvd_tensor = std::make_shared<TFTensor>(tensor);
    auto enqueue_result = EnqueueTensorCBroadcast(
      hvd_context, hvd_tensor, hvd_output, root_rank,
      ready_event, node_name, device,
      [context, done, node_name](const common::Status& status){
        context->SetStatus(ConvertStatus(status));
        #ifdef ZQDEBUG
        printf("%s[LINE:%d, Time:%s]: CBroadcastOp Over!: ", __FILE__, __LINE__, __TIME__);
        std::cout<<node_name<<std::endl;
        #endif
        done();
      },
      num_elem, batch_id
    );
    OP_REQUIRES_OK_ASYNC(context, ConvertStatus(enqueue_result), done);
    // done();
    // printf("CBroadcastOp:%d\n", __LINE__);
    // sleep(10);
  }
 private:
  int root_rank;
  // int num_elem;
  int batch_id;
};

REGISTER_KERNEL_BUILDER(Name("HorovodCBroadcast").Device(DEVICE_CPU), HorovodCBroadcastOp);

REGISTER_OP("HorovodCBroadcast")
  // .Attr("T: {float32, uint8}")
  .Input("input1: uint8")
  .Output("output1: uint8")
  .Attr("root_rank: int")
  // .Attr("num_elem: int")
  .Attr("batch_id: int")
  .Doc(R"doc(
// Perform an MPI CompressedBroadcast on a tensor.
// Inputs:
//   1. tensor to be CBroadcasted
//   2. output space provided
// Argument:
//   1. root_rank
//   2. num_elem
//   3. batch_id
// Outputs:
//   0. use input[1] as output
  )doc")
  ;

} // namespace tensorflow
} // namespace horovod
