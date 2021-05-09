// Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
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

#include <chrono>
#include <memory>
#include <thread>
#include <torch/extension.h>
#include <torch/torch.h>

#include "../common/operations.h"
#include "adapter_v2.h"
#include "cuda_util.h"
#include "handle_manager.h"
#include "ready_event.h"

namespace horovod {
namespace torch {

static HandleManager handle_manager;
static FinishedQueueForGB finished_queue_for_broadcast_;
static FinishedQueueForGB finished_queue_for_gather_;
static FinishedQueueForGB finished_queue_for_allreduce_;

namespace {

std::string GetOpName(const std::string& prefix, const std::string& name,
                      int handle) {
  if (!name.empty()) {
    return prefix + "." + std::string(name);
  }
  return prefix + ".noname." + std::to_string(handle);
}

int GetDeviceID(const ::torch::Tensor& tensor) {
  if (tensor.device().is_cuda()) {
    return tensor.device().index();
  }
  return CPU_DEVICE_ID;
}

} // namespace

//hovorod run over it.
int DoAllreduce(::torch::Tensor tensor, ::torch::Tensor output, int average,
                const std::string& name) {
  ThrowIfError(common::CheckInitialized());
  // printf("DoAllreduce...\n");

  auto handle = handle_manager.AllocateHandle();
  auto device = GetDeviceID(tensor);
  auto ready_event = RecordReadyEvent(device);
  auto hvd_tensor = std::make_shared<TorchTensor>(tensor);
  auto hvd_context = std::make_shared<TorchOpContext>(device, output);
  auto hvd_output = std::make_shared<TorchTensor>(output);

  auto enqueue_result = EnqueueTensorAllreduce(
      hvd_context, hvd_tensor, hvd_output, ready_event,
      GetOpName("allreduce", name, handle), device,
      [handle, average, output, name](const Status& status) mutable {
        // Will execute in the `device` context.
        if (average) {
          output.div_(horovod_size());
        }
        finished_queue_for_allreduce_.PushFTKsToQueue(name);
        handle_manager.MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);

  return handle;
}

int DoGather(::torch::Tensor tensor, ::torch::Tensor output, int root_rank, int num_elem, int batch_id,
              std::string& name)
{
  ThrowIfError(common::CheckInitialized());
  auto handle = handle_manager.AllocateHandle();
  auto device = GetDeviceID(tensor);
  auto ready_event = RecordReadyEvent(device);
  auto hvd_tensor = std::make_shared<TorchTensor>(tensor);
  auto hvd_context = std::make_shared<TorchOpContext>(device, output);
  auto hvd_output = std::make_shared<TorchTensor>(output);

  auto enqueue_result = EnqueueTensorGather(
    hvd_context, hvd_tensor, hvd_output, root_rank,
    nullptr, name, device,
    [handle, name](const Status& status) mutable{
      finished_queue_for_gather_.PushFTKsToQueue(name);
      handle_manager.MarkDone(handle, status);
    },
    num_elem, batch_id
  );
  ThrowIfError(enqueue_result);
  return handle;
}


int DoAllreduceCudaOnCPU(::torch::Tensor tensor, ::torch::Tensor output, int average,
                         const std::string& name) {
  ThrowIfError(common::CheckInitialized());

  // Make async copy of input tensor to CPU tensor and record completion event.
  auto device = GetDeviceID(tensor);
  auto cpu_buffer =
      tensor.to(::torch::Device(::torch::kCPU), /*non_blocking=*/true);
  auto hvd_cpu_buffer = std::make_shared<TorchTensor>(cpu_buffer);
  auto ready_event = RecordReadyEvent(device);

  auto hvd_context =
      std::make_shared<TorchOpContext>(CPU_DEVICE_ID, cpu_buffer);

  auto handle = handle_manager.AllocateHandle();
  auto enqueue_result = EnqueueTensorAllreduce(
      hvd_context, hvd_cpu_buffer, hvd_cpu_buffer, ready_event,
      GetOpName("allreduce", name, handle), CPU_DEVICE_ID,
      [handle, average, cpu_buffer, output,
       device](const Status& status) mutable {
        // Since the operation was on CPU, need to perform copy with the GPU
        // device guard.
        with_device device_guard(device);
        output.copy_(cpu_buffer);
        if (average) {
          output.div_(horovod_size());
        }
        handle_manager.MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);

  return handle;
}

int DoAllgather(::torch::Tensor tensor, ::torch::Tensor output, const std::string& name) {
  ThrowIfError(common::CheckInitialized());

  auto device = GetDeviceID(tensor);
  auto ready_event = RecordReadyEvent(device);
  auto hvd_tensor = std::make_shared<TorchTensor>(tensor);
  auto hvd_context = std::make_shared<TorchOpContext>(device, output);

  auto handle = handle_manager.AllocateHandle();
  auto enqueue_result =
      EnqueueTensorAllgather(hvd_context, hvd_tensor, ready_event,
                             GetOpName("allgather", name, handle), device,
                             [handle](const Status& status) {
                               handle_manager.MarkDone(handle, status);
                             });
  ThrowIfError(enqueue_result);

  return handle;
}

int DoAllgatherCudaOnCPU(::torch::Tensor tensor, ::torch::Tensor output,
                         const std::string& name) {
  ThrowIfError(common::CheckInitialized());

  // Make async copy of input tensor to CPU tensor and record completion event.
  auto device = GetDeviceID(tensor);
  auto cpu_tensor =
      tensor.to(::torch::Device(::torch::kCPU), /*non_blocking=*/true);
  auto hvd_cpu_tensor = std::make_shared<TorchTensor>(cpu_tensor);
  auto ready_event = RecordReadyEvent(device);

  auto cpu_output = ::torch::empty_like(cpu_tensor);
  auto hvd_cpu_output = std::make_shared<TorchTensor>(cpu_output);
  auto hvd_context =
      std::make_shared<TorchOpContext>(CPU_DEVICE_ID, cpu_output);

  auto handle = handle_manager.AllocateHandle();
  auto enqueue_result = EnqueueTensorAllgather(
      hvd_context, hvd_cpu_tensor, ready_event,
      GetOpName("allgather", name, handle), CPU_DEVICE_ID,
      [handle, cpu_output, output, device](const Status& status) mutable {
        // Since the operation was on CPU, need to perform copy with the GPU
        // device guard.
        with_device device_guard(device);
        // output needs to be resized before copying in the CPU tensor.
        output.resize_(cpu_output.sizes());
        output.copy_(cpu_output);
        handle_manager.MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);

  return handle;
}

int DoBroadcast(::torch::Tensor tensor, ::torch::Tensor output, int root_rank,
                const std::string& name) {
  ThrowIfError(common::CheckInitialized());

  auto device = GetDeviceID(tensor);
  auto ready_event = RecordReadyEvent(device);
  auto hvd_tensor = std::make_shared<TorchTensor>(tensor);
  auto hvd_context = std::make_shared<TorchOpContext>(device, output);
  std::shared_ptr<Tensor> hvd_output = nullptr;
  if (horovod_rank() == root_rank) {
    if (tensor.data_ptr() != output.data_ptr()) {
      with_device device_guard(device);
      output.copy_(tensor);
    }
  } else {
    hvd_output = std::make_shared<TorchTensor>(output);
  }

  auto handle = handle_manager.AllocateHandle();
  auto enqueue_result =
      EnqueueTensorBroadcast(hvd_context, hvd_tensor, hvd_output, root_rank,
                             ready_event, GetOpName("broadcast", name, handle),
                             device, [handle](const Status& status) {
                               handle_manager.MarkDone(handle, status);
                             });
  ThrowIfError(enqueue_result);

  return handle;
}

int DoCBroadcast(::torch::Tensor tensor, ::torch::Tensor output, int root_rank, int num_elem, int batch_id,
  std::string& name)
{
  ThrowIfError(common::CheckInitialized());
  auto handle = handle_manager.AllocateHandle();
  auto device = GetDeviceID(tensor);
  auto ready_event = RecordReadyEvent(device);
  auto hvd_tensor = std::make_shared<TorchTensor>(tensor);
  auto hvd_context = std::make_shared<TorchOpContext>(device, output);
  // auto hvd_output = std::make_shared<TorchTensor>(output);
  std::shared_ptr<Tensor> hvd_output = nullptr;
  if (horovod_rank() == root_rank){
    if (tensor.data_ptr() != output.data_ptr()){
      with_device device_guard(device);
      output.copy_(tensor);
    }
  } else{
    hvd_output = std::make_shared<TorchTensor>(output);
  }

  auto enqueue_result = EnqueueTensorCBroadcast(
    hvd_context, hvd_tensor, hvd_output, root_rank,
    nullptr, name, device,
    [handle, name](const Status& status) mutable{
      finished_queue_for_broadcast_.PushFTKsToQueue(name);
      handle_manager.MarkDone(handle, status);
    },
    num_elem, batch_id
  );
  ThrowIfError(enqueue_result);
  return handle;
}

int DoBroadcastCudaOnCPU(::torch::Tensor tensor, ::torch::Tensor output, int root_rank,
                         const std::string& name) {
  ThrowIfError(common::CheckInitialized());

  // Make async copy of input tensor to CPU tensor and record completion event.
  auto device = GetDeviceID(tensor);
  auto cpu_buffer =
      tensor.to(::torch::Device(::torch::kCPU), /*non_blocking=*/true);
  auto hvd_cpu_buffer = std::make_shared<TorchTensor>(cpu_buffer);
  auto ready_event = RecordReadyEvent(device);

  auto hvd_context =
      std::make_shared<TorchOpContext>(CPU_DEVICE_ID, cpu_buffer);

  auto handle = handle_manager.AllocateHandle();
  auto enqueue_result = EnqueueTensorBroadcast(
      hvd_context, hvd_cpu_buffer, hvd_cpu_buffer, root_rank, ready_event,
      GetOpName("broadcast", name, handle), CPU_DEVICE_ID,
      [handle, cpu_buffer, output, device](const Status& status) mutable {
        // Since the operation was on CPU, need to perform copy with the GPU
        // device guard.
        with_device device_guard(device);
        output.copy_(cpu_buffer);
        handle_manager.MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);

  return handle;
}

int PollHandle(int handle) { return handle_manager.PollHandle(handle) ? 1 : 0; }

void WaitAndClear(int handle) {
  while (!handle_manager.PollHandle(handle)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  auto status = handle_manager.ReleaseHandle(handle);
  ThrowIfError(*status);
}


std::vector<std::string> GetFTKsFromQueue(int32_t task_id){
  std::vector<std::string> FTKs;
  if (task_id == GATHER_TASK_ID_){
    FTKs = finished_queue_for_gather_.GetFTKsFromQueue();
  } else if (task_id == BROADCAT_TASK_ID_) {
    FTKs = finished_queue_for_broadcast_.GetFTKsFromQueue();
  } else {
    FTKs = finished_queue_for_allreduce_.GetFTKsFromQueue();
  }
  return FTKs;
}




PYBIND11_MODULE(mpi_lib_v2, m) {
  // allreduce
  m.def("horovod_torch_allreduce_async_torch_IntTensor", &DoAllreduce);
  m.def("horovod_torch_allreduce_async_torch_LongTensor", &DoAllreduce);
  m.def("horovod_torch_allreduce_async_torch_HalfTensor", &DoAllreduce);
  m.def("horovod_torch_allreduce_async_torch_FloatTensor", &DoAllreduce);
  m.def("horovod_torch_allreduce_async_torch_DoubleTensor", &DoAllreduce);
#if HOROVOD_GPU_ALLREDUCE
  m.def("horovod_torch_allreduce_async_torch_cuda_IntTensor", &DoAllreduce);
  m.def("horovod_torch_allreduce_async_torch_cuda_LongTensor", &DoAllreduce);
  m.def("horovod_torch_allreduce_async_torch_cuda_HalfTensor", &DoAllreduce);
  m.def("horovod_torch_allreduce_async_torch_cuda_FloatTensor", &DoAllreduce);
  m.def("horovod_torch_allreduce_async_torch_cuda_DoubleTensor", &DoAllreduce);
#else
  m.def("horovod_torch_allreduce_async_torch_cuda_IntTensor",
        &DoAllreduceCudaOnCPU);
  m.def("horovod_torch_allreduce_async_torch_cuda_LongTensor",
        &DoAllreduceCudaOnCPU);
  m.def("horovod_torch_allreduce_async_torch_cuda_HalfTensor",
        &DoAllreduceCudaOnCPU);
  m.def("horovod_torch_allreduce_async_torch_cuda_FloatTensor",
        &DoAllreduceCudaOnCPU);
  m.def("horovod_torch_allreduce_async_torch_cuda_DoubleTensor",
        &DoAllreduceCudaOnCPU);
#endif

  //cbroadcast
  m.def("horovod_torch_cbroadcast_async_torch_IntTensor", &DoCBroadcast);
  m.def("horovod_torch_cbroadcast_async_torch_LongTensor", &DoCBroadcast);
  m.def("horovod_torch_cbroadcast_async_torch_HalfTensor", &DoCBroadcast);
  m.def("horovod_torch_cbroadcast_async_torch_FloatTensor", &DoCBroadcast);
  m.def("horovod_torch_cbroadcast_async_torch_DoubleTensor", &DoCBroadcast);
  m.def("horovod_torch_cbroadcast_async_torch_ByteTensor", &DoCBroadcast);
// #if HOROVOD_GPU_GATHER
#if 1
  m.def("horovod_torch_cbroadcast_async_torch_cuda_IntTensor", &DoCBroadcast);
  m.def("horovod_torch_cbroadcast_async_torch_cuda_LongTensor", &DoCBroadcast);
  m.def("horovod_torch_cbroadcast_async_torch_cuda_HalfTensor", &DoCBroadcast);
  m.def("horovod_torch_cbroadcast_async_torch_cuda_FloatTensor", &DoCBroadcast);
  m.def("horovod_torch_cbroadcast_async_torch_cuda_DoubleTensor", &DoCBroadcast);
  m.def("horovod_torch_cbroadcast_async_torch_cuda_ByteTensor", &DoCBroadcast);
#else
  // m.def("horovod_torch_cbroadcast_async_torch_cuda_IntTensor",
  //       &DoCBroadcastCudaOnCPU);
  // m.def("horovod_torch_cbroadcast_async_torch_cuda_LongTensor",
  //       &DoCBroadcastCudaOnCPU);
  // m.def("horovod_torch_cbroadcast_async_torch_cuda_HalfTensor",
  //       &DoCBroadcastCudaOnCPU);
  // m.def("horovod_torch_cbroadcast_async_torch_cuda_FloatTensor",
  //       &DoCBroadcastCudaOnCPU);
  // m.def("horovod_torch_cbroadcast_async_torch_cuda_DoubleTensor",
  //       &DoCBroadcastCudaOnCPU);
#endif


  //gather
  m.def("horovod_torch_gather_async_torch_IntTensor", &DoGather);
  m.def("horovod_torch_gather_async_torch_LongTensor", &DoGather);
  m.def("horovod_torch_gather_async_torch_HalfTensor", &DoGather);
  m.def("horovod_torch_gather_async_torch_FloatTensor", &DoGather);
  m.def("horovod_torch_gather_async_torch_DoubleTensor", &DoGather);
  m.def("horovod_torch_gather_async_torch_ByteTensor", &DoGather);
// #if HOROVOD_GPU_GATHER
#if 1
  m.def("horovod_torch_gather_async_torch_cuda_IntTensor", &DoGather);
  m.def("horovod_torch_gather_async_torch_cuda_LongTensor", &DoGather);
  m.def("horovod_torch_gather_async_torch_cuda_HalfTensor", &DoGather);
  m.def("horovod_torch_gather_async_torch_cuda_FloatTensor", &DoGather);
  m.def("horovod_torch_gather_async_torch_cuda_DoubleTensor", &DoGather);
  m.def("horovod_torch_gather_async_torch_cuda_ByteTensor", &DoGather);
#else
  // m.def("horovod_torch_gather_async_torch_cuda_IntTensor",
  //       &DoGatherCudaOnCPU);
  // m.def("horovod_torch_gather_async_torch_cuda_LongTensor",
  //       &DoGatherCudaOnCPU);
  // m.def("horovod_torch_gather_async_torch_cuda_HalfTensor",
  //       &DoGatherCudaOnCPU);
  // m.def("horovod_torch_gather_async_torch_cuda_FloatTensor",
  //       &DoGatherCudaOnCPU);
  // m.def("horovod_torch_gather_async_torch_cuda_DoubleTensor",
  //       &DoGatherCudaOnCPU);
#endif

  // allgather
  m.def("horovod_torch_allgather_async_torch_ByteTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_CharTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_ShortTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_IntTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_LongTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_HalfTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_FloatTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_DoubleTensor", &DoAllgather);
#if HOROVOD_GPU_ALLGATHER
  m.def("horovod_torch_allgather_async_torch_cuda_ByteTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_cuda_CharTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_cuda_ShortTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_cuda_IntTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_cuda_LongTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_cuda_HalfTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_cuda_FloatTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_cuda_DoubleTensor", &DoAllgather);
#else
  m.def("horovod_torch_allgather_async_torch_cuda_ByteTensor",
        &DoAllgatherCudaOnCPU);
  m.def("horovod_torch_allgather_async_torch_cuda_CharTensor",
        &DoAllgatherCudaOnCPU);
  m.def("horovod_torch_allgather_async_torch_cuda_ShortTensor",
        &DoAllgatherCudaOnCPU);
  m.def("horovod_torch_allgather_async_torch_cuda_IntTensor",
        &DoAllgatherCudaOnCPU);
  m.def("horovod_torch_allgather_async_torch_cuda_LongTensor",
        &DoAllgatherCudaOnCPU);
  m.def("horovod_torch_allgather_async_torch_cuda_HalfTensor",
        &DoAllgatherCudaOnCPU);
  m.def("horovod_torch_allgather_async_torch_cuda_FloatTensor",
        &DoAllgatherCudaOnCPU);
  m.def("horovod_torch_allgather_async_torch_cuda_DoubleTensor",
        &DoAllgatherCudaOnCPU);
#endif

  // broadcast
  m.def("horovod_torch_broadcast_async_torch_ByteTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_CharTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_ShortTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_IntTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_LongTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_HalfTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_FloatTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_DoubleTensor", &DoBroadcast);
#if HOROVOD_GPU_BROADCAST
  m.def("horovod_torch_broadcast_async_torch_cuda_ByteTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_cuda_CharTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_cuda_ShortTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_cuda_IntTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_cuda_LongTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_cuda_HalfTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_cuda_FloatTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_cuda_DoubleTensor", &DoBroadcast);
#else
  m.def("horovod_torch_broadcast_async_torch_cuda_ByteTensor",
        &DoBroadcastCudaOnCPU);
  m.def("horovod_torch_broadcast_async_torch_cuda_CharTensor",
        &DoBroadcastCudaOnCPU);
  m.def("horovod_torch_broadcast_async_torch_cuda_ShortTensor",
        &DoBroadcastCudaOnCPU);
  m.def("horovod_torch_broadcast_async_torch_cuda_IntTensor",
        &DoBroadcastCudaOnCPU);
  m.def("horovod_torch_broadcast_async_torch_cuda_LongTensor",
        &DoBroadcastCudaOnCPU);
  m.def("horovod_torch_broadcast_async_torch_cuda_HalfTensor",
        &DoBroadcastCudaOnCPU);
  m.def("horovod_torch_broadcast_async_torch_cuda_FloatTensor",
        &DoBroadcastCudaOnCPU);
  m.def("horovod_torch_broadcast_async_torch_cuda_DoubleTensor",
        &DoBroadcastCudaOnCPU);
#endif

  // basics
  m.def("horovod_torch_poll", &PollHandle);
  m.def("horovod_torch_wait_and_clear", &WaitAndClear);

  //finished queue
  m.def("get_all_finished_keys", &GetFTKsFromQueue);
}

} // namespace torch
} // namespace horovod
