// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

#include <atomic>

#include "../common/operations.h"
#include "adapter.h"
#include "cuda_util.h"
#include "mpi_ops.h"
#include "ready_event.h"
#include "tensor_util.h"
#include <string>

namespace horovod {
namespace mxnet {

namespace {

std::atomic_int op_count;

std::string GetOpName(std::string prefix, char* name) {
  if (name != nullptr) {
    return prefix + "." + std::string(name);
  }

  op_count.fetch_add(1);
  return prefix + ".noname" + std::to_string(op_count);
}
} // namespace

inline void InvokeCompleteCallback(Callback on_complete, const Status& status) {
  if (status.ok()) {
    on_complete();
  } else {
    auto error = dmlc::Error(status.reason());
    on_complete(&error);
  }
}

void DoTimestamp(std::string& tensor_name, std::string& op_name, std::string& args,
                 Callback on_complete) {
  ThrowIfError(common::CheckInitialized());

  auto enqueue_result = EnqueueTimestamp(tensor_name, op_name, args);
  
  InvokeCompleteCallback(on_complete, enqueue_result);
  ThrowIfError(enqueue_result);
}

void DoAllreduce(NDArray* tensor, NDArray* output, const std::string& name, int batchid,
                 Callback on_complete) {
  ThrowIfError(common::CheckInitialized());

  EnqueueTimestamp(name, "SALLREDUCE", std::to_string(batchid));

  auto device = TensorUtil::GetDevice(tensor);
  auto hvd_tensor = std::make_shared<MXTensor<NDArray>>(tensor);
  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(device, output);
  auto hvd_output = std::make_shared<MXTensor<NDArray>>(output);

  auto enqueue_result =
      EnqueueTensorAllreduce(hvd_context, hvd_tensor, hvd_output, nullptr,
                             name, device,
                             [on_complete](const Status& status) {
                               InvokeCompleteCallback(on_complete, status);
                             }, batchid);
  ThrowIfError(enqueue_result);
}

#if HAVE_CUDA && !HOROVOD_GPU_ALLREDUCE
void DoAllreduceCudaOnCPU(NDArray* tensor, NDArray* output, std::string& name,
                          Callback on_complete) {
  ThrowIfError(common::CheckInitialized());

  // Make async copy of input tensor to CPU tensor and record completion event.
  auto hvd_cpu_buffer = std::make_shared<MXTemporaryBuffer<NDArray>>(
      CPU_DEVICE_ID, tensor->dtype());
  TensorUtil::AsyncCopyCudaToCPU(tensor, hvd_cpu_buffer->tensor());
  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(
      CPU_DEVICE_ID, hvd_cpu_buffer->tensor());
  auto ready_event = std::make_shared<MXReadyEvent<NDArray>>(tensor);

  auto enqueue_result = EnqueueTensorAllreduce(
      hvd_context, hvd_cpu_buffer, hvd_cpu_buffer, ready_event,
      name, CPU_DEVICE_ID,
      [hvd_cpu_buffer, output, on_complete](const Status& status) {
        TensorUtil::CopyCPUToCuda(hvd_cpu_buffer->tensor(), output);
        InvokeCompleteCallback(on_complete, status);
      });
  ThrowIfError(enqueue_result);
}
#endif

void DoAllgather(NDArray* tensor, NDArray* output, std::string& name,
                 Callback on_complete) {
  ThrowIfError(common::CheckInitialized());
  auto device = TensorUtil::GetDevice(tensor);
  auto hvd_tensor = std::make_shared<MXTensor<NDArray>>(tensor);
  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(device, output);
  auto enqueue_result =
      EnqueueTensorAllgather(hvd_context, hvd_tensor, nullptr,
                             name, device,
                             [on_complete](const Status& status) {
                               InvokeCompleteCallback(on_complete, status);
                             });
  ThrowIfError(enqueue_result);
}

void DoGatherBcast(NDArray* tensor, NDArray* output, std::string& name,
                 Callback on_complete) {
  ThrowIfError(common::CheckInitialized());
  auto device = TensorUtil::GetDevice(tensor);
  auto hvd_tensor = std::make_shared<MXTensor<NDArray>>(tensor);
  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(device, output);
  auto hvd_output = std::make_shared<MXTensor<NDArray>>(output);
  auto enqueue_result =
      EnqueueTensorGatherBcast(hvd_context, hvd_tensor, hvd_output, nullptr,
                             name, device,
                             [on_complete](const Status& status) {
                               InvokeCompleteCallback(on_complete, status);
                             });
  ThrowIfError(enqueue_result);
}

void DoGather(NDArray* tensor, NDArray* output, int root_rank, int num_elem, int batchid,
                 std::string& name, Callback on_complete) {
  ThrowIfError(common::CheckInitialized());

  // EnqueueTimestamp(name, "SGATHER", std::to_string(batchid));

  auto device = TensorUtil::GetDevice(tensor);
  auto hvd_tensor = std::make_shared<MXTensor<NDArray>>(tensor);
  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(device, output);
  auto hvd_output = std::make_shared<MXTensor<NDArray>>(output);

  auto enqueue_result = EnqueueTensorGather(
      hvd_context, hvd_tensor, hvd_output, root_rank,
      nullptr, name, device,
      [on_complete](const Status& status) {
        InvokeCompleteCallback(on_complete, status);
      }, num_elem, batchid);
  ThrowIfError(enqueue_result);
}

#if HAVE_CUDA
void DoAllgatherCudaOnCPU(NDArray* tensor, NDArray* output, std::string& name,
                          Callback on_complete) {
  ThrowIfError(common::CheckInitialized());

  // Make async copy of input tensor to CPU tensor and record completion event.
  auto hvd_cpu_tensor = std::make_shared<MXTemporaryBuffer<NDArray>>(
      CPU_DEVICE_ID, tensor->dtype());
  TensorUtil::AsyncCopyCudaToCPU(tensor, hvd_cpu_tensor->tensor());
  auto ready_event = std::make_shared<MXReadyEvent<NDArray>>(tensor);

  auto hvd_cpu_output = std::make_shared<MXTemporaryBuffer<NDArray>>(
      CPU_DEVICE_ID, output->dtype());
  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(
      CPU_DEVICE_ID, hvd_cpu_output->tensor());

  auto enqueue_result = EnqueueTensorAllgather(
      hvd_context, hvd_cpu_tensor, ready_event,
      name, CPU_DEVICE_ID,
      [hvd_cpu_output, output, on_complete](const Status& status) {
        TensorUtil::CopyCPUToCuda(hvd_cpu_output->tensor(), output);
        InvokeCompleteCallback(on_complete, status);
      });
  ThrowIfError(enqueue_result);
}
#endif

void DoBroadcast(NDArray* tensor, NDArray* output, int root_rank, int num_elem, int batchid,
                 std::string& name, Callback on_complete) {
  ThrowIfError(common::CheckInitialized());

  // EnqueueTimestamp(name, "SBCAST", std::to_string(batchid));

  auto device = TensorUtil::GetDevice(tensor);
  auto hvd_tensor = std::make_shared<MXTensor<NDArray>>(tensor);
  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(device, output);
  std::shared_ptr<Tensor> hvd_output = nullptr;
  if (horovod_rank() == root_rank) {
    if (tensor != output) {
      TensorUtil::Copy(output, tensor);
    }
  } else {
    hvd_output = std::make_shared<MXTensor<NDArray>>(output);
  }

  auto enqueue_result = EnqueueTensorBroadcast(
      hvd_context, hvd_tensor, hvd_output, root_rank,
      nullptr, name, device,
      [on_complete](const Status& status) {
        InvokeCompleteCallback(on_complete, status);
      }, num_elem, batchid);
  ThrowIfError(enqueue_result);
}

// broadcast the compressed data
void DoCBroadcast(NDArray* tensor, NDArray* output, int root_rank, int num_elem, int batchid,
                 std::string& name, Callback on_complete) {
  ThrowIfError(common::CheckInitialized());

  // EnqueueTimestamp(name, "SBCAST", std::to_string(batchid));

  auto device = TensorUtil::GetDevice(tensor);
  auto hvd_tensor = std::make_shared<MXTensor<NDArray>>(tensor);
  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(device, output);
  std::shared_ptr<Tensor> hvd_output = nullptr;
  if (horovod_rank() == root_rank) {
    if (tensor != output) {
      TensorUtil::Copy(output, tensor);
    }
  } else {
    hvd_output = std::make_shared<MXTensor<NDArray>>(output);
  }

  auto enqueue_result = EnqueueTensorCBroadcast(
      hvd_context, hvd_tensor, hvd_output, root_rank,
      nullptr, name, device,
      [on_complete](const Status& status) {
        InvokeCompleteCallback(on_complete, status);
      }, num_elem, batchid);
  ThrowIfError(enqueue_result);
}

#if HAVE_CUDA
void DoBroadcastCudaOnCPU(
    std::shared_ptr<MXTemporaryBuffer<NDArray>>& hvd_cpu_buffer, int root_rank, int num_elem, int batchid,
    std::string& name, Callback on_complete) {
  // Make async copy of input tensor to CPU tensor and record completion event.
  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(
      CPU_DEVICE_ID, hvd_cpu_buffer->tensor());
  auto ready_event =
      std::make_shared<MXReadyEvent<NDArray>>(hvd_cpu_buffer->tensor());

  auto enqueue_result = EnqueueTensorBroadcast(
      hvd_context, hvd_cpu_buffer, hvd_cpu_buffer, root_rank,
      ready_event, name, CPU_DEVICE_ID,
      [on_complete](const Status& status) {
        InvokeCompleteCallback(on_complete, status);
      }, num_elem, batchid);
  ThrowIfError(enqueue_result);
}
#endif

extern "C" int horovod_mxnet_allreduce_async(NDArray* input, NDArray* output,
                                             char* name, bool average, int batchid) {
  MX_API_BEGIN();

  std::string op_name = GetOpName("allreduce", name);
  auto allreduce_async_fn = [input, output,
                             op_name, batchid](RunContext rctx,
                                      Callback on_complete) mutable {
    DoAllreduce(input, output, op_name, batchid, on_complete);
  };

#if HAVE_CUDA && !HOROVOD_GPU_ALLREDUCE
  auto allreduce_async_cpu_fn =
      [input, output, op_name](RunContext rctx,
                               Callback on_complete) mutable {
        DoAllreduceCudaOnCPU(input, output, op_name, on_complete);
      };
#endif

#if HAVE_CUDA && !HOROVOD_GPU_ALLREDUCE
  // Not in-place
  if (input->var() != output->var()) {
    Engine::Get()->PushAsync(allreduce_async_cpu_fn, input->ctx(),
                             {input->var()}, {output->var()},
                             FnProperty::kNormal, 0, "HorovodAllreduce");
  // In-place
  } else {
    Engine::Get()->PushAsync(allreduce_async_cpu_fn, input->ctx(), {},
                             {output->var()}, FnProperty::kNormal, 0,
                             "HorovodAllreduce");
  }
#else
  // Not in-place
  if (input->var() != output->var()) {
    Engine::Get()->PushAsync(allreduce_async_fn, input->ctx(), {input->var()},
                             {output->var()}, FnProperty::kNormal, 0,
                             "HorovodAllreduce");
  // In-place
  } else {
    Engine::Get()->PushAsync(allreduce_async_fn, input->ctx(), {},
                             {output->var()}, FnProperty::kNormal, 0,
                             "HorovodAllreduce");
  }
#endif

  if (average) {
    *output /= horovod_size();
  }

  MX_API_END();
}

extern "C" int horovod_mxnet_allgather_async(NDArray* input, NDArray* output,
                                             char* name) {
  MX_API_BEGIN();

  std::string op_name = GetOpName("allgather", name);
  auto allgather_async_fn = [input, output,
                             op_name](RunContext rctx,
                                      Callback on_complete) mutable {
    DoAllgather(input, output, op_name, on_complete);
  };

#if HAVE_CUDA
  auto allgather_async_cpu_fn =
      [input, output, op_name](RunContext rctx,
                               Callback on_complete) mutable {
        DoAllgatherCudaOnCPU(input, output, op_name, on_complete);
      };
#endif

#if HAVE_CUDA && HOROVOD_GPU_ALLGATHER != 'N'
  // Not in-place
  if (input->var() != output->var()) {
    Engine::Get()->PushAsync(allgather_async_cpu_fn, input->ctx(),
                             {input->var()}, {output->var()},
                             FnProperty::kNormal, 0, "HorovodAllgather");
  // In-place
  } else {
    Engine::Get()->PushAsync(allgather_async_cpu_fn, input->ctx(), {},
                             {output->var()}, FnProperty::kNormal, 0,
                             "HorovodAllgather");
  }
#else
  // Not in-place
  if (input->var() != output->var()) {
    Engine::Get()->PushAsync(allgather_async_fn, input->ctx(),
                             {input->var()}, {output->var()},
                             FnProperty::kNormal, 0, "HorovodAllgather");
  // In-place
  } else {
    Engine::Get()->PushAsync(allgather_async_fn, input->ctx(), {},
                             {output->var()}, FnProperty::kNormal, 0,
                             "HorovodAllgather");
  }
#endif

  MX_API_END();
}

extern "C" int horovod_mxnet_gatherBcast_async(NDArray* input, NDArray* output,
                                             char* name) {
  MX_API_BEGIN();

  std::string op_name = GetOpName("gatherBcast", name);
  auto gatherBcast_async_fn = [input, output,
                             op_name](RunContext rctx,
                                      Callback on_complete) mutable {
    DoGatherBcast(input, output, op_name, on_complete);
  };
  // Not in-place
  if (input->var() != output->var()) {
    Engine::Get()->PushAsync(gatherBcast_async_fn, input->ctx(),
                             {input->var()}, {output->var()},
                             FnProperty::kNormal, 0, "HorovodGatherBcast");
  // In-place
  } else {
    Engine::Get()->PushAsync(gatherBcast_async_fn, input->ctx(), {},
                             {output->var()}, FnProperty::kNormal, 0,
                             "HorovodGatherBcast");
  }

  MX_API_END();
}

extern "C" int horovod_mxnet_gather_async(NDArray* input, NDArray* output, NDArray* dep,
                                          int root_rank, int num_elem, char* name, int batchid) {
  MX_API_BEGIN();

  std::string op_name = GetOpName("Gather", name);
  auto gather_async_fn = [input, output, root_rank,
                             op_name, num_elem, batchid](RunContext rctx,
                                      Callback on_complete) mutable {
    DoGather(input, output, root_rank, num_elem, batchid, op_name, on_complete);
  };
  // Not in-place
  if (input->var() != output->var()) {
    Engine::Get()->PushAsync(gather_async_fn, input->ctx(),
                             {input->var(), dep->var()}, {output->var()},
                             FnProperty::kNormal, 0, "HorovodGather");
  // In-place
  } else {
    Engine::Get()->PushAsync(gather_async_fn, input->ctx(), {dep->var()},
                             {output->var()}, FnProperty::kNormal, 0,
                             "HorovodGather");
  }

  MX_API_END();
}

extern "C" int horovod_mxnet_broadcast_async(NDArray* input, NDArray* output,
                                             int root_rank, int num_elem, char* name, int batchid) {
  MX_API_BEGIN();

  std::string op_name = GetOpName("Bcast", name);
  auto broadcast_async_fn = [input, output, op_name,
                             root_rank, num_elem, batchid](RunContext rctx,
                                        Callback on_complete) mutable {
    DoBroadcast(input, output, root_rank, num_elem, batchid, op_name, on_complete);
  };

#if HAVE_CUDA && HOROVOD_GPU_BROADCAST != 'M'
  ThrowIfError(common::CheckInitialized());
  // Make async copy of input tensor to CPU tensor and record completion event.
  auto hvd_cpu_buffer = std::make_shared<MXTemporaryBuffer<NDArray>>(
      CPU_DEVICE_ID, input->dtype());
  TensorUtil::AsyncCopyCudaToCPU(input, hvd_cpu_buffer->tensor());
  auto broadcast_async_cpu_fn =
        [hvd_cpu_buffer, op_name, root_rank, num_elem, batchid]
        (RunContext rctx, Callback on_complete) mutable {
          DoBroadcastCudaOnCPU(hvd_cpu_buffer, root_rank, num_elem, batchid, op_name, on_complete);
        };

  // Not in-place
  if (input->var() != output->var()) {
    Engine::Get()->PushAsync(broadcast_async_cpu_fn, input->ctx(),
                             {input->var()}, {output->var()},
                             FnProperty::kNormal, 0, "HorovodBroadcast");
  // In-place
  } else {
    Engine::Get()->PushAsync(broadcast_async_cpu_fn, input->ctx(), {},
                             {output->var()}, FnProperty::kNormal, 0,
                             "HorovodBroadcast");
  }

  TensorUtil::CopyCPUToCuda(hvd_cpu_buffer->tensor(), output);
#else
  // Not in-place
  if (input->var() != output->var()) {
    Engine::Get()->PushAsync(broadcast_async_fn, input->ctx(), {input->var()},
                             {output->var()}, FnProperty::kNormal, 0,
                             "HorovodBroadcast");
  // In-place
  } else {
    Engine::Get()->PushAsync(broadcast_async_fn, input->ctx(), {},
                             {output->var()}, FnProperty::kNormal, 0,
                             "HorovodBroadcast");
  }
#endif

  MX_API_END();
}

extern "C" int horovod_mxnet_mpibroadcast_async(NDArray* input, NDArray* output,
                                             int root_rank, int num_elem, char* name, int batchid) {
  MX_API_BEGIN();

  std::string op_name = GetOpName("CBcast", name);
  auto broadcast_async_fn = [input, output, op_name,
                             root_rank, num_elem, batchid](RunContext rctx,
                                        Callback on_complete) mutable {
    DoCBroadcast(input, output, root_rank, num_elem, batchid, op_name, on_complete);
  };

  // Not in-place
  if (input->var() != output->var()) {
    Engine::Get()->PushAsync(broadcast_async_fn, input->ctx(), {input->var()},
                             {output->var()}, FnProperty::kNormal, 0,
                             "HorovodBroadcast");
  // In-place
  } else {
    Engine::Get()->PushAsync(broadcast_async_fn, input->ctx(), {},
                             {output->var()}, FnProperty::kNormal, 0,
                             "HorovodBroadcast");
  }

  MX_API_END();
}

extern "C" int horovod_mxnet_timestamp_async(NDArray** read_tenor,
                                             int read_len,
                                             NDArray** write_tensor,
                                             int write_len,
                                             char* tensor_name, char* op_name, char* args) {
  MX_API_BEGIN();

  std::string t_name = std::string(tensor_name);
  std::string o_name = std::string(op_name);
  std::string a_name = std::string(args);
  auto timestamp_async_fn = [t_name, o_name, a_name](RunContext rctx,
                             Callback on_complete) mutable {
    DoTimestamp(t_name, o_name, a_name, on_complete);
  };

  std::vector<engine::VarHandle> read_vars(read_len);
  std::vector<engine::VarHandle> write_vars(write_len);

  for (int i = 0; i < read_len; i++) {
    read_vars[i] = read_tenor[i]->var();
  }
  
  for (int i = 0; i < write_len; i++) {
    write_vars[i] = write_tensor[i]->var();
  }

  // push timestamp operator to cpu prioritized task queue
  Engine::Get()->PushAsync(timestamp_async_fn, Context(), read_vars,
                            write_vars, FnProperty::kNormal, 0,
                            "HorovodTimestamp");

  MX_API_END();
}

} // namespace mxnet
} // namespace horovod
