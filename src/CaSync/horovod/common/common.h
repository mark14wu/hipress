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

#ifndef HOROVOD_COMMON_H
#define HOROVOD_COMMON_H

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "message.h"

namespace horovod {
namespace common {

// Activity names, see Horovod Timeline for more details.
#define INIT_FUSION_BUFFER "INIT_FUSION_BUFFER"
#define WAIT_FOR_DATA "WAIT_FOR_DATA"
#define WAIT_FOR_OTHER_TENSOR_DATA "WAIT_FOR_OTHER_TENSOR_DATA"
#define ALLOCATE_OUTPUT "ALLOCATE_OUTPUT"
#define MPI_CROSS_ALLGATHER "MPI_CROSS_ALLGATHER"
#define MPI_ALLGATHER "MPI_ALLGATHER"
#define INIT_NCCL "INIT_NCCL"
#define QUEUE "QUEUE"
#define MEMCPY_IN_FUSION_BUFFER "MEMCPY_IN_FUSION_BUFFER"
#define MEMCPY_IN_HOST_BUFFER "MEMCPY_IN_HOST_BUFFER"
#define MEMCPY_IN_SHARED_BUFFER "MEMCPY_IN_SHARED_BUFFER"
#define MPI_ALLREDUCE "MPI_ALLREDUCE"
#define MEMCPY_OUT_HOST_BUFFER "MEMCPY_OUT_HOST_BUFFER"
#define NCCL_ALLREDUCE "NCCL_ALLREDUCE"
#define MEMCPY_OUT_FUSION_BUFFER "MEMCPY_OUT_FUSION_BUFFER"
#define MPI_BCAST "MPI_BCAST"
#define MPI_GATHERBCAST "MPI_GATHERBCAST"
#define MPI_GATHER "MPI_GATHER"
#define MPI_ALLTOALL "MPI_ALLOALL"
#define MPI_BALLTOALL "MPI_BALLTOALL"
#define NCCL_REDUCESCATTER "NCCL_REDUCESCATTER"
#define NCCL_ALLGATHER "NCCL_ALLGATHER"
#define NCCL_REDUCE "NCCL_REDUCE"
#define NCCL_BCAST "NCCL_BCAST"
#define COPY_ALLGATHER_OUTPUT "COPY_ALLGATHER_OUTPUT"
#define ALLOCATE_SHARED_BUFFER "ALLOCATE_SHARED_BUFFER"

// Device ID used for CPU.
#define CPU_DEVICE_ID (-1)

// List of supported frameworks.
enum Framework { TENSORFLOW, PYTORCH, MXNET };

enum StatusType { OK, UNKNOWN_ERROR, PRECONDITION_ERROR, ABORTED, INVALID_ARGUMENT, IN_PROGRESS };

enum DeviceType { CPU, GPU };

enum Communicator {
  GLOBAL = 0,
  LOCAL = 1,
  CROSS = 2
};

inline std::string CommunicatorName(Communicator comm) {
  switch (comm) {
    case GLOBAL:
      return "global";
    case LOCAL:
      return "local";
    case CROSS:
      return "cross";
    default:
      return "<unknown>";
  }
}

class Status {
public:
  Status();
  static Status OK();
  static Status UnknownError(std::string message);
  static Status PreconditionError(std::string message);
  static Status Aborted(std::string message);
  static Status InvalidArgument(std::string message);
  static Status InProgress();
  bool ok() const;
  bool in_progress() const;
  StatusType type() const;
  const std::string& reason() const;

private:
  StatusType type_ = StatusType::OK;
  std::string reason_ = "";
  Status(StatusType type, std::string reason);
};

class TensorShape {
public:
  void AddDim(int64_t dim);
  void AppendShape(TensorShape& other);

  const std::string DebugString() const;
  int dims() const;
  int64_t dim_size(int idx) const;
  int64_t num_elements() const;

  inline bool operator==(const TensorShape& rhs) const {
    return shape_ == rhs.shape_;
  }

  inline bool operator!=(const TensorShape& rhs) const {
    return shape_ != rhs.shape_;
  }

private:
  std::vector<int64_t> shape_;
};

class ReadyEvent {
public:
  virtual bool Ready() const = 0;
  virtual ~ReadyEvent() = default;
};

class OpContext;

class PersistentBuffer {
public:
  virtual const void* AccessData(std::shared_ptr<OpContext> context) const = 0;
  virtual ~PersistentBuffer() = default;
};

class Tensor {
public:
  virtual const DataType dtype() const = 0;
  virtual const TensorShape shape() const = 0;
  virtual const void* data() const = 0;
  virtual int64_t size() const = 0;
  virtual ~Tensor() = default;
};

class OpContext {
public:
  // These allocators are fully synchronous, unlike TensorFlow counterparts.
  virtual Status
  AllocatePersistent(int64_t size,
                     std::shared_ptr<PersistentBuffer>* tensor) = 0;
  virtual Status AllocateOutput(TensorShape shape,
                                std::shared_ptr<Tensor>* tensor) = 0;
  virtual Framework framework() const = 0;
  virtual ~OpContext() = default;
};

// A callback to call after the MPI communication completes. Since the
// allreduce and allgather ops are asynchronous, this callback is what resumes
// computation after the reduction is completed.
using StatusCallback = std::function<void(const Status&)>;

// Table storing Tensors to be reduced, keyed by unique name.
// This table contains everything necessary to do the reduction.
struct TensorTableEntry {
  // Name of the tensor.
  std::string tensor_name;
  // Operation context.
  std::shared_ptr<OpContext> context;
  // Input tensor.
  std::shared_ptr<Tensor> tensor;
  // Pre-allocated output tensor.
  std::shared_ptr<Tensor> output;
  // Root rank for broadcast operation.
  int root_rank = 0;
  // Event indicating that data is ready.
  std::shared_ptr<ReadyEvent> ready_event;
  // GPU to do reduction on, or CPU_DEVICE_ID in case of CPU.
  int device = CPU_DEVICE_ID;
  // A callback to call with the status.
  StatusCallback callback;
  // number of elements in current tensor
  int num_elem = 0;
  // batch index
  int batchid = 0;
};
using TensorTable = std::unordered_map<std::string, TensorTableEntry>;

} // namespace common
} // namespace horovod

#endif // HOROVOD_COMMON_H
